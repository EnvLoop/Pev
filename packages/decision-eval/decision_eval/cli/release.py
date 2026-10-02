"""compare-release: amendment 6's TEST scoring of all models from one spec file.

Spec (JSON): {"candidate": "adapter", "models": [{"name", "label", "preds", "temperature": number|temperature.json,
"thresholds": thresholds.json|null, "local": bool}, ...]}. Relative paths resolve against the spec's directory.
Writes `<out-dir>/score.<name>.json` per model (overall + halves) and `<out-dir>/results.json` (all models, the
candidate's paired comparisons with Holm per subset, input hashes) and, with --test-results, the tech report's
TEST_RESULTS.json (`muse-test-comparison/1`, keyed by each model's label). Never prints per-question content.
Input paths are recorded relative to the repository root (never as absolute paths); see `recorded_path`.
"""
import argparse
import os
from pathlib import Path

from ..predictions import load_aligned
from ..records import load_questions, read_jsonl, sha256_file
from ..release import HALVES, SCHEMA, compare, record_halves, test_comparison
from ..stats import SAMPLES, SEED
from .common import print_json, read_json, temperature_arg, threshold_file, write_json


def _path(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def repo_root(start=None):
    """The nearest ancestor of `start` (default: the working directory) holding `.git`, else `start` itself."""
    start = Path(start or Path.cwd()).resolve()
    return next((folder for folder in (start, *start.parents) if (folder / ".git").exists()), start)


def recorded_path(path, root=None):
    """How an input path is written to the reports: relative to the repository root when inside it, else relative to
    the working directory when inside that, else the bare file name, so no home directory ever lands in a report."""
    resolved = Path(os.path.normpath(Path(path).resolve()))
    for base in (Path(root).resolve() if root else repo_root(), Path.cwd().resolve()):
        if resolved.is_relative_to(base):
            return resolved.relative_to(base).as_posix()
    return resolved.name


def load_models(spec_path, questions):
    spec = read_json(spec_path)
    base, models, files = Path(spec_path).resolve().parent, {}, {}
    for entry in spec["models"]:
        name = entry["name"]
        if name in models:
            raise SystemExit(f"duplicate model {name!r}")
        preds = _path(base, entry["preds"])
        logits, identity = load_aligned(questions, preds)
        temp_value = entry.get("temperature", 1.0)
        temp_arg = str(_path(base, temp_value)) if isinstance(temp_value, str) else str(temp_value)
        temperature, temp_sha = temperature_arg(temp_arg)
        thr_path = _path(base, entry["thresholds"]) if entry.get("thresholds") else None
        threshold = threshold_file(thr_path, temperature) if thr_path else None
        local = bool(entry.get("local", False))
        if local and thr_path is None:
            raise SystemExit(f"local model {name!r} needs its VAL thresholds")
        models[name] = {"logits": logits, "temperature": temperature, "threshold": threshold, "local": local,
                        "label": entry.get("label", name), "predictor": identity["predictor"],
                        "template": identity["template"]}
        files[name] = {"preds": {"path": recorded_path(preds), "sha256": sha256_file(preds)},
                       "temperature_sha256": temp_sha,
                       "thresholds": {"path": recorded_path(thr_path), "sha256": sha256_file(thr_path)}
                       if thr_path else None}
    return spec["candidate"], models, files


def _renderer_overrides(values):
    out = {}
    for item in values or []:
        renderer, _, half = item.partition("=")
        if half not in HALVES:
            raise SystemExit(f"--renderer-half takes RENDERER=A|B, got {item!r}")
        out[renderer] = half
    return out


def compare_release_main(argv=None):
    ap = argparse.ArgumentParser(prog="compare-release", description="amendment 6: all models on TEST, paired + Holm")
    ap.add_argument("--data", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--renderer-half", action="append", help="RENDERER=A|B (default: astra -> A, claude/opus -> B)")
    ap.add_argument("--test-results", help="also write the tech report's TEST_RESULTS.json (muse-test-comparison/1)")
    ap.add_argument("--commitment", help="TEST_COMMITMENT.json, recorded by SHA-256 in --test-results")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    questions = load_questions(a.data)
    halves = record_halves(read_jsonl(a.data), _renderer_overrides(a.renderer_half))
    candidate, models, files = load_models(a.spec, questions)
    result = compare(questions, models, candidate, halves, a.samples, a.seed)
    out = Path(a.out_dir)
    config = {"samples": a.samples, "seed": a.seed, "candidate": candidate, "halves": HALVES,
              "holm": "step-down over the candidate's comparisons, separately per subset (overall, A, B)",
              "temperatures": {n: m["temperature"] for n, m in models.items()},
              "thresholds": {n: m["threshold"] for n, m in models.items()},
              "local": {n: m["local"] for n, m in models.items()}}
    data = {"path": recorded_path(a.data), "sha256": sha256_file(a.data), "subsets": result["subset_sizes"]}
    for name, model in models.items():
        write_json(out / f"score.{name}.json",
                   {"schema": SCHEMA + "/model", "model": name, "label": model["label"], "data": data,
                    "predictor": model["predictor"], "template": model["template"], "inputs": files[name],
                    "temperature": model["temperature"], "threshold": model["threshold"], "local": model["local"],
                    "subsets": result["models"][name]})
    report = {"schema": SCHEMA, "data": data, "config": config, "inputs": files,
              "labels": {n: m["label"] for n, m in models.items()},
              "models": result["models"], "comparisons": result["comparisons"], "sensitivity": result["sensitivity"]}
    write_json(out / "results.json", report)
    if a.test_results:
        write_json(a.test_results, test_comparison(report, data["sha256"],
                                                   sha256_file(a.commitment) if a.commitment else None))
    print_json(summary(report))


def summary(report):
    """Macro accuracy per model and subset, and the candidate's pairs (delta, CI, p, Holm p)."""
    table = {name: {subset: (None if body is None else round(body["macro"]["accuracy"], 4))
                    for subset, body in subsets.items()} for name, subsets in report["models"].items()}
    pairs = {}
    for subset, block in report["comparisons"].items():
        if block is None:
            continue
        pairs[subset] = {other: {"delta": round(row["delta"], 4), "ci95": [round(x, 4) for x in row["ci95"]],
                                 "p": row["p_one_sided"], "p_holm": row["p_holm"]}
                         for other, row in block["pairs"].items()}
    sens = report["sensitivity"]
    table6 = {name: {subset: (None if body is None else round(body["macro_accuracy"], 4))
                     for subset, body in subsets.items()} for name, subsets in sens["models"].items()}
    pairs6 = {subset: {other: {"delta": round(row["delta"], 4), "ci95": [round(x, 4) for x in row["ci95"]],
                               "p": row["p_one_sided"], "p_holm": row["p_holm"]}
                       for other, row in block["pairs"].items()}
              for subset, block in sens["comparisons"].items() if block is not None}
    return {"macro_accuracy": table, "candidate_minus_other": pairs, "subsets": report["data"]["subsets"],
            "sensitivity_excl": sens["excluded_families"], "macro_accuracy_6fam": table6,
            "candidate_minus_other_6fam": pairs6}
