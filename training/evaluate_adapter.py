"""Standalone evaluation of a LoRA adapter against its base model with packages/decision-eval (the pre-registered
protocol, docs/PREREGISTRATION.md): B0 = base, C = base + adapter, same scorer and frozen template.

    uv run --project packages/decision-eval --extra fast python training/evaluate_adapter.py \
        --adapter runs/r2-mix/adapter --val data/val.jsonl --test data/test.jsonl --out eval/test \
        [--guard-data kev/evals/v7/decision-v7/test.jsonl] [--gate dev|none]

Steps (each skipped when its output already exists, so a failed run resumes):
1. predict-base on VAL and on the evaluation set, without and with the adapter;
2. calibrate (one temperature per predictor, fitted on VAL) and thresholds (5% error budget, on VAL);
3. score: paired family-macro report C vs B0 (state-cluster bootstrap, McNemar, safety false negatives, calibration,
   automation coverage), plus score-reference for each predictor alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL, REVISION, TEMPLATE = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "direct"


def plan(a) -> list[tuple[Path, list[str]]]:
    """(output, decision-eval argv) in order."""
    out, steps = a.out, []
    common = ["--model", a.model, "--revision", a.revision, "--template", a.template, "--device", a.device,
              "--dtype", a.dtype, "--batch", str(a.batch)]
    sets = {"val": a.val, "eval": a.data} | ({"guard": a.guard_data} if a.guard_data else {})
    for name, data in sets.items():
        steps.append((out / f"{name}.base.jsonl", ["predict-base", *common, "--data", str(data),
                                                   "--out", str(out / f"{name}.base.jsonl")]))
        steps.append((out / f"{name}.sft.jsonl", ["predict-base", *common, "--data", str(data), "--adapter",
                                                  str(a.adapter), "--out", str(out / f"{name}.sft.jsonl")]))
    for who in ("base", "sft"):
        temperature = out / f"temperature.{who}.json"
        steps.append((temperature, ["calibrate", "--data", str(a.val), "--preds", str(out / f"val.{who}.jsonl"),
                                    "--out", str(temperature)]))
        steps.append((out / f"thresholds.{who}.json",
                      ["thresholds", "--data", str(a.val), "--preds", str(out / f"val.{who}.jsonl"), "--temperature",
                       str(temperature), "--budget", str(a.budget), "--out", str(out / f"thresholds.{who}.json")]))
        steps.append((out / f"reference.{who}.json",
                      ["score-reference", "--data", str(a.data), "--preds", str(out / f"eval.{who}.jsonl"),
                       "--temperature", str(temperature), "--thr", str(out / f"thresholds.{who}.json"),
                       "--out", str(out / f"reference.{who}.json")]))
    score = ["score", "--data", str(a.data), "--base", str(out / "eval.base.jsonl"),
             "--cand", str(out / "eval.sft.jsonl"), "--base-temp", str(out / "temperature.base.json"),
             "--cand-temp", str(out / "temperature.sft.json"), "--base-thr", str(out / "thresholds.base.json"),
             "--cand-thr", str(out / "thresholds.sft.json"), "--gate", a.gate, "--out", str(out / "report.json")]
    if a.guard_data:
        score += ["--guard-data", str(a.guard_data), "--guard-base", str(out / "guard.base.jsonl"),
                  "--guard-cand", str(out / "guard.sft.jsonl")]
    steps.append((out / "report.json", score))
    return steps


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True, help="VAL records (temperature and thresholds)")
    ap.add_argument("--data", "--test", type=Path, required=True, help="the evaluation records (DEV or TEST)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--guard-data", type=Path, help="Kev decision-v7 test (regression guard; needed for --gate dev)")
    ap.add_argument("--gate", choices=("dev", "none"), default="none")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--template", default=TEMPLATE, help="frozen on VAL before training (amendment 1)")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print the decision-eval commands only")
    return ap


def main(argv=None) -> None:
    a = parser().parse_args(argv)
    if a.gate == "dev" and not a.guard_data:
        raise SystemExit("--gate dev needs --guard-data (the decision-v7 guard is a required check)")
    a.out.mkdir(parents=True, exist_ok=True)
    for output, command in plan(a):
        if output.exists():
            continue
        print("decision-eval " + " ".join(command), flush=True)
        if not a.dry_run:
            from decision_eval.__main__ import main as decision_eval
            decision_eval(command)


if __name__ == "__main__":
    main()
