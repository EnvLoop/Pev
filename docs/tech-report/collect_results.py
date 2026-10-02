"""Snapshot the aggregate result files the tech report cites from the ignored `work/` tree into `data/`.

Only aggregates are copied: per-question rows (`questions` in the DEV score reports) are dropped, nothing is read
from `work/muse/sealed*` or the TEST build, and the HIDDEN aggregate is the `--aggregate-only` report of the one-shot
gate (its SHA-256 is checked against the job receipt). `data/sources.json` records every source path and SHA-256.
The `work/` inputs are the authors' private run records and are not distributed; the script documents how the
snapshots were made. Infrastructure identifiers (job ids, internal record paths) are not copied, and paths are made
relative to the repository.

    python3 docs/tech-report/collect_results.py [WORK_DIR]   # default <repo>/work; stdlib only
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "work"  # optional: work/ of another checkout
OUT = Path(__file__).resolve().parent / "data"
ROUNDS = {  # display name -> run directory under work/intents
    "r1-seedA": "r1-sky5-s20260930",
    "r1-seedB": "r1-sky5-s20261003",
    "r2-mix": "r2-mix-s20260930",
    "r2-7k": "r2-7k-s20260930",
}
HIDDEN = "intents/hidden-r2-mix-run/report"
SOURCES: dict[str, str] = {}


def load(rel: str) -> dict:
    path = WORK / rel
    data = path.read_bytes()
    SOURCES[f"work/{rel}"] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def scrub(obj):
    """Local absolute paths -> paths relative to the repository (no home directory in the snapshot)."""
    if isinstance(obj, dict):
        return {key: scrub(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [scrub(value) for value in obj]
    if isinstance(obj, str) and str(ROOT) in obj:
        return obj.replace(str(ROOT) + "/", "")
    return obj


def pick(obj: dict, *keys: str) -> dict:
    return {key: obj[key] for key in keys if key in obj}


def rounds() -> dict:
    base = load("intents/baseline-1.intent.result.json")["baseline_evaluation"]
    out = {"baseline": pick(base, "selected_template", "template_scores", "noise_floor", "validity", "references",
                            "dev_sha256", "val_sha256", "guard_sha256")}
    out["baseline"]["base_dev"] = base["base_dev"]
    for name, intent in ROUNDS.items():
        score = load(f"intents/{intent}/report/dev-score.json")
        decision = load(f"intents/{intent}/report/classification-result.json")
        recipe = load(f"intents/{intent}/recipe.json")
        report = load(f"intents/{intent}/report.json")
        entry = pick(score, "bootstrap", "config", "families", "gate", "guard", "macro", "mcnemar", "n_ambiguous",
                     "n_questions", "n_states", "safety", "schema")
        entry["decision"] = decision["decision"]
        entry["recipe"] = {"name": recipe["name"], "train_rows": recipe["dataset"]["train"]["count"],
                           "train_records": recipe["dataset"]["records"]["train"]["count"],
                           "max_steps": recipe["training"]["max_steps"], "seed": recipe["training"]["seed"]}
        entry["job"] = pick(report, "gpu_class", "status")
        receipt = WORK / f"intents/{intent}/report/receipt.json"
        if receipt.exists():
            data = load(f"intents/{intent}/report/receipt.json")
            entry["job"]["wall_hours"] = round((data["finished_epoch"] - data["started_epoch"]) / 3600, 3)
            entry["job"]["device"] = data["device"]
        out[name] = entry
    return out


def dataset() -> dict:
    split = load("muse/kb/split_manifest.json")
    gen = load("muse/data/GEN_RECEIPT.v2-5k.json")
    mix = load("muse/data/GEN_RECEIPT.r2-mix.json")
    dev = load("muse/eval-dev/DEV_RECEIPT.json")
    hidden = load("muse/HIDDEN_COMMITMENT.json")
    build = load("muse/kb/BUILD_RECEIPT.json")
    splits = {name: {k: v for k, v in info.items() if k not in ("file", "style_counts")}
              for name, info in gen["splits"].items()}
    return {
        "raw_stats": load("muse/raw/STATS.json")["row_counts"],
        "population": load("muse/kb/population_receipt.json"),
        "kb_extractor": pick(build["extractor"], "facts_llm", "facts_rule", "facts_total", "share_llm_of_all_facts",
                             "total_input_tokens", "total_output_tokens", "total_requests", "mode"),
        "split_users": {name: shard["users"] for name, shard in split["shards"].items()},
        "split_leakage": split["leakage"],
        "train_val": {"generator_commit": gen["generator_commit"], "family_weights": gen["family_weights"],
                      "p_buried": gen["p_buried"], "seeds": gen["seeds"], "splits": splits,
                      "render_tokens_all_generation": gen["render_tokens_all_generation"]},
        "general_mix": pick(mix, "seed", "general_share_target", "source", "leakage_audit", "families",
                            "general_mix", "train", "mix"),
        "dev": pick(dev, "anchors", "balance", "counts", "generator", "noise_floor_oracle_free", "renderer_model",
                    "seed", "sha256", "shortcut_baselines", "styles_public", "tokens"),
        "hidden_commitment": hidden,
        # the public snapshot carries the cleaned public copy (docs/TEST_COMMITMENT.json; data-file hashes unchanged)
        "test_commitment": load("muse/TEST_COMMITMENT.json"),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    aggregate = load(f"{HIDDEN}/hidden-aggregate.json")
    receipt = load(f"{HIDDEN}/receipt.json")
    expected = receipt["objects"]["attempt-1/hidden-aggregate.json"]["sha256"]
    if SOURCES[f"work/{HIDDEN}/hidden-aggregate.json"] != expected or aggregate.get("aggregate_only") is not True:
        raise SystemExit("hidden aggregate does not match its job receipt or is not aggregate-only")
    hidden = {"aggregate": aggregate, "classification": load(f"{HIDDEN}/hidden-classification-result.json"),
              "job": pick(load("intents/hidden-r2-mix-run/report.json"), "gpu_class",
                          "independent_improvement_verified", "status", "evaluation"),
              "wall_hours": round((receipt["finished_epoch"] - receipt["started_epoch"]) / 3600, 3)}
    hidden["job"].get("evaluation", {}).pop("consumption_record_path", None)   # internal record path
    data = dataset()
    tokens = data["dev"].get("tokens", {})
    for key in [k for k in tokens if k.endswith("_errors_concurrency4_phase") and not k.startswith("api_")]:
        tokens["api_errors_concurrency4_phase"] = tokens.pop(key)   # neutral key name in the public snapshot
    files = {"dev-rounds.json": rounds(), "dataset.json": data, "hidden.json": hidden}
    for name, payload in files.items():
        (OUT / name).write_text(json.dumps(scrub(payload), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "sources.json").write_text(json.dumps(dict(sorted(SOURCES.items())), indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": sorted(files), "sources": len(SOURCES)}))


if __name__ == "__main__":
    main()
