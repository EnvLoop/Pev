"""VAL-only fits: select-template (B0's prompt), calibrate (one temperature), thresholds (automation threshold)."""
import argparse

from ..calibration import BUDGET, fit_temperature, fit_threshold
from ..metrics import family_accuracy, score_questions
from ..predictions import load_aligned
from ..records import load_questions, sha256_file
from .common import print_json, temperature_arg, write_json


def select_template(questions, preds_paths):
    """Best family-macro accuracy on VAL; ties go to the first file given."""
    candidates = []
    for path in preds_paths:
        logits, identity = load_aligned(questions, path)
        per_family = family_accuracy(questions, score_questions(questions, logits))   # ambiguous questions left out
        candidates.append({"template": identity["template"], "preds": str(path), "preds_sha256": sha256_file(path),
                           "macro_accuracy": sum(per_family.values()) / len(per_family), "families": per_family})
    templates = [c["template"] for c in candidates]
    if len(set(templates)) != len(templates):
        raise SystemExit("select-template needs prediction files of distinct templates")
    best = max(range(len(candidates)), key=lambda i: (candidates[i]["macro_accuracy"], -i))
    return {"chosen": candidates[best]["template"], "chosen_preds": candidates[best]["preds"], "candidates": candidates}


def select_template_main(argv=None):
    ap = argparse.ArgumentParser(prog="select-template", description="pick B0's template by VAL macro accuracy")
    ap.add_argument("--data", required=True, help="VAL records")
    ap.add_argument("--preds", required=True, nargs="+", help="one base prediction file per template, in order")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = {**select_template(load_questions(a.data), a.preds), "data_sha256": sha256_file(a.data)}
    if a.out:
        write_json(a.out, result)
    print_json(result)


def calibrate_main(argv=None):
    ap = argparse.ArgumentParser(prog="calibrate", description="fit one temperature on VAL by NLL")
    ap.add_argument("--data", required=True, help="VAL records")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    questions = load_questions(a.data)
    logits, identity = load_aligned(questions, a.preds)
    result = {**fit_temperature(questions, logits), **identity, "n": len(questions),
              "data_sha256": sha256_file(a.data), "preds_sha256": sha256_file(a.preds)}
    write_json(a.out, result)
    print_json(result)


def thresholds_main(argv=None):
    ap = argparse.ArgumentParser(prog="thresholds", description="lowest max-prob threshold within the VAL error budget")
    ap.add_argument("--data", required=True, help="VAL records")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--temperature", required=True, help="a number or the temperature.json from calibrate")
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not 0 <= a.budget <= 1:
        raise SystemExit("--budget must be in [0, 1]")
    questions = load_questions(a.data)
    logits, identity = load_aligned(questions, a.preds)
    temperature, temperature_sha = temperature_arg(a.temperature)
    result = {**fit_threshold(questions, logits, temperature, a.budget), **identity,
              "temperature_sha256": temperature_sha, "data_sha256": sha256_file(a.data),
              "preds_sha256": sha256_file(a.preds)}
    write_json(a.out, result)
    print_json(result)
