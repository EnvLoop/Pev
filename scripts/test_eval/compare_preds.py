"""Reproduction check: predictions on a DEV slice vs the frozen DEV predictions for the same questions.

Usage: python compare_preds.py NEW.jsonl REFERENCE.jsonl [--tol 0.05]
Reports, over the questions in NEW: argmax agreement, max / mean absolute difference of the raw logits and of the
softmax probabilities (T = 1). Exit 1 when an argmax differs or the max probability difference exceeds --tol.
"""
import argparse
import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        return {(r["id"], r["qid"]): r for r in map(json.loads, filter(str.strip, handle))}


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("reference")
    ap.add_argument("--tol", type=float, default=0.05, help="max allowed |p_new - p_ref| (bf16 batch effects)")
    a = ap.parse_args(argv)
    new, ref = load(a.new), load(a.reference)
    missing = [k for k in new if k not in ref]
    if missing:
        raise SystemExit(f"{len(missing)} questions absent from the reference")
    agree, dz, dp = 0, [], []
    for key, row in new.items():
        other = ref[key]
        if row["options"] != other["options"] or row["predictor"] != other["predictor"]:
            raise SystemExit("options or predictor differ")
        z, w = row["logits"], other["logits"]
        agree += max(range(len(z)), key=z.__getitem__) == max(range(len(w)), key=w.__getitem__)
        dz += [abs(x - y) for x, y in zip(z, w)]
        dp += [abs(x - y) for x, y in zip(softmax(z), softmax(w))]
    report = {"questions": len(new), "argmax_agree": agree, "logit_max_abs_diff": max(dz),
              "logit_mean_abs_diff": sum(dz) / len(dz), "prob_max_abs_diff": max(dp),
              "prob_mean_abs_diff": sum(dp) / len(dp), "tol": a.tol}
    report["passed"] = agree == len(new) and report["prob_max_abs_diff"] <= a.tol
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
