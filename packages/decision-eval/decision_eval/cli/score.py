"""score (paired B0 vs C, with the gate) and score-reference (one predictor, e.g. the gpt-6-astra reference on DEV).

`score` exits 0 whatever the gate decides: read `gate.passed`. With --aggregate-only (the sealed HIDDEN set) the output
holds no record id, question id, path or per-question value, and errors name counts only.
"""
import argparse

from ..predictions import load_aligned
from ..records import load_questions
from ..report import paired_report, reference_report
from ..stats import SAMPLES, SEED
from .common import inputs, print_json, temperature_arg, threshold_file, write_json


def _summary(report):
    lines = {"gate": report["gate"], "macro_accuracy_delta": report["macro"]["accuracy_delta"],
             "bootstrap": {k: report["bootstrap"][k] for k in ("delta", "ci95", "p_one_sided")}}
    if report["guard"] is not None:
        lines["guard_delta"] = report["guard"]["delta"]
    return lines


def score_main(argv=None):
    ap = argparse.ArgumentParser(prog="score", description="paired B0 vs C report and the pre-registered gate")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True, help="B0 predictions")
    ap.add_argument("--cand", required=True, help="C predictions (predict-base --adapter)")
    ap.add_argument("--base-temp", required=True, help="a number or B0's temperature.json")
    ap.add_argument("--cand-temp", required=True, help="a number or C's temperature.json")
    ap.add_argument("--base-thr", required=True, help="B0's thresholds json (fitted on VAL)")
    ap.add_argument("--cand-thr", required=True, help="C's thresholds json (fitted on VAL)")
    ap.add_argument("--gate", required=True, choices=("dev", "hidden", "none"))
    ap.add_argument("--guard-data", help="kev decision-v7 test records")
    ap.add_argument("--guard-base")
    ap.add_argument("--guard-cand")
    ap.add_argument("--aggregate-only", action="store_true", help="no per-question or per-id content (sealed data)")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    guard_args = (a.guard_data, a.guard_base, a.guard_cand)
    if any(guard_args) and not all(guard_args):
        raise SystemExit("--guard-data, --guard-base and --guard-cand go together")
    redact = a.aggregate_only
    questions = load_questions(a.data)
    base_logits, base_id = load_aligned(questions, a.base, redact)
    cand_logits, cand_id = load_aligned(questions, a.cand, redact)
    if base_id["predictor"] != "base" or cand_id["predictor"] != "sft":
        raise SystemExit("--base must hold predictor 'base' and --cand predictor 'sft' predictions")
    if base_id["template"] != cand_id["template"]:
        raise SystemExit("B0 and C must be scored with the same template")
    (base_t, base_t_sha), (cand_t, cand_t_sha) = temperature_arg(a.base_temp), temperature_arg(a.cand_temp)
    temperatures = {"base": base_t, "cand": cand_t}
    thresholds = {"base": threshold_file(a.base_thr, base_t), "cand": threshold_file(a.cand_thr, cand_t)}
    guard = None
    if a.guard_data:
        guard_questions = load_questions(a.guard_data)
        (guard_base, g_base_id), (guard_cand, g_cand_id) = (load_aligned(guard_questions, a.guard_base),
                                                            load_aligned(guard_questions, a.guard_cand))
        if (g_base_id["predictor"], g_cand_id["predictor"]) != ("base", "sft"):
            raise SystemExit("--guard-base must hold 'base' and --guard-cand 'sft' predictions")
        if not g_base_id["template"] == g_cand_id["template"] == base_id["template"]:
            raise SystemExit("the guard's B0 predictions must use the template B0 is scored with")
        guard = (guard_questions, guard_base, guard_cand)
    files = inputs(data=a.data, base=a.base, cand=a.cand, guard_data=a.guard_data, guard_base=a.guard_base,
                   guard_cand=a.guard_cand, base_thr=a.base_thr, cand_thr=a.cand_thr,
                   base_temp=a.base_temp if base_t_sha else None, cand_temp=a.cand_temp if cand_t_sha else None)
    report = paired_report(questions, {"base": base_logits, "cand": cand_logits}, temperatures, thresholds, gate=a.gate,
                           guard=guard, inputs=files, aggregate_only=a.aggregate_only, samples=a.samples, seed=a.seed)
    report["config"]["templates"] = {"base": base_id["template"], "cand": cand_id["template"]}
    write_json(a.out, report)
    print_json(_summary(report))


def score_reference_main(argv=None):
    ap = argparse.ArgumentParser(prog="score-reference", description="metrics of one predictor (report only)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--temperature", default="1.0", help="a number or a temperature.json (default 1.0: as returned)")
    ap.add_argument("--thr", default=None, help="optional thresholds json fitted on VAL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    questions = load_questions(a.data)
    logits, identity = load_aligned(questions, a.preds)
    temperature, _ = temperature_arg(a.temperature)
    threshold = threshold_file(a.thr, temperature) if a.thr else None
    report = reference_report(questions, logits, temperature, threshold, inputs(data=a.data, preds=a.preds, thr=a.thr))
    report["config"].update(identity)
    write_json(a.out, report)
    print_json({"macro": report["macro"], "safety": report["safety"]})
