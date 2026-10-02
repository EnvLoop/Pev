"""Score reports: the paired B0-vs-C report (with the gate) and the single-predictor reference report.

Aggregate-only reports (the sealed HIDDEN set) hold exactly AGGREGATE_KEYS: counts, per-family and macro metrics,
statistics, the gate and input hashes; never a record id, question id, path or per-question value.
"""

from . import gate as gate_module
from .metrics import (ECE_BINS, at_risk, families_of, family_accuracy, macro_accuracy, predictor_report,
                      safety_rates, score_questions)
from .stats import SAMPLES, SEED, mcnemar, paired_bootstrap, rate_delta_bootstrap

SCHEMA = "decision-eval-score-v1"
REFERENCE_SCHEMA = "decision-eval-reference-v1"
AGGREGATE_KEYS = ("schema", "gate", "inputs", "config", "n_questions", "n_states", "n_ambiguous", "families", "macro",
                  "safety", "bootstrap", "mcnemar", "guard", "aggregate_only")


def _pair(base, cand):
    return {"base": base, "cand": cand}


def question_rows(questions, base, cand):
    def correct(scored, i):
        return None if questions[i].ambiguous else bool(scored.correct[i])
    return [{"id": q.record_id, "qid": q.qid, "family": q.family, "state_id": q.state_id,
             "label": None if q.ambiguous else q.keys[q.label], "ambiguous": q.ambiguous,
             "base_pred": q.keys[int(base.prediction[i])], "cand_pred": q.keys[int(cand.prediction[i])],
             "base_conf": float(base.confidence[i]), "cand_conf": float(cand.confidence[i]),
             "base_correct": correct(base, i), "cand_correct": correct(cand, i)}
            for i, q in enumerate(questions)]


def _delta(base, cand):
    return None if base is None or cand is None else cand - base


def safety_report(questions, base, cand, samples, seed):
    """Per safety family: false-negative rates and the paired bootstrap CI of their delta (cand - base)."""
    base_rates, cand_rates, out = safety_rates(questions, base), safety_rates(questions, cand), {}
    for family in base_rates:
        index = at_risk(questions, family)
        base_miss = [int(base.prediction[i] != questions[i].label) for i in index]
        cand_miss = [int(cand.prediction[i] != questions[i].label) for i in index]
        boot = rate_delta_bootstrap([questions[i] for i in index], base_miss, cand_miss, samples, seed)
        out[family] = {"n_at_risk": len(index), "base_fn_rate": base_rates[family]["fn_rate"],
                       "cand_fn_rate": cand_rates[family]["fn_rate"], "fn_delta": boot["delta"],
                       "fn_delta_ci95": boot["ci95"]}
    return out


def guard_report(questions, base_logits, cand_logits):
    """Accuracy only (argmax is temperature-invariant): family-macro over the guard's families (kev `src`)."""
    base, cand = score_questions(questions, base_logits), score_questions(questions, cand_logits)
    base_acc, cand_acc = family_accuracy(questions, base), family_accuracy(questions, cand)
    counts = {name: len(index) for name, index in families_of(questions).items()}
    families = {name: {"n": counts[name], "base": base_acc[name], "cand": cand_acc[name]} for name in base_acc}
    base_macro, cand_macro = macro_accuracy(questions, base), macro_accuracy(questions, cand)
    return {"n": len(questions), "base_macro_accuracy": base_macro, "cand_macro_accuracy": cand_macro,
            "delta": cand_macro - base_macro, "families": families}


def paired_report(questions, logits, temperatures, thresholds, gate="none", guard=None, inputs=None,
                  aggregate_only=False, samples=SAMPLES, seed=SEED):
    """logits / temperatures / thresholds: {"base": ..., "cand": ...}; guard: (questions, base logits, cand logits)."""
    base = score_questions(questions, logits["base"], temperatures["base"])
    cand = score_questions(questions, logits["cand"], temperatures["cand"])
    base_report = predictor_report(questions, base, thresholds["base"])
    cand_report = predictor_report(questions, cand, thresholds["cand"])
    scorable = [q for q in questions if not q.ambiguous]
    boot = paired_bootstrap(scorable, base.correct[base.scorable], cand.correct[cand.scorable], samples, seed)
    families = {}
    for name, b in base_report["families"].items():
        c = cand_report["families"][name]
        ci = boot["families"].get(name, {}).get("ci95", [None, None])
        families[name] = {"n": b["n"], "n_ambiguous": b["n_ambiguous"], **_pair(b, c),
                          "accuracy_delta": _delta(b["accuracy"], c["accuracy"]), "accuracy_delta_ci95": ci}
    report = {
        "schema": SCHEMA,
        "inputs": inputs or {},
        "config": {"seed": seed, "samples": samples, "ece_bins": ECE_BINS, "budget": 0.05,
                   "temperatures": temperatures, "thresholds": thresholds},
        "n_questions": len(questions), "n_states": len({q.state_id for q in questions}),
        "families": families,
        "macro": {**_pair(base_report["macro"], cand_report["macro"]),
                  "accuracy_delta": cand_report["macro"]["accuracy"] - base_report["macro"]["accuracy"],
                  "micro": _pair(base_report["micro"], cand_report["micro"])},
        "safety": safety_report(questions, base, cand, samples, seed),
        "n_ambiguous": int((~base.scorable).sum()),
        "bootstrap": {k: v for k, v in boot.items() if k != "families"},   # per-family CIs sit in `families`
        "mcnemar": mcnemar(base.correct[base.scorable], cand.correct[cand.scorable]),
        "guard": guard_report(*guard) if guard is not None else None,
        "aggregate_only": aggregate_only,
    }
    report["gate"] = gate_module.decide(gate, report)
    if aggregate_only:
        report["inputs"] = {role: {"sha256": item["sha256"]} for role, item in report["inputs"].items()}
        return {key: report[key] for key in AGGREGATE_KEYS}
    report["questions"] = question_rows(questions, base, cand)
    return report


def reference_report(questions, logits, temperature=1.0, threshold=None, inputs=None):
    scored = score_questions(questions, logits, temperature)
    body = predictor_report(questions, scored, threshold)
    return {"schema": REFERENCE_SCHEMA, "inputs": inputs or {},
            "config": {"temperature": temperature, "threshold": threshold, "ece_bins": ECE_BINS},
            "n_questions": len(questions), "n_states": len({q.state_id for q in questions}),
            "n_ambiguous": sum(q.ambiguous for q in questions), "families": body["families"],
            "macro": body["macro"], "micro": body["micro"],
            "safety": safety_rates(questions, scored)}
