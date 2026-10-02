"""Amendment 6 (the public TEST set): every model scored once with the same metrics as DEV/HIDDEN, and the candidate
(the frozen adapter) compared with each of the other models.

Per model and per subset (overall, half A = gpt-6-astra-rendered, half B = Claude-rendered): family-macro accuracy and
per-family accuracy, Brier / ECE (API models as returned, T = 1: uncalibrated), safety false-negative rates, and the
automation coverage at the VAL-fitted threshold (local models only; None for API models). Pairs: candidate - other on
the state-clustered paired bootstrap (decision_eval.stats: 10,000 resamples, seed 20260930, family-macro accuracy
recomputed per resample, one-sided p for H0: delta <= 0), with Holm's step-down adjustment over the comparisons of
each subset. A record's half comes from its id prefix (`muse/test-a/...` = A, `muse/test-b/...` = B, as the TEST
builder writes them), else from its renderer (`meta.render.model`).
"""
from collections import Counter

import numpy as np

from .metrics import predictor_report, safety_rates, score_questions
from .stats import SAMPLES, SEED, paired_bootstrap

SCHEMA = "decision-eval-release-v1"
HALVES = {"A": "gpt-6-astra-rendered", "B": "Claude-rendered"}
AUTOMATION_KEYS = ("coverage", "accepted", "errors", "realized_error", "coverage_at_5pct_in_sample")


def half_of_renderer(model, overrides=None):
    """'A' for a gpt-6-astra render, 'B' for a Claude render; an explicit {renderer: half} map wins."""
    if overrides and model in overrides:
        return overrides[model]
    name = str(model or "").lower()
    if "astra" in name:
        return "A"
    if "claude" in name or "opus" in name:
        return "B"
    return None


ID_PREFIXES = {"muse/test-a/": "A", "muse/test-b/": "B"}


def half_of_id(record_id):
    return next((half for prefix, half in ID_PREFIXES.items() if str(record_id).startswith(prefix)), None)


def record_halves(records, overrides=None):
    """{record id: half}, by id prefix, else by renderer; refuses records it cannot place (counts only, no ids) and
    a record whose id prefix and renderer disagree."""
    out, unknown, conflicts = {}, Counter(), 0
    for record in records:
        renderer = ((record.get("meta") or {}).get("render") or {}).get("model")
        by_id, by_renderer = half_of_id(record["id"]), half_of_renderer(renderer, overrides)
        conflicts += bool(by_id and by_renderer and by_id != by_renderer)
        half = by_id or by_renderer
        if half not in HALVES:
            unknown[str(renderer)] += 1
            continue
        out[str(record["id"])] = half
    if conflicts:
        raise ValueError(f"{conflicts} records whose id prefix and renderer name different halves")
    if unknown:
        raise ValueError(f"records with an unplaceable renderer: {dict(unknown)} (pass --renderer-half)")
    return out


def holm(p_values):
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running, m = [0.0] * len(p_values), 0.0, len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _scored(questions, logits, temperature):
    return score_questions(questions, logits, temperature)


def model_metrics(questions, scored, threshold, local):
    """One model on one subset: the DEV report body plus safety false negatives."""
    report = predictor_report(questions, scored, threshold)
    if not local:   # API models: no VAL fit, so no automation threshold or coverage
        for block in (report["macro"], report["micro"], *report["families"].values()):
            for key in AUTOMATION_KEYS:
                if key in block:
                    block[key] = None
    return {"n_questions": len(questions), "n_states": len({q.state_id for q in questions}),
            "n_ambiguous": sum(q.ambiguous for q in questions), "macro": report["macro"],
            "families": report["families"], "micro": report["micro"], "safety": safety_rates(questions, scored)}


def _subset(index, items):
    return [items[i] for i in index]


SENSITIVITY_EXCLUDED = ("pick_option",)   # amendment 7: the residual pick_option shortcut on TEST


def compare(questions, models, candidate, halves, samples=SAMPLES, seed=SEED, exclude=SENSITIVITY_EXCLUDED):
    """questions: all TEST questions; models: {name: {"logits", "temperature", "threshold", "local", ...}};
    halves: {record id: "A"|"B"} -> {"models": {name: {subset: metrics}}, "comparisons": {subset: {other: ...}},
    "sensitivity": the same macro accuracy and pairs without the `exclude` families (amendment 7)}."""
    if candidate not in models:
        raise ValueError(f"candidate {candidate!r} is not among the models")
    subsets = {"overall": list(range(len(questions)))}
    for half in HALVES:
        subsets[half] = [i for i, q in enumerate(questions) if halves.get(q.record_id) == half]
    scored = {name: _scored(questions, m["logits"], m["temperature"]) for name, m in models.items()}
    out_models, comparisons = {name: {} for name in models}, {}
    sensitivity = {"excluded_families": list(exclude), "models": {name: {} for name in models}, "comparisons": {}}
    for subset, index in subsets.items():
        if not index:
            comparisons[subset] = sensitivity["comparisons"][subset] = None
            for name in models:
                out_models[name][subset] = sensitivity["models"][name][subset] = None
            continue
        sub_q = _subset(index, questions)
        for name, m in models.items():
            sub_scored = _restrict(scored[name], index)
            out_models[name][subset] = model_metrics(sub_q, sub_scored, m["threshold"], m["local"])
        comparisons[subset] = _pairs(sub_q, {n: _restrict(s, index) for n, s in scored.items()}, candidate,
                                     samples, seed)
        kept = [i for i in index if questions[i].family not in exclude]
        for name in models:
            accuracies = [m["accuracy"] for f, m in out_models[name][subset]["families"].items()
                          if f not in exclude and m["accuracy"] is not None]
            sensitivity["models"][name][subset] = {"macro_accuracy": float(np.mean(accuracies)),
                                                   "families": len(accuracies)}
        sensitivity["comparisons"][subset] = _pairs(_subset(kept, questions),
                                                    {n: _restrict(s, kept) for n, s in scored.items()}, candidate,
                                                    samples, seed)
    return {"models": out_models, "comparisons": comparisons, "sensitivity": sensitivity,
            "subset_sizes": {k: {"questions": len(v), "states": len({questions[i].state_id for i in v})}
                             for k, v in subsets.items()}}


def _restrict(scored, index):
    index = np.asarray(index, dtype=int)
    return type(scored)(probs=[scored.probs[i] for i in index], confidence=scored.confidence[index],
                        prediction=scored.prediction[index], correct=scored.correct[index],
                        brier=scored.brier[index], nll=scored.nll[index], scorable=scored.scorable[index],
                        expected_correct=scored.expected_correct[index])


def _pairs(questions, scored, candidate, samples, seed):
    scorable_index = [i for i, q in enumerate(questions) if not q.ambiguous]
    scorable = _subset(scorable_index, questions)
    cand = scored[candidate].correct[scorable_index]
    rows = {}
    for other in (name for name in scored if name != candidate):
        boot = paired_bootstrap(scorable, scored[other].correct[scorable_index], cand, samples, seed)
        rows[other] = {"delta": boot["delta"], "ci95": boot["ci95"], "p_one_sided": boot["p_one_sided"],
                       "families": boot["families"], "clusters": boot["clusters"]}
    adjusted = holm([row["p_one_sided"] for row in rows.values()])
    for row, p in zip(rows.values(), adjusted):
        row["p_holm"] = p
    return {"candidate": candidate, "holm_m": len(rows), "pairs": rows}


TEST_SCHEMA = "muse-test-comparison/1"
SENSITIVITY_FIELD = "sensitivity_excl_pick_option"
ALPHA = 0.05
SUBSET_KEYS = {"overall": "all", "A": "A", "B": "B"}


def _predictor_block(body):
    macro = body["macro"]
    block = {"macro_accuracy": macro["accuracy"], "macro_brier": macro["brier"], "macro_ece": macro["ece"],
             "families": {f: {"accuracy": m["accuracy"], "n_scored": m["n_scored"]}
                          for f, m in body["families"].items()},
             "safety_fn_rate": {f: s["fn_rate"] for f, s in body["safety"].items()}}
    if macro.get("coverage") is not None:
        block["automation_coverage"] = macro["coverage"]
    return block


def test_comparison(report, test_sha256, commitment_sha256=None):
    """The tech report's TEST_RESULTS.json ("muse-test-comparison/1", docs/tech-report/make_figures.py): predictors
    and pairs keyed by display label, halves "all" / "A" / "B" (an empty half is omitted)."""
    labels, candidate = report["labels"], report["config"]["candidate"]
    n, predictors, pairs = {}, {}, {}
    for subset, key in SUBSET_KEYS.items():
        some = report["models"][candidate][subset]
        if some is None:
            continue
        n[key] = {"states": some["n_states"], "questions": some["n_questions"],
                  "scorable": some["n_questions"] - some["n_ambiguous"]}
        for name, subsets in report["models"].items():
            predictors.setdefault(labels[name], {})[key] = _predictor_block(subsets[subset])
        for other, row in report["comparisons"][subset]["pairs"].items():
            pairs.setdefault(f"adapter - {labels[other]}", {})[key] = {
                k: row[k] for k in ("delta", "ci95", "p_one_sided", "p_holm", "clusters", "families")}
    return {"schema": TEST_SCHEMA, "test_sha256": test_sha256, "commitment_sha256": commitment_sha256,
            "adapter": labels[candidate], "n": n, "predictors": predictors, "paired_adapter_vs": pairs,
            SENSITIVITY_FIELD: _sensitivity_block(report, pairs),
            "config": {k: report["config"][k] for k in ("samples", "seed", "holm", "temperatures", "thresholds",
                                                         "local")},
            "inputs": report["inputs"]}


def _sensitivity_block(report, primary_pairs):
    """Amendment 7: 6-family macro accuracy (pick_option excluded) and the same paired comparisons, plus whether each
    comparison agrees with the primary 7-family one (same sign of delta, same Holm significance at ALPHA)."""
    sens, labels = report["sensitivity"], report["labels"]
    predictors, pairs, agreement = {}, {}, {}
    for subset, key in SUBSET_KEYS.items():
        if sens["comparisons"].get(subset) is None:
            continue
        for name, subsets in sens["models"].items():
            predictors.setdefault(labels[name], {})[key] = subsets[subset]
        for other, row in sens["comparisons"][subset]["pairs"].items():
            label = f"adapter - {labels[other]}"
            pairs.setdefault(label, {})[key] = {k: row[k] for k in ("delta", "ci95", "p_one_sided", "p_holm",
                                                                   "clusters", "families")}
            main = primary_pairs[label][key]
            agreement.setdefault(label, {})[key] = (
                (main["delta"] > 0) == (row["delta"] > 0) and (main["p_holm"] < ALPHA) == (row["p_holm"] < ALPHA))
    return {"excluded_families": sens["excluded_families"], "primary": False, "alpha": ALPHA,
            "predictors": predictors, "paired_adapter_vs": pairs, "agrees_with_primary": agreement}
