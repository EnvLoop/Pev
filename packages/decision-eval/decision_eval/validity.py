"""Eval validity checks (docs/PREREGISTRATION.md amendment 4), run before the first paid run; results go into receipts.

- oracle: the labels themselves as predictions, through the full prediction -> alignment -> scoring path, must score
  100% on every scorable question (and Brier ~0 on hard labels).
- null baselines: majority label, best constant option position, shuffled labels (label positions permuted within a
  family among questions with the same option count) must sit at chance (mean 1/K) within binomial noise; majority and
  constant only fail above it (they are maxima). Shuffled is the mean over PERMUTATIONS fixed-seed permutations (one
  permutation alone strays outside the noise band by chance at n ~ 100) with their 2.5-97.5% range, and fails on
  either side of chance; its mean estimates sum_k p_k^2 over label positions, so it flags position imbalance.
- ladder: predictors ordered weakest -> strongest must not decrease significantly (each paired bootstrap CI upper
  bound of stronger - weaker >= 0, macro and per family), and the strongest must stay below 95% (headroom).
"""
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .metrics import families_of, family_accuracy, score_questions
from .noise import Z95
from .predictions import align, prediction_row
from .stats import SEED, paired_bootstrap

CEILING = 0.95
PERMUTATIONS = 50


def _check(passed, **values):
    return {"passed": bool(passed), **values}


def oracle(questions):
    rows = [prediction_row(q.key, "base", "oracle", q.keys, [math.log(max(t, 1e-12)) for t in q.target])
            for q in questions]
    logits, _ = align(questions, rows)
    scored = score_questions(questions, logits)
    accuracy = family_accuracy(questions, scored)
    hard = [i for i, q in enumerate(questions) if max(q.target) == 1.0]
    max_brier = float(max(scored.brier[hard])) if hard else 0.0
    return _check(all(a == 1.0 for a in accuracy.values()) and max_brier < 1e-6, families=accuracy,
                  max_brier_hard_labels=max_brier)


def _shuffled(questions, labels, rng):
    """Label positions permuted among the family's questions with the same option count."""
    shuffled = list(labels)
    by_width = defaultdict(list)
    for i, q in enumerate(questions):
        by_width[len(q.keys)].append(i)
    for index in by_width.values():
        permuted = [labels[i] for i in index]
        rng.shuffle(permuted)
        for i, label in zip(index, permuted):
            shuffled[i] = label
    return shuffled


def _baseline_predictions(questions, seed, permutations):
    """(majority, {position: constant}, [shuffled per permutation]) option indices for one family's questions."""
    labels = [q.label for q in questions]
    majority_key = Counter(q.keys[q.label] for q in questions).most_common(1)[0][0]
    majority_index = Counter(labels).most_common(1)[0][0]
    majority = [q.keys.index(majority_key) if majority_key in q.keys else min(majority_index, len(q.keys) - 1)
                for q in questions]
    width = max(len(q.keys) for q in questions)
    constants = {j: [min(j, len(q.keys) - 1) for q in questions] for j in range(width)}
    rng = random.Random(seed)
    return majority, constants, [_shuffled(questions, labels, rng) for _ in range(permutations)]


def null_baselines(questions, seed=SEED, permutations=PERMUTATIONS):
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    kept = [q for q in questions if not q.ambiguous]
    families, checks = {}, []
    for name, index in families_of(kept).items():
        group = [kept[i] for i in index]
        labels = np.asarray([q.label for q in group])
        chance = float(np.mean([1 / len(q.keys) for q in group]))
        half_width = Z95 * math.sqrt(chance * (1 - chance) / len(group))
        majority, constants, shuffled = _baseline_predictions(group, seed, permutations)
        per_permutation = np.asarray([np.mean(np.asarray(p) == labels) for p in shuffled])
        acc = {"majority": float(np.mean(np.asarray(majority) == labels)),
               "shuffled": float(per_permutation.mean()),
               "shuffled_range": [float(x) for x in np.quantile(per_permutation, [0.025, 0.975])]}
        by_position = {str(j): float(np.mean(np.asarray(p) == labels)) for j, p in constants.items()}
        best = max(by_position, key=by_position.get)
        acc["constant"] = by_position[best]
        ceiling = chance + half_width
        passed = {"majority": acc["majority"] <= ceiling, "constant": acc["constant"] <= ceiling,
                  "shuffled": abs(acc["shuffled"] - chance) <= half_width}
        families[name] = {"n": len(group), "chance": chance, "half_width": half_width, **acc,
                          "constant_position": int(best), "permutations": permutations, "passed": passed}
        checks += passed.values()
    return _check(all(checks), families=families,
                  rule="majority/constant <= chance + noise; |mean shuffled - chance| <= noise")


def ladder(questions, named_logits, samples=2000, seed=SEED):
    """named_logits: [(name, logits)] from weakest to strongest."""
    kept = [q for q in questions if not q.ambiguous]
    scored = [(name, score_questions(questions, logits)) for name, logits in named_logits]
    rungs = [{"name": name, "macro_accuracy": float(np.mean(list(family_accuracy(questions, s).values()))),
              "families": family_accuracy(questions, s)} for name, s in scored]
    steps, inversions = [], []
    for (weak, a), (strong, b) in zip(scored, scored[1:]):
        boot = paired_bootstrap(kept, a.correct[a.scorable], b.correct[b.scorable], samples, seed)
        steps.append({"weaker": weak, "stronger": strong, "delta": boot["delta"], "ci95": boot["ci95"],
                      "passed": boot["ci95"][1] >= 0})
        inversions += [{"weaker": weak, "stronger": strong, "family": f, **v}
                       for f, v in boot["families"].items() if v["ci95"][1] is not None and v["ci95"][1] < 0]
    best = max(rungs, key=lambda r: r["macro_accuracy"]) if rungs else None
    return {"rungs": rungs,
            "monotone": _check(all(s["passed"] for s in steps), steps=steps),
            "no_family_inversion": _check(not inversions, inversions=inversions),
            "headroom": _check(best is not None and best["macro_accuracy"] < CEILING, ceiling=CEILING,
                               best=None if best is None else best["name"],
                               best_macro_accuracy=None if best is None else best["macro_accuracy"])}


def validity(questions, named_logits=(), samples=2000, seed=SEED, permutations=PERMUTATIONS):
    checks = {"oracle": oracle(questions), "null_baselines": null_baselines(questions, seed, permutations)}
    rungs = None
    if named_logits:
        result = ladder(questions, list(named_logits), samples, seed)
        rungs = result.pop("rungs")
        checks.update({f"ladder_{name}": check for name, check in result.items()})
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks, "ladder": rungs,
            "n_questions": len(questions), "n_ambiguous": sum(q.ambiguous for q in questions)}
