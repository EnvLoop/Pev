"""Noise floor and build variance for the hillclimb (docs/PREREGISTRATION.md amendment 4).

noise_floor: for one predictor at the data's current size, the state-clustered bootstrap 95% CI half-width of each
family's accuracy and of the family-macro accuracy, and the paired-difference half-width a C - B0 comparison of two
predictors of this accuracy would have if their errors were independent (sqrt(2) x; predictors sharing a base are
positively correlated, so the real paired floor is smaller: use seed_spread or a paired score to measure it). A round
whose expected gain is below `min_detectable_gain` is not worth running.

seed_spread: two candidates from identically configured trainings (seeds only differ): the family and macro accuracy
deltas between them with their paired CIs, i.e. how much "changing nothing" moves DEV.
"""
import math

import numpy as np

from .metrics import score_questions
from .stats import SEED, group_rate_bootstrap, mcnemar, paired_bootstrap

Z95 = 1.959964


def _half_width(values):
    values = values[~np.isnan(values)]
    low, high = np.quantile(values, [0.025, 0.975])
    return float((high - low) / 2)


def _row(n, accuracy, half_width):
    return {"n": n, "accuracy": accuracy, "headroom": 1 - accuracy, "accuracy_half_width": half_width,
            "binomial_half_width": Z95 * math.sqrt(accuracy * (1 - accuracy) / n) if n else None,
            "paired_half_width_independent": math.sqrt(2) * half_width}


def noise_floor(questions, logits, samples=2000, seed=SEED):
    scored = score_questions(questions, logits)
    kept = [q for q in questions if not q.ambiguous]
    correct = scored.correct[scored.scorable]
    zeros = np.zeros(len(kept))
    families, accuracy, resampled, clusters = group_rate_bootstrap(kept, [q.family for q in kept], zeros, correct,
                                                                   samples, seed)
    counts = {f: sum(q.family == f for q in kept) for f in families}
    per_family = {f: _row(counts[f], float(accuracy[i]), _half_width(resampled[:, i])) for i, f in enumerate(families)}
    macro = _row(len(kept), float(np.mean(accuracy)), _half_width(np.nanmean(resampled, axis=1)))
    family_widths = [r["binomial_half_width"] for r in per_family.values()]
    macro["binomial_half_width"] = math.sqrt(sum(w ** 2 for w in family_widths)) / len(family_widths)
    return {"families": per_family, "macro": macro, "min_detectable_gain": macro["paired_half_width_independent"],
            "n_scored": len(kept), "n_ambiguous": len(questions) - len(kept), "clusters": clusters,
            "samples": samples, "seed": seed,
            "note": "paired_half_width_independent assumes independent errors; correlated predictors have a smaller "
                    "paired floor (measure it with seed-spread or score)"}


def seed_spread(questions, logits_a, logits_b, samples=2000, seed=SEED):
    a, b = score_questions(questions, logits_a), score_questions(questions, logits_b)
    kept = [q for q in questions if not q.ambiguous]
    boot = paired_bootstrap(kept, a.correct[a.scorable], b.correct[b.scorable], samples, seed)
    return {"macro_delta": boot["delta"], "macro_ci95": boot["ci95"], "families": boot["families"],
            "build_variance": abs(boot["delta"]),
            "significant": boot["ci95"][0] > 0 or boot["ci95"][1] < 0,
            "mcnemar": mcnemar(a.correct[a.scorable], b.correct[b.scorable]),
            "n_scored": len(kept), "samples": samples, "seed": seed,
            "note": "delta = second - first file; a round whose expected gain is below build_variance is not run"}
