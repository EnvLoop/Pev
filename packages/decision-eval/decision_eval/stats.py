"""Paired statistics (docs/PREREGISTRATION.md, amendments 1 and 4): state-clustered paired bootstraps and McNemar.

The bootstrap resamples whole states (every question of a state moves together, as kev's cluster_resamples keeps
sibling questions together) and, on each resample, recomputes a rate per group (a family's accuracy, a safety family's
false-negative rate) for both predictors. Reported: the observed delta (cand - base), its percentile 95% interval, per
group and, for accuracy, the unweighted family-macro delta with the one-sided p for H0: delta <= 0,
p = (1 + #{resampled delta <= 0}) / (samples + 1). A group absent from a resample is skipped in that resample's macro.
"""
import math

import numpy as np

SAMPLES = 10_000
SEED = 20_260_930
CHUNK = 500


def _tables(questions, groups, arrays):
    """[clusters, groups] sums of each value array (plus "n": question counts) over the questions."""
    clusters = sorted({q.state_id for q in questions})
    c_index = {c: i for i, c in enumerate(clusters)}
    g_index = {g: i for i, g in enumerate(sorted(set(groups)))}
    out = {name: np.zeros((len(clusters), len(g_index))) for name in ("n", *arrays)}
    for i, (q, g) in enumerate(zip(questions, groups)):
        cell = c_index[q.state_id], g_index[g]
        out["n"][cell] += 1
        for name, values in arrays.items():
            out[name][cell] += float(values[i])
    return out, sorted(g_index)


def _rates(n, k):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, k / np.where(n > 0, n, 1), np.nan)


def group_rate_bootstrap(questions, groups, base_values, cand_values, samples=SAMPLES, seed=SEED):
    """-> (group names, observed per-group delta [G], resampled per-group deltas [samples, G] with NaN where absent)."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    tables, names = _tables(questions, groups, {"base": base_values, "cand": cand_values})
    n, base, cand = tables["n"], tables["base"], tables["cand"]
    observed = _rates(n.sum(0), cand.sum(0)) - _rates(n.sum(0), base.sum(0))
    rng, n_clusters, deltas = np.random.default_rng(seed), n.shape[0], []
    for start in range(0, samples, CHUNK):
        draws = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=min(CHUNK, samples - start))
        counts = draws @ n
        deltas.append(_rates(counts, draws @ cand) - _rates(counts, draws @ base))
    return names, observed, np.concatenate(deltas), n_clusters


def _ci(values):
    values = values[~np.isnan(values)]
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if len(values) else [None, None]


def paired_bootstrap(questions, base_correct, cand_correct, samples=SAMPLES, seed=SEED):
    """Family-macro accuracy delta with CI and one-sided p, plus every family's delta and CI (same resamples)."""
    families, observed, deltas, clusters = group_rate_bootstrap(questions, [q.family for q in questions],
                                                                base_correct, cand_correct, samples, seed)
    macro = np.nanmean(deltas, axis=1)
    low, high = np.quantile(macro, [0.025, 0.975])
    return {"delta": float(np.nanmean(observed)), "ci95": [float(low), float(high)],
            "p_one_sided": float((1 + np.sum(macro <= 0)) / (samples + 1)),
            "families": {f: {"delta": float(observed[i]), "ci95": _ci(deltas[:, i])} for i, f in enumerate(families)},
            "samples": samples, "seed": seed, "clusters": clusters,
            "method": "paired percentile bootstrap over state clusters; family-macro accuracy recomputed per resample"}


def rate_delta_bootstrap(questions, base_values, cand_values, samples=SAMPLES, seed=SEED):
    """One rate (e.g. a safety false-negative rate over its at-risk questions): delta cand - base and its 95% CI."""
    if not questions:
        return {"delta": None, "ci95": [None, None]}
    _, observed, deltas, _ = group_rate_bootstrap(questions, [0] * len(questions), base_values, cand_values,
                                                  samples, seed)
    return {"delta": float(observed[0]), "ci95": _ci(deltas[:, 0])}


def mcnemar(base_correct, cand_correct):
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    base_correct, cand_correct = np.asarray(base_correct, bool), np.asarray(cand_correct, bool)
    b = int(np.sum(base_correct & ~cand_correct))
    c = int(np.sum(~base_correct & cand_correct))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
        p = min(1.0, 2 * tail / 2 ** n)
    return {"base_only_correct": b, "cand_only_correct": c, "p_two_sided": float(p), "method": "exact binomial"}
