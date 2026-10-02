"""VAL-only fits applied unchanged to DEV/HIDDEN: one temperature per predictor (NLL) and one automation threshold.

Temperature: minimum mean cross-entropy against each question's target (the one-hot label, or the soft label), every
question weighted equally (kev's TEMPERATURE_FIT aggregation "micro"), over a log grid. kev ships 121 points on
0.25..4 for its pointer heads; the zero-shot base's label log-probabilities can be far sharper, so the same kind of grid
is widened to 0.05..50 (481 points) for both predictors. Dividing logits by T > 0 never changes the argmax.

Threshold: the lowest max-probability threshold whose accepted VAL questions have error <= budget
(kev's select_threshold, decision_eval.conventions: the largest confidence-ordered accepted set within the
budget); None = abstain on all.
"""
import numpy as np

from .conventions import evaluate_threshold, select_threshold
from .metrics import BUDGET, score_questions

GRID = np.exp(np.linspace(np.log(0.05), np.log(50.0), 481))
METHOD = "min micro mean cross-entropy vs target over a 481-point log grid on 0.05..50"


def _padded(questions, logits):
    width = max(len(z) for z in logits)
    z = np.full((len(logits), width), -np.inf)
    target = np.zeros((len(logits), width))
    for i, (q, row) in enumerate(zip(questions, logits)):
        z[i, : len(row)] = row
        target[i, : len(row)] = q.target
    return z, target


def _mean_nll(z, target, temperature):
    """Mean cross-entropy at `temperature` (same floor as metrics.score_questions: p >= 1e-12)."""
    scaled = z / temperature
    scaled = scaled - scaled.max(1, keepdims=True)
    log_p = scaled - np.log(np.exp(scaled).sum(1, keepdims=True))
    log_p = np.maximum(log_p, np.log(1e-12))   # padded options become the floor, and their target is 0
    return float(-(target * log_p).sum(1).mean())


def mean_nll(questions, logits, temperature):
    return _mean_nll(*_padded(questions, logits), temperature)


def fit_temperature(questions, logits):
    z, target = _padded(questions, logits)
    losses = [_mean_nll(z, target, float(t)) for t in GRID]
    best = int(np.argmin(losses))
    return {"temperature": float(GRID[best]), "nll_raw": _mean_nll(z, target, 1.0), "nll_calibrated": losses[best],
            "at_grid_edge": best in (0, len(GRID) - 1), "method": METHOD}


def fit_threshold(questions, logits, temperature, budget=BUDGET):
    scored = score_questions(questions, logits, temperature)
    confidence, correct = scored.confidence[scored.scorable], scored.correct[scored.scorable]   # no ambiguous ones
    threshold = select_threshold(confidence, correct, budget)
    val = evaluate_threshold(confidence, correct, threshold)
    return {"threshold": threshold, "budget": budget, "temperature": temperature,
            "val": {k: val[k] for k in ("n", "accepted", "errors", "coverage", "risk")}}
