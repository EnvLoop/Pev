"""Kev's record and scoring conventions, copied from jaredpalmer/kev@0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91
(Apache-2.0; kev/api.py and kev/metrics.py) so the default install does not need kev or its torch<2.9 bound.

The copies are verbatim in behaviour: tests/test_conventions.py compares them with kev itself when the `kev` extra is
installed. Rendering decides the exact prompt bytes (frozen by tests/test_prompts.py), so it must never drift.
"""
import math

import numpy as np

JSONContent = str | dict | list | int | float | bool | None


def render(v, indent=0):
    """kev.api.render: flatten str | object | array into text the model sees; field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name, desc):
    """kev.api.option_text."""
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype, criteria):
    """kev.api.question_keys: criteria names (choice), ["false", "true"] (noul), level indices as strings (score)."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def ece(conf, correct, bins=10):
    """kev.metrics.ece: equal-width binned expected calibration error."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _selective_inputs(confidence, correct):
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct)
    if confidence.ndim != 1 or correct.shape != confidence.shape:
        raise ValueError("confidence and correctness must be equal-length vectors")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.isin(correct, [False, True]).all():
        raise ValueError("correctness must be boolean")
    return confidence, correct.astype(bool)


def _risk_curve_arrays(confidence, correct):
    confidence, correct = _selective_inputs(confidence, correct)
    if not len(confidence):
        return confidence, np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-confidence, kind="stable")
    confidence, errors = confidence[order], np.cumsum(~correct[order])
    ends = np.r_[np.flatnonzero(confidence[1:] != confidence[:-1]), len(order) - 1]
    return confidence[ends], ends + 1, errors[ends]


def coverage_at_error(confidence, correct, budget):
    """kev.metrics.coverage_at_error: the largest confidence-ordered accepted share with error <= budget."""
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("error budget must be in [0, 1]")
    _, accepted, errors = _risk_curve_arrays(confidence, correct)
    ok = np.flatnonzero(errors <= budget * accepted)
    return float(accepted[ok[-1]] / accepted[-1]) if len(ok) else 0.0


def select_threshold(confidence, correct, budget, min_accepted=1):
    """kev.metrics.select_threshold: the lowest confidence threshold whose accepted set keeps error <= budget."""
    if not math.isfinite(budget) or not 0 <= budget <= 1 or min_accepted < 1:
        raise ValueError("invalid error budget or minimum accepted count")
    thresholds, accepted, errors = _risk_curve_arrays(confidence, correct)
    ok = np.flatnonzero((errors <= budget * accepted) & (accepted >= min_accepted))
    return float(thresholds[ok[-1]]) if len(ok) else None


def evaluate_threshold(confidence, correct, threshold):
    """kev.metrics.evaluate_threshold: accepted count, errors, coverage and risk at a threshold (None = abstain)."""
    confidence, correct = _selective_inputs(confidence, correct)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1] or None for abstain-all")
    accepted = confidence >= threshold if threshold is not None else np.zeros(len(confidence), dtype=bool)
    n, errors = int(accepted.sum()), int((accepted & ~correct).sum())
    return {"threshold": threshold, "n": len(confidence), "accepted": n, "errors": errors,
            "coverage": n / len(confidence) if len(confidence) else 0.0, "risk": errors / n if n else None}
