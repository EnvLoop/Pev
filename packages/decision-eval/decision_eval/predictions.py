"""The prediction file: one JSON line per question.

    {"id", "qid", "predictor": "base"|"sft"|"kev"|"openai"|"jev", "template": str|null, "options": [keys in order],
     "logits": [float], "probs": [float]}

`logits` are pre-temperature (log-probabilities for the base and the reference, the pointer head's raw logits for Kev),
so a temperature can be refit; every metric is recomputed from them. `probs` is softmax(logits) at T=1, for reading.
"""
import math

import numpy as np

from .records import read_jsonl, write_jsonl

PREDICTORS = ("base", "sft", "kev", "openai", "jev")   # B0, C, and the DEV references
FIELDS = ("id", "qid", "predictor", "template", "options", "logits", "probs")


def softmax(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = np.asarray(logits, dtype=float) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def prediction_row(question_key, predictor, template, options, logits):
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor {predictor!r}")
    logits = [float(z) for z in logits]
    if len(logits) != len(options) or not all(math.isfinite(z) for z in logits):
        raise ValueError("logits must be finite, one per option")
    record_id, qid = question_key
    return {"id": record_id, "qid": qid, "predictor": predictor, "template": template, "options": list(options),
            "logits": logits, "probs": softmax(logits).tolist()}


def write_predictions(path, rows):
    write_jsonl(path, rows)


def read_predictions(path):
    rows = read_jsonl(path)
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("prediction rows must have exactly the fields " + ", ".join(FIELDS))
    return rows


def _examples(keys, redact):
    return "" if redact else f" (e.g. {sorted(keys)[:3]})"


def align(questions, rows, redact=False):
    """Logit vectors in question order. Every question needs exactly one prediction with its option keys in order, and
    the file must hold one predictor and one template. With `redact`, errors name counts only (sealed data)."""
    by_key = {}
    for row in rows:
        key = (str(row["id"]), str(row["qid"]))
        if key in by_key:
            raise ValueError("duplicate prediction" + _examples([key], redact))
        by_key[key] = row
    wanted = {q.key for q in questions}
    missing, extra = wanted - set(by_key), set(by_key) - wanted
    if missing or extra:
        raise ValueError(f"predictions do not match the data: {len(missing)} missing, {len(extra)} extra"
                         + _examples(missing or extra, redact))
    identities = {(row["predictor"], row["template"]) for row in rows}
    if len(identities) != 1:
        raise ValueError("a prediction file must hold one predictor and one template")
    out = []
    for q in questions:
        row = by_key[q.key]
        if tuple(row["options"]) != q.keys:
            raise ValueError("prediction options differ from the question's option keys" + _examples([q.key], redact))
        z = np.asarray(row["logits"], dtype=float)
        if z.shape != (len(q.keys),) or not np.isfinite(z).all():
            raise ValueError("prediction logits must be finite, one per option" + _examples([q.key], redact))
        out.append(z)
    predictor, template = identities.pop()
    return out, {"predictor": predictor, "template": template}


def load_aligned(questions, path, redact=False):
    return align(questions, read_predictions(path), redact)
