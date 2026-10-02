"""Synthetic MUSE-style and plain Kev records and fake predictions for tests (never real DEV/HIDDEN data)."""
import random

from decision_eval.predictions import prediction_row
from decision_eval.records import questions_of

FAMILIES = ("pick_option", "needs_approval", "apply_memory", "forgotten_violation", "share_ok", "notify_level", "route")


def muse_question(family, rng):
    if family in ("pick_option", "route"):
        names = ["alpha", "beta", "gamma", "delta"][: rng.randint(2, 4)]
        return {"type": "choice", "instructions": f"Which option fits ({family})?",
                "criteria": {n: f"option {n}" for n in names}, "label": rng.choice(names), "src": f"muse/{family}"}
    if family == "notify_level":
        return {"type": "score", "instructions": "How urgently should the user be told?",
                "criteria": ["silent", "digest", "notify", "interrupt"], "label": rng.randint(0, 3),
                "src": f"muse/{family}"}
    return {"type": "noul", "instructions": f"Decide {family}.", "criteria": {"true": "yes, it does", "false": "no"},
            "label": rng.random() < 0.5, "src": f"muse/{family}"}


def muse_records(n_states=30, seed=0, per_state=3, prefix="rec"):
    rng = random.Random(seed)
    records = []
    for s in range(n_states):
        qids = {}
        for k in range(per_state):
            family = FAMILIES[(s + k) % len(FAMILIES)]
            qids[f"q{k}"] = (family, muse_question(family, rng))
        records.append({"id": f"{prefix}-{s:03d}", "state": f"Synthetic state {s}. The user likes tea.",
                        "questions": {qid: q for qid, (_, q) in qids.items()},
                        "meta": {"user_id": f"user-{s % 7}", "state_id": f"state-{s:03d}",
                                 "families": {qid: family for qid, (family, _) in qids.items()}}})
    return records


def kev_records(n=12, seed=1):
    """Plain Kev records (decision-v7 layout): src is the family, _meta carries id and group_id."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        src = ("sst5", "boolq", "agnews")[i % 3]
        if src == "sst5":
            q = {"type": "score", "instructions": "Rate the sentiment.", "criteria": ["1", "2", "3", "4", "5"],
                 "label": rng.randint(0, 4), "src": src}
        elif src == "boolq":
            q = {"type": "noul", "instructions": "Is the answer yes?", "label": rng.random() < 0.5, "src": src}
        else:
            q = {"type": "choice", "instructions": "Topic?", "criteria": {"world": None, "sports": None, "tech": None},
                 "label": rng.choice(["world", "sports", "tech"]), "src": src}
        out.append({"state": f"Kev text {i}", "questions": {"answer": q},
                    "_meta": {"id": f"{src}/test/{i}", "group_id": f"{src}/test/{i}", "variant": "clean"}})
    return out


def fake_logits(question, p_correct, rng, sharpness=3.0):
    """Logits whose argmax is the label with probability p_correct (else a random wrong option)."""
    k = len(question.keys)
    hit = rng.random() < p_correct
    top = question.label if hit or k == 1 else rng.choice([i for i in range(k) if i != question.label])
    z = [rng.gauss(0, 0.3) for _ in range(k)]
    z[top] = max(z) + sharpness * rng.random() + 0.1
    return z


def fake_predictions(records, predictor, p_correct, seed, template=None, family_bias=None):
    rng = random.Random(seed)
    rows = []
    for q in questions_of(records):
        p = (family_bias or {}).get(q.family, p_correct)
        rows.append(prediction_row(q.key, predictor, template, q.keys, fake_logits(q, p, rng)))
    return rows

