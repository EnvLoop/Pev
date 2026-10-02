"""Decision records -> scored questions.

Two record layouts are accepted:
- MUSE records: {"id", "state", "questions": {qid: {...}}, "meta": {"user_id", "state_id", "families": {qid: family}}}.
- Plain Kev records (e.g. kev evals/v7/decision-v7/test.jsonl): {"state", "questions", "_meta": {"id", "group_id"}};
  the family is the question's `src`, the bootstrap cluster the record's `group_id`.

Labels follow Kev: choice -> option name, noul -> true/false, score -> zero-based level index. An optional
`soft_label` {option key: p} becomes the question's target (Brier, ECE, NLL); its argmax is the label when it is
unique. A soft label without a unique argmax (a tie, e.g. uniform) makes the question *ambiguous*: it has no correct
answer, so it is left out of accuracy, McNemar, safety false negatives and the automation counts, and enters only the
target-based metrics.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .conventions import question_keys

MUSE_PREFIX = "muse/"
TIE = 1e-9   # soft-label probabilities this close to the maximum tie with it


@dataclass(frozen=True)
class Question:
    record_id: str
    qid: str
    family: str
    state_id: str
    type: str
    keys: tuple[str, ...]
    label: int                   # index into keys: the hard label, or the soft label's argmax (first on ties)
    target: tuple[float, ...]    # one-hot of the hard label, or the normalised soft label
    ambiguous: bool = False      # soft label without a unique argmax: no correct answer, target-based metrics only

    @property
    def key(self):
        return self.record_id, self.qid


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")   # never sort: option order is meaning


def record_id(record):
    rid = record.get("id") or (record.get("_meta") or {}).get("id")
    if not rid:
        raise ValueError("record without an id (neither `id` nor `_meta.id`)")
    return str(rid)


def state_id(record):
    meta, kev_meta = record.get("meta") or {}, record.get("_meta") or {}
    return str(meta.get("state_id") or kev_meta.get("group_id") or record_id(record))


def family_of(record, qid):
    families = (record.get("meta") or {}).get("families") or {}
    if qid in families:
        return str(families[qid])
    src = record["questions"][qid].get("src")
    if not src:
        raise ValueError("question without a family (no meta.families entry and no src)")
    return src[len(MUSE_PREFIX):] if src.startswith(MUSE_PREFIX) else src


def hard_label_index(qtype, keys, label):
    if qtype == "choice":
        if label not in keys:
            raise ValueError("choice label is not one of the option names")
        return keys.index(label)
    if qtype == "noul":
        if isinstance(label, str) and label in ("true", "false"):
            return 1 if label == "true" else 0
        if isinstance(label, bool):
            return int(label)
        raise ValueError("noul label must be true or false")
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(keys):
        raise ValueError("score label must be a level index")
    return label


def _option_key(key):
    return ("true" if key else "false") if isinstance(key, bool) else str(key)


def soft_target(keys, soft):
    named = {_option_key(k): float(v) for k, v in soft.items()}
    if set(named) - set(keys):
        raise ValueError("soft_label names options the question does not have")
    values = [named.get(k, 0.0) for k in keys]
    if any(v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("soft_label must be non-negative with positive mass")
    total = sum(values)
    return tuple(v / total for v in values)


def question_of(record, qid):
    q = record["questions"][qid]
    keys = tuple(question_keys(q["type"], q.get("criteria")))
    if q.get("soft_label") is not None:
        target = soft_target(keys, q["soft_label"])
        label = max(range(len(keys)), key=lambda i: (target[i], -i))
        ambiguous = sum(t >= target[label] - TIE for t in target) > 1
    else:
        label = hard_label_index(q["type"], list(keys), q.get("label"))
        target = tuple(float(i == label) for i in range(len(keys)))
        ambiguous = False
    return Question(record_id(record), str(qid), family_of(record, qid), state_id(record), q["type"], keys, label,
                    target, ambiguous)


def questions_of(records):
    out, seen = [], set()
    for record in records:
        for qid in record["questions"]:
            question = question_of(record, qid)
            if question.key in seen:
                raise ValueError("duplicate (record id, question id) in the data")
            seen.add(question.key)
            out.append(question)
    if not out:
        raise ValueError("no questions in the data")
    return out


def load_questions(path):
    return questions_of(read_jsonl(path))


def kev_request(record):
    """The record as kev's predictors take it: typed questions with src and a Kev label (materialize needs both). A
    soft-labelled question carries its argmax as the label; labels never reach the model."""
    questions = {}
    for qid, q in record["questions"].items():
        question = question_of(record, qid)
        key = question.keys[question.label]
        label = key if q["type"] == "choice" else (key == "true" if q["type"] == "noul" else question.label)
        questions[qid] = {"type": q["type"], "instructions": q.get("instructions"), "criteria": q.get("criteria"),
                          "src": q.get("src") or question.family, "label": label}
    return {"state": record["state"], "questions": questions}
