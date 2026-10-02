"""`to-kev`: records -> Kev labelled requests (kev.data.load_records at 0fe8fc97).

Kev keeps unknown top-level keys and ignores unknown question keys, so the records load as they are; but Kev reads
soft targets only from a question's `target` field ({option: weight}; option names for choice, "false"/"true" for
noul, level indices as strings for score; normalized by Kev) and ignores `soft_label`. This converter drops `meta`,
moves `soft_label` to `target` (or drops it with keep_soft=False) and sets `_meta` so Kev's ids and groups follow
ours (id = record id, group_id = state id, source = "muse").
"""
from __future__ import annotations

from pathlib import Path

from .jsonl import read_jsonl, write_jsonl

QUESTION_KEYS = ("type", "instructions", "criteria", "label", "src")


def to_kev(record: dict, keep_soft: bool = True) -> dict:
    questions = {}
    for qid, question in record["questions"].items():
        entry = {name: question[name] for name in QUESTION_KEYS if name in question}
        if keep_soft and question.get("soft_label"):
            entry["target"] = dict(question["soft_label"])
        questions[qid] = entry
    meta = record.get("meta", {})
    return {"state": record["state"], "questions": questions,
            "_meta": {"id": record["id"], "group_id": meta.get("state_id", record["id"]), "source": "muse"}}


def convert(records_path: str | Path, out: str | Path, keep_soft: bool = True) -> dict:
    rows = [to_kev(record, keep_soft) for record in read_jsonl(records_path)]
    count = write_jsonl(out, rows, sort_keys=False)
    targets = sum(1 for row in rows for question in row["questions"].values() if "target" in question)
    return {"records": count, "questions_with_target": targets, "out": str(out)}
