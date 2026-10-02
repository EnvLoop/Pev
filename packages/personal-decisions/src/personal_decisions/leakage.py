"""Cross-shard leakage audit of a user-id split: collision counts only, never ids or contents.

Checked for every shard pair (TRAIN vs each evaluation shard, and the evaluation shards against each other; the
first shard of a pair in EVAL_ORDER is the one whose later choices are looked up):
- `users`: the same user id in both shards.
- `user_target_pairs`: the same (user id, item id) real later choice in both shards.
- `target_review_text`: a later-choice review of the evaluation shard whose exact text (>= 40 characters) appears
  anywhere in the other shard's histories (a duplicated account or copied review would reveal the choice).
- `target_events`: a later choice whose (item id, timestamp, rating) appears in the other shard's histories.
A TRAIN user may well have reviewed the same restaurant or product; that alone is not counted.
"""
from __future__ import annotations

from itertools import combinations

from .jsonl import text_sha256

MIN_TEXT = 40
EVAL_ORDER = ("test", "hidden", "dev", "val", "train")  # shards named otherwise follow, in the order given


def fingerprints(users: list[dict]) -> dict[str, set]:
    ids, pairs, texts, events = set(), set(), set(), set()
    targets_text, targets_event, targets_pair = [], [], []
    for user in users:
        ids.add(user["user_id"])
        for row in user["memory_interactions"] + user["later_interactions"]:
            pairs.add((user["user_id"], row["item_id"]))
            events.add((row["item_id"], row["timestamp"], row["rating"]))
            if len(row["text"].strip()) >= MIN_TEXT:
                texts.add(text_sha256(row["text"].strip()))
        for row in user["later_interactions"]:
            targets_pair.append((user["user_id"], row["item_id"]))
            targets_event.append((row["item_id"], row["timestamp"], row["rating"]))
            if len(row["text"].strip()) >= MIN_TEXT:
                targets_text.append(text_sha256(row["text"].strip()))
    return {"ids": ids, "pairs": pairs, "texts": texts, "events": events, "targets_text": targets_text,
            "targets_event": targets_event, "targets_pair": targets_pair}


def audit(shards: dict[str, list[dict]]) -> dict:
    """{"<eval>_vs_<other>": {check: collisions}}; the evaluation shard's later choices are looked up in the other."""
    prints = {name: fingerprints(users) for name, users in shards.items()}
    names = [name for name in EVAL_ORDER if name in prints] + [name for name in prints if name not in EVAL_ORDER]
    report = {}
    for first, second in combinations(names, 2):
        a, b = prints[first], prints[second]
        report[f"{first}_vs_{second}"] = {
            "users": len(a["ids"] & b["ids"]),
            "user_target_pairs": sum(1 for pair in a["targets_pair"] if pair in b["pairs"]),
            "target_review_text": sum(1 for digest in a["targets_text"] if digest in b["texts"]),
            "target_events": sum(1 for event in a["targets_event"] if event in b["events"])}
    return report
