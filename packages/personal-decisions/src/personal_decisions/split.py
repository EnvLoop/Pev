"""`split`: KB -> disjoint shards by user id (train / val / dev / hidden) plus a contents-free manifest.

Users are ordered by sha256(salt:user_id); hidden, dev and val take fixed user counts from the front of that order
and train takes the rest, so one real person lives in exactly one shard. Distractor emails are partitioned by
sha256(salt:email_id) buckets; the item catalog and the OpenFlights world are public and copied to every shard.

Default user counts assume about two states per user in the evaluation shards: val 150 users (~300 states),
dev 100 (~200 states, ~600 questions), hidden 200 (~400 states, ~1200 questions); train needs >= 300 users to
yield >= 3000 states at <= 10 states per user.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from . import leakage
from .jsonl import read_jsonl, stable_int, write_json, write_jsonl
from .kb.build import file_manifest

SPLIT_SALT = "muse-split-v1"
SHARDS = ("train", "val", "dev", "hidden")
PARTITION_SALT = "muse-partition-v1"
DEFAULT_USERS = {"hidden": 200, "dev": 100, "val": 150}
# Email buckets (sha256 % 100) per shard: distractor text never repeats across shards.
EMAIL_BUCKETS = {"hidden": range(0, 15), "dev": range(15, 25), "val": range(25, 35), "train": range(35, 100)}
SHARED_FILES = ("items.jsonl", "airports.jsonl", "routes.jsonl", "airlines.json")
MIN_TRAIN_USERS = 300


def assign_users(user_ids: list[str], counts: dict[str, int], salt: str = SPLIT_SALT) -> dict[str, list[str]]:
    ordered = sorted(user_ids, key=lambda user_id: (stable_int(salt, user_id), user_id))
    shards, start = {}, 0
    for name in ("hidden", "dev", "val"):
        shards[name] = ordered[start:start + counts.get(name, 0)]
        start += len(shards[name])
    shards["train"] = ordered[start:]
    return shards


def email_shard(email_id: str, salt: str = SPLIT_SALT) -> str:
    bucket = stable_int(salt, "email", email_id) % 100
    return next(name for name, buckets in EMAIL_BUCKETS.items() if bucket in buckets)


def split_kb(kb: Path, outputs: dict[str, Path], manifest_path: Path | None = None,
             counts: dict[str, int] | None = None, salt: str = SPLIT_SALT) -> dict:
    kb = Path(kb)
    counts = {**DEFAULT_USERS, **(counts or {})}
    users = list(read_jsonl(kb / "users.jsonl"))
    if len({user["user_id"] for user in users}) != len(users):
        raise ValueError("users.jsonl has duplicate user ids")
    assigned = assign_users([user["user_id"] for user in users], counts, salt)
    shard_of = {user_id: name for name, ids in assigned.items() for user_id in ids}
    emails = list(read_jsonl(kb / "emails.jsonl")) if (kb / "emails.jsonl").is_file() else []
    manifest = {"kind": "muse-kb-split", "salt": salt, "requested_users": counts, "shards": {}, "warnings": []}
    for name in SHARDS:
        if name not in outputs:
            continue
        folder = Path(outputs[name])
        folder.mkdir(parents=True, exist_ok=True)
        write_jsonl(folder / "users.jsonl", (user for user in users if shard_of[user["user_id"]] == name))
        write_jsonl(folder / "emails.jsonl",
                    (email for email in emails if email_shard(email["email_id"], salt) == name))
        for shared in SHARED_FILES:
            if (kb / shared).is_file():
                shutil.copyfile(kb / shared, folder / shared)
        entry = {"shard": name, "users": len(assigned[name]), **file_manifest(folder)}
        write_json(folder / "manifest.json", entry)
        manifest["shards"][name] = entry
    manifest["leakage"] = leakage.audit({name: [user for user in users if shard_of[user["user_id"]] == name]
                                         for name in SHARDS})
    if any(entry["users"] or entry["user_target_pairs"] for entry in manifest["leakage"].values()):
        manifest["warnings"].append("a user or (user, target) pair appears in two shards")
    if len(assigned["train"]) < MIN_TRAIN_USERS:
        manifest["warnings"].append(f"train has {len(assigned['train'])} users (< {MIN_TRAIN_USERS})")
    if any(len(assigned[name]) < counts[name] for name in DEFAULT_USERS):
        manifest["warnings"].append("an evaluation shard got fewer users than requested")
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def user_domains(kb: Path, users: list[dict]) -> dict[str, str]:
    """user id -> the domain of their history ("mixed" when it spans several), from the KB's item catalog."""
    wanted = {row["item_id"] for user in users for row in user["memory_interactions"] + user["later_interactions"]}
    domain = {row["item_id"]: row["domain"] for row in read_jsonl(Path(kb) / "items.jsonl") if row["item_id"] in wanted}
    out = {}
    for user in users:
        seen = {domain.get(row["item_id"]) for row in user["memory_interactions"] + user["later_interactions"]}
        out[user["user_id"]] = seen.pop() if len(seen) == 1 else "mixed"
    return out


def partition_kb(kb: Path, outputs: dict[str, Path], salt: str = PARTITION_SALT,
                 email_pool: str | None = None) -> dict:
    """Split a whole KB (e.g. a fresh test population) into equal named parts by user id, stratified by domain:
    within each domain users are ordered by sha256(salt:user_id) and dealt round robin over the parts. Emails are
    those of `email_pool` (a `split` email shard: train / val / dev / hidden) or all of them; the item catalog and
    OpenFlights files are copied. Each part gets a manifest naming its shard (the generator's RNG namespace)."""
    kb, names = Path(kb), list(outputs)
    users = list(read_jsonl(kb / "users.jsonl"))
    if len({user["user_id"] for user in users}) != len(users):
        raise ValueError("users.jsonl has duplicate user ids")
    domains = user_domains(kb, users)
    part_of = {}
    for domain in sorted(set(domains.values())):
        ordered = sorted((uid for uid, value in domains.items() if value == domain),
                         key=lambda user_id: (stable_int(salt, user_id), user_id))
        part_of.update({user_id: names[number % len(names)] for number, user_id in enumerate(ordered)})
    emails = list(read_jsonl(kb / "emails.jsonl")) if (kb / "emails.jsonl").is_file() else []
    if email_pool is not None:
        if email_pool not in EMAIL_BUCKETS:
            raise ValueError(f"unknown email pool {email_pool!r} (expected one of {', '.join(EMAIL_BUCKETS)})")
        emails = [email for email in emails if email_shard(email["email_id"]) == email_pool]
    manifest = {"kind": "muse-kb-partition", "salt": salt, "email_pool": email_pool or "all", "parts": {}}
    for name, folder in outputs.items():
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        mine = [user for user in users if part_of[user["user_id"]] == name]
        write_jsonl(folder / "users.jsonl", mine)
        write_jsonl(folder / "emails.jsonl", emails)
        for shared in SHARED_FILES:
            if (kb / shared).is_file():
                shutil.copyfile(kb / shared, folder / shared)
        per_domain = {domain: sum(1 for user in mine if domains[user["user_id"]] == domain)
                      for domain in sorted(set(domains.values()))}
        entry = {"shard": name, "users": len(mine), "per_domain": per_domain, **file_manifest(folder)}
        write_json(folder / "manifest.json", entry)
        manifest["parts"][name] = entry
    manifest["leakage"] = leakage.audit({name: [user for user in users if part_of[user["user_id"]] == name]
                                         for name in outputs})
    return manifest
