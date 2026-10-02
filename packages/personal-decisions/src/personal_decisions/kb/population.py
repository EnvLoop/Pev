"""`population`: choose the KB's users from the raw tables with a deterministic, content-free rule.

A user is eligible when their deduplicated history (the same dedupe and time split `build` uses) has between
`min_items` and `max_items` distinct items, at least `min_text_share` of the memory-side interactions carry review
text, and the later (choice-side) interactions hold at least one rating >= 4 and, with `require_low_later`, at least
one rating <= 2 (so pick_option can offer the user's own real low-rated later item). Eligible users are ordered per
domain by sha256("muse-population":seed:user_id) and the first N per domain are taken.

The output file holds the chosen user ids (one per line); the receipt holds the rule and counts only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..jsonl import read_jsonl, stable_int
from .build import dedupe, time_split
from .schema import Interaction

POPULATION_SALT = "muse-population"


@dataclass(frozen=True)
class PopulationRule:
    seed: int
    per_domain: dict[str, int]
    min_items: int = 8
    max_items: int = 30
    min_text_share: float = 0.8
    require_low_later: bool = True
    memory_fraction: float = 0.6


def eligible(history: list[Interaction], rule: PopulationRule) -> bool:
    rows = dedupe(history)
    if not rule.min_items <= len(rows) <= rule.max_items:
        return False
    early, later = time_split(rows, rule.memory_fraction)
    if sum(1 for row in early if row.text.strip()) < rule.min_text_share * len(early):
        return False
    if not any(row.rating >= 4 for row in later):
        return False
    return not rule.require_low_later or any(row.rating <= 2 for row in later)


def select_population(raw: Path, rule: PopulationRule) -> tuple[list[str], dict]:
    """(chosen user ids sorted, receipt with the rule and per-domain counts)."""
    raw = Path(raw)
    domain_of = {row["item_id"]: row["domain"] for row in read_jsonl(raw / "items.jsonl")}
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for row in read_jsonl(raw / "interactions.jsonl"):
        if str(row["item_id"]) in domain_of:
            by_user[str(row["user_id"])].append(Interaction(
                user_id=str(row["user_id"]), item_id=str(row["item_id"]), rating=int(row["rating"]),
                timestamp=str(row["timestamp"]), text=str(row.get("text") or "")))
    pools: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    for user_id, history in by_user.items():
        domains = {domain_of[row.item_id] for row in history}
        domain = domains.pop() if len(domains) == 1 else "mixed"
        seen[domain] += 1
        if domain in rule.per_domain and eligible(history, rule):
            pools[domain].append(user_id)
    chosen, counts = [], {}
    for domain, wanted in sorted(rule.per_domain.items()):
        ordered = sorted(pools[domain], key=lambda user_id: (stable_int(POPULATION_SALT, rule.seed, user_id), user_id))
        chosen += ordered[:wanted]
        counts[domain] = {"users_seen": seen[domain], "eligible": len(pools[domain]),
                          "chosen": len(ordered[:wanted]), "requested": wanted}
    receipt = {"rule": {**asdict(rule), "salt": POPULATION_SALT}, "domains": counts, "users": len(chosen)}
    return sorted(chosen), receipt


def write_population(path: Path, user_ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{user_id}\n" for user_id in user_ids), encoding="utf-8")


def read_population(path: Path) -> set[str]:
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
