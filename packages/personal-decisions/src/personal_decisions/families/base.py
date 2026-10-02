"""Shared types for the question families.

A family is a pure function `build(ctx) -> Fragment | None`: everything it draws comes from `ctx.rng` (seeded by the
generator) and the KB shard slice in `ctx`, so the same (shard, seed, index) always yields the same fragment. A
fragment contributes lines to the structured state's sections, one pending item (request, proposed action or
incoming event) and one question with its Kev label and anchors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import random

from ..kb.schema import Item, MemoryFact, UserProfile
from .shard import ShardData

SECTIONS = ("profile", "rules", "preference_updates", "forget_requests", "contacts", "calendar")
VARIANTS = ("clean", "override", "removed")


@dataclass
class Context:
    user: UserProfile
    shard: ShardData
    rng: random.Random
    now: datetime  # the decision time: a real later interaction's timestamp
    target_index: int  # index into user.later_interactions of the real choice this state is built around
    letter: str  # pending-item letter assigned by the composer (A, B, C)
    variant: str = "clean"


@dataclass
class Question:
    family: str
    type: str  # "noul" | "choice" | "score"
    instructions: str
    criteria: dict | list | None
    label: str | bool | int
    required: list[str]
    forbidden: list[str] = field(default_factory=list)
    soft_label: dict[str, float] | None = None
    variant: str = "clean"
    option_features: dict | None = None  # choice options -> {"price", "popularity"} for shortcut baselines


@dataclass
class Fragment:
    family: str
    question: Question
    pending: dict  # {"label": "Request A", "text": ..., optional "options": [...]}
    lines: dict[str, list[str]] = field(default_factory=dict)  # section -> lines this family needs
    memory_ids: list[str] = field(default_factory=list)  # facts that must be in the memory section
    excluded_memory_ids: list[str] = field(default_factory=list)  # facts that must not appear (removed evidence)

    def add(self, section: str, *lines: str) -> None:
        bucket = self.lines.setdefault(section, [])
        bucket.extend(line for line in lines if line not in bucket)


def money(value: float) -> str:
    return f"${value:,.2f}"


def whole_money(value: int) -> str:
    return f"${value:,}"


def stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def pending_label(kind: str, letter: str) -> str:
    return f"{kind} {letter}"


def active_facts(user: UserProfile, include_forgotten: bool = False) -> list[MemoryFact]:
    forgotten = {request.fact_id for request in user.forget_requests}
    return [fact for fact in user.memory_facts if include_forgotten or fact.fact_id not in forgotten]


def touched_items(user: UserProfile) -> set[str]:
    return {row.item_id for row in user.memory_interactions + user.later_interactions}


def item_price_text(item: Item) -> str:
    if item.price is not None:
        return money(item.price)
    if item.price_level:
        return "$" * item.price_level
    return "price not listed"


def noul_soft(probability_true: float) -> dict[str, float]:
    return {"true": round(probability_true, 4), "false": round(1 - probability_true, 4)}


def fact_line(fact: MemoryFact) -> str:
    return f"{fact.date}: {fact.text}"
