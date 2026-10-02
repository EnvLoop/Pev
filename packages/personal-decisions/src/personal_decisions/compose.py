"""Compose family fragments into one structured state (the input of the renderer) with up to three questions.

Everything is drawn from RNGs seeded by (shard, seed, state index, family), so a state is a pure function of its id.
A fragment is dropped when it conflicts with the ones already accepted (a fact one needs is one another withholds,
or a forbidden anchor of any question shows up in the merged state).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random

from . import anchors
from .balance import MAX_ATTEMPTS, LabelSchedule
from .families import FAMILIES
from .families.base import SECTIONS, Context, Fragment, Question, fact_line, stamp
from .families.shard import ShardData
from .jsonl import seeded_rng
from .kb.rules import parse_time
from .kb.schema import UserProfile

DEFAULT_VARIANT_WEIGHTS = {"clean": 0.5, "override": 0.25, "removed": 0.25}
LETTERS = "ABCDEF"


@dataclass
class Draft:
    state_id: str
    user_id: str
    structured: dict
    questions: dict[str, Question]
    documents: list[str] = field(default_factory=list)  # distractor text inserted verbatim after rendering

    @property
    def buried(self) -> bool:
        return bool(self.documents)

    def variants(self) -> dict[str, str]:
        return {qid: ("buried+" if self.buried else "") + question.variant for qid, question in self.questions.items()}


def pick_variant(rng: random.Random, supported: tuple[str, ...], weights: dict[str, float]) -> str:
    names = [name for name in supported if weights.get(name, 0) > 0] or ["clean"]
    return rng.choices(names, weights=[weights.get(name, 1.0) for name in names])[0]


def target_index(user: UserProfile, visit: int, rng: random.Random) -> int:
    liked = [index for index, row in enumerate(user.later_interactions) if row.rating >= 4]
    return liked[visit % len(liked)] if liked else rng.randrange(len(user.later_interactions))


def memory_lines(user: UserProfile, fragments: list[Fragment], rng: random.Random, cap: int) -> list[str]:
    by_id = {fact.fact_id: fact for fact in user.memory_facts}
    needed = [fact_id for fragment in fragments for fact_id in fragment.memory_ids]
    excluded = {fact_id for fragment in fragments for fact_id in fragment.excluded_memory_ids}
    excluded |= {request.fact_id for request in user.forget_requests} - set(needed)
    filler = [fact.fact_id for fact in user.memory_facts if fact.fact_id not in excluded and fact.fact_id not in needed]
    chosen = list(dict.fromkeys(needed)) + rng.sample(filler, max(0, min(len(filler), cap - len(set(needed)))))
    facts = sorted((by_id[fact_id] for fact_id in chosen), key=lambda fact: (fact.date, fact.fact_id))
    return [fact_line(fact) for fact in facts]


def structured_state(now, user: UserProfile, fragments: list[Fragment], rng: random.Random, cap: int) -> dict:
    state: dict = {"now": stamp(now)}
    if user.city:
        state["profile"] = [f"Home city: {user.city}"]
    state["memory"] = memory_lines(user, fragments, rng, cap)
    for section in SECTIONS:
        lines = list(state.get(section, []))
        for fragment in fragments:
            lines.extend(line for line in fragment.lines.get(section, []) if line not in lines)
        if lines:
            state[section] = lines
    state["pending"] = [fragment.pending for fragment in fragments]
    return {key: value for key, value in state.items() if value}


def conflicts(fragment: Fragment, accepted: list[Fragment]) -> bool:
    needed = {fact_id for other in accepted for fact_id in other.memory_ids}
    withheld = {fact_id for other in accepted for fact_id in other.excluded_memory_ids}
    return bool(set(fragment.excluded_memory_ids) & needed or set(fragment.memory_ids) & withheld)


def state_text(value) -> str:
    """Every string in the structured state, one per line (what any faithful rendering must contain)."""
    if isinstance(value, dict):
        return "\n".join(state_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(state_text(item) for item in value)
    return str(value)


def consistent(structured: dict, fragments: list[Fragment]) -> bool:
    text = state_text(structured)
    return all(anchors.check(text, fragment.question.required, fragment.question.forbidden) is None
               for fragment in fragments)


def build_fragment(shard: ShardData, user: UserProfile, family: str, *, seed: int, index: int, now, chosen: int,
                   letter: str, weights: dict[str, float], schedule: LabelSchedule | None) -> Fragment | None:
    """The family's fragment; with a schedule, rebuilt with fresh seeded RNGs until its label fits the schedule."""
    build, supported = FAMILIES[family]
    variant = None
    for attempt in range(MAX_ATTEMPTS if schedule is not None else 1):
        family_rng = seeded_rng("fragment", shard.name, seed, index, family, *([attempt] if attempt else []))
        drawn = pick_variant(family_rng, supported, weights)
        variant = variant or drawn  # the first draw's variant for every attempt: retries keep the variant mix
        fragment = build(Context(user=user, shard=shard, rng=family_rng, now=now, target_index=chosen,
                                 letter=letter, variant=variant))
        if fragment is None or (attempt and fragment.question.soft_label):
            continue  # a retry may not turn a hard question soft, so retries do not inflate the soft share
        if schedule is None or schedule.fit(fragment):
            return fragment
    return None


def compose(shard: ShardData, user: UserProfile, *, seed: int, index: int, visit: int, families: list[str],
            variant_weights: dict[str, float] | None = None, memory_cap: int = 8,
            schedule: LabelSchedule | None = None) -> Draft | None:
    """families[0] is the primary family (the state is dropped without it); the rest are added when compatible.
    With a `schedule`, every accepted hard label follows it (see balance.py) and consumes it."""
    weights = variant_weights or DEFAULT_VARIANT_WEIGHTS
    rng = seeded_rng("state", shard.name, seed, index)
    chosen = target_index(user, visit, rng)
    now = parse_time(user.later_interactions[chosen].timestamp)
    accepted: list[Fragment] = []
    for family in families:
        fragment = build_fragment(shard, user, family, seed=seed, index=index, now=now, chosen=chosen,
                                  letter=LETTERS[len(accepted)], weights=weights, schedule=schedule)
        if fragment is None or conflicts(fragment, accepted):
            if not accepted:
                return None
            continue
        memory_rng = seeded_rng("memory", shard.name, seed, index)
        if not consistent(structured_state(now, user, accepted + [fragment], memory_rng, memory_cap),
                          accepted + [fragment]):
            if not accepted:
                return None
            continue
        accepted.append(fragment)
    if schedule is not None:
        for fragment in accepted:
            schedule.commit(fragment)
    structured = structured_state(now, user, accepted, seeded_rng("memory", shard.name, seed, index), memory_cap)
    return Draft(state_id=f"{shard.name}/{seed}/{index}", user_id=user.user_id, structured=structured,
                 questions={fragment.family: fragment.question for fragment in accepted})
