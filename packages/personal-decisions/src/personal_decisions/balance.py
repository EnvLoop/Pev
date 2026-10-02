"""Exact label balance by construction: a stratified schedule of target labels per family.

Every hard-labelled question consumes the next value of its family's schedule:
- noul: true / false alternate in seeded shuffled blocks of two, so a family is 50/50 up to one question;
- score: the levels in seeded shuffled blocks of all levels, so the level distribution is uniform;
- choice: the correct option's position (per option count) in seeded shuffled blocks of all positions, so the best
  fixed position is right 1/n of the time. The options are reordered to put the label there. For fixed-name classes
  (`NAME_BALANCED`: route's services and ask_user) the label name follows its own block schedule too, answer first:
  a fragment with another name is rebuilt.
A fragment whose noul/score label differs from the scheduled one is rebuilt with a fresh seeded RNG (the family's
own answer-first logic stays untouched); after `MAX_ATTEMPTS` the family is left out of that state and the schedule
does not advance. Soft-labelled questions (removed evidence) do not consume the schedule: they must be tied (no
unique argmax), so they are never scored for accuracy; a soft label with a unique argmax is rebuilt.

The schedule is consumed in state order, so state i depends on the states before it; `generate` replays the drafts of
already-finished ids on resume to keep every state identical.
"""
from __future__ import annotations

from collections import defaultdict

from .families.base import Fragment
from .jsonl import seeded_rng

MAX_ATTEMPTS = 12


NAME_BALANCED = ("route",)  # choice families whose label NAME is also scheduled (their classes are fixed names)


def tied(soft: dict) -> bool:
    """True when the soft label's maximum is shared (no unique argmax): the question is ambiguous, never scored."""
    top = max(soft.values())
    return sum(1 for value in soft.values() if value == top) > 1


def name_key(fragment: Fragment) -> tuple | None:
    if fragment.family not in NAME_BALANCED or fragment.question.type != "choice":
        return None
    return (fragment.family, "name", len(fragment.question.criteria))


def schedule_key(fragment: Fragment) -> tuple:
    question = fragment.question
    if question.type == "noul":
        return (fragment.family, "noul", 2)
    if question.type == "score":
        return (fragment.family, "score", len(question.criteria))
    return (fragment.family, "position", len(question.criteria))


class LabelSchedule:
    def __init__(self, *parts):
        self.parts = parts
        self.used: dict[tuple, int] = defaultdict(int)

    def target(self, key: tuple):
        count = self.used[key]
        size = key[2]
        block = list(range(size))
        seeded_rng("label-schedule", *self.parts, *key, count // size).shuffle(block)
        value = block[count % size]
        return bool(value) if key[1] == "noul" else value

    def fit(self, fragment: Fragment) -> bool:
        """Make the fragment match its scheduled value (choice: reorder options); False when it cannot."""
        question = fragment.question
        if question.soft_label:
            # A soft label skips the schedule, so it must be tied; one with a unique argmax would be an unbalanced
            # scored label and is rebuilt instead.
            return tied(question.soft_label)
        key = schedule_key(fragment)
        wanted = self.target(key)
        if question.type != "choice":
            return question.label == wanted
        names = name_key(fragment)
        if names is not None and question.label != sorted(question.criteria)[self.target(names)]:
            return False
        place_label(fragment, wanted)
        return True

    def commit(self, fragment: Fragment) -> None:
        if not fragment.question.soft_label:
            self.used[schedule_key(fragment)] += 1
            names = name_key(fragment)
            if names is not None:
                self.used[names] += 1

    def counts(self) -> dict[str, int]:
        return {"/".join(map(str, key)): count for key, count in sorted(self.used.items())}


def place_label(fragment: Fragment, position: int) -> None:
    """Reorder a choice question's criteria (and the pending option list) so the label sits at `position`."""
    question = fragment.question
    names = [name for name in question.criteria if name != question.label]
    names.insert(position, question.label)
    question.criteria = {name: question.criteria[name] for name in names}
    listed = fragment.pending.get("options")
    if listed:
        by_name = {}
        for line in listed:
            owner = max((name for name in names if line.startswith(name)), key=len, default=None)
            by_name[owner] = line
        if set(by_name) == set(names):
            fragment.pending["options"] = [by_name[name] for name in names]
