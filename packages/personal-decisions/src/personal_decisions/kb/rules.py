"""Synthesized-but-seeded personal rules: the same user id always yields the same rules (seed = sha256(salt:user_id)).

Real history decides what a user likes; these rules decide what the assistant may do on their behalf.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import random

from ..jsonl import seeded_rng
from .schema import (EVENT_TYPES, PRIVACY_LEVELS, SERVICES, ApprovalRules, CalendarEvent, Contact, ForgetRequest,
                     MemoryFact, NotificationRules, PrivacyRules)

RULE_SALT = "muse-rules-v1"
FIRST_NAMES = ("Avery", "Blake", "Camila", "Dmitri", "Elena", "Farah", "Gustavo", "Hana", "Idris", "Jonas", "Keiko",
               "Luca", "Maya", "Nikhil", "Olga", "Priya", "Quentin", "Rosa", "Samir", "Tamsin", "Umar", "Vera",
               "Wen", "Ximena", "Yusuf", "Zara")
LAST_NAMES = ("Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Castillo", "Brennan", "Iyer", "Kowalski",
              "Achebe", "Fischer", "Varga", "Saito", "Delgado", "Osei")
RELATIONSHIPS = ("family", "family", "close_friend", "close_friend", "coworker", "coworker", "service_provider")
CALENDAR_TITLES = ("Team standup", "Dentist appointment", "Quarterly review", "Yoga class", "Parent-teacher meeting",
                   "Call with landlord", "Project deadline sync", "Lunch with a friend", "Flight check-in reminder")
THRESHOLDS = (25, 40, 50, 75, 100, 150, 200, 300)
# Default clearance per recipient type before per-user jitter (index into PRIVACY_LEVELS).
BASE_CLEARANCE = {"family": 3, "close_friend": 2, "coworker": 1, "service_provider": 0, "public": 0}


def user_rng(user_id: str, part: str) -> random.Random:
    return seeded_rng(RULE_SALT, user_id, part)


def parse_time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def between(start: str, end: str, rng: random.Random) -> str:
    a, b = parse_time(start), parse_time(end)
    if b <= a:
        return a.date().isoformat()
    return (a + (b - a) * rng.uniform(0.2, 0.8)).date().isoformat()


def approval_rules(user_id: str, first_stamp: str, cut_stamp: str) -> ApprovalRules:
    rng = user_rng(user_id, "approval")
    threshold = rng.choice(THRESHOLDS)
    rules = ApprovalRules(spend_threshold=threshold, new_recipient=rng.random() < 0.85,
                          irreversible=rng.random() < 0.9)
    if rng.random() < 0.5:
        higher = [value for value in THRESHOLDS if value > threshold]
        lower = [value for value in THRESHOLDS if value < threshold]
        choices = higher if higher and (rng.random() < 0.6 or not lower) else lower
        if choices:
            rules.updated_threshold = rng.choice(choices)
            rules.threshold_update_date = between(first_stamp, cut_stamp, rng)
    return rules


def privacy_rules(user_id: str) -> PrivacyRules:
    rng = user_rng(user_id, "privacy")
    clearance = {}
    for recipient, base in BASE_CLEARANCE.items():
        level = min(len(PRIVACY_LEVELS) - 2, max(0, base + rng.choice((-1, 0, 0, 0, 1))))
        clearance[recipient] = PRIVACY_LEVELS[0 if recipient == "public" else level]
    return PrivacyRules(clearance=clearance)


def assign_privacy(user_id: str, facts: list[MemoryFact]) -> list[MemoryFact]:
    rng = user_rng(user_id, "fact-privacy")
    return [fact.model_copy(update={"privacy": rng.choices(PRIVACY_LEVELS, weights=(2, 2, 3, 2, 2))[0]})
            for fact in facts]


def contacts(user_id: str) -> list[Contact]:
    rng = user_rng(user_id, "contacts")
    people = []
    firsts = rng.sample(FIRST_NAMES, rng.randint(4, 7))
    for first in firsts:
        last = rng.choice(LAST_NAMES)
        relationship = rng.choice(RELATIONSHIPS)
        people.append(Contact(name=f"{first} {last}", relationship=relationship,
                              email=f"{first.lower()}.{last.lower()}@example.com"))
    return people


def forget_requests(user_id: str, facts: list[MemoryFact], names: dict[str, str],
                    cut_stamp: str) -> list[ForgetRequest]:
    """One or two early facts the user asked the assistant to forget (never the only fact)."""
    rng = user_rng(user_id, "forget")
    eligible = [fact for fact in facts if fact.item_id in names]
    if len(eligible) < 3:
        return []
    chosen = rng.sample(eligible, 1 if rng.random() < 0.6 else 2)
    out = []
    for fact in chosen:
        name = names[fact.item_id]
        date = between(fact.date + "T00:00:00+00:00", cut_stamp, rng)
        phrase = rng.choice(("Please forget that I ever went to {}.", "Delete anything you remember about {}.",
                             "Forget my history with {}; don't bring it up again."))
        out.append(ForgetRequest(date=date, fact_id=fact.fact_id, item_id=fact.item_id, item_name=name,
                                 text=phrase.format(name)))
    return out


def notification_rules(user_id: str, people: list[Contact], first_stamp: str, cut_stamp: str) -> NotificationRules:
    rng = user_rng(user_id, "notify")
    start, end = rng.choice(((22, 7), (23, 7), (21, 6), (22, 8), (0, 6)))
    vip = [person.name for person in people if person.relationship == "family" and rng.random() < 0.7]
    max_level = {event: rng.choice((1, 2, 2, 3, 3)) for event in EVENT_TYPES}
    rules = NotificationRules(quiet_start=start, quiet_end=end, vip_contacts=vip, max_level=max_level,
                              meeting_cap=rng.choice((1, 1, 2)))
    if rng.random() < 0.4:
        new_start, new_end = rng.choice([pair for pair in ((20, 7), (21, 9), (23, 6), (22, 6)) if pair != (start, end)])
        rules.updated_quiet_start, rules.updated_quiet_end = new_start, new_end
        rules.quiet_update_date = between(first_stamp, cut_stamp, rng)
    return rules


def calendar(user_id: str, cut_stamp: str) -> list[CalendarEvent]:
    rng = user_rng(user_id, "calendar")
    day = parse_time(cut_stamp).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    events = []
    for title in rng.sample(CALENDAR_TITLES, rng.randint(3, 5)):
        start = day + timedelta(days=rng.randint(0, 6), hours=rng.randint(8, 18) - day.hour)
        end = start + timedelta(minutes=rng.choice((30, 45, 60, 90)))
        events.append(CalendarEvent(title=title, start=start.isoformat(timespec="minutes"),
                                    end=end.isoformat(timespec="minutes")))
    return sorted(events, key=lambda event: event.start)


def connected_services(user_id: str) -> list[str]:
    rng = user_rng(user_id, "services")
    return [service for service in SERVICES if rng.random() < 0.8] or ["email"]
