"""notify_level (score): how strongly to notify the user about an incoming event.

Level = urgency (none 0, low 1, medium 2, high 3) capped by the per-type maximum; unless the sender is a VIP contact,
quiet hours cap it at digest (1) and a calendar event in progress caps it at the meeting cap. Quiet hours and meetings
are judged at the event's own time (the question says "when it arrives"), which may differ from the current time.
Variants: override (a later change of quiet hours flips the answer), removed (the event's urgency is withheld; the
soft label is exactly uniform over the levels the unknown urgency could reach, so it is tied and never scored for
accuracy; scenarios that reach a single level are not used for it). Scenarios are drawn uniformly over the reachable
levels first, so hard labels are balanced by construction.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..kb.rules import between
from ..kb.schema import EVENT_TYPES, NOTIFY_LEVELS
from .base import Context, Fragment, Question, pending_label

FAMILY = "notify_level"
VARIANTS = ("clean", "override", "removed")
URGENCY = {"none": 0, "low": 1, "medium": 2, "high": 3}
LEVEL_NAMES = tuple(level.split(":")[0] for level in NOTIFY_LEVELS)
TYPE_LABEL = {"flight_change": "Flight changes", "price_drop": "Price drops", "package_delivery": "Package updates",
              "message": "Personal messages", "calendar_conflict": "Calendar conflicts", "bill_due": "Bill reminders"}
EVENT_TEXT = {
    "flight_change": {"none": "your airline sent its monthly newsletter",
                      "low": "the gate for your next flight changed",
                      "medium": "your next flight is delayed 3 hours",
                      "high": "your flight tonight was cancelled"},
    "price_drop": {"none": "a store you follow posted new arrivals",
                   "low": "a saved item dropped 5% in price",
                   "medium": "a saved item is 30% off until tomorrow",
                   "high": "a saved item is 60% off for the next hour only"},
    "package_delivery": {"none": "a delivery survey is available",
                         "low": "your package shipped",
                         "medium": "your package is out for delivery",
                         "high": "the courier is at the door and needs a signature now"},
    "message": {"none": "{sender} reacted to your photo",
                "low": "{sender} shared a funny video",
                "medium": "{sender} asks about weekend plans",
                "high": "{sender} says it's urgent and asks you to call back"},
    "calendar_conflict": {"none": "a calendar sync finished",
                          "low": "two optional events overlap next month",
                          "medium": "two meetings overlap tomorrow",
                          "high": "two meetings overlap and the first starts in 10 minutes"},
    "bill_due": {"none": "your phone statement is ready and nothing is due",
                 "low": "your phone bill is due in two weeks",
                 "medium": "your phone bill is due tomorrow",
                 "high": "your phone bill is overdue and a late fee applies today"}}
NEUTRAL_TEXT = {"flight_change": "an update about your flight", "price_drop": "an update about a saved item",
                "package_delivery": "an update about your package", "message": "a message from {sender}",
                "calendar_conflict": "a calendar notice", "bill_due": "a notice about your phone bill"}


def in_window(hour: int, start: int, end: int) -> bool:
    return start <= hour < end if start < end else hour >= start or hour < end


def level_for(base: int, event: str, vip: bool, quiet: bool, meeting: bool, ctx: Context) -> int:
    rules = ctx.user.notifications
    level = min(base, rules.max_level[event])
    if vip:
        return level
    if quiet:
        level = min(level, 1)
    if meeting:
        level = min(level, rules.meeting_cap)
    return level


def reachable(combo: tuple, ctx: Context) -> list[int]:
    """The distinct levels a scenario reaches over every urgency (what is left open when the urgency is withheld)."""
    event, _, in_quiet, meeting, vip = combo
    return sorted({level_for(base, event, vip, in_quiet, meeting, ctx) for base in URGENCY.values()})


def windows(ctx: Context) -> tuple[tuple[int, int], tuple[int, int] | None]:
    rules = ctx.user.notifications
    old = (rules.quiet_start, rules.quiet_end)
    if rules.quiet_update_date and rules.quiet_update_date <= ctx.now.date().isoformat():
        return old, (rules.updated_quiet_start, rules.updated_quiet_end)
    return old, None


def pick_hour(ctx: Context, quiet: tuple[int, int], want_quiet: bool, other: tuple[int, int] | None) -> int | None:
    hours = [hour for hour in range(24) if in_window(hour, *quiet) == want_quiet
             and (other is None or in_window(hour, *other) != want_quiet)]
    return ctx.rng.choice(hours) if hours else None


def scenarios(ctx: Context, quiet_options: tuple[bool, ...]) -> list[tuple[str, str, bool, bool, bool]]:
    """Every (event, urgency, in quiet hours, in a meeting, VIP sender) the user's rules can face."""
    vips = bool(ctx.user.notifications.vip_contacts)
    return [(event, urgency, quiet, meeting, vip)
            for event in EVENT_TYPES for urgency in URGENCY for quiet in quiet_options
            for meeting in ((False,) if quiet else (False, True))
            for vip in ((False, True) if event == "message" and vips else (False,))]


def choose_scenario(ctx: Context, combos: list[tuple], score) -> tuple:
    """Uniform over the reachable levels first, then over the scenarios with that level (balanced labels)."""
    by_level: dict[int, list[tuple]] = {}
    for combo in combos:
        by_level.setdefault(score(combo), []).append(combo)
    return ctx.rng.choice(by_level[ctx.rng.choice(sorted(by_level))])


def build(ctx: Context) -> Fragment | None:
    rng, rules = ctx.rng, ctx.user.notifications
    old, new = windows(ctx)
    quiet_window = new or old

    def score(combo, quiet=None):
        event, urgency, in_quiet, meeting, vip = combo
        return level_for(URGENCY[urgency], event, vip, in_quiet if quiet is None else quiet, meeting, ctx)

    variant, hour, combo = ctx.variant, None, None
    flips = [h for h in range(24) if new and in_window(h, *new) != in_window(h, *old)]
    if variant == "override" and flips:
        hour = rng.choice(flips)
        now_quiet = in_window(hour, *new)
        combos = [c for c in scenarios(ctx, (now_quiet,)) if score(c) != score(c, quiet=not now_quiet)]
        combo = choose_scenario(ctx, combos, score) if combos else None
    if combo is None:
        variant = "removed" if variant == "removed" else "clean"
        combos = scenarios(ctx, (True, False))
        if variant == "removed":
            # Withholding the urgency must leave the answer open: keep scenarios where it reaches 2+ levels.
            combos = [c for c in combos if len(reachable(c, ctx)) > 1] or combos
            variant = "removed" if any(len(reachable(c, ctx)) > 1 for c in combos) else "clean"
        combo = choose_scenario(ctx, combos, score)
        hour = pick_hour(ctx, quiet_window, combo[2], None)
    if hour is None:
        return None
    event, urgency, in_quiet, meeting, vip = combo
    others = [person.name for person in ctx.user.contacts if person.name not in rules.vip_contacts]
    sender = rng.choice(rules.vip_contacts) if vip else (rng.choice(others) if others else "Sam Rivera")
    day = ctx.now.replace(hour=hour, minute=rng.choice((5, 20, 35, 50)), second=0, microsecond=0)
    label = pending_label("Incoming event", ctx.letter)
    when = day.strftime("%Y-%m-%d %H:%M")
    fragment = Fragment(FAMILY, Question(FAMILY, "score", f"{label}: how strongly should the assistant notify the user "
                                                          "about this when it arrives?", list(NOTIFY_LEVELS), 0, [label, when]),
                        {"label": label, "text": ""})
    add_calendar(ctx, fragment, day, meeting)
    add_rules(ctx, fragment, event, old, new)
    question = fragment.question
    question.required += [f"{quiet_window[0]:02d}:00-{quiet_window[1]:02d}:00"]
    if variant == "removed":
        fragment.pending["text"] = f"At {when}: {NEUTRAL_TEXT[event].format(sender=sender)}."
        # Exactly uniform (tied) over the levels the unknown urgency could reach: the answer is genuinely unknown,
        # so the question only enters Brier / ECE, never accuracy. The label is the real scenario's level.
        levels = reachable(combo, ctx)
        question.soft_label = {str(level): round(1 / len(levels), 4) for level in levels}
        question.label = score(combo)
        question.forbidden = ["(urgency:"]
    else:
        text = EVENT_TEXT[event][urgency].format(sender=sender)
        fragment.pending["text"] = f"At {when}: {text} (urgency: {urgency})."
        question.required.append(f"(urgency: {urgency})")
        question.label = score(combo)
    question.variant = variant
    return fragment


def add_calendar(ctx: Context, fragment: Fragment, day: datetime, meeting: bool) -> None:
    titles = [event.title for event in ctx.user.calendar] or ["Team standup"]
    title = ctx.rng.choice(titles)
    if meeting:
        start = day - timedelta(minutes=ctx.rng.choice((5, 15, 25)))
    else:
        start = day + timedelta(hours=ctx.rng.choice((2, 3, 4)))
    end = start + timedelta(minutes=45)
    fragment.add("calendar", f"{title}: {start:%Y-%m-%d %H:%M} to {end:%H:%M}")
    fragment.question.required.append(f"{start:%H:%M}")


def add_rules(ctx: Context, fragment: Fragment, event: str, old: tuple[int, int], new: tuple[int, int] | None) -> None:
    rules = ctx.user.notifications
    fragment.add("rules", f"Quiet hours {old[0]:02d}:00-{old[1]:02d}:00: nothing above digest unless it's from a "
                          "VIP contact.")
    if new:
        date = rules.quiet_update_date or between(ctx.user.cut_timestamp, ctx.now.isoformat(), ctx.rng)
        fragment.add("rules", f"{date}: Changed quiet hours to {new[0]:02d}:00-{new[1]:02d}:00.")
    fragment.add("rules", f"While I'm in a calendar event, notifications go no higher than "
                          f"{LEVEL_NAMES[rules.meeting_cap]}.",
                 f"{TYPE_LABEL[event]}: never higher than {LEVEL_NAMES[rules.max_level[event]]}.")
    if rules.vip_contacts:
        fragment.add("rules", "VIP contacts: " + ", ".join(rules.vip_contacts) + ".")
    fragment.question.required.append(f"{TYPE_LABEL[event]}: never higher than")
