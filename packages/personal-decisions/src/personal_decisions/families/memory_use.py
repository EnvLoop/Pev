"""apply_memory and forgotten_violation (noul): when to use memory, and when using it breaks a forget request.

apply_memory: is a remembered fact relevant to the current request? Half the requests share the fact's category
(true); the rest are other product departments (never another restaurant category) or unrelated tasks (false:
guards against over-personalization).

forgotten_violation: does a proposed action rely on something the user asked the assistant to forget? Negatives use a
different, non-forgotten remembered item (same category when possible). The removed variant is a contrast pair: the
forget request is withheld (must not appear) and the fact is in memory, so the same action is not a violation.
"""
from __future__ import annotations

from ..kb.memory import category_label
from ..kb.schema import MemoryFact
from .base import Context, Fragment, Question, active_facts, pending_label

APPLY = "apply_memory"
FORGOTTEN = "forgotten_violation"
APPLY_VARIANTS = ("clean",)
FORGOTTEN_VARIANTS = ("clean", "removed")
APPLY_CRITERIA = {"true": "Relevant: the assistant should use it",
                  "false": "Unrelated: using it would over-personalize"}
FORGOTTEN_CRITERIA = {"true": "Relies on something the user asked to forget",
                      "false": "Does not rely on anything the user asked to forget"}
UNRELATED_REQUESTS = ("Summarize my unread email from this morning.", "What time is my {event}?",
                      "Remind me to call {contact} tomorrow.", "Move my {event} to next week.")
ACTIONS = ("Recommend {item} again, since it came up in your history.",
           "Tell {contact} about your experience with {item}.",
           "Add {item} to this week's personalized suggestions based on your past experience with it.")


def request_for(category: str, domain: str, city: str | None) -> str:
    label = category_label(category)
    if domain == "restaurant":
        where = f" in {city}" if city else ""
        return f"Find me a {label}{where} for Saturday night."
    return f"I need to order something from {label}; what should I get?"


def unrelated_request(ctx: Context) -> str:
    event = ctx.user.calendar[0].title if ctx.user.calendar else "dentist appointment"
    contact = ctx.user.contacts[0].name if ctx.user.contacts else "my sister"
    return ctx.rng.choice(UNRELATED_REQUESTS).format(event=event, contact=contact)


def other_category(ctx: Context, fact: MemoryFact) -> tuple[str, str, str | None] | None:
    """(category, domain, city) of an unrelated request. Product facts: another top-level product department.
    Restaurant facts: a product department, never another restaurant category: dining facts (service, waits, taste)
    bear on any restaurant search, so a different cuisine would make "unrelated" ambiguous."""
    items = ctx.shard.by_domain.get("product", [])
    sample = ctx.rng.sample(items, min(len(items), 40))
    for item in sample:
        if item.category != fact.category:
            return item.category, item.domain, item.city
    return None


def build_apply(ctx: Context) -> Fragment | None:
    facts = [fact for fact in active_facts(ctx.user) if fact.item_id in ctx.shard.items]
    if not facts:
        return None
    fact = ctx.rng.choice(facts)
    item = ctx.shard.items[fact.item_id]
    relevant = ctx.rng.random() < 0.5
    if relevant:
        text = request_for(fact.category, fact.domain, item.city or ctx.user.city)
    else:
        other = other_category(ctx, fact) if ctx.rng.random() < 0.6 else None
        text = request_for(other[0], other[1], other[2] or ctx.user.city) if other else unrelated_request(ctx)
    label = pending_label("Request", ctx.letter)
    required = [label, item.name] + ([fact.evidence] if len(fact.evidence) >= 12 else [])
    question = Question(APPLY, "noul", f"{label}: should the assistant take the remembered fact about {item.name} "
                                       "into account for this request?", dict(APPLY_CRITERIA), relevant, required)
    fragment = Fragment(APPLY, question, {"label": label, "text": text}, memory_ids=[fact.fact_id])
    return fragment


def action_text(ctx: Context, item_name: str) -> str:
    contact = ctx.rng.choice(ctx.user.contacts).name if ctx.user.contacts else "a friend"
    return ctx.rng.choice(ACTIONS).format(item=item_name, contact=contact)


def build_forgotten(ctx: Context) -> Fragment | None:
    requests = [request for request in ctx.user.forget_requests
                if request.date <= ctx.now.date().isoformat()]
    if not requests:
        return None
    request = ctx.rng.choice(requests)
    label = pending_label("Proposed action", ctx.letter)
    # Label first (true with probability 0.5), so the removed contrast (always false) cannot skew the balance.
    violate = ctx.rng.random() < 0.5
    contrast = ctx.variant == "removed" and not violate
    forget_line = f"{request.date}: {request.text}"
    question = Question(FORGOTTEN, "noul", f"{label}: does this action rely on something the user asked the assistant "
                                           "to forget?", dict(FORGOTTEN_CRITERIA), violate, [label, request.item_name])
    fragment = Fragment(FORGOTTEN, question, {"label": label, "text": ""})
    if violate or contrast:
        fragment.pending["text"] = action_text(ctx, request.item_name)
        if contrast:
            fragment.memory_ids = [request.fact_id]
            question.label, question.forbidden, question.variant = False, [request.text], "removed"
            return fragment
        fragment.excluded_memory_ids = [request.fact_id]
        fragment.add("forget_requests", forget_line)
        return fragment
    forgotten_items = {entry.item_id for entry in ctx.user.forget_requests}
    others = [fact for fact in active_facts(ctx.user) if fact.item_id in ctx.shard.items
              and fact.item_id not in forgotten_items and ctx.shard.items[fact.item_id].name != request.item_name]
    if not others:
        return None
    same = [fact for fact in others if ctx.shard.items[fact.item_id].category == ctx.shard.items.get(
        request.item_id, ctx.shard.items[fact.item_id]).category]
    fact = ctx.rng.choice(same or others)
    name = ctx.shard.items[fact.item_id].name
    fragment.pending["text"] = action_text(ctx, name)
    fragment.memory_ids = [fact.fact_id]
    fragment.excluded_memory_ids = [request.fact_id]
    fragment.add("forget_requests", forget_line)
    question.required.append(name)
    return fragment
