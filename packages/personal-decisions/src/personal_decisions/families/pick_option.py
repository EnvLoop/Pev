"""pick_option (choice): which candidate will the user really choose?

Target: a real later interaction rated >= 4. Candidates: the target, the user's real later low-rated (<= 2) item of the
same category when there is one and the target can still be balanced around it, and untouched items of the same
category (and city for restaurants). The target sits among them where a sampled distractor would on price,
popularity, medoid and centroid distance (families/candidates.py), so no cheap-feature shortcut carries signal.

Variants: override (a later preference update supersedes an older one), removed (every memory fact about the
category is withheld and must not appear; soft label uniform over the candidates).
"""
from __future__ import annotations

import math

from ..kb.memory import category_label
from ..kb.schema import Item
from .base import Context, Fragment, Question, active_facts, item_price_text, pending_label, touched_items, whole_money
from .candidates import price_value, sample_options

FAMILY = "pick_option"
VARIANTS = ("clean", "override", "removed")
LOW_SHARE = 1.0  # chance of offering a same-category real later low-rated item when the user has one


def subcategory(item: Item) -> str | None:
    """The most specific Amazon subcategory, when listed (restaurant categories are already specific)."""
    values = item.attributes.get("subcategories") if isinstance(item.attributes, dict) else None
    return values[-1] if isinstance(values, list) and values and isinstance(values[-1], str) else None


def describe(item: Item) -> str:
    where = f", {item.city}" if item.domain == "restaurant" and item.city else ""
    return f"{category_label(item.category)}, {item_price_text(item)}{where}"


def candidates(ctx: Context, target: Item) -> list[Item] | None:
    """The target, in half of the questions (seeded) the user's own real later low-rated item, and untouched
    same-category distractors; see `candidates.sample_options` for the shortcut-free sampling."""
    user, items, rng = ctx.user, ctx.shard.items, ctx.rng
    touched = touched_items(user)
    low = [items[row.item_id] for row in user.later_interactions
           if row.rating <= 2 and row.item_id in items and row.item_id != target.item_id
           and items[row.item_id].domain == target.domain and items[row.item_id].name != target.name
           and (price_value(items[row.item_id]) is None) == (price_value(target) is None)]
    # Only a same-category low item: the request names the category, so any other category would give it away.
    same_category = [item for item in low if item.category == target.category]
    include_low = bool(same_category) and rng.random() < LOW_SHARE
    chosen_low = [rng.choice(same_category)] if include_low else []
    size = rng.choice((3, 4, 5))
    wanted = size - 1 - len(chosen_low)
    pool = [item for item in ctx.shard.similar_items(target) if item.item_id not in touched]
    narrow = subcategory(target)
    same_subcategory = [item for item in pool if narrow and subcategory(item) == narrow]
    if len(same_subcategory) >= max(wanted * 3, 8):
        pool = same_subcategory
    options = sample_options(target, chosen_low, pool, size, rng)
    if options is None:
        return None
    rng.shuffle(options)
    return options


def request_text(target: Item) -> str:
    if target.domain == "restaurant":
        where = f" in {target.city}" if target.city else ""
        return f"Pick one {category_label(target.category)}{where} for me to try next."
    return f"Help me choose one {category_label(target.category)} product to buy."


def override_lines(ctx: Context, target: Item, options: list[Item], fragment: Fragment) -> list[str] | None:
    """Anchors of a preference update that supersedes an older preference; None when none applies.

    A real category-opinion change is used when the KB has one. Otherwise an older budget is lifted to cover every
    candidate. With equal odds the stale budget excluded the target (following it picks wrong) or only excluded a
    pricier candidate, so "pick what the old budget excluded" is no shortcut and the new budget singles out nothing.
    """
    real = [update for update in ctx.user.preference_updates
            if update.kind == "category_opinion" and update.category == target.category
            and update.date <= ctx.now.date().isoformat()]
    if real:
        fragment.add("preference_updates", *(f"{update.date}: {update.text}" for update in real))
        return [real[-1].text]
    facts = active_facts(ctx.user)
    if not facts:
        return None
    label = category_label(target.category)
    first, later = ctx.rng.choice(facts).date, ctx.now.date().isoformat()
    exclude_target = ctx.rng.random() < 0.5
    if target.price is not None:
        prices = sorted({item.price for item in options if item.price is not None})
        position = prices.index(target.price)
        cut = position if exclude_target else position + 1  # budget between prices[cut - 1] and prices[cut]
        if not 0 < cut < len(prices):
            cut = position + 1 if exclude_target else position
        if not 0 < cut < len(prices) or int(prices[cut]) <= prices[cut - 1]:
            return None
        old, new = int(prices[cut]), int(math.ceil(prices[-1] * 1.25 / 5) * 5)
        before, after = whole_money(old), whole_money(new)
        fragment.add("preference_updates", f"{first}: Asked to keep {label} purchases under {before}.",
                     f"{later}: Raised the {label} budget to {after}; ignore the older limit.")
        return [before, after]
    levels = sorted({item.price_level for item in options if item.price_level})
    if not target.price_level or len(levels) < 2:
        return None
    position = levels.index(target.price_level)
    cut = position if exclude_target else position + 1
    if not 0 < cut < len(levels):
        cut = position + 1 if exclude_target else position
    if not 0 < cut < len(levels):
        return None
    before, after = f"price level {levels[cut - 1]} of 4", "price level 4 of 4"
    fragment.add("preference_updates", f"{first}: Only wanted {label} places at {before} or cheaper.",
                 f"{later}: Now happy to go up to {after} for {label}.")
    return [before, after]


def build(ctx: Context) -> Fragment | None:
    row = ctx.user.later_interactions[ctx.target_index]
    target = ctx.shard.items.get(row.item_id)
    if target is None or row.rating < 4:
        return None
    options = candidates(ctx, target)
    if options is None:
        return None
    label = pending_label("Request", ctx.letter)
    criteria = {item.name: describe(item) for item in options}
    question = Question(
        family=FAMILY, type="choice", criteria=criteria, label=target.name, required=[item.name for item in options],
        instructions=f"{label}: which option will this user actually choose and end up rating highly?",
        option_features={item.name: {"price": price_value(item), "popularity": item.popularity} for item in options})
    fragment = Fragment(FAMILY, question, {"label": label, "text": request_text(target),
                                           "options": [f"{name} ({text})" for name, text in criteria.items()]})
    option_ids = {item.item_id for item in options}
    relevant = [fact for fact in active_facts(ctx.user)
                if fact.category == target.category or fact.item_id in option_ids]
    variant = ctx.variant
    if variant == "removed" and relevant:
        fragment.excluded_memory_ids = [fact.fact_id for fact in relevant]
        question.forbidden = sorted({fact.evidence for fact in relevant if len(fact.evidence) >= 12})
        question.soft_label = {name: round(1 / len(options), 4) for name in criteria}
    else:
        variant = "override" if variant == "override" else "clean"
        chosen = sorted(ctx.rng.sample(relevant, min(3, len(relevant))), key=lambda fact: fact.date)
        fragment.memory_ids = [fact.fact_id for fact in chosen]
        question.required += [fact.evidence for fact in chosen[:2] if len(fact.evidence) >= 12]
        if variant == "override":
            anchors = override_lines(ctx, target, options, fragment)
            if anchors is None:
                variant = "clean"
            else:
                question.required += anchors
    question.variant = variant
    return fragment

