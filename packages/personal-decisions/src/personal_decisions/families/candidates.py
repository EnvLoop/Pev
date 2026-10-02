"""pick_option candidate sampling with no cheap-feature shortcut.

Nearest neighbours of the target put the target at the centre of its candidates (a medoid shortcut), and real
choices skew popular (a most-popular shortcut). The target should instead sit among its candidates where a sampled
distractor sits, on every cheap feature `shortcut-baselines` has a predictor for: price, popularity, medoid distance
and distance to the centroid. Per question:

1. SAMPLES option sets are proposed: half "shell" sets (distractors about as far from a random anchor near the
   target as the target is, so the target is not the centre) and half "directed" sets (a random number of
   distractors below the target on popularity or price, so extreme target positions are proposed too).
2. For each set and feature, the chance that an argmin/argmax predictor facing the options in random order finds the
   target first, middle or last is computed (ties shared).
3. The sets are weighted by iterative proportional fitting so that, per feature, the target is first with weight
   1/size and last with 1/size (with a fixed real item, the rates a random sampled option has next to it), and one
   set is drawn by weight. A question whose target cannot be balanced within FIT_TOLERANCE is skipped, whatever its
   draw, so skipping favours no position.

Distractors also match the target on price availability, so "the only priced option" is no cue either. A fixed real
item (the user's same-category low-rated later choice) keeps its real position; it often makes balancing impossible
(it sits on one side of the target on every feature), so fewer questions carry it than requested.
"""
from __future__ import annotations

import math
import random

from ..kb.schema import Item

POSITIONS = ("first", "middle", "last")
FEATURES = ("popularity", "price", "medoid", "centroid")
NEAR = 20  # anchors are drawn among the target's NEAR nearest distractor-eligible items
SHELL = 3  # distractors are drawn from the wanted * SHELL items whose distance to the anchor is closest to the target's
WIDE = 150  # directed proposals draw from the target's WIDE nearest items
FIT_TOLERANCE = 0.02  # largest allowed gap between the raked and the goal position rates
SAMPLES = 120  # option sets proposed per question (half shell, half directed)


def price_value(item: Item) -> float | None:
    if item.price is not None:
        return item.price
    return float(item.price_level) if item.price_level else None


def distance(a: Item, b: Item) -> float:
    if a.price is not None and b.price is not None:
        price = abs(math.log1p(a.price) - math.log1p(b.price))
    elif a.price_level and b.price_level:
        price = 0.7 * abs(a.price_level - b.price_level)
    else:
        price = 1.0
    return price + abs(math.log1p(a.popularity) - math.log1p(b.popularity))


def point(item: Item) -> tuple[float, float]:
    price = price_value(item)
    return (math.log1p(price) if price is not None else 0.0, math.log1p(item.popularity))


def gap(a: Item, b: Item) -> float:
    """The log-price + log-popularity distance `shortcut-baselines` uses for its medoid predictor."""
    pa, pb = price_value(a), price_value(b)
    price = abs(math.log1p(pa) - math.log1p(pb)) if pa is not None and pb is not None else 1.0
    return price + abs(math.log1p(a.popularity) - math.log1p(b.popularity))


def feature_values(options: list[Item]) -> dict[str, list[float]]:
    """Per feature, one value per option; lower is what the matching shortcut predictor picks."""
    points = [point(item) for item in options]
    centre = tuple(sum(values) / len(values) for values in zip(*points))
    out = {"popularity": [-float(item.popularity) for item in options],
           "medoid": [sum(gap(item, other) for other in options if other is not item) for item in options],
           "centroid": [math.dist(p, centre) for p in points]}
    prices = [price_value(item) for item in options]
    if all(price is not None for price in prices):
        out["price"] = prices
    return out


def member_classes(options: list[Item]) -> list[list[dict[str, float] | None]]:
    """Per option and feature, the probability that an argmin predictor facing the options in random order finds
    that option first, last or in the middle; ties are shared. None for a feature some option lacks."""
    values, size = feature_values(options), len(options)
    out = []
    for index in range(size):
        row = []
        for name in FEATURES:
            if name not in values:
                row.append(None)
                continue
            column = values[name]
            below = sum(1 for value in column if value < column[index])
            tied = sum(1 for value in column if value == column[index])  # includes the option itself
            share = {"first": 0.0, "middle": 0.0, "last": 0.0}
            for rank in range(below, below + tied):
                share[position(rank, size)] += 1 / tied
            row.append(share)
        out.append(row)
    return out


def sample_options(target: Item, fixed: list[Item], pool: list[Item], size: int,
                   rng: random.Random) -> list[Item] | None:
    """[target, *fixed, distractors...] (unshuffled) whose target positions are those of a random member, or None."""
    wanted = size - 1 - len(fixed)
    has_price = price_value(target) is not None
    names = {target.name, *(item.name for item in fixed)}
    pool = list({item.name: item for item in pool
                 if (price_value(item) is not None) == has_price and item.name not in names}.values())
    if wanted < 1 or len(pool) < wanted * SHELL:
        return None
    ranked = sorted(pool, key=lambda item: (distance(item, target), item.item_id))
    near, wide = ranked[:NEAR], ranked[:WIDE]
    draws, classes, fixed_classes = [], [], []
    for number in range(SAMPLES):
        chosen = directed(target, wide, wanted, rng) if number % 2 else None
        if chosen is None:
            anchor = rng.choice(near)
            radius = distance(target, anchor)
            shell = sorted((item for item in pool if item is not anchor),
                           key=lambda item: (abs(distance(item, anchor) - radius), item.item_id))[:wanted * SHELL]
            chosen = rng.sample(shell, wanted)
        draws.append([target, *fixed, *chosen])
        found = member_classes(draws[-1])
        classes.append(found[0])
        fixed_classes.extend(found[1:1 + len(fixed)])
    # The target should sit where a random non-fixed member sits: each position is held by exactly one option
    # (middle by size - 2), a fixed real item keeps its real position and the other options share the rest equally.
    # With no fixed item that is 1/size first and 1/size last on every feature.
    slots = {"first": 1.0, "middle": size - 2.0, "last": 1.0}
    goal = []
    for index in range(len(FEATURES)):
        if classes[0][index] is None:
            goal.append(None)
            continue
        taken = {name: sum(entry[index][name] for entry in fixed_classes) / len(draws) for name in POSITIONS}
        goal.append({name: max(0.0, slots[name] - taken[name]) / (size - len(fixed)) for name in POSITIONS})
    weights = rake(classes, goal)
    if weights is None:
        return None
    return draws[rng.choices(range(len(draws)), weights=weights)[0]]


def rake(classes: list[list[dict[str, float] | None]], goal: list[dict[str, float] | None],
         rounds: int = 40) -> list[float] | None:
    """Weights over the sampled option sets (iterative proportional fitting) so that, per feature, the weighted
    chance that the target is first / middle / last equals `goal`. None when a position the goal needs is never
    reached (the question is skipped whatever its draw, so skipping favours no position)."""
    weights = [1.0] * len(classes)
    features = [index for index, target in enumerate(goal) if target is not None]
    for index in features:
        if any(goal[index][name] > 0.02 and not any(entry[index][name] for entry in classes) for name in POSITIONS):
            return None
    for _ in range(rounds):
        for index in features:
            totals = {name: sum(weight * entry[index][name] for entry, weight in zip(classes, weights))
                      for name in POSITIONS}
            everything = sum(totals.values())
            scale = {name: goal[index][name] * everything / totals[name] if totals[name] else 0.0
                     for name in POSITIONS}
            weights = [weight * sum(entry[index][name] * scale[name] for name in POSITIONS)
                       for entry, weight in zip(classes, weights)]
    everything = sum(weights)
    for index in features:
        for name in POSITIONS:
            reached = sum(weight * entry[index][name] for entry, weight in zip(classes, weights)) / everything
            if abs(reached - goal[index][name]) > FIT_TOLERANCE:
                return None  # the features' positions cannot all be balanced for this target: skip it
    return weights


def directed(target: Item, wide: list[Item], wanted: int, rng: random.Random) -> list[Item] | None:
    """Distractors with a random number below the target on popularity or price (so extreme target positions are
    proposed too); None when the neighbourhood cannot supply that split."""
    key = rng.choice(("popularity", "price") if price_value(target) is not None else ("popularity",))
    value = (lambda item: item.popularity) if key == "popularity" else price_value
    below = [item for item in wide if value(item) < value(target)]
    above = [item for item in wide if value(item) > value(target)]
    count = rng.choice((0, wanted, rng.randint(0, wanted)))
    if len(below) < count or len(above) < wanted - count:
        return None
    return rng.sample(below, count) + rng.sample(above, wanted - count)


def position(rank: int, size: int) -> str:
    """What an argmin / argmax predictor sees: the target first, last, or in between."""
    return "first" if rank == 0 else "last" if rank == size - 1 else "middle"
