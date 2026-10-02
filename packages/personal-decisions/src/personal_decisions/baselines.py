"""`shortcut-baselines`: accuracy of trivial predictors per family on a records file.

Predictors: majority (the most common label of that family in the file), first_option (first criteria key; `false`
for noul, level 0 for score) and, for choice questions with option features (pick_option), one predictor per cheap
feature the candidate sampler uses: most_popular, least_popular, cheapest, priciest, medoid (smallest summed
log-price + log-popularity distance to the other options) and centroid (nearest to the options' mean point).
Only hard-labelled questions count; soft-labelled ones are reported separately. The evaluation must beat every one of
these by design.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from .jsonl import read_jsonl

PREDICTORS = ("majority", "most_popular", "least_popular", "cheapest", "priciest", "medoid", "centroid",
              "first_option")


def key(label) -> str:
    return str(label).lower() if isinstance(label, bool) else str(label)


def first_option(question: dict):
    if question["type"] == "choice":
        return next(iter(question["criteria"]))
    return False if question["type"] == "noul" else 0


def by_feature(features: dict | None, name: str, lowest: bool):
    if not features:
        return None
    values = [(option, stats.get(name)) for option, stats in features.items() if stats.get(name) is not None]
    if not values:
        return None
    pick = min if lowest else max
    return pick(values, key=lambda pair: pair[1])[0]


def point(stats: dict) -> tuple[float, float]:
    price = stats.get("price")
    return (math.log1p(price) if price is not None else 0.0, math.log1p(stats.get("popularity") or 0))


def by_geometry(features: dict | None, kind: str):
    """medoid: least summed distance to the other options; centroid: nearest to the mean point."""
    if not features:
        return None
    points = {option: point(stats) for option, stats in features.items()}
    if kind == "centroid":
        centre = tuple(sum(values) / len(values) for values in zip(*points.values()))
        return min(points, key=lambda option: math.dist(points[option], centre))

    def gap(a, b):
        price = abs(points[a][0] - points[b][0]) if None not in (features[a].get("price"), features[b].get("price")) \
            else 1.0
        return price + abs(points[a][1] - points[b][1])
    return min(points, key=lambda option: sum(gap(option, other) for other in points if other != option))


def predict(predictor: str, question: dict, features: dict | None, majority: str):
    if predictor == "majority":
        return majority
    if predictor == "first_option":
        return first_option(question)
    if predictor in ("medoid", "centroid"):
        return by_geometry(features, predictor)
    name = "popularity" if predictor.endswith("popular") else "price"
    return by_feature(features, name, lowest=predictor in ("least_popular", "cheapest"))


def rows(records: list[dict]):
    for record in records:
        meta = record.get("meta", {})
        for qid, question in record["questions"].items():
            family = meta.get("families", {}).get(qid) or question.get("src", "unknown").removeprefix("muse/")
            yield family, question, meta.get("option_features", {}).get(qid)


def shortcut_baselines(path: str | Path) -> dict:
    records = list(read_jsonl(path))
    hard = [(family, question, features) for family, question, features in rows(records)
            if "soft_label" not in question]
    soft = Counter(family for family, question, _ in rows(records) if "soft_label" in question)
    counts: dict[str, Counter] = defaultdict(Counter)
    for family, question, _ in hard:
        counts[family][key(question["label"])] += 1
    majority = {family: counter.most_common(1)[0][0] for family, counter in counts.items()}
    chance: dict[str, list[float]] = defaultdict(list)
    table: dict[str, dict] = {}
    for family in sorted(counts):
        items = [(question, features) for name, question, features in hard if name == family]
        entry = {"n": len(items), "soft_labelled_excluded": soft.get(family, 0)}
        for predictor in PREDICTORS:
            predictions = [predict(predictor, question, features, majority[family]) for question, features in items]
            scored = [(guess, question) for guess, (question, _) in zip(predictions, items) if guess is not None]
            entry[predictor] = None if not scored else round(
                sum(key(guess) == key(question["label"]) for guess, question in scored) / len(scored), 4)
        for question, _ in items:
            size = len(question["criteria"]) if question["type"] != "noul" else 2
            chance[family].append(1 / size)
        entry["chance"] = round(sum(chance[family]) / len(chance[family]), 4)
        table[family] = entry
    macro = {}
    for predictor in (*PREDICTORS, "chance"):
        values = [entry[predictor] for entry in table.values() if entry.get(predictor) is not None]
        macro[predictor] = round(sum(values) / len(values), 4) if values else None
    return {"records": len(records), "per_family": table, "macro_over_families_with_value": macro,
            "best_shortcut_per_family": {family: max((entry[p] or 0) for p in PREDICTORS) for family, entry in
                                         table.items()}}
