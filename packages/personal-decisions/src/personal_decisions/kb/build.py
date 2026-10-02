"""`build`: raw tables -> knowledge base directory.

Layout (also the layout of every shard written by `split`):
  users.jsonl     one UserProfile per line (time-split real history + seeded personal rules)
  items.jsonl     the item catalog with popularity (shared world; no user text)
  emails.jsonl    distractor emails
  airports.jsonl, routes.jsonl, airlines.json   OpenFlights world
  manifest.json   counts, SHA-256 of every file, build parameters and extractor statistics
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from ..jsonl import file_sha256, seeded_rng, write_json, write_jsonl
from . import raw_tables, rules
from .memory import Extractor, category_label, extract_facts
from .schema import Airport, Interaction, Item, PreferenceUpdate, Route, UserProfile

KB_FILES = ("users.jsonl", "items.jsonl", "emails.jsonl", "airports.jsonl", "routes.jsonl", "airlines.json")


def time_split(history: list[Interaction], memory_fraction: float) -> tuple[list[Interaction], list[Interaction]]:
    ordered = sorted(history, key=lambda row: (row.timestamp, row.item_id))
    cut = min(len(ordered) - 1, max(3, round(len(ordered) * memory_fraction)))
    return ordered[:cut], ordered[cut:]


def opinion_updates(user_id: str, early: list[Interaction], later: list[Interaction],
                    items: dict[str, Item]) -> list[PreferenceUpdate]:
    """Real preference changes: a category the user rated >=4 on average early and <=2 later, or the reverse."""
    def means(rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[items[row.item_id].category].append(row.rating)
        return {category: sum(values) / len(values) for category, values in grouped.items()}

    before, after = means(early), means(later)
    updates = []
    for category in sorted(set(before) & set(after)):
        old, new = before[category], after[category]
        if not ((old >= 4 and new <= 2) or (old <= 2 and new >= 4)):
            continue
        date = min(row.timestamp for row in later if items[row.item_id].category == category)[:10]
        label = category_label(category)
        text = (f"Since {date}, no longer enjoys {label}." if new < old
                else f"Since {date}, has come around to {label} and now enjoys it.")
        updates.append(PreferenceUpdate(update_id=f"{user_id}/u{len(updates)}", date=date, kind="category_opinion",
                                        text=text, category=category, old=round(old, 2), new=round(new, 2)))
    return updates


def home_airport(user_id: str, city: str | None, airports: list[Airport], route_counts: Counter) -> str | None:
    if not airports:
        return None
    served = [airport for airport in airports if route_counts.get(airport.iata)]
    local = [airport for airport in served if city and airport.city.casefold() == city.casefold()]
    if local:
        return max(local, key=lambda airport: (route_counts[airport.iata], airport.iata)).iata
    hubs = sorted(served, key=lambda airport: (-route_counts[airport.iata], airport.iata))[:50]
    return seeded_rng(rules.RULE_SALT, user_id, "airport").choice(hubs).iata if hubs else None


def build_user(user_id: str, history: list[Interaction], items: dict[str, Item], airports: list[Airport],
               route_counts: Counter, memory_fraction: float, extractor: Extractor | None,
               stats: dict) -> UserProfile | None:
    early, later = time_split(history, memory_fraction)
    if not any(row.rating >= 4 for row in later):
        return None
    cities = Counter(items[row.item_id].city for row in history if items[row.item_id].city)
    city = cities.most_common(1)[0][0] if cities else None
    facts = rules.assign_privacy(user_id, extract_facts(user_id, early, items, extractor, stats))
    first, cut = early[0].timestamp, early[-1].timestamp
    people = rules.contacts(user_id)
    names = {item_id: item.name for item_id, item in items.items()}
    return UserProfile(
        user_id=user_id, city=city, home_airport=home_airport(user_id, city, airports, route_counts),
        connected_services=rules.connected_services(user_id), cut_timestamp=cut, memory_interactions=early,
        later_interactions=later, memory_facts=facts,
        preference_updates=opinion_updates(user_id, early, later, items),
        approval=rules.approval_rules(user_id, first, cut), privacy=rules.privacy_rules(user_id),
        forget_requests=rules.forget_requests(user_id, facts, names, cut),
        notifications=rules.notification_rules(user_id, people, first, cut), contacts=people,
        calendar=rules.calendar(user_id, later[0].timestamp))


# (user id, [(early review text, item)]) jobs -> an Extractor (e.g. llm_extract.prefetch over the LLM API).
Prefetcher = Callable[[list[tuple[str, list[tuple[str, Item]]]]], Extractor]


def build_kb(raw: Path, out: Path, *, min_history: int = 5, memory_fraction: float = 0.6,
             extractor: Extractor | None = None, max_users: int | None = None, user_ids: set[str] | None = None,
             prefetcher: Prefetcher | None = None) -> dict:
    """`user_ids` restricts the KB to a chosen population; `prefetcher` builds the extractor from the early
    reviews of exactly the users that will be built (used for batched, concurrent LLM extraction)."""
    raw, out = Path(raw), Path(out)
    interactions = raw_tables.load_interactions(raw)
    items = {item.item_id: item for item in raw_tables.with_popularity(raw_tables.load_items(raw), interactions)}
    airports, routes = raw_tables.load_airports(raw), raw_tables.load_routes(raw)
    route_counts = Counter(route.src for route in routes)
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        if row.item_id in items and (user_ids is None or row.user_id in user_ids):
            by_user[row.user_id].append(row)
    stats: dict = {"users_seen": len(by_user), "skipped_short_history": 0, "skipped_no_liked_choice": 0}
    if user_ids is not None:
        stats["population_requested"] = len(user_ids)
    selected, early_of = [], {}
    for user_id in sorted(by_user):
        history = dedupe(by_user[user_id])
        if len(history) < min_history:
            stats["skipped_short_history"] += 1
            continue
        early, later = time_split(history, memory_fraction)
        if not any(row.rating >= 4 for row in later):
            stats["skipped_no_liked_choice"] += 1
            continue
        selected.append(user_id)
        early_of[user_id] = early
        if max_users and len(selected) >= max_users:
            break
    if prefetcher is not None:
        extractor = prefetcher([(user_id, [(row.text, items[row.item_id]) for row in early_of[user_id]
                                           if row.text.strip()]) for user_id in selected])
    users = [build_user(user_id, dedupe(by_user[user_id]), items, airports, route_counts, memory_fraction, extractor,
                        stats) for user_id in selected]
    write_kb(out, users, list(items.values()), raw_tables.load_emails(raw), airports, routes,
             raw_tables.load_airlines(raw))
    manifest = {"kind": "muse-kb", "min_history": min_history, "memory_fraction": memory_fraction,
                "extractor": "llm" if extractor else "rule", "stats": stats, **file_manifest(out)}
    write_json(out / "manifest.json", manifest)
    return manifest


def dedupe(history: list[Interaction]) -> list[Interaction]:
    """Keep each (user, item) once: the latest interaction, since it reflects the current opinion."""
    latest: dict[str, Interaction] = {}
    for row in sorted(history, key=lambda row: row.timestamp):
        latest[row.item_id] = row
    return list(latest.values())


def write_kb(out: Path, users: list[UserProfile], items: list[Item], emails, airports: list[Airport],
             routes: list[Route], airlines: dict[str, str]) -> None:
    write_jsonl(out / "users.jsonl", (user.model_dump(mode="json") for user in users))
    write_jsonl(out / "items.jsonl", (item.model_dump(mode="json") for item in items))
    write_jsonl(out / "emails.jsonl", (email.model_dump(mode="json") for email in emails))
    write_jsonl(out / "airports.jsonl", (airport.model_dump(mode="json") for airport in airports))
    write_jsonl(out / "routes.jsonl", (route.model_dump(mode="json") for route in routes))
    write_json(out / "airlines.json", airlines)


def file_manifest(folder: Path) -> dict:
    """Per-file SHA-256 and line counts (never contents)."""
    files = {}
    for name in KB_FILES:
        path = folder / name
        if path.is_file():
            count = None
            if name.endswith(".jsonl"):
                with path.open(encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
            files[name] = {"sha256": file_sha256(path), "count": count}
    return {"files": files}
