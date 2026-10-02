"""Restaurants from UCSD Google Local 2021 (per-state 10-core reviews + metadata) -> staging JSONL.

Keeps food businesses only, then users with >= MIN_INTERACTIONS restaurant reviews (distinct businesses)
spanning >= 30 days with >= 5 texted reviews. Reviewer names, photos and owner responses are dropped.
"""
import gzip
import json
import re
from collections import defaultdict

from common import DL, MIN_INTERACTIONS, STAGING, TEXT_MAX, anon_user, clip, iso_from_ms, write_jsonl

SRC = "google_local_2021"
STATES = ["District_of_Columbia", "Delaware", "Rhode_Island"]
FOOD = re.compile(
    r"restaurant|\bcaf[eé]\b|coffee|bakery|pizz|diner|\bdeli\b|bistro|brewpub|gastropub|bar & grill|grill\b|"
    r"fast food|sandwich|bagel|ice cream|dessert|donut|doughnut|taco|burrito|sushi|steak|barbecue|bbq|brunch|"
    r"breakfast|buffet|noodle|ramen|seafood|burger|hamburger|chicken wings|juice|tea house|bubble tea|creperie|"
    r"food court|food truck|eatery|pub\b|wine bar|oyster bar|tapas|pho\b|dumpling|hot dog|frozen yogurt|"
    r"chocolate cafe|pastry|patisserie|cupcake|poke",
    re.I,
)
NOT_FOOD = re.compile(r"supply|supplier|equipment|wholesaler|distributor|manufacturer|repair|consultant|"
                      r"service establishment|delivery service|store$|market$|bed & breakfast|dog cafe|cat cafe|internet cafe|"
                      r"bbq area|wholesale", re.I)
DAY_MS = 86_400_000


def food_categories(cats):
    return [c for c in (cats or []) if FOOD.search(c) and not NOT_FOOD.search(c)]


def city_of(address: str | None):
    parts = [p.strip() for p in (address or "").split(",")]
    return parts[-2] if len(parts) >= 3 else None


def load_items():
    items = {}
    for st in STATES:
        with gzip.open(DL / "google" / f"meta-{st}.json.gz", "rt", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                food = food_categories(m.get("category"))
                if not food or m["gmap_id"] in items:
                    continue
                price = m.get("price")
                items[m["gmap_id"]] = {
                    "item_id": f"gmap:{m['gmap_id']}", "domain": "restaurant", "name": m.get("name") or "",
                    "category": food[0], "price": None,
                    "price_level": len(price) if price and set(price) == {"$"} else None,
                    "attributes": {
                        "categories": m.get("category") or [], "state": st, "address": m.get("address"),
                        "avg_rating": m.get("avg_rating"), "num_of_reviews": m.get("num_of_reviews"),
                        "latitude": m.get("latitude"), "longitude": m.get("longitude"),
                        "description": clip(m.get("description"), 500) or None, "hours": m.get("hours"),
                        "misc": m.get("MISC"),
                    },
                    "city": city_of(m.get("address")), "source": SRC,
                }
    return items


def load_reviews(items):
    by_user = defaultdict(dict)  # original user id -> gmap_id -> latest review
    for st in STATES:
        with gzip.open(DL / "google" / f"review-{st}_10.json.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["gmap_id"] not in items or not r.get("rating") or not r.get("time"):
                    continue
                prev = by_user[r["user_id"]].get(r["gmap_id"])
                if prev is None or r["time"] > prev["time"]:
                    by_user[r["user_id"]][r["gmap_id"]] = r
    return by_user


def eligible(reviews) -> bool:
    if len(reviews) < MIN_INTERACTIONS:
        return False
    times = [r["time"] for r in reviews]
    texted = sum(1 for r in reviews if (r.get("text") or "").strip())
    return max(times) - min(times) >= 30 * DAY_MS and texted >= 5


def main() -> None:
    items = load_items()
    by_user = load_reviews(items)
    rows, used = [], set()
    for uid, revs in by_user.items():
        revs = list(revs.values())
        if not eligible(revs):
            continue
        anon = anon_user(SRC, uid)
        for r in sorted(revs, key=lambda x: x["time"]):
            used.add(r["gmap_id"])
            rows.append({
                "user_id": anon, "item_id": f"gmap:{r['gmap_id']}", "rating": int(r["rating"]),
                "timestamp": iso_from_ms(r["time"]), "text": clip(r.get("text"), TEXT_MAX), "source": SRC,
            })
    rows.sort(key=lambda r: (r["user_id"], r["timestamp"]))
    n_items = write_jsonl(STAGING / "items_restaurant.jsonl", (items[g] for g in sorted(items)))
    n_rows = write_jsonl(STAGING / "interactions_restaurant.jsonl", rows)
    print(f"food businesses {n_items} (reviewed by kept users: {len(used)}); interactions {n_rows}; "
          f"users {len({r['user_id'] for r in rows})}")


if __name__ == "__main__":
    main()
