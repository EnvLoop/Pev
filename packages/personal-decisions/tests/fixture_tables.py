"""Synthetic raw tables in the agreed work/muse/raw/ schema (no real data): a few dozen rows, fully deterministic."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import random

PRODUCT_CATEGORIES = ("Grocery_and_Gourmet_Food", "Books")
RESTAURANT_CATEGORIES = ("Italian restaurant", "Thai restaurant")
SENTENCES = ("The flavor was rich and the portions were generous.", "Shipping took longer than promised.",
             "I would happily order this again for the family.", "The staff were friendly and quick to help.",
             "It tasted stale and the packaging was damaged.", "Great value for the price, honestly.",
             "The noodles were far too salty for my taste.", "A cozy place with a quiet patio out back.")
AIRPORTS = (("SPI", "Abraham Lincoln Capital Airport", "Springfield", "United States"),
            ("ORD", "Chicago O'Hare International Airport", "Chicago", "United States"),
            ("LHR", "London Heathrow Airport", "London", "United Kingdom"),
            ("YXU", "London International Airport", "London", "Canada"),
            ("DEN", "Denver International Airport", "Denver", "United States"))
ROUTES = (("AA", "SPI", "ORD"), ("AA", "ORD", "SPI"), ("UA", "SPI", "DEN"), ("BA", "ORD", "LHR"),
          ("AC", "ORD", "YXU"), ("UA", "DEN", "ORD"))


def items(rng: random.Random) -> list[dict]:
    rows = []
    for category in PRODUCT_CATEGORIES:
        for index in range(60):
            rows.append({"item_id": f"amz:{category[:4]}{index:02d}", "domain": "product",
                         "name": f"{category.split('_')[0]} Pick {index:02d}", "category": category,
                         "price": round(rng.uniform(4, 90), 2), "price_level": None,
                         "attributes": {"rating_number": rng.randint(5, 5000), "average_rating": 4.1},
                         "city": None, "source": "amazon_reviews_2023"})
    for category in RESTAURANT_CATEGORIES:
        for index in range(40):
            rows.append({"item_id": f"gmap:{category[:4]}{index:02d}", "domain": "restaurant",
                         "name": f"{category.split()[0]} Kitchen {index:02d}", "category": category, "price": None,
                         "price_level": rng.randint(1, 4), "attributes": {"num_of_reviews": rng.randint(10, 900)},
                         "city": "Springfield", "source": "google_local_2021"})
    return rows


def review(rng: random.Random) -> str:
    return " ".join(rng.sample(SENTENCES, rng.randint(2, 5)))


def interactions(rng: random.Random, catalog: list[dict], users: int) -> list[dict]:
    rows = []
    start = datetime(2019, 1, 1)
    for number in range(users):
        user_id = f"u{number:03d}"
        picks = rng.sample(catalog, 9)
        for step, item in enumerate(picks):
            rating = rng.choice((1, 2, 3, 4, 5, 5)) if step < 7 else 5
            when = start + timedelta(days=40 * step + number, hours=rng.randint(8, 20))
            rows.append({"user_id": user_id, "item_id": item["item_id"], "rating": rating,
                         "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "text": review(rng),
                         "source": item["source"]})
    return rows


def emails(rng: random.Random) -> list[dict]:
    return [{"email_id": f"e{index:03d}", "date": f"2001-0{1 + index % 9}-1{index % 9}",
             "subject": f"Quarterly planning notes {index}",
             "body": " ".join(rng.choice(("Please review the attached figures before Monday.",
                                          "The committee meets again next week to discuss the budget.",
                                          "Forwarding the minutes from yesterday's call for reference."))
                              for _ in range(40))} for index in range(30)]


def write_raw(folder: Path, users: int = 14, seed: int = 7) -> Path:
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    catalog = items(rng)
    tables = {
        "items": catalog, "interactions": interactions(rng, catalog, users), "emails": emails(rng),
        "airports": [{"airport_id": index, "name": name, "city": city, "country": country, "iata": code,
                      "icao": None, "source": "openflights"} for index, (code, name, city, country) in
                     enumerate(AIRPORTS)],
        "routes": [{"airline": airline, "src_airport": src, "dst_airport": dst, "stops": 0, "source": "openflights"}
                   for airline, src, dst in ROUTES],
        "airlines": [{"airline_id": 1, "name": "American Airlines", "iata": "AA", "source": "openflights"}]}
    for name, rows in tables.items():
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return folder
