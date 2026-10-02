"""Readers for the raw normalized tables in work/muse/raw/ (read-only input).

items/interactions/emails follow the agreed schema. The OpenFlights tables are read tolerantly: the field names below
are tried in order, because the acquisition step may keep OpenFlights' own column names.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..jsonl import read_jsonl
from .schema import Airport, Email, Interaction, Item, Route

AIRPORT_CODE = ("iata", "IATA", "iata_code", "code")
AIRPORT_NAME = ("name", "airport", "airport_name")
AIRPORT_CITY = ("city", "municipality")
ROUTE_SRC = ("src_airport", "source_airport", "src", "source_iata", "from")
ROUTE_DST = ("dst_airport", "dest_airport", "dst", "destination_airport", "dest_iata", "to", "destination")
ROUTE_AIRLINE = ("airline", "airline_code", "carrier")
AIRLINE_CODE = ("iata", "IATA", "code", "airline_code")
ITEM_FIELDS = set(Item.model_fields) - {"popularity"}


def first(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in ("\\N", "-"):
            return value.strip()
    return ""


def is_iata(code: str) -> bool:
    return len(code) == 3 and code.isalpha() and code.isupper()


def optional(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.is_file() else []


def load_items(raw: Path) -> list[Item]:
    rows = [row for row in read_jsonl(raw / "items.jsonl")]
    items = []
    for row in rows:
        clean = {key: row.get(key) for key in ITEM_FIELDS if key in row}
        clean["item_id"] = str(clean["item_id"])
        if clean.get("price") is not None:
            clean["price"] = float(clean["price"])
        items.append(Item.model_validate(clean))
    return items


def load_interactions(raw: Path) -> list[Interaction]:
    out = []
    for row in read_jsonl(raw / "interactions.jsonl"):
        out.append(Interaction(user_id=str(row["user_id"]), item_id=str(row["item_id"]), rating=int(row["rating"]),
                               timestamp=str(row["timestamp"]), text=str(row.get("text") or ""),
                               source=row.get("source")))
    return out


def load_airports(raw: Path) -> list[Airport]:
    out = []
    for row in optional(raw / "airports.jsonl"):
        code = first(row, AIRPORT_CODE)
        if is_iata(code):
            out.append(Airport(iata=code, name=first(row, AIRPORT_NAME) or code, city=first(row, AIRPORT_CITY),
                               country=first(row, ("country",))))
    return out


def load_routes(raw: Path) -> list[Route]:
    out = []
    for row in optional(raw / "routes.jsonl"):
        src, dst = first(row, ROUTE_SRC), first(row, ROUTE_DST)
        if not src and is_iata(str(row.get("source", ""))):  # OpenFlights' own name collides with provenance
            src = row["source"]
        if is_iata(src) and is_iata(dst) and src != dst:
            out.append(Route(airline=first(row, ROUTE_AIRLINE) or "??", src=src, dst=dst))
    return out


def load_airlines(raw: Path) -> dict[str, str]:
    names = {}
    for row in optional(raw / "airlines.jsonl"):
        code = first(row, AIRLINE_CODE)
        if code:
            names[code] = first(row, ("name", "airline_name")) or code
    return names


def load_emails(raw: Path) -> list[Email]:
    return [Email(email_id=str(row["email_id"]), date=str(row.get("date") or ""), subject=str(row.get("subject") or ""),
                  body=str(row.get("body") or "")) for row in optional(raw / "emails.jsonl")]


CATALOG_POPULARITY = ("rating_number", "num_of_reviews")


def catalog_popularity(item: Item) -> int | None:
    """The public review count the source catalog reports (Amazon rating_number, Google num_of_reviews)."""
    if isinstance(item.attributes, dict):
        for key in CATALOG_POPULARITY:
            value = item.attributes.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
    return None


def with_popularity(items: list[Item], interactions: list[Interaction]) -> list[Item]:
    """Popularity = the catalog's public review count when present, else interactions in the raw table."""
    counts = Counter(interaction.item_id for interaction in interactions)
    out = []
    for item in items:
        popularity = catalog_popularity(item)
        out.append(item.model_copy(update={"popularity": counts.get(item.item_id, 0) if popularity is None
                                           else popularity}))
    return out
