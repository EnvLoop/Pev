"""One KB shard loaded into memory with the indexes the families need."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path

from ..jsonl import read_jsonl
from ..kb.schema import Airport, Email, Item, Route, UserProfile


@dataclass
class ShardData:
    name: str
    users: list[UserProfile]
    items: dict[str, Item]
    emails: list[Email] = field(default_factory=list)
    airports: list[Airport] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    airlines: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.by_group: dict[tuple, list[Item]] = defaultdict(list)
        for item in self.items.values():
            # Restaurants are indexed under every Google category they list (primary first), so "a Thai
            # restaurant in X" also finds places whose secondary category is Thai restaurant.
            listed = item.attributes.get("categories") if isinstance(item.attributes, dict) else None
            names = [item.category] if item.domain != "restaurant" or not isinstance(listed, list) else \
                list(dict.fromkeys([item.category, *(name for name in listed if isinstance(name, str))]))
            for category in names:
                self.by_group[(item.domain, category, item.city or "")].append(item)
                self.by_group[(item.domain, category, "*")].append(item)
        self.item_list = list(self.items.values())
        self.by_domain: dict[str, list[Item]] = defaultdict(list)
        for item in self.items.values():
            self.by_domain[item.domain].append(item)
        self.airports_by_code = {airport.iata: airport for airport in self.airports}
        self.airports_by_city: dict[str, list[Airport]] = defaultdict(list)
        for airport in self.airports:
            if airport.city:
                self.airports_by_city[airport.city.casefold()].append(airport)
        self.destinations: dict[str, set[str]] = defaultdict(set)
        for route in self.routes:
            self.destinations[route.src].add(route.dst)
        self.ambiguous_cities = sorted(city for city, ports in self.airports_by_city.items()
                                       if len({port.country for port in ports}) > 1)
        self._served: dict[str, list[str]] = {}
        # Other users' reviews in this shard only: distractor text never crosses shards.
        self.reviews = [(user.user_id, row) for user in self.users for row in user.memory_interactions
                        if len(row.text) >= 200]

    def served_cities(self, home: str) -> list[str]:
        """Unambiguous destination cities with a direct route from `home` (cached per airport)."""
        if home not in self._served:
            reachable = self.destinations.get(home, set())
            ambiguous = set(self.ambiguous_cities)
            self._served[home] = sorted(city for city, ports in self.airports_by_city.items()
                                        if city not in ambiguous and any(port.iata in reachable for port in ports))
        return self._served[home]

    def similar_items(self, item: Item) -> list[Item]:
        """Same domain and category; restaurants also the same city when the target has one."""
        city = (item.city or "") if item.domain == "restaurant" else "*"
        return [other for other in self.by_group.get((item.domain, item.category, city), [])
                if other.item_id != item.item_id]


def load_shard(folder: str | Path, name: str | None = None) -> ShardData:
    folder = Path(folder)

    def rows(file_name):
        path = folder / file_name
        return list(read_jsonl(path)) if path.is_file() else []

    manifest = folder / "manifest.json"
    shard_name = name or (json.loads(manifest.read_text()).get("shard") if manifest.is_file() else None) or folder.name
    airlines = folder / "airlines.json"
    return ShardData(
        name=shard_name,
        users=[UserProfile.model_validate(row) for row in rows("users.jsonl")],
        items={row["item_id"]: Item.model_validate(row) for row in rows("items.jsonl")},
        emails=[Email.model_validate(row) for row in rows("emails.jsonl")],
        airports=[Airport.model_validate(row) for row in rows("airports.jsonl")],
        routes=[Route.model_validate(row) for row in rows("routes.jsonl")],
        airlines=json.loads(airlines.read_text()) if airlines.is_file() else {})
