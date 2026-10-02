"""route (choice): which service should handle the request, or should the assistant ask the user first?

The request needs one service; the answer is that service when it is connected, else ask_user. Flight requests
use OpenFlights: a destination city whose name matches airports in different countries (listed in the state) is
ambiguous and needs a question; the home airport comes from the user's city. Variant: override (a later
connect/disconnect of the needed service flips the answer).
"""
from __future__ import annotations

from ..kb.rules import between
from ..kb.schema import SERVICES
from .base import Context, Fragment, Question, pending_label

FAMILY = "route"
VARIANTS = ("clean", "override")
ASK = "ask_user"
SERVICE_TEXT = {"flights": "Search and book flights", "restaurants": "Find restaurants and reserve tables",
                "shopping": "Buy and reorder products", "email": "Read, draft and send email",
                "calendar": "Create and move calendar events",
                ASK: "Ask the user first: a detail is ambiguous or the needed service is not connected"}


def flight_request(ctx: Context, fragment: Fragment) -> tuple[str, bool] | None:
    """(request text, ambiguous) for a destination city from the shard's airports."""
    home = ctx.shard.airports_by_code.get(ctx.user.home_airport or "")
    if home is None:
        return None
    cities, ambiguous = ctx.shard.airports_by_city, ctx.shard.ambiguous_cities
    want_ambiguous = bool(ambiguous) and ctx.rng.random() < 0.25
    pool = ambiguous if want_ambiguous else ctx.shard.served_cities(home.iata)
    if not pool:
        return None
    city_key = ctx.rng.choice(pool)
    by_country: dict[str, list] = {}
    for port in sorted(cities[city_key], key=lambda port: port.iata):
        by_country.setdefault(port.country, []).append(port)
    # One airport per country first, so an ambiguous city always shows at least two countries.
    ports = [group[0] for group in by_country.values()][:4]
    ports += [port for group in by_country.values() for port in group[1:]][:max(0, 4 - len(ports))]
    city = ports[0].city
    fragment.add("profile", f"Home airport: {home.iata} ({home.name}, {home.city})")
    fragment.add("profile", f"Airport lookup for {city}: " + "; ".join(
        f"{port.iata} ({port.name}, {port.country or 'unknown country'})" for port in ports))
    fragment.question.required += [home.iata, ports[0].iata]
    return f"Book me a flight to {city} next Friday.", want_ambiguous


def request_for(ctx: Context, service: str, fragment: Fragment) -> tuple[str, bool] | None:
    rng, user, shard = ctx.rng, ctx.user, ctx.shard
    if service == "flights":
        return flight_request(ctx, fragment)
    if service in ("restaurants", "shopping"):
        domain = "restaurant" if service == "restaurants" else "product"
        items = shard.by_domain.get(domain, [])
        if not items:
            return None
        item = rng.choice(items)
        fragment.question.required.append(item.name)
        text = f"Reserve a table for two at {item.name} this Saturday." if domain == "restaurant" else \
            f"Reorder {item.name}."
        return text, False
    if service == "email":
        if not user.contacts:
            return None
        name = rng.choice(user.contacts).name
        fragment.question.required.append(name)
        return f"Email {name} that I'll be 15 minutes late.", False
    title = rng.choice(user.calendar).title if user.calendar else "Team standup"
    fragment.question.required.append(title)
    return f"Move my {title} to Thursday afternoon.", False


def override_connection(ctx: Context, fragment: Fragment, service: str, effective: set[str]) -> set[str]:
    """Later (dis)connections of the needed service; the final state is connected or not with equal odds, and the
    listed service set is always superseded by at least one dated change."""
    rng = ctx.rng
    first, second = sorted(between(ctx.user.cut_timestamp, ctx.now.isoformat(), rng) for _ in range(2))
    connected_now = service in effective
    want_connected = rng.random() < 0.5
    if connected_now == want_connected:
        steps = [("Disconnected" if connected_now else "Connected", first),
                 ("Reconnected" if connected_now else "Disconnected", second)]
    else:
        steps = [("Connected" if want_connected else "Disconnected", second)]
    for verb, date in steps:
        fragment.add("profile", f"{date}: {verb} the {service} service.")
        fragment.question.required.append(f"{verb} the {service} service")
    return effective | {service} if want_connected else effective - {service}


def build(ctx: Context) -> Fragment | None:
    rng = ctx.rng
    options = [*SERVICES, ASK]
    rng.shuffle(options)
    label = pending_label("Request", ctx.letter)
    question = Question(FAMILY, "choice", f"{label}: which service should handle this, or should the assistant ask "
                                          "the user first?", {name: SERVICE_TEXT[name] for name in options}, ASK,
                        [label])
    fragment = Fragment(FAMILY, question, {"label": label, "text": ""})
    service = rng.choice(SERVICES)
    if ctx.variant != "override" and service not in ctx.user.connected_services and rng.random() < 0.5:
        # Halve "not connected" asks so ask_user is not the majority answer.
        service = rng.choice(ctx.user.connected_services)
    made = request_for(ctx, service, fragment)
    if made is None:
        return None
    fragment.pending["text"], ambiguous = made
    connected = list(ctx.user.connected_services)
    connected_line = "Connected services: " + ", ".join(connected) + "."
    fragment.add("profile", connected_line)
    effective = set(connected)
    if ctx.variant == "override" and not ambiguous:
        effective = override_connection(ctx, fragment, service, effective)
        question.variant = "override"
    question.label = ASK if ambiguous or service not in effective else service
    question.required.append("Connected services:")
    return fragment
