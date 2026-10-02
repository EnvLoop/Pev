"""Knowledge-base schema: the single source of truth every question family reads.

Real data (items, interactions, flights, emails) is normalized as-is; personal rules (approval, privacy, forget
requests, notification rules, contacts, calendar) are synthesized from a seed derived from the user id so that the
same user always gets the same rules.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Ordered from least to most sensitive; a recipient may receive a fact when its clearance is >= the fact's level.
PRIVACY_LEVELS = ("public", "acquaintances", "friends", "family", "private")
RECIPIENT_TYPES = ("family", "close_friend", "coworker", "service_provider", "public")
# Score levels of notify_level, lowest first (the Kev label is the zero-based index).
NOTIFY_LEVELS = ("silent: log it only", "digest: include in the daily summary", "notify: send a normal notification",
                 "interrupt: alert immediately, even if busy")
EVENT_TYPES = ("flight_change", "price_drop", "package_delivery", "message", "calendar_conflict", "bill_due")
SERVICES = ("flights", "restaurants", "shopping", "email", "calendar")

PrivacyLevel = Literal["public", "acquaintances", "friends", "family", "private"]
RecipientType = Literal["family", "close_friend", "coworker", "service_provider", "public"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Item(Strict):
    item_id: str
    domain: Literal["product", "restaurant"]
    name: str
    category: str
    price: float | None = None
    price_level: int | None = None
    attributes: dict | list | None = None
    city: str | None = None
    source: str | None = None
    popularity: int = 0  # number of interactions with this item across the whole raw table


class Interaction(Strict):
    user_id: str
    item_id: str
    rating: int = Field(ge=1, le=5)
    timestamp: str
    text: str = ""
    source: str | None = None


class Airport(Strict):
    iata: str
    name: str
    city: str
    country: str = ""


class Route(Strict):
    airline: str
    src: str
    dst: str


class Email(Strict):
    email_id: str
    date: str = ""
    subject: str = ""
    body: str = ""


class MemoryFact(Strict):
    fact_id: str
    date: str
    text: str  # the fact as the assistant remembers it; always contains `evidence` verbatim
    evidence: str  # a verbatim span of the source review (or the rating statement when the review is empty)
    item_id: str | None = None
    domain: str
    category: str
    polarity: Literal["likes", "dislikes", "mixed"]
    privacy: PrivacyLevel = "friends"
    extractor: Literal["rule", "llm"] = "rule"


class PreferenceUpdate(Strict):
    update_id: str
    date: str
    kind: Literal["budget", "category_opinion"]
    text: str
    category: str | None = None
    old: float | str | None = None
    new: float | str | None = None


class ApprovalRules(Strict):
    spend_threshold: int  # purchases above this many dollars need approval
    new_recipient: bool  # messages/payments to someone not in contacts need approval
    irreversible: bool  # deleting, cancelling or non-refundable bookings need approval
    threshold_update_date: str | None = None
    updated_threshold: int | None = None


class PrivacyRules(Strict):
    clearance: dict[RecipientType, PrivacyLevel]


class ForgetRequest(Strict):
    date: str
    fact_id: str
    item_id: str | None = None
    item_name: str
    text: str


class NotificationRules(Strict):
    quiet_start: int  # hour of day, local time
    quiet_end: int
    vip_contacts: list[str]
    max_level: dict[str, int]  # event type -> highest allowed level index
    meeting_cap: int  # highest level while a calendar event is in progress
    quiet_update_date: str | None = None
    updated_quiet_start: int | None = None
    updated_quiet_end: int | None = None


class Contact(Strict):
    name: str
    relationship: RecipientType
    email: str


class CalendarEvent(Strict):
    title: str
    start: str
    end: str


class UserProfile(Strict):
    user_id: str
    city: str | None = None
    home_airport: str | None = None
    connected_services: list[str]
    cut_timestamp: str  # interactions at or before this are memory; later ones are real choices
    memory_interactions: list[Interaction]
    later_interactions: list[Interaction]
    memory_facts: list[MemoryFact]
    preference_updates: list[PreferenceUpdate]
    approval: ApprovalRules
    privacy: PrivacyRules
    forget_requests: list[ForgetRequest]
    notifications: NotificationRules
    contacts: list[Contact]
    calendar: list[CalendarEvent]
