"""Heuristic PII detectors for the public release: emails, phones, URLs, street addresses and person names.

Deliberately simple and dependency-free (no NER model), so counts are reproducible. Person names are found two ways:
a courtesy title before a capitalized word ("Ms. Shackleton"), or a common English first name followed on the same
line by a capitalized surname ("Sara Shackleton"). Both over-match brand names built from a person's name
("Bob Evans", "Robert Mondavi"), so callers pass the names that must survive (item names, synthetic contacts) as
``protected`` phrases. Names in a protected phrase are left alone. The synthetic contacts that the rule generator
writes (``kb/rules.py``: first x last name lists, ``@example.com``) are never PII and are always skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from ..kb.rules import FIRST_NAMES as SYNTHETIC_FIRST, LAST_NAMES as SYNTHETIC_LAST

# Common US first names (SSA lists), minus ones that are mostly ordinary words in this corpus (May, June, Will,
# Grant, Bill, Rose, Hope, Joy, Faith, Art, Pat, Sue, Mark is kept: "Mark Taylor" is far more frequent than the verb
# capitalized at sentence start followed by a capitalized word).
FIRST_NAMES = frozenset("""
Aaron Abigail Adam Alan Albert Alex Alexander Alexis Alice Allison Amanda Amber Amy Andrea Andrew Angela Ann Anna Anne
Anthony Ashley Austin Barbara Benjamin Beth Betty Beverly Billy Bob Bobby Brandon Brenda Brian Brittany Bruce Bryan
Carl Carol Carolyn Catherine Charles Charlotte Cheryl Chris Christian Christina Christine Christopher Cindy Craig
Cynthia Dan Daniel Danielle Dave David Debbie Deborah Debra Denise Dennis Diana Diane Donald Donna Doris Dorothy
Douglas Dylan Ed Edward Elizabeth Emily Emma Eric Ethan Eugene Evelyn Frances Frank Gabriel Gary George Gerald Gloria
Greg Gregory Hannah Harold Heather Helen Henry Isabella Jack Jacob Jacqueline Jake James Jamie Jane Janet Janice Jason
Jean Jeff Jeffrey Jennifer Jeremy Jerry Jesse Jessica Jim Jimmy Joan Joe Joel John Jonathan Jordan Jose Joseph
Joshua Joyce Juan Judith Judy Julia Julie Justin Karen Kate Katherine Kathleen Kathryn Kathy Kay Kayla Keith Kelly
Kenneth Kevin Kim Kimberly Kyle Larry Laura Lauren Lawrence Linda Lisa Logan Lori Louis Madison Margaret Maria Marie
Marilyn Mark Martha Mary Mason Matt Matthew Megan Melissa Michael Michelle Mike Nancy Natalie Nathan Nicholas Nicole
Noah Olivia Pamela Patricia Patrick Paul Peter Philip Rachel Ralph Randy Raymond Rebecca Richard Rick Robert Roger
Ronald Roy Russell Ruth Ryan Sally Sam Samantha Samuel Sandra Sara Sarah Scott Sean Sharon Shirley Sophia Stephanie
Stephen Steve Steven Susan Teresa Terry Theresa Thomas Tim Timothy Tom Tony Tyler Victoria Vincent Virginia Walter
Wayne Willie William Zachary
""".split())
# Capitalized words that follow a first name but are not surnames (email headers, weekdays, common words).
NOT_SURNAMES = frozenset("""
Subject Sent From To Cc Bcc Date Re Fw Fwd Original Message Thanks Thank Regards Best Cheers Hi Hello Dear The And
Or But If In On At Of For With This That These Those Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October November December Inc Corp Company Co Ltd LLC
Group Brand Original Classic Organic Natural Premium Pack Size Count Ounce Oz Lb Free Fresh New Old Great Good Best
Kitchen Cafe Restaurant Bar Grill Pizza House Street Avenue Road Says Said Will Would Can Could Should Is Was Has Had
Mentioned Called Wrote Asked Told Replied Here There Not No Yes Also Just Please Note
""".split())
# Organisation words that header lines put next to names ("From: Enron North America Corp.").
NOT_NAME_WORDS = NOT_SURNAMES | frozenset("""
Enron North America American Corp Energy Services Power Gas Capital Trade Trading Resources Global Markets Risk
Management Legal Department Team Office Online Networks Industrial Markets Wholesale Retail Pipeline Transwestern
Northern Natural Portland General Electric Houston London Dow Jones Newswires News Daily Weekly Report Update Meeting
Conference Committee Board University College School Bank Fund Partners Associates Holdings International Mail
Customer Support Admin Administrator Calendar Announcement Announcements Lunch Dinner Notice Reminder Request
""".split())
TITLED = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?[ \t]+[A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,})?\b")
FIRST_LAST = re.compile(r"\b(" + "|".join(sorted(FIRST_NAMES)) + r")[ \t]+([A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?)\b")
EMAIL = re.compile(r"\b[\w.+'-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE = re.compile(r"(?:(?<!\w)\+?1[\s.-]?)?(?:\(\d{3}\)\s?|(?<!\d)\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)")
URL = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"')\]]*[^\s<>\"')\].,;:!?]", re.I)
STREET = re.compile(r"\b\d{1,5}[ \t]+(?:[A-Z][a-z]+[ \t]+){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|"
                    r"Drive|Ln|Lane|Way|Ct|Court|Pl|Place|Hwy|Highway|Pkwy|Parkway)\b\.?")
NAME2 = r"[A-Z][a-z]+(?:[ \t](?:[A-Z]\.?|[A-Z][a-z]+))?[ \t][A-Z][a-z]+(?:-[A-Z][a-z]+)?"
# Lotus Notes addresses left by the Enron export: "Benjamin Rogers/HOU/ECT", "Kay Mann/Corp/Enron".
NOTES_ORG = r"(?:[A-Z][A-Z&_]+|Corp|Enron)"
NOTES_PATH = re.compile(r"\b(" + NAME2 + r")(?=/" + NOTES_ORG + r"(?:/" + NOTES_ORG + r")*(?![\w/]))")
HEADER_LINE = re.compile(r"^[ \t>]*(?:From|To|Cc|Bcc|Sent by|Forwarded by)\b[: \t]*(.*)$", re.M)
HEADER_NAME = re.compile(r"\b" + NAME2 + r"\b")
SYNTHETIC_EMAIL = re.compile(r"@example\.(?:com|org|net)$", re.I)
SYNTHETIC_NAMES = frozenset(f"{first} {last}" for first in SYNTHETIC_FIRST for last in SYNTHETIC_LAST)
KINDS = ("email", "phone", "url", "street_address", "person_name")


@dataclass(frozen=True)
class Hit:
    kind: str
    start: int
    end: int
    text: str


def _protected_spans(text: str, protected: tuple[str, ...]) -> list[tuple[int, int]]:
    spans = []
    for phrase in protected:
        if phrase and len(phrase) >= 3:
            start = text.find(phrase)
            while start != -1:
                spans.append((start, start + len(phrase)))
                start = text.find(phrase, start + 1)
    return spans


def _inside(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def _names(text: str, allow: frozenset[str]) -> list[Hit]:
    hits = [Hit("person_name", m.start(), m.end(), m.group(0)) for m in TITLED.finditer(text)]
    for match in FIRST_LAST.finditer(text):
        if match.group(2) in NOT_SURNAMES or match.group(0) in SYNTHETIC_NAMES or match.group(0) in allow:
            continue
        hits.append(Hit("person_name", match.start(), match.end(), match.group(0)))
    for match in NOTES_PATH.finditer(text):
        hits.append(Hit("person_name", match.start(1), match.end(1), match.group(1)))
    for line in HEADER_LINE.finditer(text):
        for match in HEADER_NAME.finditer(line.group(1)):
            words = match.group(0).replace("-", " ").replace(".", " ").split()
            if not any(word in NOT_NAME_WORDS for word in words) and match.group(0) not in allow \
                    and match.group(0) not in SYNTHETIC_NAMES:
                start = line.start(1) + match.start()
                hits.append(Hit("person_name", start, start + len(match.group(0)), match.group(0)))
    return [hit for hit in hits if hit.text not in allow and hit.text not in SYNTHETIC_NAMES]


def brand_names(catalog_names) -> frozenset[str]:
    """Every first-name + surname pair or titled name inside an item name ("Bob Evans", "Dr. Noy's", "Multipet
    Mr. Bill Dog Toy"): not people. A titled match also allows its two-word prefix ("Mr. Bill")."""
    names = set()
    for name in catalog_names:
        names.update(m.group(0) for m in FIRST_LAST.finditer(name))
        for match in TITLED.finditer(name):
            names.add(match.group(0))
            names.add(" ".join(match.group(0).split()[:2]))
    return frozenset(names)


def find_pii(text: str, protected: tuple[str, ...] = (), allow: frozenset[str] = frozenset()) -> list[Hit]:
    """Non-overlapping hits, leftmost-longest, skipping synthetic contacts, ``allow``-listed names (brand names
    from the item catalog) and anything inside a ``protected`` phrase (e.g. the option texts of a question)."""
    hits = [Hit("email", m.start(), m.end(), m.group(0)) for m in EMAIL.finditer(text)
            if not SYNTHETIC_EMAIL.search(m.group(0))]
    hits += [Hit("phone", m.start(), m.end(), m.group(0)) for m in PHONE.finditer(text)]
    hits += [Hit("url", m.start(), m.end(), m.group(0)) for m in URL.finditer(text)]
    hits += [Hit("street_address", m.start(), m.end(), m.group(0)) for m in STREET.finditer(text)]
    hits += _names(text, allow)
    spans = _protected_spans(text, protected)
    kept, end = [], -1
    for hit in sorted(hits, key=lambda h: (h.start, -(h.end - h.start))):
        if hit.start >= end and not _inside((hit.start, hit.end), spans):
            kept.append(hit)
            end = hit.end
    return kept


def redact_example(text: str) -> str:
    """For audit examples only: keep each word's first character, mask the rest ("Sara Shackleton" -> "S*** S*****")."""
    out, previous = [], " "
    for char in text:
        out.append(char if not char.isalnum() or not previous.isalnum() else "*")
        previous = char
    return "".join(out)
