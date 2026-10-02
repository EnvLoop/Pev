"""Deterministic anchor checks: required anchors must appear verbatim in the state text, forbidden ones must not.

Matching is exact and case-sensitive after two normalizations only: runs of whitespace become one space, and
typographic quotes/dashes become their ASCII forms. An anchor that starts or ends with a letter or digit must not
continue another word or number there ("$50" does not match "$500" or "$50.99"; "Ann" does not match "Anna").
"""
from __future__ import annotations

from functools import lru_cache
import re

TYPOGRAPHY = str.maketrans({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2014: "-",
                            0x00A0: " ", 0x2009: " ", 0x202F: " "})


def normalize(text: str) -> str:
    return " ".join(text.translate(TYPOGRAPHY).split())


@lru_cache(maxsize=65536)
def anchor_pattern(anchor: str) -> re.Pattern:
    anchor = normalize(anchor)
    start = r"(?<![A-Za-z0-9])" if anchor[:1].isalnum() else ""
    end = r"(?![A-Za-z0-9]|[.,]\d)" if anchor[-1:].isalnum() else ""
    return re.compile(start + re.escape(anchor) + end)


def contains(text: str, anchor: str, normalized: bool = False) -> bool:
    if not anchor.strip():
        return True
    return anchor_pattern(anchor).search(text if normalized else normalize(text)) is not None


def check(text: str, required: list[str], forbidden: list[str]) -> str | None:
    """None when every required anchor is present and no forbidden one is; else the failure reason."""
    flat = normalize(text)
    if any(not contains(flat, anchor, normalized=True) for anchor in required):
        return "missing_required_anchor"
    if any(contains(flat, anchor, normalized=True) for anchor in forbidden):
        return "forbidden_anchor_present"
    return None


def missing(text: str, required: list[str]) -> list[str]:
    flat = normalize(text)
    return [anchor for anchor in required if not contains(flat, anchor, normalized=True)]
