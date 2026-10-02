"""Shared helpers for the MUSE raw-data normalizers."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

SEED = 20260930
RAW = Path(__file__).resolve().parents[1]
DL = RAW / "_downloads"
STAGING = RAW / "_staging"
MIN_INTERACTIONS = 8
TEXT_MAX = 2000
EMAIL_BODY_MAX = 4000


def anon_user(source: str, original_id: str) -> str:
    return hashlib.sha256(f"{source}:{original_id}".encode()).hexdigest()[:16]


def seeded_key(value: str) -> str:
    """Deterministic pseudo-random sort key for sampling with the fixed seed."""
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def iso_from_ms(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clip(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


URL_RE = re.compile(r"(?:https?:/{1,2}|ftp://|www\.)[^\s<>\"')\]]+", re.I)
EMAIL_RE = re.compile(r"[\w.+'-]*@[\w-]+(?:\.[\w-]+)*|[\w.+'-]+@")
NOTES_RE = re.compile(r"\b[A-Z][\w.' -]{1,40}(?:/[\w-]{2,15})+@[\w-]{2,20}")
NOTES_AT_RE = re.compile(r"(?:\b[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,3}|\[EMAIL\]>?\]?)\s*@\s*[A-Z][A-Za-z_]{1,20}\b")
PHONE_RE = re.compile(
    r"(?:(?<!\w)\+?1[\s.-]?)?(?:\(\d{3}\)\s?|(?<!\d)\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)"
    r"|(?<![\d-])(?:\d{3}|\d)-\d{4}(?![\d-])"
)


def scrub(text: str) -> str:
    text = URL_RE.sub("[URL]", text)
    text = NOTES_RE.sub("[EMAIL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = NOTES_AT_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
