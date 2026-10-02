"""``export-release``: the pseudonymization pass every publicly released record goes through.

This is consistent, *reversible* pseudonymization for readability and consistency, not anonymization: the key is
published with the release (``release/pseudonym_key.txt``) so the export is exactly reproducible from the public
source datasets, and therefore anyone can recompute the mapping. All source data were already public.

Per record (deterministic given the key, so re-running gives byte-identical output):

1. ``meta.user_id`` becomes ``HMAC-SHA256(key, "user:" + id)[:16]`` (the internal id is itself a hash of the public
   source id).
2. Person names, email addresses, phone numbers, URLs and street addresses that ``pii.find_pii`` finds in the state
   or the question instructions are replaced everywhere in the record (state, instructions, anchors) by the same
   string: names by a keyed pseudonym that keeps the shape ("Mr. Kean" -> "Mr. <P>", "Steve Kean" -> "<F> <P>", with
   one pseudonym per real first / last name), the rest by ``[EMAIL]`` / ``[PHONE]`` / ``[URL]`` / ``[ADDRESS]``
   (the acquisition scrubber's placeholders).
3. A string that occurs inside any option name or option description is never replaced, so options, labels and soft
   labels are untouched by construction.
4. The anchor contract is re-checked on the new text (required anchors present, forbidden ones absent, after the
   same replacement). A record that fails is reported by id and the command stops: nothing is half-anonymized.

Records without ``meta`` (Kev-format rows such as the general mix) are not ours to rewrite: they are counted and left
out unless ``keep_foreign`` is set, in which case they pass through unchanged.

The key (>= 32 hex characters) is read from ``--key-file`` (the published ``release/pseudonym_key.txt``), else
``RELEASE_PSEUDONYM_KEY`` in the environment or the repository ``.env``. The command never prints it.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import hmac
import os
from pathlib import Path
import re

from .. import PersonalDecisionsError
from ..anchors import check
from .pii import find_pii

KEY_VARIABLE = "RELEASE_PSEUDONYM_KEY"
PLACEHOLDER = {"email": "[EMAIL]", "phone": "[PHONE]", "url": "[URL]", "street_address": "[ADDRESS]"}
TITLE = re.compile(r"^(Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+")
# Pseudonyms: disjoint from the pii first-name lexicon and from the synthetic contacts in kb/rules.py, so a re-scan
# after the pass measures what the pass missed, not what it wrote.
PSEUDO_FIRST = ("Arlen", "Bexley", "Calder", "Dorian", "Ellery", "Fenwick", "Garnet", "Halden", "Iver", "Jessamy",
                "Kestrel", "Lorcan", "Marlowe", "Niamh", "Orrin", "Perpetua", "Quillon", "Rhiannon", "Soren",
                "Tamsen", "Ulric", "Verity", "Wystan", "Xanthe", "Yorick", "Zephyrine", "Anselm", "Briony",
                "Caspian", "Delphine", "Emrys", "Fiora")
PSEUDO_LAST = ("Ashdown", "Birkett", "Cawthorne", "Dunmore", "Ellwood", "Fairclough", "Gilchrist", "Hollinghurst",
               "Inchbald", "Jessop", "Kingsmill", "Larkworthy", "Merriweather", "Northcott", "Ormerod", "Pennington",
               "Quarrington", "Rushworth", "Southwell", "Thistlewood", "Underhill", "Vexley", "Wetherby",
               "Yarborough", "Ambleside", "Blackwood", "Coldridge", "Dewhurst", "Eastleigh", "Foxcroft", "Greenhalgh",
               "Hartington")
PSEUDO_WORDS = frozenset(PSEUDO_FIRST + PSEUDO_LAST)


def load_key(env_file: Path | None = None, key_file: Path | None = None) -> bytes:
    value = Path(key_file).read_text(encoding="utf-8").strip() if key_file else os.environ.get(KEY_VARIABLE)
    if not value and not key_file:
        from ..llm import default_env_file
        env_file = env_file or default_env_file()
        if env_file is not None and Path(env_file).is_file():
            from dotenv import dotenv_values
            value = dotenv_values(env_file).get(KEY_VARIABLE)
    if not value or len(value) < 32 or re.fullmatch(r"[0-9a-fA-F]+", value) is None:
        raise PersonalDecisionsError(f"no pseudonym key (>= 32 hex characters) in --key-file, {KEY_VARIABLE} or .env")
    return bytes.fromhex(value if len(value) % 2 == 0 else value[:-1])


class Pseudonymizer:
    def __init__(self, key: bytes):
        self._key = key

    def _pick(self, kind: str, value: str, choices: tuple[str, ...]) -> str:
        digest = hmac.new(self._key, f"{kind}:{value}".encode(), hashlib.sha256).digest()
        return choices[int.from_bytes(digest[:8], "big") % len(choices)]

    def user_id(self, user_id: str) -> str:
        return hmac.new(self._key, f"user:{user_id}".encode(), hashlib.sha256).hexdigest()[:16]

    def name(self, text: str) -> str:
        title = TITLE.match(text)
        words = text[title.end():].split() if title else text.split()
        last = self._pick("last", words[-1].lower(), PSEUDO_LAST)
        if title:
            return f"{title.group(1)}. {last}"
        return f"{self._pick('first', words[0].lower(), PSEUDO_FIRST)} {last}"


def is_pseudonym(text: str) -> bool:
    words = TITLE.sub("", text).split()
    return bool(words) and all(word in PSEUDO_WORDS for word in words)


def _protected(record: dict) -> tuple[str, ...]:
    phrases = []
    for question in record["questions"].values():
        criteria = question.get("criteria") or {}
        if isinstance(criteria, dict):
            phrases.extend(str(key) for key in criteria)
            phrases.extend(str(value) for value in criteria.values() if value)
        else:
            phrases.extend(str(value) for value in criteria)
    return tuple(phrases)


def _replacements(record: dict, pseudo: Pseudonymizer, allow: frozenset[str]) -> dict[str, tuple[str, str]]:
    protected = _protected(record)
    texts = [record["state"]] + [q.get("instructions", "") for q in record["questions"].values()]
    found = {}
    for text in texts:
        for hit in find_pii(text, protected, allow):
            value = hit.text.rstrip(".,;:!?") if hit.kind in ("url", "street_address") else hit.text
            if any(value in phrase for phrase in protected) or (hit.kind == "person_name" and is_pseudonym(value)):
                continue
            found[value] = (hit.kind, pseudo.name(value) if hit.kind == "person_name" else PLACEHOLDER[hit.kind])
    return found


def _substitute(text: str, table: dict[str, str], pattern: re.Pattern | None) -> str:
    return pattern.sub(lambda match: table[match.group(0)], text) if pattern else text


def anonymize_record(record: dict, pseudo: Pseudonymizer, allow: frozenset[str] = frozenset()) -> tuple[dict, Counter]:
    """The released form of one record and the per-kind replacement counts; raises on a broken anchor contract."""
    found = _replacements(record, pseudo, allow)
    table = {value: replacement for value, (_, replacement) in found.items()}
    pattern = None
    if table:
        alternatives = "|".join(re.escape(value) for value in sorted(table, key=len, reverse=True))
        pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + alternatives + r")(?![A-Za-z0-9])")
    counts = Counter()
    if pattern:
        for match in pattern.finditer(record["state"]):
            counts[found[match.group(0)][0]] += 1
    out = dict(record)
    out["state"] = _substitute(record["state"], table, pattern)
    out["questions"] = {qid: {**q, "instructions": _substitute(q.get("instructions", ""), table, pattern)}
                        if "instructions" in q else dict(q) for qid, q in record["questions"].items()}
    meta = dict(record.get("meta") or {})
    if meta.get("user_id"):
        meta["user_id"] = pseudo.user_id(meta["user_id"])
    if meta.get("anchors"):
        meta["anchors"] = {qid: {side: [_substitute(anchor, table, pattern) for anchor in anchors]
                                 for side, anchors in entry.items()} for qid, entry in meta["anchors"].items()}
        for qid, entry in meta["anchors"].items():
            failure = check(out["state"], entry.get("required", []), entry.get("forbidden", []))
            if failure:
                raise PersonalDecisionsError(f"{record.get('id')}/{qid}: {failure} after anonymization")
    meta["release"] = {"anonymized": True, "replacements": dict(sorted(counts.items()))}
    out["meta"] = meta
    return out, counts
