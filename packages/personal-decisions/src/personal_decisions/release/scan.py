"""``privacy-scan``: aggregate PII counts over decision records and KB shards, for the release privacy audit.

Output is counts per kind (hits, distinct strings, records/rows containing one), placeholder counts left by the
acquisition scrubber, verbatim-review and buried-distractor exposure, and pseudonymous user ids. With
``--examples K`` it adds up to K *redacted* examples per kind (first character of each word only); never use
examples on DEV/HIDDEN/TEST (``--counts-only`` refuses them).
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from ..jsonl import read_jsonl
from .pii import KINDS, brand_names, find_pii, redact_example

PLACEHOLDERS = ("[EMAIL]", "[PHONE]", "[URL]")
LONG_ANCHOR = 40  # a required anchor this long is (almost always) a verbatim review span


def catalog_allow(items_path: Path | None) -> frozenset[str]:
    if not items_path:
        return frozenset()
    return brand_names(row.get("name", "") for row in read_jsonl(items_path))


class Tally:
    def __init__(self, examples: int):
        self.examples = examples
        self.hits, self.units_with = Counter(), Counter()
        self.distinct: dict[str, set[str]] = {kind: set() for kind in KINDS}
        self.samples: dict[str, list[str]] = {kind: [] for kind in KINDS}
        self.placeholders, self.units = Counter(), 0

    def add(self, text: str, protected: tuple[str, ...] = (), allow: frozenset[str] = frozenset()) -> None:
        self.units += 1
        found = find_pii(text, protected, allow)
        for kind in {hit.kind for hit in found}:
            self.units_with[kind] += 1
        for hit in found:
            self.hits[hit.kind] += 1
            self.distinct[hit.kind].add(hit.text)
            if len(self.samples[hit.kind]) < self.examples and redact_example(hit.text) not in self.samples[hit.kind]:
                self.samples[hit.kind].append(redact_example(hit.text))
        for placeholder in PLACEHOLDERS:
            self.placeholders[placeholder] += text.count(placeholder)

    def report(self, unit: str) -> dict:
        out = {unit: self.units, "placeholders": dict(self.placeholders), "kinds": {}}
        for kind in KINDS:
            out["kinds"][kind] = {"hits": self.hits[kind], "distinct": len(self.distinct[kind]),
                                  f"{unit}_with": self.units_with[kind],
                                  f"{unit}_with_share": round(self.units_with[kind] / self.units, 4)
                                  if self.units else 0.0}
            if self.examples:
                out["kinds"][kind]["redacted_examples"] = self.samples[kind]
        return out


def option_texts(record: dict) -> tuple[str, ...]:
    texts = []
    for question in record.get("questions", {}).values():
        criteria = question.get("criteria") or {}
        texts.extend(str(value) for value in (criteria.values() if isinstance(criteria, dict) else criteria))
    return tuple(texts)


def scan_records(path: Path, allow: frozenset[str], examples: int = 0) -> dict:
    tally, users, buried, verbatim = Tally(examples), set(), 0, 0
    for record in read_jsonl(path):
        meta = record.get("meta", {})
        state = record["state"] if isinstance(record["state"], str) else json.dumps(record["state"], ensure_ascii=False)
        text = state + "\n" + "\n".join(str(q.get("instructions", "")) for q in record["questions"].values())
        tally.add(text, option_texts(record), allow)
        users.add(meta.get("user_id") or (record.get("_meta") or {}).get("group_id"))
        variants = (meta.get("variants") or {}).values()
        buried += any(str(variant).startswith("buried+") for variant in variants)
        anchors = (meta.get("anchors") or {}).values()
        verbatim += any(len(anchor) >= LONG_ANCHOR for entry in anchors for anchor in entry.get("required", []))
    report = tally.report("records")
    report.update(distinct_user_ids=len(users - {None}), records_with_buried_distractors=buried,
                  records_with_verbatim_review_anchor=verbatim)
    return report


def scan_kb(folder: Path, allow: frozenset[str], examples: int = 0) -> dict:
    reviews, facts, emails, users = Tally(examples), Tally(examples), Tally(examples), 0
    for user in read_jsonl(folder / "users.jsonl"):
        users += 1
        for row in user["memory_interactions"] + user["later_interactions"]:
            if row.get("text"):
                reviews.add(row["text"], allow=allow)
        for fact in user["memory_facts"]:
            facts.add(fact["text"], allow=allow)
    if (folder / "emails.jsonl").is_file():
        for email in read_jsonl(folder / "emails.jsonl"):
            emails.add(email.get("subject", "") + "\n" + email.get("body", ""), allow=allow)
    return {"users": users, "review_texts": reviews.report("texts"), "memory_facts": facts.report("texts"),
            "enron_emails": emails.report("texts")}


def privacy_scan(records: list[Path], kbs: list[Path], items: Path | None, examples: int,
                 counts_only: list[Path]) -> dict:
    allow = catalog_allow(items)
    sealed = [p for p in records + kbs if any(part in ("sealed", "sealed-test") for part in p.parts)]
    if sealed:
        raise ValueError("privacy-scan refuses sealed paths")
    out = {"allow_listed_brand_names": len(allow), "records": {}, "kb": {}}
    quiet = {p.resolve() for p in counts_only}
    for path in records:
        out["records"][path.name] = scan_records(path, allow, 0 if path.resolve() in quiet else examples)
    for folder in kbs:
        out["kb"]["/".join(folder.parts[-2:])] = scan_kb(folder, allow, 0 if folder.resolve() in quiet else examples)
    return out
