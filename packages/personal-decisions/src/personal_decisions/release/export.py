"""Whole-file driver of the anonymization pass (``export-release``) and its contents-free receipt."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..jsonl import file_sha256, read_jsonl, write_jsonl
from .anonymize import Pseudonymizer, anonymize_record, is_pseudonym
from .pii import find_pii
from .scan import catalog_allow


def residual_counts(record: dict, allow: frozenset[str]) -> Counter:
    """What the detector still finds after the pass (pseudonyms excluded): should be zero by construction, except
    strings kept because they occur in an option (counted here so the audit sees them)."""
    counts = Counter()
    for hit in find_pii(record["state"], allow=allow):
        if not (hit.kind == "person_name" and is_pseudonym(hit.text)):
            counts[hit.kind] += 1
    return counts


def export_release(records: Path, out: Path, key: bytes, items: Path | None = None,
                   keep_foreign: bool = False) -> dict:
    if records.resolve() == out.resolve():
        raise ValueError("export-release writes a new file; --out must differ from --records")
    allow = catalog_allow(items)
    pseudo = Pseudonymizer(key)
    replaced, residual, kept, foreign, users = Counter(), Counter(), 0, 0, set()

    def rows():
        nonlocal kept, foreign
        for record in read_jsonl(records):
            if not isinstance(record.get("meta"), dict):
                foreign += 1
                if keep_foreign:
                    kept += 1
                    yield record
                continue
            released, counts = anonymize_record(record, pseudo, allow)
            replaced.update(counts)
            residual.update(residual_counts(released, allow))
            users.add(released["meta"].get("user_id"))
            kept += 1
            yield released

    # Records keep their key order (a choice question's criteria order is its option order).
    write_jsonl(out, rows(), sort_keys=False)
    return {"records_in_sha256": file_sha256(records), "out": str(out), "out_sha256": file_sha256(out),
            "records_written": kept, "foreign_records": foreign, "foreign_kept": keep_foreign,
            "pseudonymous_users": len(users - {None}), "replacements": dict(sorted(replaced.items())),
            "residual_detections": dict(sorted(residual.items())),
            "catalog_sha256": file_sha256(items) if items else None, "brand_names_allowed": len(allow)}
