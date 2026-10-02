"""Render style specs: a JSON file each agent keeps private (voice, format, language mix, verbosity).

Only the style id is written into records (meta.render.style_id); the spec itself stays with its owner.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..jsonl import dumps, text_sha256

STYLE_KEYS = ("voice", "format", "language_mix", "verbosity")
DEFAULT_STYLE = {"style_id": "default", "voice": "neutral assistant notes, third person about the user",
                 "format": "short paragraphs with a heading per topic", "language_mix": "English only",
                 "verbosity": "concise"}


def load_style(path: str | Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_STYLE)
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: a style spec is a JSON object")
    unknown = set(spec) - set(STYLE_KEYS) - {"style_id", "notes"}
    if unknown:
        raise ValueError(f"{path}: unknown style keys {sorted(unknown)}")
    body = {key: spec[key] for key in STYLE_KEYS if key in spec}
    return {"style_id": str(spec.get("style_id") or "sha256:" + text_sha256(dumps(body))[:16]), **body,
            **({"notes": spec["notes"]} if "notes" in spec else {})}


def style_prompt(style: dict) -> str:
    lines = [f"- {key.replace('_', ' ')}: {style[key]}" for key in (*STYLE_KEYS, "notes") if style.get(key)]
    return "\n".join(lines) or "- plain, neutral prose"
