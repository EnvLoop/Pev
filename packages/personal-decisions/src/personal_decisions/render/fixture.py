"""Deterministic offline renderer: a fixed template over the structured state (tests and dry runs; no network)."""
from __future__ import annotations

from ..jsonl import dumps, text_sha256
from . import Rendered

MODEL = "fixture-template"
TITLES = (("profile", "About the user"), ("memory", "What the assistant remembers"),
          ("preference_updates", "Preference updates"), ("rules", "Rules the user set"),
          ("forget_requests", "Things the user asked to forget"), ("contacts", "Contacts"),
          ("calendar", "Calendar"))


def template(structured: dict) -> str:
    blocks = [f"Current time: {structured['now']}"]
    for key, title in TITLES:
        if structured.get(key):
            blocks.append(title + ":\n" + "\n".join(f"- {line}" for line in structured[key]))
    for pending in structured.get("pending", []):
        block = f"{pending['label']}: {pending['text']}"
        if pending.get("options"):
            block += "\nOptions:\n" + "\n".join(f"- {option}" for option in pending["options"])
        blocks.append(block)
    return "\n\n".join(blocks)


class FixtureRenderer:
    model = MODEL

    def render(self, structured: dict, required: list[str], style: dict) -> Rendered:
        prompt = dumps({"renderer": MODEL, "state": structured, "style_id": style.get("style_id")})
        return Rendered(text=template(structured), model=MODEL, prompt_sha256=text_sha256(prompt))
