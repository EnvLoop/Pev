"""gpt-6-astra renderer: rewrites the structured state as natural prose in a given style.

The model only renders; it never sees labels or questions. Every required anchor is listed as a VERBATIM string; the
generator drops the record when one is missing or a forbidden anchor appears. prompt_sha256 covers the exact
instructions and input sent.
"""
from __future__ import annotations

import json

from ..jsonl import text_sha256
from . import Rendered
from .style import style_prompt

INSTRUCTIONS = """You turn a structured snapshot of a personal assistant's context into natural-language text that \
the assistant will read before acting for its user.

Hard rules:
1. Keep every fact, rule, date, time, amount, name, update, pending item and option from the snapshot. Do not drop, \
merge away, or generalize any of them, and do not add facts, opinions or advice that are not in the snapshot.
2. Never say or hint which option is best, whether an action is allowed, or what the assistant should do.
3. Copy every string listed under VERBATIM exactly, character for character (same case, digits, symbols and \
punctuation). Do not translate them, even if the style mixes languages.
4. Keep each pending item's label (for example "Request A", "Proposed action B", "Incoming event C") exactly, \
followed by its content; list all of its options.
5. Keep dated entries with their dates; keep "Current time".
6. Separate topics with blank lines. Output only the rendered text, no preamble.

Style:
{style}"""


def request_input(structured: dict, required: list[str]) -> str:
    verbatim = "\n".join(f"- {anchor}" for anchor in dict.fromkeys(required))
    return "SNAPSHOT (JSON):\n" + json.dumps(structured, ensure_ascii=False, indent=1) + "\n\nVERBATIM:\n" + verbatim


class AstraRenderer:
    def __init__(self, client, max_output_tokens: int = 6000):
        self.client = client
        self.model = client.model
        self.max_output_tokens = max_output_tokens

    def render(self, structured: dict, required: list[str], style: dict) -> Rendered:
        instructions = INSTRUCTIONS.format(style=style_prompt(style))
        text = request_input(structured, required)
        reply = self.client.complete(instructions, text, max_output_tokens=self.max_output_tokens)
        return Rendered(text=reply.text.strip(), model=reply.model or self.model,
                        prompt_sha256=text_sha256(instructions + "\n\n" + text), usage=reply.usage)
