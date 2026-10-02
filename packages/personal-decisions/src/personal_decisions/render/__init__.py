"""Renderers: structured state -> natural-language state text.

`FixtureRenderer` is deterministic and offline (tests, dry runs); `AstraRenderer` calls gpt-6-astra through the
OpenAI-compatible API. Both return a `Rendered`; the generator verifies anchors on the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rendered:
    text: str
    model: str
    prompt_sha256: str
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


def make_renderer(name: str, env_file=None, model: str | None = None, max_attempts: int | None = None):
    if name == "fixture":
        from .fixture import FixtureRenderer
        return FixtureRenderer()
    if name == "astra":
        from ..llm import open_client
        from .astra import AstraRenderer
        return AstraRenderer(open_client(env_file, model, max_attempts))
    raise ValueError(f"unknown renderer {name!r} (expected astra or fixture)")
