"""Memory facts from a user's early reviews.

The deterministic rule extractor (default, used by tests) quotes the first informative sentence of the review. The
optional LLM extractor asks the model for a one-line fact plus a verbatim evidence span; a span that is not found
verbatim in the review is rejected and the rule extractor is used instead, so every fact stays grounded in real text.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import re
import threading

from .schema import Interaction, Item, MemoryFact

SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
MIN_EVIDENCE, MAX_EVIDENCE = 12, 200
LLM_INSTRUCTIONS = (
    "You extract one durable personal preference from a customer review. Reply with JSON only: "
    '{"fact": "<one short sentence about what this person likes or dislikes>", '
    '"evidence": "<an exact, contiguous quote copied character for character from the review>"}. '
    "The evidence must be 12 to 200 characters and appear verbatim in the review. "
    "If the review states no preference, reply {\"fact\": \"\", \"evidence\": \"\"}.")

# (review text, item) -> (fact sentence, evidence) or None; implemented by LlmFactExtractor or test doubles.
Extractor = Callable[[str, Item], "tuple[str, str] | None"]


def category_label(category: str) -> str:
    return " ".join(category.replace("_", " ").split())


def polarity(rating: int) -> str:
    return "likes" if rating >= 4 else "dislikes" if rating <= 2 else "mixed"


def evidence_span(text: str) -> str:
    """The first sentence of MIN..MAX characters, verbatim; else a verbatim prefix; '' for an empty review."""
    text = text.strip()
    for match in SENTENCE.finditer(text):
        sentence = match.group(0).strip()
        if MIN_EVIDENCE <= len(sentence) <= MAX_EVIDENCE:
            return sentence
    if not text:
        return ""
    prefix = text[:MAX_EVIDENCE]
    cut = prefix.rfind(" ")
    return prefix[:cut] if len(text) > MAX_EVIDENCE and cut >= MIN_EVIDENCE else prefix


def describe_item(item: Item) -> str:
    where = f", {item.city}" if item.city else ""
    return f"{item.name} ({category_label(item.category)}{where})"


def rule_fact(interaction: Interaction, item: Item) -> tuple[str, str]:
    evidence = evidence_span(interaction.text) or f"rated it {interaction.rating} out of 5"
    verb = {"likes": "Loved", "dislikes": "Disliked", "mixed": "Felt mixed about"}[polarity(interaction.rating)]
    return f'{verb} {describe_item(item)}: "{evidence}"', evidence


def verified_llm_fact(interaction: Interaction, item: Item, extractor: Extractor) -> tuple[str, str] | None:
    """The extractor's fact only when its evidence is a verbatim span of the review of acceptable length."""
    result = extractor(interaction.text, item)
    if not result:
        return None
    fact, evidence = (part.strip() for part in result)
    if not fact or not (MIN_EVIDENCE <= len(evidence) <= MAX_EVIDENCE) or evidence not in interaction.text:
        return None
    return f'{fact.rstrip(".")} ({describe_item(item)}): "{evidence}"', evidence


def extract_facts(user_id: str, interactions: list[Interaction], items: dict[str, Item],
                  extractor: Extractor | None = None, stats: dict | None = None) -> list[MemoryFact]:
    facts = []
    stats = stats if stats is not None else {}
    for index, interaction in enumerate(interactions):
        item = items.get(interaction.item_id)
        if item is None:
            continue
        source = "rule"
        found = None
        if extractor is not None and interaction.text.strip():
            covers = getattr(extractor, "covers", None)
            if covers is not None and not covers(interaction.text, item):
                outcome = "llm_unavailable"  # the LLM pass never saw this review (deadline, failure, abort)
            else:
                found = verified_llm_fact(interaction, item, extractor)
                outcome = "llm_accepted" if found else "llm_rejected"
            stats[outcome] = stats.get(outcome, 0) + 1
            source = "llm" if found else "rule"
        text, evidence = found or rule_fact(interaction, item)
        facts.append(MemoryFact(fact_id=f"{user_id}/f{index}", date=interaction.timestamp[:10], text=text,
                                evidence=evidence, item_id=item.item_id, domain=item.domain, category=item.category,
                                polarity=polarity(interaction.rating), extractor=source))
    return facts


class LlmFactExtractor:
    """Extractor over the OpenAI-compatible client (llm.TextClient); counts tokens into `usage`."""

    def __init__(self, client):
        self.client = client
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        self._lock = threading.Lock()

    def __call__(self, review: str, item: Item) -> tuple[str, str] | None:
        prompt = f"Item: {describe_item(item)}\nReview:\n{review[:4000]}"
        reply = self.client.complete(LLM_INSTRUCTIONS, prompt, max_output_tokens=200)
        with self._lock:
            for key in ("input_tokens", "output_tokens"):
                self.usage[key] += reply.usage.get(key, 0)
            self.usage["requests"] += 1
        try:
            parsed = json.loads(reply.text.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return str(parsed.get("fact") or ""), str(parsed.get("evidence") or "")
