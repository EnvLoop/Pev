"""Hard variant "buried": 1k-6k tokens of real distractor text (shard emails, other users' reviews in the same shard)
interleaved with the rendered state, so the deciding evidence sits in the middle of unrelated text.

Distractor text is inserted verbatim after rendering (never sent through the renderer). A document that contains any
question's required or forbidden anchor is skipped, so distractors can neither supply nor contradict evidence.
"""
from __future__ import annotations

import random

from . import anchors
from .compose import Draft
from .families.shard import ShardData

CHARS_PER_TOKEN = 4
MAX_DOCUMENT_CHARS = 3000


def clip(text: str, limit: int = MAX_DOCUMENT_CHARS) -> str:
    text = " ".join(text.split()) if len(text) > limit else text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > 0 else limit] + " [...]"


def candidate_document(shard: ShardData, user_id: str, rng: random.Random) -> str | None:
    use_email = shard.emails and (not shard.reviews or rng.random() < 0.5)
    if use_email:
        email = rng.choice(shard.emails)
        header = f"--- Archived email ({email.date or 'undated'}): {email.subject or 'no subject'} ---"
        return f"{header}\n{clip(email.body)}"
    if not shard.reviews:
        return None
    owner, row = rng.choice(shard.reviews)
    if owner == user_id:
        return None
    item = shard.items.get(row.item_id)
    about = item.name if item else "a place"
    return f"--- A review written by someone else about {about} ---\n{clip(row.text)}"


def pick_documents(shard: ShardData, draft: Draft, rng: random.Random, min_tokens: int = 1000,
                   max_tokens: int = 6000) -> list[str]:
    budget = rng.randint(min_tokens, max_tokens) * CHARS_PER_TOKEN
    guarded = [anchor for question in draft.questions.values() for anchor in question.required + question.forbidden
               if not anchor.startswith(("Request ", "Proposed action ", "Incoming event "))]
    documents, used, size = [], set(), 0
    for _ in range(400):
        if size >= budget:
            break
        document = candidate_document(shard, draft.user_id, rng)
        if document is None or document in used:
            continue
        flat = anchors.normalize(document)
        if any(anchors.contains(flat, anchor, normalized=True) for anchor in guarded):
            continue
        used.add(document)
        documents.append(document)
        size += len(document)
    return documents if size >= min_tokens * CHARS_PER_TOKEN else []


def insert_documents(text: str, documents: list[str], rng: random.Random) -> str:
    """Interleave documents between the rendered paragraphs at seeded positions."""
    paragraphs = [part for part in text.split("\n\n") if part.strip()] or [text]
    slots: dict[int, list[str]] = {}
    for document in documents:
        slots.setdefault(rng.randint(0, len(paragraphs)), []).append(document)
    out = []
    for position in range(len(paragraphs) + 1):
        out.extend(slots.get(position, []))
        if position < len(paragraphs):
            out.append(paragraphs[position])
    return "\n\n".join(out)
