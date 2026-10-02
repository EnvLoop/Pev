"""Batched, concurrent, resumable LLM memory-fact extraction (the `--extractor llm` path of `build`).

One request covers up to BATCH_SIZE early reviews of one user and asks for a fact plus a verbatim evidence span per
review. Replies are only proposals: `memory.verified_llm_fact` still rejects any evidence that is not a verbatim span
of the review, and every rejected, missing or unavailable review falls back to the rule extractor.

Each finished user is appended to a JSONL cache (results + token usage), so an interrupted build resumes without
repeating requests. A deadline or a run of consecutive API failures stops the LLM pass; the remaining users are
then built with the rule extractor and the receipt records the split. Nothing here prints review text or credentials.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import threading
import time

from .. import PersonalDecisionsError
from ..jsonl import dumps, read_jsonl, text_sha256
from .memory import describe_item
from .schema import Item

BATCH_SIZE = 12
MAX_REVIEW_CHARS = 2000
BATCH_INSTRUCTIONS = (
    "You extract durable personal preferences from one person's customer reviews. For EVERY numbered review, give "
    "one short sentence about what this person likes or dislikes, plus an exact, contiguous quote copied character "
    "for character from that review (12 to 200 characters) that supports it. Reply with JSON only: "
    '{"facts": [{"review": 1, "fact": "...", "evidence": "..."}, ...]} with one entry per review. '
    'If a review states no preference, use "fact": "" and "evidence": "".')

Job = tuple[str, list[tuple[str, Item]]]  # (user id, [(review text, item)])


def review_key(text: str, item: Item) -> str:
    return text_sha256(f"{item.item_id}\n{text}")


def batch_prompt(reviews: list[tuple[str, Item]]) -> str:
    return "\n\n".join(f"Review {number} (item: {describe_item(item)}):\n{text[:MAX_REVIEW_CHARS]}"
                       for number, (text, item) in enumerate(reviews, 1))


def parse_batch(reply: str, count: int) -> list[tuple[str, str] | None] | None:
    """Per-review (fact, evidence) proposals; None when the reply is not the expected JSON."""
    cleaned = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    entries = parsed.get("facts") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return None
    out: list[tuple[str, str] | None] = [None] * count
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        number = entry.get("review")
        if isinstance(number, int) and 1 <= number <= count and out[number - 1] is None:
            out[number - 1] = (str(entry.get("fact") or ""), str(entry.get("evidence") or ""))
    return out


class Prefetched:
    """An `Extractor` backed by prefetched proposals; `covers` tells extract_facts whether the LLM saw a review."""

    def __init__(self, results: dict[str, list | None] | None = None):
        self.results = results or {}

    def covers(self, text: str, item: Item) -> bool:
        return review_key(text, item) in self.results

    def __call__(self, text: str, item: Item) -> tuple[str, str] | None:
        found = self.results.get(review_key(text, item))
        return (found[0], found[1]) if found else None


@dataclass
class PrefetchStats:
    users_total: int = 0
    users_cached: int = 0
    users_extracted: int = 0
    users_failed: int = 0
    users_skipped_deadline: int = 0
    users_skipped_abort: int = 0
    requests: int = 0
    unparsed_replies: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_requests: int = 0
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    wall_s: float = 0.0
    aborted: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def as_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


def extract_user(client, reviews: list[tuple[str, Item]], stats: PrefetchStats) -> tuple[dict, dict]:
    """One user's reviews in batches -> ({key: [fact, evidence] | None}, usage)."""
    results, usage = {}, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    for start in range(0, len(reviews), BATCH_SIZE):
        chunk = reviews[start:start + BATCH_SIZE]
        reply = client.complete(BATCH_INSTRUCTIONS, batch_prompt(chunk), max_output_tokens=300 * len(chunk) + 200)
        usage["requests"] += 1
        for key in ("input_tokens", "output_tokens"):
            usage[key] += reply.usage.get(key, 0)
        proposals = parse_batch(reply.text, len(chunk))
        if proposals is None:
            with stats._lock:
                stats.unparsed_replies += 1
            proposals = [None] * len(chunk)
        for (text, item), proposal in zip(chunk, proposals):
            results[review_key(text, item)] = list(proposal) if proposal else None
    return results, usage


def load_cache(path: Path | None, stats: PrefetchStats) -> tuple[dict[str, list | None], set[str]]:
    results, done = {}, set()
    if path is None or not Path(path).is_file():
        return results, done
    for row in read_jsonl(path):
        results.update(row["results"])
        done.add(row["user_id"])
        stats.cached_requests += row["usage"]["requests"]
        stats.cached_input_tokens += row["usage"]["input_tokens"]
        stats.cached_output_tokens += row["usage"]["output_tokens"]
    return results, done


def prefetch(jobs: list[Job], client, *, concurrency: int = 8, cache_path: Path | None = None,
             deadline_s: float | None = None, max_consecutive_failures: int = 20,
             progress_every: int = 100) -> tuple[Prefetched, PrefetchStats]:
    stats = PrefetchStats(users_total=len(jobs))
    results, done = load_cache(cache_path, stats)
    stats.users_cached = sum(1 for user_id, _ in jobs if user_id in done)
    todo = [(user_id, reviews) for user_id, reviews in jobs if user_id not in done and reviews]
    started, lock, failures = time.monotonic(), threading.Lock(), [0]
    handle = Path(cache_path).open("a", encoding="utf-8") if cache_path else None

    def work(job: Job) -> None:
        user_id, reviews = job
        if stats.aborted:
            with lock:
                stats.users_skipped_abort += 1
            return
        if deadline_s is not None and time.monotonic() - started > deadline_s:
            with lock:
                stats.users_skipped_deadline += 1
            return
        try:
            found, usage = extract_user(client, reviews, stats)
        except PersonalDecisionsError as error:
            with lock:
                stats.users_failed += 1
                failures[0] += 1
                if failures[0] >= max_consecutive_failures and not stats.aborted:
                    stats.aborted = f"{failures[0]} consecutive API failures ({error})"
            return
        with lock:
            failures[0] = 0
            results.update(found)
            stats.users_extracted += 1
            stats.requests += usage["requests"]
            stats.input_tokens += usage["input_tokens"]
            stats.output_tokens += usage["output_tokens"]
            if handle:
                handle.write(dumps({"user_id": user_id, "results": found, "usage": usage}) + "\n")
                handle.flush()
            finished = stats.users_extracted + stats.users_failed
            if progress_every and finished % progress_every == 0:
                print(json.dumps({"llm_progress": finished, "of": len(todo), "requests": stats.requests,
                                  "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens,
                                  "elapsed_s": round(time.monotonic() - started)}), file=sys.stderr, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(work, todo))
    finally:
        if handle:
            handle.close()
    stats.wall_s = round(time.monotonic() - started, 1)
    return Prefetched(results), stats
