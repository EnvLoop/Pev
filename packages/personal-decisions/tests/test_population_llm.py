import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions import PersonalDecisionsError
from personal_decisions.jsonl import read_jsonl
from personal_decisions.kb.build import build_kb
from personal_decisions.kb.llm_extract import parse_batch, prefetch
from personal_decisions.kb.population import PopulationRule, select_population
from personal_decisions.llm import Reply

from .fixture_tables import write_raw
from .support import workspace


class FakeClient:
    """Answers every batch by quoting each review's first 20 characters; can fail on demand."""

    model = "fake"

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0

    def complete(self, instructions, text, max_output_tokens=4000):
        self.calls += 1
        if self.fail:
            raise PersonalDecisionsError("LLM API failed 5 times: APIConnectionError")
        facts = []
        for number, block in enumerate(text.split("\n\n"), 1):
            review = block.split("\n", 1)[1]
            facts.append({"review": number, "fact": "Has an opinion", "evidence": review[:20]})
        return Reply(text=json.dumps({"facts": facts}), usage={"input_tokens": 10, "output_tokens": 5}, model="fake")


class PopulationTest(unittest.TestCase):
    def test_selection_is_deterministic_and_respects_the_rule(self):
        raw = workspace() / "raw"
        rule = PopulationRule(seed=1, per_domain={"product": 3, "restaurant": 3}, min_items=5, max_items=30,
                              require_low_later=False)
        first, receipt = select_population(raw, rule)
        self.assertEqual(first, select_population(raw, rule)[0])
        self.assertLessEqual(len(first), 6)
        self.assertNotIn("u0", json.dumps(receipt))
        other, _ = select_population(raw, PopulationRule(seed=2, per_domain=rule.per_domain, min_items=5,
                                                          require_low_later=False))
        self.assertEqual(len(other), len(first))
        strict, _ = select_population(raw, PopulationRule(seed=1, per_domain=rule.per_domain, min_items=10))
        self.assertEqual(strict, [])  # fixture users have 9 items

    def test_build_keeps_only_the_population(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_raw(root / "raw", users=12)
            chosen = {"u001", "u004", "u007"}
            build_kb(root / "raw", root / "kb", user_ids=chosen)
            self.assertEqual({row["user_id"] for row in read_jsonl(root / "kb" / "users.jsonl")}, chosen)


class BatchExtractionTest(unittest.TestCase):
    def test_parse_batch_tolerates_fences_and_missing_entries(self):
        reply = '```json\n{"facts": [{"review": 2, "fact": "x", "evidence": "y"}, {"review": 9}]}\n```'
        self.assertEqual(parse_batch(reply, 2), [None, ("x", "y")])
        self.assertIsNone(parse_batch("not json", 2))

    def test_llm_build_uses_verbatim_evidence_and_resumes_from_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_raw(root / "raw", users=6)
            client, cache, usage = FakeClient(), root / "cache.jsonl", {}

            def prefetcher(jobs):
                extractor, stats = prefetch(jobs, client, concurrency=3, cache_path=cache)
                usage.update(stats.as_dict())
                return extractor
            manifest = build_kb(root / "raw", root / "kb", prefetcher=prefetcher)
            self.assertEqual(usage["users_extracted"], 6)
            self.assertEqual(sum(1 for _ in read_jsonl(cache)), 6)
            self.assertGreater(manifest["stats"]["llm_accepted"], 0)
            self.assertNotIn("llm_unavailable", manifest["stats"])
            for user in read_jsonl(root / "kb" / "users.jsonl"):
                texts = {row["item_id"]: row["text"] for row in user["memory_interactions"]}
                for fact in user["memory_facts"]:
                    self.assertEqual(fact["extractor"], "llm")
                    self.assertTrue(texts[fact["item_id"]].startswith(fact["evidence"]))
            calls = client.calls
            build_kb(root / "raw", root / "kb2", prefetcher=prefetcher)
            self.assertEqual(client.calls, calls)  # everything came from the cache
            self.assertEqual(usage["users_cached"], 6)

    def test_api_failures_abort_and_fall_back_to_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_raw(root / "raw", users=6)
            usage = {}

            def prefetcher(jobs):
                extractor, stats = prefetch(jobs, FakeClient(fail=True), concurrency=1, max_consecutive_failures=2)
                usage.update(stats.as_dict())
                return extractor
            manifest = build_kb(root / "raw", root / "kb", prefetcher=prefetcher)
            self.assertTrue(usage["aborted"])
            self.assertEqual(usage["users_failed"], 2)
            self.assertEqual(usage["users_skipped_abort"], 4)
            self.assertGreater(manifest["stats"]["llm_unavailable"], 0)
            self.assertNotIn("llm_accepted", manifest["stats"])
            for user in read_jsonl(root / "kb" / "users.jsonl"):
                self.assertTrue(all(fact["extractor"] == "rule" for fact in user["memory_facts"]))

    def test_deadline_skips_remaining_users(self):
        _, stats = prefetch([("a", [])], FakeClient(), deadline_s=0)
        self.assertEqual(stats.users_extracted, 0)
        extractor, stats = prefetch([("a", [("some review text here", workspace_item())])], FakeClient(),
                                    deadline_s=-1)
        self.assertEqual(stats.users_skipped_deadline, 1)
        self.assertFalse(extractor.covers("some review text here", workspace_item()))


def workspace_item():
    from personal_decisions.kb.schema import Item
    return Item(item_id="x", domain="product", name="Thing", category="Books")


if __name__ == "__main__":
    unittest.main()
