import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions.jsonl import file_sha256, read_jsonl
from personal_decisions.kb.build import build_kb
from personal_decisions.kb.memory import evidence_span, extract_facts
from personal_decisions.kb.schema import Interaction, Item, UserProfile

from .fixture_tables import write_raw
from .support import workspace


def users(folder: Path) -> list[UserProfile]:
    return [UserProfile.model_validate(row) for row in read_jsonl(folder / "users.jsonl")]


class BuildTest(unittest.TestCase):
    def test_time_split_keeps_memory_before_choices(self):
        for user in users(workspace() / "kb"):
            self.assertTrue(all(row.timestamp <= user.cut_timestamp for row in user.memory_interactions))
            self.assertTrue(all(row.timestamp > user.cut_timestamp for row in user.later_interactions))
            self.assertGreaterEqual(len(user.memory_interactions), 3)
            self.assertTrue(any(row.rating >= 4 for row in user.later_interactions))

    def test_facts_quote_the_review_verbatim(self):
        for user in users(workspace() / "kb"):
            texts = {row.item_id: row.text for row in user.memory_interactions}
            for fact in user.memory_facts:
                self.assertIn(fact.evidence, texts[fact.item_id])
                self.assertIn(fact.evidence, fact.text)

    def test_rules_are_seeded_by_user_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_raw(root / "raw", users=40)
            build_kb(root / "raw", root / "kb")
            self.assertEqual(file_sha256(root / "kb" / "users.jsonl"),
                             file_sha256(workspace() / "kb" / "users.jsonl"))
        profiles = users(workspace() / "kb")
        self.assertGreater(len({user.approval.spend_threshold for user in profiles}), 1)
        for user in profiles:
            forgotten = {request.fact_id for request in user.forget_requests}
            self.assertTrue(forgotten <= {fact.fact_id for fact in user.memory_facts})

    def test_manifest_hashes_match_files(self):
        manifest = json.loads((workspace() / "kb" / "manifest.json").read_text())
        for name, entry in manifest["files"].items():
            self.assertEqual(entry["sha256"], file_sha256(workspace() / "kb" / name))
        self.assertEqual(manifest["files"]["users.jsonl"]["count"], 40)

    def test_popularity_prefers_catalog_review_counts(self):
        items = {row["item_id"]: row for row in read_jsonl(workspace() / "kb" / "items.jsonl")}
        raw = {row["item_id"]: row for row in read_jsonl(workspace() / "raw" / "items.jsonl")}
        for item_id, row in items.items():
            attributes = raw[item_id]["attributes"]
            expected = attributes.get("rating_number", attributes.get("num_of_reviews"))
            self.assertEqual(row["popularity"], expected)


class MemoryExtractionTest(unittest.TestCase):
    item = Item(item_id="x", domain="restaurant", name="Casa Verde", category="Mexican restaurant", city="Austin")

    def interaction(self, text: str) -> Interaction:
        return Interaction(user_id="u", item_id="x", rating=5, timestamp="2020-01-01T10:00:00Z", text=text)

    def test_rule_evidence_is_first_informative_sentence(self):
        self.assertEqual(evidence_span("Great! The mole was the best I have had in years. Slow service."),
                         "The mole was the best I have had in years.")
        self.assertEqual(evidence_span(""), "")

    def test_llm_fact_with_verbatim_evidence_is_accepted(self):
        stats = {}
        review = "We came for tacos. The salsa verde was fiery and bright."
        facts = extract_facts("u", [self.interaction(review)], {"x": self.item},
                              lambda text, item: ("Loves spicy salsa", "salsa verde was fiery"), stats)
        self.assertEqual(facts[0].extractor, "llm")
        self.assertEqual(facts[0].evidence, "salsa verde was fiery")
        self.assertEqual(stats, {"llm_accepted": 1})

    def test_llm_fact_with_invented_evidence_falls_back_to_rule(self):
        stats = {}
        review = "We came for tacos. The salsa verde was fiery and bright."
        facts = extract_facts("u", [self.interaction(review)], {"x": self.item},
                              lambda text, item: ("Loves spicy food", "the salsa was extremely spicy"), stats)
        self.assertEqual(facts[0].extractor, "rule")
        self.assertIn(facts[0].evidence, review)
        self.assertEqual(stats, {"llm_rejected": 1})

    def test_empty_review_uses_the_rating(self):
        facts = extract_facts("u", [self.interaction("")], {"x": self.item})
        self.assertEqual(facts[0].evidence, "rated it 5 out of 5")
        self.assertTrue(facts[0].text.startswith("Loved Casa Verde (Mexican restaurant, Austin)"))


if __name__ == "__main__":
    unittest.main()
