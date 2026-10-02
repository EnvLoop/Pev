"""Unit tests for scripts/general_mix.py on synthetic records (run with the decision-eval environment):

    uv run --project packages/decision-eval python -m unittest discover -s scripts -p 'test_*.py'
"""
import random
import unittest

from general_mix import TargetIndex, audit, normalize, quotas, select


def kev(i, src, text, qids=("answer",)):
    return {"state": text, "_meta": {"id": f"{src}/train/{i}", "group_id": f"{src}/train/{i}"},
            "questions": {q: {"type": "noul", "instructions": "yes?", "label": True, "src": src} for q in qids}}


class GeneralMixTests(unittest.TestCase):
    def test_target_index_finds_exact_normalized_and_ngram_matches(self):
        target = "The quick brown fox jumps over the lazy dog near the old river bank today."
        index = TargetIndex([target])
        self.assertEqual(index.match(target), "exact")
        self.assertEqual(index.match("the QUICK brown fox, jumps over the lazy dog near the old river bank today"),
                         "normalized")
        near = "Yesterday the quick brown fox jumps over the lazy dog near the old river bank today."
        self.assertEqual(index.match(near), "ngram")
        self.assertIsNone(index.match("A completely different sentence about banking fees and card limits."))
        self.assertEqual(normalize("\uff21\uff42\uff43,  DEF!"), "abc def")

    def test_audit_counts_and_flags(self):
        records = [kev(0, "imdb", "great film, loved it"), kev(1, "imdb", "terrible plot and acting")]
        counts, flagged = audit(records, {"guard": TargetIndex(["Great film, loved it!"])})
        self.assertEqual(counts["guard"], {"exact": 0, "normalized": 1, "ngram": 0})
        self.assertEqual(flagged, {"imdb/train/0"})

    def test_quotas_water_fill_to_the_exact_total(self):
        got = quotas({"a": 72, "b": 300, "c": 300, "d": 600}, 400)
        self.assertEqual(sum(got.values()), 400)
        self.assertEqual(got["a"], 72)
        self.assertLessEqual(max(got.values()) - min(v for k, v in got.items() if k != "a"), 1)

    def test_select_balances_families_and_keeps_only_chosen_questions(self):
        records = [kev(i, "agnews", f"news {i}", ("topic", "yn")) for i in range(50)]
        for record in records:
            record["questions"]["yn"]["src"] = "agnews_yn"
        records += [kev(100 + i, "boolq", f"passage {i}") for i in range(10)]
        chosen, quota, available = select(records, {"agnews/train/3"}, 30, random.Random(1))
        self.assertEqual(available, {"agnews": 49, "agnews_yn": 49, "boolq": 10})
        self.assertEqual(quota, {"boolq": 10, "agnews": 10, "agnews_yn": 10})
        self.assertEqual(sum(len(r["questions"]) for r in chosen), 30)
        self.assertNotIn("agnews/train/3", {r["_meta"]["id"] for r in chosen})


if __name__ == "__main__":
    unittest.main()
