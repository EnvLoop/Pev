from pathlib import Path
import tempfile
import unittest

from personal_decisions.baselines import shortcut_baselines
from personal_decisions.jsonl import read_jsonl, write_jsonl
from personal_decisions.to_kev import convert, to_kev


def record(index: int, label: str, soft: dict | None = None) -> dict:
    features = {"A": {"price": 10.0, "popularity": 5}, "B": {"price": 20.0, "popularity": 50},
                "C": {"price": 30.0, "popularity": 1}}
    pick = {"type": "choice", "instructions": "Request A: which?", "criteria": {"A": "x", "B": "y", "C": "z"},
            "label": label, "src": "muse/pick_option"}
    if soft:
        pick["soft_label"] = soft
    ask = {"type": "noul", "instructions": "Proposed action B: approval?", "criteria": {"true": "t", "false": "f"},
           "label": index % 3 == 0, "src": "muse/needs_approval"}
    level = {"type": "score", "instructions": "Incoming event C: level?", "criteria": ["a", "b", "c", "d"],
             "label": 2, "src": "muse/notify_level"}
    return {"id": f"muse/train/1/{index}", "state": "state text",
            "questions": {"pick_option": pick, "needs_approval": ask, "notify_level": level},
            "meta": {"user_id": "u", "shard": "train", "state_id": f"train/1/{index}",
                     "families": {"pick_option": "pick_option", "needs_approval": "needs_approval",
                                  "notify_level": "notify_level"},
                     "anchors": {}, "render": {"model": "m", "style_id": "s", "prompt_sha256": "0" * 64},
                     "option_features": {"pick_option": features}}}


class BaselinesTest(unittest.TestCase):
    def test_trivial_predictors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            rows = [record(0, "A"), record(1, "B"), record(2, "B"), record(3, "C", soft={"A": 0.34, "B": 0.33,
                                                                                         "C": 0.33})]
            write_jsonl(path, rows)
            report = shortcut_baselines(path)
        pick = report["per_family"]["pick_option"]
        self.assertEqual((pick["n"], pick["soft_labelled_excluded"]), (3, 1))
        self.assertEqual(pick["cheapest"], round(1 / 3, 4))
        self.assertEqual(pick["most_popular"], round(2 / 3, 4))
        self.assertEqual(pick["first_option"], round(1 / 3, 4))
        self.assertEqual(pick["majority"], round(2 / 3, 4))
        approval = report["per_family"]["needs_approval"]
        self.assertEqual(approval["majority"], 0.5)
        self.assertIsNone(approval["cheapest"])
        self.assertEqual(report["per_family"]["notify_level"]["first_option"], 0.0)


class ToKevTest(unittest.TestCase):
    def test_soft_label_becomes_target_and_meta_is_dropped(self):
        converted = to_kev(record(3, "C", soft={"A": 0.5, "B": 0.25, "C": 0.25}))
        self.assertEqual(set(converted), {"state", "questions", "_meta"})
        self.assertEqual(converted["_meta"], {"id": "muse/train/1/3", "group_id": "train/1/3", "source": "muse"})
        pick = converted["questions"]["pick_option"]
        self.assertEqual(pick["target"], {"A": 0.5, "B": 0.25, "C": 0.25})
        self.assertNotIn("soft_label", pick)
        self.assertNotIn("target", to_kev(record(3, "C", soft={"A": 1.0}), keep_soft=False)["questions"]["pick_option"])

    def test_convert_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "r.jsonl", Path(folder) / "k.jsonl"
            write_jsonl(source, [record(0, "A"), record(1, "B", soft={"A": 0.5, "B": 0.5})])
            receipt = convert(source, out)
            self.assertEqual((receipt["records"], receipt["questions_with_target"]), (2, 1))
            self.assertEqual(len(list(read_jsonl(out))), 2)


if __name__ == "__main__":
    unittest.main()
