"""DEV null-baseline fixes: every soft label is tied (never scored), notify_level's withheld urgency leaves the answer
open, and route labels are balanced by NAME (services and ask_user) as well as by position."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions.balance import LabelSchedule, tied
from personal_decisions.families.base import Fragment, Question
from personal_decisions.generate import Options, run
from personal_decisions.jsonl import read_jsonl
from personal_decisions.render.fixture import FixtureRenderer

from .support import workspace
from .test_families import fragments


class ShortcutFixesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(tempfile.mkdtemp())
        style = folder / "style.json"
        style.write_text(json.dumps({"style_id": "t", "voice": "plain"}))
        cls.out = folder / "records.jsonl"
        run(workspace() / "train", cls.out, Options(n=160, seed=3, concurrency=2), FixtureRenderer(), style)
        cls.records = list(read_jsonl(cls.out))

    def test_every_soft_label_is_tied(self):
        soft = 0
        for record in self.records:
            for question in record["questions"].values():
                if "soft_label" in question:
                    soft += 1
                    self.assertTrue(tied(question["soft_label"]), question["soft_label"])
        self.assertGreater(soft, 0)

    def test_route_labels_are_balanced_by_name_and_position(self):
        names, positions = Counter(), Counter()
        for record in self.records:
            question = record["questions"].get("route")
            if question is None:
                continue
            names[question["label"]] += 1
            positions[list(question["criteria"]).index(question["label"])] += 1
        self.assertEqual(len(names), 6)
        self.assertLessEqual(max(names.values()) - min(names.values()), 1)
        self.assertLessEqual(max(positions.values()) - min(positions.values()), 1)

    def test_withheld_urgency_is_uniform_over_two_or_more_levels(self):
        seen = 0
        for _, fragment in fragments("notify_level", "removed", 200):
            question = fragment.question
            if question.variant != "removed":
                self.assertIsNone(question.soft_label)
                continue
            seen += 1
            self.assertGreaterEqual(len(question.soft_label), 2)
            self.assertEqual(len(set(question.soft_label.values())), 1)
            self.assertIn(str(question.label), question.soft_label)
        self.assertGreater(seen, 20)

    def test_schedule_rejects_a_soft_label_with_a_unique_argmax(self):
        schedule = LabelSchedule("train", 1)
        skewed = Question("notify_level", "score", "i", ["a", "b", "c", "d"], 1, [], soft_label={"0": 0.25, "1": 0.75})
        even = Question("notify_level", "score", "i", ["a", "b", "c", "d"], 1, [], soft_label={"0": 0.5, "1": 0.5})
        self.assertFalse(schedule.fit(Fragment("notify_level", skewed, {"label": "x", "text": "t"})))
        self.assertTrue(schedule.fit(Fragment("notify_level", even, {"label": "x", "text": "t"})))


if __name__ == "__main__":
    unittest.main()
