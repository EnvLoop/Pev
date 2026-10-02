"""decision_eval.conventions must behave exactly like kev@0fe8fc97 (checked when the kev extra is installed)."""
import importlib.util
import unittest

import numpy as np

from decision_eval import conventions as ours

VALUES = [None, "text", 3, 2.5, True, ["a", {"k": "v", "n": [1, 2]}], {"memory": ["tea", "Oslo"], "request": {"x": 1}},
          {"deep": {"deeper": [{"a": None}, "b"]}, "flag": False}]


@unittest.skipUnless(importlib.util.find_spec("kev"), "the kev extra is not installed")
class ConventionParityTests(unittest.TestCase):
    def test_rendering(self):
        from kev import api
        for value in VALUES:
            self.assertEqual(ours.render(value), api.render(value))
            self.assertEqual(ours.option_text("name", value), api.option_text("name", value))
        for qtype, criteria in (("choice", {"a": None, "b": "x"}), ("noul", None), ("score", ["lo", "hi", "top"])):
            self.assertEqual(ours.question_keys(qtype, criteria), api.question_keys(qtype, criteria))

    def test_selective_prediction_and_ece(self):
        from kev import metrics
        rng = np.random.default_rng(0)
        for _ in range(20):
            confidence = np.round(rng.uniform(0.3, 1.0, size=60), 2)       # rounding makes ties
            correct = rng.random(60) < confidence
            for bins in (10, 15):
                self.assertEqual(ours.ece(confidence, correct, bins), metrics.ece(confidence, correct, bins))
            for budget in (0.0, 0.05, 0.2):
                self.assertEqual(ours.coverage_at_error(confidence, correct, budget),
                                 metrics.coverage_at_error(confidence, correct, budget))
                threshold = ours.select_threshold(confidence, correct, budget)
                self.assertEqual(threshold, metrics.select_threshold(confidence, correct, budget))
                self.assertEqual(ours.evaluate_threshold(confidence, correct, threshold),
                                 metrics.evaluate_threshold(confidence, correct, threshold))


class ConventionTests(unittest.TestCase):
    def test_render_examples(self):
        self.assertEqual(ours.render({"a": 1, "b": ["x", "y"]}), "a: 1\nb:\n  - x\n  - y")
        self.assertEqual(ours.option_text("cafe", None), "cafe")
        self.assertEqual(ours.question_keys("score", ["a", "b"]), ["0", "1"])


if __name__ == "__main__":
    unittest.main()
