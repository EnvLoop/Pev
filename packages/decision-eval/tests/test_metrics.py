import math
import os
import unittest

import numpy as np

from decision_eval.metrics import (ECE_BINS, macro, macro_accuracy, predictor_report, safety_rates,
                                   score_questions)
from decision_eval.records import Question, load_questions


def q(family, label, k=2, target=None, state="s", qtype="noul", i=0):
    keys = ("false", "true") if qtype == "noul" else tuple(str(j) for j in range(k))
    target = target or tuple(float(j == label) for j in range(len(keys)))
    return Question(f"r{i}", "q", family, state, qtype, keys, label, target)


def logits_for(probs):
    return [np.log(np.asarray(p, dtype=float)) for p in probs]


class MetricTests(unittest.TestCase):
    def test_accuracy_brier_nll(self):
        questions = [q("a", 1, i=0), q("a", 0, i=1)]
        scored = score_questions(questions, logits_for([[0.2, 0.8], [0.4, 0.6]]))
        self.assertEqual(scored.correct.tolist(), [True, False])
        self.assertAlmostEqual(scored.brier[0], 0.2 ** 2 + 0.2 ** 2)
        self.assertAlmostEqual(scored.brier[1], 0.6 ** 2 + 0.6 ** 2)
        self.assertAlmostEqual(scored.nll[0], -math.log(0.8))

    def test_brier_against_soft_target_and_accuracy_against_its_argmax(self):
        question = q("a", 1, target=(0.3, 0.7))
        scored = score_questions([question], logits_for([[0.5, 0.5]]))
        self.assertAlmostEqual(scored.brier[0], 0.2 ** 2 + 0.2 ** 2)
        self.assertFalse(scored.correct[0])            # a tie predicts option 0; the soft argmax is 1
        self.assertAlmostEqual(scored.nll[0], -math.log(0.5))

    def test_temperature_flattens_but_keeps_argmax(self):
        questions = [q("a", 1)]
        hot = score_questions(questions, [np.array([0.0, 2.0])], temperature=4.0)
        cold = score_questions(questions, [np.array([0.0, 2.0])])
        self.assertLess(hot.confidence[0], cold.confidence[0])
        self.assertEqual(hot.prediction.tolist(), cold.prediction.tolist())

    def test_ece_uses_15_equal_width_bins(self):
        self.assertEqual(ECE_BINS, 15)
        # 0.62 and 0.64 share the 15-bin [0.600, 0.667): one bin, |accuracy 0.5 - mean confidence 0.63|
        questions = [q("a", 1, i=0), q("a", 0, i=1)]
        scored = score_questions(questions, logits_for([[0.38, 0.62], [0.36, 0.64]]))
        report = predictor_report(questions, scored, threshold=None)
        self.assertAlmostEqual(report["families"]["a"]["ece"], abs(0.5 - 0.63), places=9)
        # 0.61 and 0.69 share a 10-bin but not a 15-bin (edge 0.667): 0.5 * |1 - 0.61| + 0.5 * |0 - 0.69|
        scored = score_questions(questions, logits_for([[0.39, 0.61], [0.31, 0.69]]))
        ece = predictor_report(questions, scored, None)["families"]["a"]["ece"]
        self.assertAlmostEqual(ece, 0.5 * 0.39 + 0.5 * 0.69, places=9)

    def test_automation_coverage_and_realized_error_at_threshold(self):
        questions = [q("a", 1, i=i) for i in range(4)]
        scored = score_questions(questions, logits_for([[0.05, 0.95], [0.9, 0.1], [0.3, 0.7], [0.45, 0.55]]))
        family = predictor_report(questions, scored, threshold=0.85)["families"]["a"]
        self.assertEqual((family["accepted"], family["errors"]), (2, 1))
        self.assertAlmostEqual(family["coverage"], 0.5)
        self.assertAlmostEqual(family["realized_error"], 0.5)
        none = predictor_report(questions, scored, threshold=None)["families"]["a"]
        self.assertEqual((none["coverage"], none["realized_error"]), (0.0, None))

    def test_macro_is_unweighted_over_families(self):
        questions = [q("a", 1, i=0), q("b", 1, i=1), q("b", 1, i=2), q("b", 1, i=3)]
        scored = score_questions(questions, logits_for([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]))
        report = predictor_report(questions, scored, None)
        self.assertAlmostEqual(report["macro"]["accuracy"], (1.0 + 1 / 3) / 2)
        self.assertAlmostEqual(report["micro"]["accuracy"], 0.5)
        self.assertAlmostEqual(macro_accuracy(questions, scored), (1.0 + 1 / 3) / 2)
        self.assertIsNone(macro({"a": {**report["families"]["a"], "realized_error": None}})["realized_error"])

    def test_safety_false_negative_rates(self):
        questions = [q("needs_approval", 1, i=0), q("needs_approval", 1, i=1), q("needs_approval", 0, i=2),
                     q("share_ok", 0, i=3), q("share_ok", 1, i=4),
                     q("forgotten_violation", 0, i=5), q("route", 0, k=3, qtype="choice", i=6)]
        # needs_approval: true->false miss, true->true hit; share_ok: false->true miss; no forgotten positives
        probs = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.2, 0.8], [0.9, 0.1], [0.8, 0.1, 0.1]]
        rates = safety_rates(questions, score_questions(questions, logits_for(probs)))
        self.assertEqual(rates["needs_approval"], {"n_at_risk": 2, "fn_rate": 0.5})
        self.assertEqual(rates["share_ok"], {"n_at_risk": 1, "fn_rate": 1.0})
        self.assertEqual(rates["forgotten_violation"], {"n_at_risk": 0, "fn_rate": None})
        self.assertNotIn("route", rates)

    @unittest.skipUnless(os.environ.get("DECISION_EVAL_V7"), "set DECISION_EVAL_V7 to kev's decision-v7 test.jsonl")
    def test_reads_kev_decision_v7_test(self):
        questions = load_questions(os.environ["DECISION_EVAL_V7"])
        self.assertEqual(len(questions), 1440)
        self.assertEqual(len({q.family for q in questions}), 16)


if __name__ == "__main__":
    unittest.main()
