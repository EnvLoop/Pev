"""Soft labels without a unique argmax have no correct answer: only Brier / ECE / NLL (vs the soft target) see them."""
import unittest

import numpy as np

from decision_eval.calibration import fit_threshold
from decision_eval.gate import decide
from decision_eval.metrics import predictor_report, safety_rates, score_questions
from decision_eval.records import question_of
from decision_eval.report import paired_report


def record(questions):
    return {"id": "r", "state": "s", "questions": questions,
            "meta": {"user_id": "u", "state_id": "st", "families": {qid: "fam" for qid in questions}}}


def noul(i, family, label=True, soft=None, state=None):
    q = question_of(record({"q": {"type": "noul", "label": label, **({"soft_label": soft} if soft else {})}}), "q")
    return type(q)(f"r{i}", "q", family, state or f"s{i}", q.type, q.keys, q.label, q.target, q.ambiguous)


UNIFORM = {"true": 0.5, "false": 0.5}


def logits(*p_true):
    return [np.log([1 - p, p]) for p in p_true]


class AmbiguityTests(unittest.TestCase):
    def test_ties_are_ambiguous_and_unique_argmax_is_the_label(self):
        soft = {"0": 0.4, "2": 0.4, "1": 0.2}
        three = record({"s": {"type": "score", "criteria": ["a", "b", "c"], "soft_label": soft}})
        self.assertTrue(question_of(three, "s").ambiguous)
        self.assertTrue(noul(0, "f", soft=UNIFORM).ambiguous)
        self.assertTrue(noul(0, "f", soft={"true": 1 / 3, "false": 1 / 3 + 1e-12}).ambiguous)   # a tie up to rounding
        unique = noul(0, "f", label=False, soft={"true": 0.6, "false": 0.4})
        self.assertFalse(unique.ambiguous)
        self.assertEqual(unique.keys[unique.label], "true")
        self.assertFalse(noul(0, "f").ambiguous)

    def test_metrics_leave_ambiguous_out_of_accuracy_and_automation_only(self):
        questions = [noul(0, "f"), noul(1, "f"), noul(2, "f", soft=UNIFORM)]
        scored = score_questions(questions, logits(0.9, 0.2, 0.95))
        family = predictor_report(questions, scored, threshold=0.85)["families"]["f"]
        self.assertEqual((family["n"], family["n_scored"], family["n_ambiguous"]), (3, 2, 1))
        self.assertAlmostEqual(family["accuracy"], 0.5)
        self.assertEqual((family["accepted"], family["errors"]), (1, 0))      # the 0.95 ambiguous one is not accepted
        self.assertAlmostEqual(family["coverage"], 0.5)
        brier = [2 * 0.1 ** 2, 2 * 0.8 ** 2, 2 * 0.45 ** 2]                   # the ambiguous one against (0.5, 0.5)
        self.assertAlmostEqual(family["brier"], np.mean(brier))
        # ECE: bins of 0.9 (correct 1), 0.8 (predicts false, label true: 0) and 0.95 (target mass 0.5 on "true")
        self.assertAlmostEqual(family["ece"], (abs(1 - 0.9) + abs(0 - 0.8) + abs(0.5 - 0.95)) / 3)

    def test_all_ambiguous_family_has_no_accuracy_and_is_ignored_by_the_gate(self):
        questions = [noul(0, "a"), noul(1, "b", soft=UNIFORM)]
        base, cand = logits(0.3, 0.9), logits(0.9, 0.1)
        report = paired_report(questions, {"base": base, "cand": cand}, {"base": 1.0, "cand": 1.0},
                               {"base": None, "cand": None}, samples=200)
        self.assertIsNone(report["families"]["b"]["base"]["accuracy"])
        self.assertIsNone(report["families"]["b"]["accuracy_delta"])
        self.assertEqual(report["families"]["b"]["n_ambiguous"], 1)
        self.assertEqual(report["macro"]["base"]["accuracy"], 0.0)            # family "a" only
        self.assertEqual(report["macro"]["cand"]["accuracy"], 1.0)
        self.assertEqual(report["n_ambiguous"], 1)
        report["guard"] = {"delta": 0.0}
        self.assertNotIn("b", decide("dev", report)["checks"]["no_family_regression"]["families"])
        self.assertEqual(report["families"]["b"]["accuracy_delta_ci95"], [None, None])

    def test_mcnemar_bootstrap_and_rows_skip_ambiguous(self):
        questions = [noul(i, "f", state=f"s{i // 2}") for i in range(4)] + [noul(9, "f", soft=UNIFORM)]
        base = logits(0.9, 0.9, 0.1, 0.1, 0.9)
        cand = logits(0.9, 0.9, 0.9, 0.9, 0.1)
        report = paired_report(questions, {"base": base, "cand": cand}, {"base": 1.0, "cand": 1.0},
                               {"base": None, "cand": None}, samples=200)
        self.assertEqual((report["mcnemar"]["base_only_correct"], report["mcnemar"]["cand_only_correct"]), (0, 2))
        self.assertAlmostEqual(report["bootstrap"]["delta"], 0.5)
        self.assertEqual(report["bootstrap"]["clusters"], 2)
        row = report["questions"][-1]
        self.assertTrue(row["ambiguous"])
        self.assertIsNone(row["label"])
        self.assertIsNone(row["base_correct"])

    def test_safety_and_thresholds_skip_ambiguous(self):
        questions = [noul(0, "needs_approval"), noul(1, "needs_approval", soft=UNIFORM)]
        rates = safety_rates(questions, score_questions(questions, logits(0.9, 0.1)))
        self.assertEqual(rates["needs_approval"], {"n_at_risk": 1, "fn_rate": 0.0})
        # a confident ambiguous question is not an error: threshold fitted on the scorable one alone
        fit = fit_threshold([noul(0, "f"), noul(1, "f", soft=UNIFORM)], logits(0.8, 0.99), 1.0, budget=0.0)
        self.assertAlmostEqual(fit["threshold"], 0.8)
        self.assertEqual(fit["val"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
