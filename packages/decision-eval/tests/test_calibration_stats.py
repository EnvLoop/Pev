import random
import unittest

import numpy as np

from decision_eval.calibration import GRID, fit_temperature, fit_threshold, mean_nll
from decision_eval.metrics import score_questions
from decision_eval.records import Question
from decision_eval.stats import mcnemar, paired_bootstrap


def sampled(n, true_temperature, seed=0, k=3):
    """Questions whose labels are drawn from softmax(z / T): the NLL-optimal temperature is ~T."""
    rng = np.random.default_rng(seed)
    questions, logits = [], []
    for i in range(n):
        z = rng.normal(0, 3, size=k)
        p = np.exp(z / true_temperature)
        p /= p.sum()
        label = int(rng.choice(k, p=p))
        questions.append(Question(f"r{i}", "q", "f", f"s{i}", "choice", tuple("abc"[:k]), label,
                                  tuple(float(j == label) for j in range(k))))
        logits.append(z)
    return questions, logits


class CalibrationTests(unittest.TestCase):
    def test_fit_recovers_the_generating_temperature(self):
        questions, logits = sampled(4000, 2.5)
        fit = fit_temperature(questions, logits)
        self.assertAlmostEqual(fit["temperature"], 2.5, delta=0.35)
        self.assertLess(fit["nll_calibrated"], fit["nll_raw"])
        self.assertFalse(fit["at_grid_edge"])
        self.assertAlmostEqual(fit["nll_calibrated"], mean_nll(questions, logits, fit["temperature"]))
        self.assertIn(fit["temperature"], GRID)

    def test_vectorised_nll_matches_per_question_scoring(self):
        questions, logits = sampled(50, 1.0, seed=3)
        for t in (0.3, 1.0, 7.0):
            self.assertAlmostEqual(mean_nll(questions, logits, t), score_questions(questions, logits, t).nll.mean())

    def test_threshold_is_lowest_within_budget(self):
        conf = [0.99, 0.98, 0.97, 0.96, 0.9, 0.8]
        right = [True, True, True, True, False, False]
        questions = [Question(f"r{i}", "q", "f", "s", "noul", ("false", "true"), 1, (0.0, 1.0)) for i in range(6)]
        logits = [np.log([1 - c, c]) if ok else np.log([c, 1 - c]) for c, ok in zip(conf, right)]
        fit = fit_threshold(questions, logits, 1.0, budget=0.2)
        self.assertAlmostEqual(fit["threshold"], 0.9)          # 5 accepted, 1 error = 0.2 <= 0.2
        self.assertEqual(fit["val"]["accepted"], 5)
        self.assertIsNone(fit_threshold(questions[4:], logits[4:], 1.0, budget=0.05)["threshold"])


def paired(n_states, base_rate, cand_rate, seed=0):
    rng = random.Random(seed)
    questions, base, cand = [], [], []
    for s in range(n_states):
        for k in range(3):
            questions.append(Question(f"r{s}", f"q{k}", ("f1", "f2", "f3")[k], f"s{s}", "noul", ("false", "true"), 1,
                                      (0.0, 1.0)))
            base.append(rng.random() < base_rate)
            cand.append(rng.random() < cand_rate)
    return questions, np.array(base), np.array(cand)


class StatsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_for_a_seed(self):
        questions, base, cand = paired(40, 0.6, 0.8)
        a = paired_bootstrap(questions, base, cand, samples=2000, seed=7)
        b = paired_bootstrap(questions, base, cand, samples=2000, seed=7)
        c = paired_bootstrap(questions, base, cand, samples=2000, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a["ci95"], c["ci95"])
        self.assertEqual(a["clusters"], 40)

    def test_bootstrap_delta_is_family_macro_and_inside_ci(self):
        questions, base, cand = paired(60, 0.5, 0.9, seed=2)
        result = paired_bootstrap(questions, base, cand, samples=3000)
        families = np.array([q.family for q in questions])
        expected = np.mean([cand[families == f].mean() - base[families == f].mean() for f in ("f1", "f2", "f3")])
        self.assertAlmostEqual(result["delta"], expected)
        self.assertLess(result["ci95"][0], result["delta"])
        self.assertGreater(result["ci95"][1], result["delta"])
        self.assertGreater(result["ci95"][0], 0)
        self.assertLess(result["p_one_sided"], 0.01)
        self.assertEqual(result["samples"], 3000)

    def test_bootstrap_null_gives_large_p(self):
        questions, base, _ = paired(50, 0.7, 0.7, seed=4)
        result = paired_bootstrap(questions, base, base.copy(), samples=1000)
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual(result["p_one_sided"], 1.0)

    def test_bootstrap_keeps_states_together(self):
        """Every question of one state flips together: resampling states, not questions, widens the interval."""
        questions, base, cand = [], [], []
        for s in range(30):
            flip = s % 2 == 0
            for k in range(10):
                questions.append(Question(f"r{s}", f"q{k}", "f", f"s{s}", "noul", ("false", "true"), 1, (0.0, 1.0)))
                base.append(not flip)
                cand.append(flip)
        clustered = paired_bootstrap(questions, np.array(base), np.array(cand), samples=2000)
        split = [Question(q.record_id, q.qid, q.family, f"{q.state_id}/{q.qid}", q.type, q.keys, q.label, q.target)
                 for q in questions]
        independent = paired_bootstrap(split, np.array(base), np.array(cand), samples=2000)
        width = lambda r: r["ci95"][1] - r["ci95"][0]
        self.assertGreater(width(clustered), 2 * width(independent))

    def test_mcnemar_exact(self):
        base = np.array([False] * 5 + [True] * 10)
        cand = np.array([True] * 5 + [True] * 10)
        result = mcnemar(base, cand)
        self.assertEqual((result["base_only_correct"], result["cand_only_correct"]), (0, 5))
        self.assertAlmostEqual(result["p_two_sided"], 2 / 32)
        self.assertEqual(mcnemar(base, base)["p_two_sided"], 1.0)
        self.assertAlmostEqual(mcnemar(np.array([True, False]), np.array([False, True]))["p_two_sided"], 1.0)


if __name__ == "__main__":
    unittest.main()
