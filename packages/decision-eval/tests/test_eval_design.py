"""Amendment 4 tooling on synthetic data: per-family / safety CIs, noise floor, seed spread, validity checks."""
import random
import unittest

import numpy as np

from decision_eval.noise import noise_floor, seed_spread
from decision_eval.predictions import align
from decision_eval.records import questions_of
from decision_eval.stats import paired_bootstrap, rate_delta_bootstrap
from decision_eval.validity import ladder, null_baselines, oracle, validity
from tests.synthetic import fake_predictions, muse_records


def balanced_records(n_states, seed=0, p_true=0.5, choice_position=None):
    """Noul family `flag` with exactly a p_true share of true labels, and choice family `pick` (4 options) whose label
    positions are exactly balanced, or always `choice_position` (both in shuffled order)."""
    rng, records = random.Random(seed), []
    flags = [s < p_true * n_states for s in range(n_states)]
    positions = [s % 4 if choice_position is None else choice_position for s in range(n_states)]
    rng.shuffle(flags)
    rng.shuffle(positions)
    for s, (flag, position) in enumerate(zip(flags, positions)):
        records.append({"id": f"r{s}", "state": f"state {s}", "meta": {"state_id": f"s{s}",
                                                                          "families": {"a": "flag", "b": "pick"}},
                        "questions": {"a": {"type": "noul", "instructions": "flag?", "label": flag},
                                      "b": {"type": "choice", "instructions": "pick?",
                                            "criteria": {f"o{j}": None for j in range(4)}, "label": f"o{position}"}}})
    return records


def logits_of(questions, rows):
    return align(questions, rows)[0]


class StatsTests(unittest.TestCase):
    def test_family_cis_and_safety_rate_delta(self):
        records = muse_records(80, seed=5)
        questions = questions_of(records)
        base = np.array([random.Random(i).random() < 0.6 for i in range(len(questions))])
        cand = np.array([random.Random(i + 999).random() < 0.8 for i in range(len(questions))])
        boot = paired_bootstrap(questions, base, cand, samples=1000)
        self.assertEqual(len(boot["families"]), 7)
        for family in boot["families"].values():
            self.assertLessEqual(family["ci95"][0], family["delta"])
            self.assertGreaterEqual(family["ci95"][1], family["delta"])
        self.assertEqual(rate_delta_bootstrap([], [], []), {"delta": None, "ci95": [None, None]})
        rate = rate_delta_bootstrap(questions[:40], [1] * 40, [0] * 40, samples=200)
        self.assertEqual((rate["delta"], rate["ci95"]), (-1.0, [-1.0, -1.0]))


class NoiseTests(unittest.TestCase):
    def test_floor_shrinks_with_n_and_reports_headroom(self):
        small, large = muse_records(50, seed=1), muse_records(800, seed=1)
        floors = []
        for records in (small, large):
            questions = questions_of(records)
            floors.append(noise_floor(questions, logits_of(questions, fake_predictions(records, "base", 0.7, 3)),
                                      samples=800))
        self.assertLess(floors[1]["macro"]["accuracy_half_width"], floors[0]["macro"]["accuracy_half_width"] / 2.5)
        macro = floors[1]["macro"]
        self.assertAlmostEqual(macro["headroom"], 1 - macro["accuracy"])
        self.assertAlmostEqual(floors[1]["min_detectable_gain"], np.sqrt(2) * macro["accuracy_half_width"])
        family = next(iter(floors[1]["families"].values()))
        self.assertGreater(family["accuracy_half_width"], macro["accuracy_half_width"])
        self.assertAlmostEqual(family["accuracy_half_width"], family["binomial_half_width"], delta=0.03)

    def test_seed_spread(self):
        records = muse_records(120, seed=2)
        questions = questions_of(records)
        a = logits_of(questions, fake_predictions(records, "sft", 0.8, 11))
        same = seed_spread(questions, a, a, samples=500)
        self.assertEqual((same["macro_delta"], same["build_variance"], same["significant"]), (0.0, 0.0, False))
        b = logits_of(questions, fake_predictions(records, "sft", 0.8, 12))
        spread = seed_spread(questions, a, b, samples=500)
        self.assertEqual(set(spread["families"]), {q.family for q in questions})
        self.assertAlmostEqual(spread["build_variance"], abs(spread["macro_delta"]))


class ValidityTests(unittest.TestCase):
    def test_oracle_scores_everything_including_soft_labels(self):
        records = muse_records(30, seed=3)
        first = next(iter(records[0]["questions"].values()))
        first.update({"type": "noul", "criteria": None, "label": True, "soft_label": {"true": 0.5, "false": 0.5}})
        records[1]["questions"]["q0"].update({"type": "noul", "criteria": None, "soft_label": {"true": 0.8}})
        result = oracle(questions_of(records))
        self.assertTrue(result["passed"])
        self.assertTrue(all(a == 1.0 for a in result["families"].values()))

    def test_null_baselines_pass_on_balanced_and_catch_shortcuts(self):
        self.assertTrue(null_baselines(questions_of(balanced_records(400, seed=4)))["passed"])
        skewed = null_baselines(questions_of(balanced_records(400, seed=4, p_true=0.85)))
        self.assertFalse(skewed["families"]["flag"]["passed"]["majority"])
        self.assertTrue(skewed["families"]["pick"]["passed"]["constant"])
        position = null_baselines(questions_of(balanced_records(400, seed=4, choice_position=2)))
        pick = position["families"]["pick"]
        self.assertEqual((pick["constant"], pick["constant_position"]), (1.0, 2))
        self.assertFalse(pick["passed"]["constant"] or pick["passed"]["shuffled"])
        self.assertFalse(position["passed"])

    def test_shuffled_baseline_is_a_mean_over_fixed_permutations(self):
        # a 4-level family of ~110 exactly balanced questions, like VAL notify_level
        questions = [q for q in questions_of(balanced_records(112, seed=5)) if q.family == "pick"]
        single = None
        for seed in range(300):                      # find a seed whose single permutation strays outside the band
            one = null_baselines(questions, seed=seed, permutations=1)["families"]["pick"]
            if not one["passed"]["shuffled"]:
                single = (seed, one)
                break
        self.assertIsNotNone(single, "no single permutation strayed; the fixture is too easy")
        seed, one = single
        many = null_baselines(questions, seed=seed)["families"]["pick"]
        self.assertTrue(many["passed"]["shuffled"])
        self.assertEqual(many["permutations"], 50)
        low, high = many["shuffled_range"]
        self.assertLessEqual(low, many["shuffled"])
        self.assertGreaterEqual(high, many["shuffled"])
        self.assertLess(low, high)
        self.assertLessEqual(low, one["shuffled"])  # the stray single permutation is within the permutation spread
        self.assertEqual(null_baselines(questions, seed=seed), null_baselines(questions, seed=seed))   # fixed seeds

    def test_shuffled_mean_flags_label_imbalance(self):
        skewed = null_baselines(questions_of(balanced_records(400, seed=6, p_true=0.7)))["families"]["flag"]
        self.assertAlmostEqual(skewed["shuffled"], 0.7 ** 2 + 0.3 ** 2, delta=0.02)
        self.assertFalse(skewed["passed"]["shuffled"])

    def test_ladder_monotone_and_headroom(self):
        records = muse_records(300, seed=6)
        questions = questions_of(records)
        rungs = [(name, logits_of(questions, fake_predictions(records, "base", p, seed)))
                 for name, p, seed in (("small", 0.45, 1), ("b0", 0.65, 2), ("ref", 0.85, 3))]
        good = ladder(questions, rungs, samples=500)
        self.assertTrue(good["monotone"]["passed"] and good["headroom"]["passed"])
        self.assertEqual(good["headroom"]["best"], "ref")
        inverted = ladder(questions, list(reversed(rungs)), samples=500)
        self.assertFalse(inverted["monotone"]["passed"])
        self.assertFalse(inverted["no_family_inversion"]["passed"])
        oracle_like = rungs + [("perfect", logits_of(questions, fake_predictions(records, "base", 1.0, 4)))]
        self.assertFalse(ladder(questions, oracle_like, samples=300)["headroom"]["passed"])

    def test_validity_report_is_machine_readable(self):
        records = balanced_records(300, seed=7)
        questions = questions_of(records)
        rungs = [(name, logits_of(questions, fake_predictions(records, "base", p, s)))
                 for name, p, s in (("weak", 0.5, 1), ("strong", 0.8, 2))]
        result = validity(questions, rungs, samples=300)
        self.assertEqual(set(result["checks"]), {"oracle", "null_baselines", "ladder_monotone",
                                                 "ladder_no_family_inversion", "ladder_headroom"})
        self.assertTrue(all(isinstance(c["passed"], bool) for c in result["checks"].values()))
        self.assertEqual(result["passed"], all(c["passed"] for c in result["checks"].values()))
        self.assertEqual([r["name"] for r in result["ladder"]], ["weak", "strong"])


if __name__ == "__main__":
    unittest.main()
