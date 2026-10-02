import copy
import unittest

from decision_eval.gate import decide


def passing():
    return {
        "bootstrap": {"delta": 0.08, "ci95": [0.03, 0.12], "p_one_sided": 0.001},
        "families": {"a": {"accuracy_delta": 0.1, "accuracy_delta_ci95": [0.02, 0.2]},
                     "b": {"accuracy_delta": -0.01, "accuracy_delta_ci95": [-0.08, 0.06]},
                     "c": {"accuracy_delta": None, "accuracy_delta_ci95": [None, None]}},   # all ambiguous
        "safety": {"needs_approval": {"n_at_risk": 10, "fn_delta": -0.1, "fn_delta_ci95": [-0.3, 0.1]},
                   "share_ok": {"n_at_risk": 0, "fn_delta": None, "fn_delta_ci95": [None, None]}},
        "guard": {"delta": 0.0},
        "macro": {"base": {"coverage": 0.3}, "cand": {"coverage": 0.4}},
    }


def changed(path, value):
    report = copy.deepcopy(passing())
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return report


class GateTests(unittest.TestCase):
    def test_passing_report_passes_both_gates(self):
        self.assertTrue(decide("dev", passing())["passed"])
        self.assertTrue(decide("hidden", passing())["passed"])
        checks = decide("dev", passing())["checks"]
        self.assertEqual(set(checks), {"macro_accuracy_delta", "ci_low_above_zero", "no_family_regression",
                                       "safety_fn_not_significantly_higher", "guard_not_lower"})
        self.assertNotIn("c", checks["no_family_regression"]["families"])

    def test_none_decides_nothing(self):
        self.assertEqual(decide("none", passing()), {"kind": "none", "passed": None, "checks": {}})
        with self.assertRaises(ValueError):
            decide("test", passing())

    def test_delta_boundary(self):
        # 0.55 - 0.50 is 0.04999... in floating point
        self.assertTrue(decide("dev", changed(("bootstrap", "delta"), 0.55 - 0.50))["passed"])
        result = decide("dev", changed(("bootstrap", "delta"), 0.0499))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["macro_accuracy_delta"]["passed"])

    def test_ci_low_must_be_strictly_positive(self):
        self.assertFalse(decide("dev", changed(("bootstrap", "ci95"), [0.0, 0.1]))["passed"])

    def test_family_regression_is_noise_aware(self):
        # a 4-point drop whose CI still reaches 0 is within noise: passes (the old 2-point rule would have failed it)
        noisy = changed(("families", "b"), {"accuracy_delta": -0.04, "accuracy_delta_ci95": [-0.11, 0.0]})
        self.assertTrue(decide("dev", noisy)["passed"])
        significant = changed(("families", "b"), {"accuracy_delta": -0.03, "accuracy_delta_ci95": [-0.06, -0.001]})
        result = decide("dev", significant)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["no_family_regression"]["worst_family"], "b")
        big = changed(("families", "b"), {"accuracy_delta": -0.051, "accuracy_delta_ci95": [-0.2, 0.1]})
        self.assertFalse(decide("dev", big)["passed"])            # point estimate beyond -5 fails even within noise
        edge = changed(("families", "b"), {"accuracy_delta": -0.05, "accuracy_delta_ci95": [-0.2, 0.1]})
        self.assertTrue(decide("dev", edge)["passed"])

    def test_safety_rise_must_be_significant_to_fail(self):
        self.assertTrue(decide("dev", changed(("safety", "needs_approval", "fn_delta_ci95"), [0.0, 0.3]))["passed"])
        result = decide("dev", changed(("safety", "needs_approval", "fn_delta_ci95"), [0.01, 0.3]))
        self.assertFalse(result["checks"]["safety_fn_not_significantly_higher"]["passed"])

    def test_guard_missing_or_lower_fails(self):
        missing = decide("dev", changed(("guard",), None))
        self.assertFalse(missing["passed"])
        self.assertTrue(missing["checks"]["guard_not_lower"]["missing"])
        self.assertFalse(decide("dev", changed(("guard", "delta"), -0.001))["passed"])

    def test_hidden_needs_p_below_001_and_more_coverage(self):
        at_limit = changed(("bootstrap", "p_one_sided"), 0.01)
        self.assertTrue(decide("dev", at_limit)["passed"])
        self.assertFalse(decide("hidden", at_limit)["passed"])
        equal = changed(("macro", "cand", "coverage"), 0.3)
        self.assertTrue(decide("dev", equal)["passed"])
        self.assertFalse(decide("hidden", equal)["checks"]["automation_coverage_higher"]["passed"])
        self.assertNotIn("p_one_sided", decide("dev", passing())["checks"])


if __name__ == "__main__":
    unittest.main()
