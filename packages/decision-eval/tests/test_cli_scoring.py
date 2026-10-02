import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from decision_eval.__main__ import main
from decision_eval.predictions import write_predictions
from decision_eval.records import read_jsonl, sha256_file, write_jsonl
from decision_eval.report import AGGREGATE_KEYS
from tests.synthetic import fake_predictions, kev_records, muse_records


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(list(argv))
    return json.loads(out.getvalue())


class ScoringPipelineTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.files = {}
        sets = {"val": muse_records(40, seed=1, prefix="val"), "dev": muse_records(60, seed=2, prefix="dev"),
                "hidden": muse_records(80, seed=3, prefix="sealedrecord"), "guard": kev_records(30)}
        for name, records in sets.items():
            self.files[name] = self.dir / f"{name}.jsonl"
            write_jsonl(self.files[name], records)
            base_bias = {"needs_approval": 0.9} if name != "guard" else None
            for role, predictor, p, seed in (("base", "base", 0.55, 10), ("cand", "sft", 0.9, 20)):
                path = self.dir / f"{name}.{role}.jsonl"
                bias = base_bias if role == "base" else None
                template = "direct"
                write_predictions(path, fake_predictions(records, predictor, p, seed, template, bias))
                self.files[f"{name}.{role}"] = path

    def fit(self, role):
        temp, thr = self.dir / f"{role}.temperature.json", self.dir / f"{role}.thresholds.json"
        fitted = run("calibrate", "--data", str(self.files["val"]), "--preds", str(self.files[f"val.{role}"]),
                     "--out", str(temp))
        self.assertEqual(json.loads(temp.read_text())["temperature"], fitted["temperature"])
        run("thresholds", "--data", str(self.files["val"]), "--preds", str(self.files[f"val.{role}"]),
            "--temperature", str(temp), "--budget", "0.05", "--out", str(thr))
        return temp, thr

    def score(self, split, gate, *extra):
        (bt, bthr), (ct, cthr) = self.fit("base"), self.fit("cand")
        out = self.dir / f"{split}.{gate}.score.json"
        run("score", "--data", str(self.files[split]), "--base", str(self.files[f"{split}.base"]),
            "--cand", str(self.files[f"{split}.cand"]), "--base-temp", str(bt), "--cand-temp", str(ct),
            "--base-thr", str(bthr), "--cand-thr", str(cthr), "--gate", gate, "--samples", "500",
            "--guard-data", str(self.files["guard"]), "--guard-base", str(self.files["guard.base"]),
            "--guard-cand", str(self.files["guard.cand"]), "--out", str(out), *extra)
        return out

    def test_dev_score_report_and_gate(self):
        report = json.loads(self.score("dev", "dev").read_text())
        self.assertEqual(report["n_questions"], 180)
        self.assertEqual(report["n_states"], 60)
        self.assertEqual(len(report["families"]), 7)
        self.assertGreater(report["macro"]["accuracy_delta"], 0.1)
        self.assertEqual(report["inputs"]["data"]["sha256"], sha256_file(self.files["dev"]))
        self.assertEqual(set(report["inputs"]), {"data", "base", "cand", "guard_data", "guard_base", "guard_cand",
                                                 "base_thr", "cand_thr", "base_temp", "cand_temp"})
        self.assertIn("needs_approval", report["safety"])
        self.assertIsNotNone(report["guard"])
        self.assertEqual(len(report["questions"]), 180)
        self.assertIn(report["gate"]["passed"], (True, False))
        self.assertEqual(set(report["gate"]["checks"]), {"macro_accuracy_delta", "ci_low_above_zero",
                                                         "no_family_regression", "safety_fn_not_significantly_higher",
                                                         "guard_not_lower"})
        for family in report["families"].values():
            low, high = family["accuracy_delta_ci95"]
            self.assertLessEqual(low, family["accuracy_delta"])
            self.assertGreaterEqual(high, family["accuracy_delta"])
        safety = report["safety"]["needs_approval"]
        self.assertAlmostEqual(safety["fn_delta"], safety["cand_fn_rate"] - safety["base_fn_rate"])
        self.assertEqual(len(safety["fn_delta_ci95"]), 2)

    def test_aggregate_only_leaks_no_ids_paths_or_rows(self):
        out = self.score("hidden", "hidden", "--aggregate-only")
        report = json.loads(out.read_text())
        self.assertEqual(tuple(report), tuple(sorted(AGGREGATE_KEYS)))
        self.assertTrue(report["aggregate_only"])
        text = out.read_text()
        for secret in ("sealedrecord", "state-0", "q0", "hidden.jsonl", str(self.dir)):
            self.assertNotIn(secret, text)
        self.assertEqual(report["inputs"]["data"], {"sha256": sha256_file(self.files["hidden"])})
        self.assertIn("p_one_sided", report["gate"]["checks"])

    def test_aggregate_only_errors_name_counts_not_ids(self):
        rows = [json.loads(line) for line in self.files["hidden.cand"].read_text().splitlines()][:-1]
        write_predictions(self.files["hidden.cand"], rows)
        with self.assertRaises(ValueError) as caught:
            self.score("hidden", "hidden", "--aggregate-only")
        self.assertIn("1 missing", str(caught.exception))
        self.assertNotIn("sealedrecord", str(caught.exception))

    def test_select_template_prefers_best_then_first(self):
        paths, records = [], read_jsonl(self.files["val"])
        for name, p in (("direct", 0.5), ("assistant", 0.8), ("evidence", 0.8)):
            path = self.dir / f"val.{name}.jsonl"
            write_predictions(path, fake_predictions(records, "base", p, 5, name))
            paths.append(str(path))
        result = run("select-template", "--data", str(self.files["val"]), "--preds", *paths)
        self.assertEqual(result["chosen"], "assistant")          # identical seeds: assistant ties evidence, first wins
        self.assertEqual([c["template"] for c in result["candidates"]], ["direct", "assistant", "evidence"])

    def test_threshold_from_another_temperature_is_refused(self):
        (bt, bthr), (ct, cthr) = self.fit("base"), self.fit("cand")
        with self.assertRaises(SystemExit):
            run("score", "--data", str(self.files["dev"]), "--base", str(self.files["dev.base"]),
                "--cand", str(self.files["dev.cand"]), "--base-temp", "3.3", "--cand-temp", str(ct),
                "--base-thr", str(bthr), "--cand-thr", str(cthr), "--gate", "none", "--out", str(self.dir / "x.json"))

    def test_c_must_be_sft_predictions_with_b0s_template(self):
        (bt, bthr), (ct, cthr) = self.fit("base"), self.fit("cand")
        other = self.dir / "dev.cand.evidence.jsonl"
        write_predictions(other, fake_predictions(read_jsonl(self.files["dev"]), "sft", 0.9, 20, "evidence"))
        for cand in (other, self.files["dev.base"]):
            with self.assertRaises(SystemExit):
                run("score", "--data", str(self.files["dev"]), "--base", str(self.files["dev.base"]),
                    "--cand", str(cand), "--base-temp", str(bt), "--cand-temp", str(ct), "--base-thr", str(bthr),
                    "--cand-thr", str(cthr), "--gate", "none", "--out", str(self.dir / "x.json"))

    def test_score_reference(self):
        out = self.dir / "ref.json"
        result = run("score-reference", "--data", str(self.files["dev"]), "--preds", str(self.files["dev.cand"]),
                     "--out", str(out))
        report = json.loads(out.read_text())
        self.assertEqual(report["n_questions"], 180)
        self.assertEqual(result["macro"]["accuracy"], report["macro"]["accuracy"])
        self.assertEqual(report["config"]["predictor"], "sft")


if __name__ == "__main__":
    unittest.main()
