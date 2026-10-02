import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from decision_eval.__main__ import main
from decision_eval.cli.release import recorded_path
from decision_eval.predictions import write_predictions
from decision_eval.records import write_jsonl
from decision_eval.release import half_of_renderer, holm, record_halves
from tests.synthetic import fake_predictions, muse_records


class HolmTests(unittest.TestCase):
    def test_step_down(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

    def test_capped_and_monotone(self):
        adjusted = holm([0.5, 0.4, 0.9])
        self.assertEqual(adjusted, [1.0, 1.0, 1.0])


class HalvesTests(unittest.TestCase):
    def test_renderer_rule(self):
        self.assertEqual(half_of_renderer("gpt-6-astra"), "A")
        self.assertEqual(half_of_renderer("claude-opus-5-5"), "B")
        self.assertIsNone(half_of_renderer("someone-else"))
        self.assertEqual(half_of_renderer("someone-else", {"someone-else": "B"}), "B")

    def test_id_prefix_wins_and_conflicts_refused(self):
        records = muse_records(2, seed=0)
        records[0]["id"], records[1]["id"] = "muse/test-a/1/0", "muse/test-b/1/1"
        records[0]["meta"]["render"] = {"model": "api-returned-name"}
        records[1]["meta"]["render"] = {"model": "claude-opus-5-5"}
        self.assertEqual(record_halves(records), {"muse/test-a/1/0": "A", "muse/test-b/1/1": "B"})
        records[1]["meta"]["render"] = {"model": "gpt-6-astra"}
        with self.assertRaises(ValueError):
            record_halves(records)

    def test_unknown_renderer_refused_with_counts_only(self):
        records = muse_records(4, seed=0)
        for r in records:
            r["meta"]["render"] = {"model": "mystery"}
        with self.assertRaises(ValueError) as ctx:
            record_halves(records)
        self.assertIn("mystery", str(ctx.exception))
        self.assertNotIn("rec-000", str(ctx.exception))


class CompareReleaseTests(unittest.TestCase):
    def test_end_to_end(self):
        d = Path(tempfile.mkdtemp())
        records = muse_records(60, seed=4, prefix="test")
        for i, r in enumerate(records):
            r["meta"]["render"] = {"model": "gpt-6-astra" if i % 2 else "claude-opus-5-5"}
        write_jsonl(d / "test.jsonl", records)
        spec = {"candidate": "cand", "models": []}
        for k, (name, p) in enumerate({"cand": 0.95, "base": 0.6, "api": 0.7}.items()):
            predictor = {"cand": "sft", "base": "base", "api": "openai"}[name]
            write_predictions(d / f"{name}.jsonl", fake_predictions(records, predictor, p, seed=k))
            entry = {"name": name, "preds": f"{name}.jsonl", "temperature": 1.0, "local": name != "api",
                     "thresholds": None}
            if name != "api":
                (d / f"{name}-thr.json").write_text(json.dumps({"temperature": 1.0, "threshold": 0.6}))
                entry["thresholds"] = f"{name}-thr.json"
            spec["models"].append(entry)
        (d / "spec.json").write_text(json.dumps(spec))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["compare-release", "--data", str(d / "test.jsonl"), "--spec", str(d / "spec.json"),
                  "--out-dir", str(d / "out"), "--samples", "200"])
        summary = json.loads(out.getvalue())
        self.assertEqual(set(summary["candidate_minus_other"]), {"overall", "A", "B"})
        self.assertEqual(set(summary["candidate_minus_other"]["overall"]), {"base", "api"})
        results = json.loads((d / "out" / "results.json").read_text())
        self.assertIsNone(results["models"]["api"]["overall"]["macro"]["coverage"])
        self.assertIsNotNone(results["models"]["base"]["overall"]["macro"]["coverage"])
        for name in ("cand", "base", "api"):
            self.assertTrue((d / "out" / f"score.{name}.json").is_file())
        self.assertEqual(results["sensitivity"]["excluded_families"], ["pick_option"])
        self.assertEqual(results["sensitivity"]["models"]["cand"]["overall"]["families"], 6)
        self.assertIn("base", results["sensitivity"]["comparisons"]["A"]["pairs"])
        test_results = d / "TEST_RESULTS.json"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["compare-release", "--data", str(d / "test.jsonl"), "--spec", str(d / "spec.json"),
                  "--out-dir", str(d / "out"), "--samples", "50", "--test-results", str(test_results)])
        block = json.loads(test_results.read_text())["sensitivity_excl_pick_option"]
        self.assertEqual(set(block["paired_adapter_vs"]["adapter - base"]), {"all", "A", "B"})
        self.assertIn("all", block["agrees_with_primary"]["adapter - api"])
        sizes = results["data"]["subsets"]
        self.assertEqual(sizes["A"]["questions"] + sizes["B"]["questions"], sizes["overall"]["questions"])
        recorded = [results["data"]["path"]] + [f["preds"]["path"] for f in results["inputs"].values()]
        self.assertFalse(any(Path(p).is_absolute() for p in recorded))      # no local directories in the reports
        self.assertEqual(json.loads(test_results.read_text())["inputs"]["base"]["preds"]["path"], "base.jsonl")

    def test_recorded_path_is_repository_relative(self):
        root = Path(tempfile.mkdtemp())
        (root / "work" / "preds").mkdir(parents=True)
        inside = root / "scripts" / ".." / "work" / "preds" / "b0.jsonl"
        self.assertEqual(recorded_path(inside, root), "work/preds/b0.jsonl")
        self.assertEqual(recorded_path(Path(tempfile.mkdtemp()) / "x.jsonl", root), "x.jsonl")


if __name__ == "__main__":
    unittest.main()
