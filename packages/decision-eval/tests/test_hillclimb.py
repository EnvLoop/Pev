"""export-hillclimb writes the claude-api hillclimb layout; noise-floor and validity run end to end from the CLI."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from decision_eval.__main__ import main
from decision_eval.predictions import write_predictions
from decision_eval.records import write_jsonl
from tests.synthetic import fake_predictions, muse_records

# Optional: path to the claude-api skill's `evals/report/build-report-lite.mjs` (the renderer test skips when unset).
BUILDER = Path(os.environ["HILLCLIMB_REPORT_BUILDER"]) if os.environ.get("HILLCLIMB_REPORT_BUILDER") else None


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(list(argv))
    return json.loads(out.getvalue())


class HillclimbExportTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.records = muse_records(20, seed=8)
        first = next(iter(self.records[0]["questions"].values()))
        first.update({"type": "noul", "criteria": None, "soft_label": {"true": 0.5, "false": 0.5}})
        self.data, self.flow = self.dir / "val.jsonl", self.dir / "flow"
        write_jsonl(self.data, self.records)
        self.preds = {}
        for variant, p in (("baseline", 0.5), ("v1", 0.8)):
            self.preds[variant] = self.dir / f"{variant}.jsonl"
            write_predictions(self.preds[variant], fake_predictions(self.records, "base", p, 3, "direct"))

    def export(self, variant, split="val"):
        return run("export-hillclimb", "--data", str(self.data), "--preds", str(self.preds[variant]),
                   "--variant", variant, "--split", split, "--out", str(self.flow))

    def test_val_rows_follow_the_results_contract(self):
        summary = self.export("baseline")
        rows = [json.loads(x) for x in (self.flow / "baseline" / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 60)
        row = rows[1]
        self.assertTrue(set(row) >= {"prompt_id", "rep", "prompt", "tags", "grade", "model", "usage"})
        self.assertEqual((row["rep"], row["tags"][1], row["model"], row["usage"]), (0, "baseline", "base/direct", {}))
        self.assertIn(self.records[0]["state"], row["prompt"])
        self.assertEqual(set(row["grade"]), {"correct", "brier"})
        self.assertEqual(set(rows[0]["grade"]), {"brier"})                 # ambiguous: no correct grade
        state = json.loads((self.flow / "_state.json").read_text())
        self.assertEqual([m["id"] for m in state["metrics"]], ["correct", "brier"])
        self.assertEqual(len(state["train_ids"]), 60)
        self.assertEqual(summary["label"], "baseline")
        self.assertEqual(json.loads((self.flow / "baseline" / "summary.json").read_text())["target"], "code")

    def test_dev_rows_withhold_prompts(self):
        self.export("v1", split="test")
        text = (self.flow / "v1" / "results.jsonl").read_text()
        self.assertNotIn(self.records[0]["state"], text)
        self.assertIn("withheld", text)
        self.assertEqual(len(json.loads((self.flow / "_state.json").read_text())["test_ids"]), 60)

    def test_refuses_sealed_paths_and_bad_variants(self):
        sealed = self.dir / "sealed" / "x.jsonl"
        write_jsonl(sealed, self.records)
        with self.assertRaises(SystemExit):
            run("export-hillclimb", "--data", str(sealed), "--preds", str(self.preds["v1"]), "--variant", "v1",
                "--out", str(self.flow))
        with self.assertRaises(SystemExit):
            run("export-hillclimb", "--data", str(self.data), "--preds", str(self.preds["v1"]), "--variant", "v1",
                "--out", str(self.dir / "hidden-flow"))
        with self.assertRaises(ValueError):
            self.export_variant("round1")

    def export_variant(self, variant):
        return run("export-hillclimb", "--data", str(self.data), "--preds", str(self.preds["v1"]),
                   "--variant", variant, "--out", str(self.flow))

    @unittest.skipUnless(shutil.which("node") and BUILDER and BUILDER.is_file(),
                         "node or HILLCLIMB_REPORT_BUILDER (claude-api report builder) missing")
    def test_lite_report_builder_renders_the_rounds(self):
        self.export("baseline")
        self.export("v1")
        subprocess.run(["node", str(BUILDER), str(self.flow)], check=True, capture_output=True)
        self.assertTrue((self.flow / "report.html").is_file())
        scores = (self.flow / "trajectory" / "scores.tsv").read_text().splitlines()
        self.assertEqual(len(scores), 61)                                  # header + one row per question


class DiagnoseCliTests(unittest.TestCase):
    def test_noise_floor_and_validity_commands(self):
        directory = Path(tempfile.mkdtemp())
        records = muse_records(60, seed=9)
        data, a, b = directory / "dev.jsonl", directory / "a.jsonl", directory / "b.jsonl"
        write_jsonl(data, records)
        write_predictions(a, fake_predictions(records, "base", 0.5, 1, "direct"))
        write_predictions(b, fake_predictions(records, "sft", 0.8, 2, "direct"))
        floor = run("noise-floor", "--data", str(data), "--preds", str(a), "--reps", "300")
        self.assertEqual(floor["mode"], "noise-floor")
        self.assertIn("min_detectable_gain", floor)
        spread = run("noise-floor", "--data", str(data), "--seed-spread", str(a), str(b), "--reps", "300",
                     "--out", str(directory / "spread.json"))
        self.assertEqual(spread["mode"], "seed-spread")
        self.assertTrue((directory / "spread.json").is_file())
        result = run("validity", "--data", str(data), "--preds-ladder", f"b0={a}", f"c={b}", "--samples", "300")
        self.assertIn("ladder_monotone", result["checks"])
        self.assertEqual(set(result["inputs"]), {"data", "ladder_b0", "ladder_c"})


if __name__ == "__main__":
    unittest.main()
