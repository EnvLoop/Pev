import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions import anchors
from personal_decisions.generate import Options, run
from personal_decisions.jsonl import file_sha256, read_jsonl
from personal_decisions.render import Rendered
from personal_decisions.render.fixture import FixtureRenderer

from .support import workspace
from .test_cli import run_cli

TOP_KEYS = {"id", "state", "questions", "meta"}
META_KEYS = {"user_id", "shard", "state_id", "families", "anchors", "render"}


class DroppingRenderer(FixtureRenderer):
    """Loses every option line: records with a pick_option question must be dropped."""

    def render(self, structured, required, style):
        rendered = super().render(structured, required, style)
        text = "\n".join(line for line in rendered.text.splitlines() if not line.startswith("- ") or "(" not in line)
        return Rendered(text, rendered.model, rendered.prompt_sha256, {"input_tokens": 10, "output_tokens": 5})


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.style = self.folder / "style.json"
        self.style.write_text(json.dumps({"style_id": "test-style", "voice": "plain"}))

    def generate(self, name: str, n: int, seed: int = 5, renderer=None, **options) -> tuple[Path, dict]:
        out = self.folder / name
        stats = run(workspace() / "train", out, Options(n=n, seed=seed, concurrency=2, **options),
                    renderer or FixtureRenderer(), self.style)
        return out, stats

    def test_records_follow_the_contract(self):
        out, stats = self.generate("a.jsonl", 25)
        records = list(read_jsonl(out))
        self.assertEqual(len(records), 25)
        self.assertEqual(len({record["id"] for record in records}), 25)
        for record in records:
            self.assertEqual(set(record), TOP_KEYS)
            self.assertTrue(META_KEYS <= set(record["meta"]))
            self.assertEqual(record["meta"]["shard"], "train")
            self.assertEqual(record["meta"]["render"]["style_id"], "test-style")
            self.assertEqual(set(record["meta"]["render"]), {"model", "style_id", "prompt_sha256"})
            for qid, question in record["questions"].items():
                self.assertEqual(question["src"], "muse/" + record["meta"]["families"][qid])
                self.assertIn(question["type"], ("noul", "choice", "score"))
                self.assertTrue({"type", "instructions", "criteria", "label", "src"} <= set(question))
                marks = record["meta"]["anchors"][qid]
                self.assertIsNone(anchors.check(record["state"], marks["required"], marks["forbidden"]))
        self.assertEqual(stats["output_sha256"], file_sha256(out))
        self.assertTrue(out.with_name("a.jsonl.stats.json").is_file())
        self.assertIn(stats["output_sha256"], out.with_name("a.jsonl.sha256").read_text())
        self.assertEqual(stats["questions"], sum(stats["per_family"].values()))

    def test_generation_is_deterministic_and_resumable(self):
        first, _ = self.generate("full.jsonl", 20)
        partial, _ = self.generate("resumed.jsonl", 8)
        resumed, stats = self.generate("resumed.jsonl", 20)
        self.assertEqual(file_sha256(first), file_sha256(resumed))
        self.assertEqual(stats["states"], 20)

    def test_buried_states_carry_distractor_text(self):
        out, _ = self.generate("buried.jsonl", 12, p_buried=1.0)
        records = list(read_jsonl(out))
        buried = [r for r in records if any(v.startswith("buried+") for v in r["meta"]["variants"].values())]
        self.assertTrue(buried)
        for record in buried:
            self.assertGreater(len(record["state"]), 4000)
            self.assertIn("--- ", record["state"])

    def test_missing_anchor_drops_the_record(self):
        out, stats = self.generate("drops.jsonl", 10, renderer=DroppingRenderer(), families=("pick_option",))
        self.assertEqual(stats["states"], 0)
        self.assertGreater(stats["drops_by_reason"]["missing_required_anchor"], 0)
        self.assertGreater(stats["tokens"]["input_tokens"], 0)

    def test_cli_generate_with_fixture_renderer(self):
        out = self.folder / "cli.jsonl"
        code, _ = run_cli("generate", "--shard", str(workspace() / "dev"), "--out", str(out), "--n", "4",
                          "--seed", "1", "--style-file", str(self.style), "--renderer", "fixture",
                          "--families", "route", "share_ok")
        self.assertEqual(code, 0)
        families = {family for record in read_jsonl(out) for family in record["meta"]["families"].values()}
        self.assertTrue(families <= {"route", "share_ok"})


if __name__ == "__main__":
    unittest.main()
