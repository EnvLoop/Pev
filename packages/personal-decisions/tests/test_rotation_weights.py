"""Style rotation over several style files and family oversampling in `generate`."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions.families.shard import load_shard
from personal_decisions.generate import Options, plan, run, style_for
from personal_decisions.jsonl import read_jsonl
from personal_decisions.render.fixture import FixtureRenderer

from .support import workspace
from .test_cli import run_cli


class RotationWeightsTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.styles = []
        for name in ("s-one", "s-two", "s-three"):
            path = self.folder / f"{name}.json"
            path.write_text(json.dumps({"style_id": name, "voice": name}))
            self.styles.append(path)

    def test_style_blocks_use_every_style_once(self):
        styles = [{"style_id": str(i)} for i in range(4)]
        for block in range(5):
            used = [style_for(styles, "train", 3, block * 4 + i)["style_id"] for i in range(4)]
            self.assertEqual(sorted(used), ["0", "1", "2", "3"])

    def test_records_rotate_styles_and_log_them(self):
        out = self.folder / "r.jsonl"
        stats = run(workspace() / "train", out, Options(n=18, seed=4, concurrency=2), FixtureRenderer(), self.styles)
        records = list(read_jsonl(out))
        used = Counter(record["meta"]["render"]["style_id"] for record in records)
        self.assertEqual(set(used), {"s-one", "s-two", "s-three"})
        self.assertEqual(stats["style_ids"], ["s-one", "s-two", "s-three"])
        self.assertEqual(stats["style_counts"], dict(used))
        logged = {row["id"]: row["style_id"] for row in read_jsonl(out.with_name("r.jsonl.log.jsonl"))
                  if row["status"] == "kept"}
        for record in records:  # the log may hold a few more kept ids past N (the last batch overshoots)
            self.assertEqual(logged[record["id"]], record["meta"]["render"]["style_id"])

    def test_duplicate_style_ids_are_refused(self):
        with self.assertRaises(ValueError):
            run(workspace() / "train", self.folder / "d.jsonl", Options(n=2, seed=1), FixtureRenderer(),
                [self.styles[0], self.styles[0]])

    def test_weights_oversample_a_family(self):
        shard = load_shard(workspace() / "train")
        plain, weighted = Options(n=1, seed=2), Options(n=1, seed=2, family_weights={"pick_option": 6.0})
        count = {name: Counter() for name in ("plain", "weighted")}
        for index in range(700):
            count["plain"].update(plan(shard, plain, index)[2])
            count["weighted"].update(plan(shard, weighted, index)[2])
        self.assertGreater(count["weighted"]["pick_option"], 1.8 * count["plain"]["pick_option"])
        for index in range(50):
            families = plan(shard, weighted, index)[2]
            self.assertEqual(len(families), len(set(families)))
            self.assertEqual(len(families), 3)

    def test_unweighted_plan_is_unchanged_by_the_option(self):
        shard = load_shard(workspace() / "train")
        self.assertEqual([plan(shard, Options(n=1, seed=7), i)[2] for i in range(30)],
                         [plan(shard, Options(n=1, seed=7, family_weights=None), i)[2] for i in range(30)])

    def test_cli_accepts_several_styles_and_weights(self):
        out = self.folder / "c.jsonl"
        code, receipt = run_cli("generate", "--shard", str(workspace() / "train"), "--out", str(out), "--n", "6",
                                "--seed", "2", "--renderer", "fixture", "--family-weights", "pick_option=2",
                                "--style-file", *map(str, self.styles))
        self.assertEqual(code, 0)
        self.assertEqual(receipt["states"], 6)
        code, _ = run_cli("generate", "--shard", str(workspace() / "train"), "--out", str(out), "--n", "1",
                          "--seed", "2", "--renderer", "fixture", "--family-weights", "bogus=2",
                          "--style-file", str(self.styles[0]))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
