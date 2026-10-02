import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions.cli import main
from personal_decisions.jsonl import read_jsonl

from .fixture_tables import write_raw


def run_cli(*argv: str) -> tuple[int, dict]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(argv))
    text = (stdout.getvalue() or stderr.getvalue()).strip().splitlines()[-1]
    return code, json.loads(text)


class CliTest(unittest.TestCase):
    def test_build_split_generate_baselines_to_kev(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_raw(root / "raw", users=30)
            code, receipt = run_cli("build", "--raw", str(root / "raw"), "--out", str(root / "kb"))
            self.assertEqual((code, receipt["files"]["users.jsonl"]), (0, 30))
            code, receipt = run_cli("split", "--kb", str(root / "kb"), "--train", str(root / "train"),
                                    "--dev", str(root / "dev"), "--users", "hidden=3,dev=3,val=3",
                                    "--manifest", str(root / "split.json"))
            self.assertEqual(receipt["users"], {"train": 21, "dev": 3})
            style = root / "style.json"
            style.write_text(json.dumps({"voice": "plain"}))
            code, receipt = run_cli("generate", "--shard", str(root / "train"), "--out", str(root / "t.jsonl"),
                                    "--n", "6", "--seed", "2", "--style-file", str(style), "--renderer", "fixture")
            self.assertEqual((code, receipt["states"]), (0, 6))
            code, receipt = run_cli("shortcut-baselines", "--records", str(root / "t.jsonl"))
            self.assertEqual((code, receipt["records"]), (0, 6))
            code, receipt = run_cli("to-kev", "--records", str(root / "t.jsonl"), "--out", str(root / "k.jsonl"))
            self.assertEqual(code, 0)
            self.assertTrue(all("_meta" in row and "meta" not in row for row in read_jsonl(root / "k.jsonl")))

    def test_errors_are_reported_without_traceback(self):
        code, receipt = run_cli("generate", "--shard", "/nonexistent", "--out", "/tmp/x.jsonl", "--n", "1",
                                "--seed", "1", "--style-file", "/nonexistent/style.json", "--renderer", "fixture",
                                "--families", "nope")
        self.assertEqual(code, 1)
        self.assertFalse(receipt["ok"])
        self.assertIn("unknown families", receipt["error"])


if __name__ == "__main__":
    unittest.main()
