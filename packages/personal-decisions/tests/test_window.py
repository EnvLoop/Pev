"""The sliding render window: an API failure mid-run loses nothing, and a rerun matches an uninterrupted run."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from personal_decisions import PersonalDecisionsError
from personal_decisions.generate import Options, run
from personal_decisions.jsonl import file_sha256, read_jsonl
from personal_decisions.render.fixture import FixtureRenderer

from .support import workspace


class FlakyRenderer(FixtureRenderer):
    """Fails once, on the `fail_at`-th render call, like an API that exhausted its retries."""

    def __init__(self, fail_at: int):
        self.calls, self.fail_at, self.lock = 0, fail_at, threading.Lock()

    def render(self, structured, required, style):
        with self.lock:
            self.calls += 1
            fail = self.calls == self.fail_at
        if fail:
            raise PersonalDecisionsError("LLM API failed 5 times: RateLimitError")
        return super().render(structured, required, style)


class WindowTest(unittest.TestCase):
    def test_failure_then_rerun_equals_uninterrupted_run(self):
        folder = Path(tempfile.mkdtemp())
        style = folder / "style.json"
        style.write_text(json.dumps({"style_id": "w", "voice": "plain"}))
        clean = folder / "clean.jsonl"
        run(workspace() / "train", clean, Options(n=24, seed=6, concurrency=3), FixtureRenderer(), style)
        flaky = folder / "flaky.jsonl"
        with self.assertRaises(PersonalDecisionsError):
            run(workspace() / "train", flaky, Options(n=24, seed=6, concurrency=3), FlakyRenderer(10), style)
        partial = {row["id"] for row in read_jsonl(flaky)}
        self.assertGreater(len(partial), 0)
        self.assertLess(len(partial), 24)
        run(workspace() / "train", flaky, Options(n=24, seed=6, concurrency=3), FixtureRenderer(), style)
        self.assertEqual(file_sha256(clean), file_sha256(flaky))
        self.assertEqual(len(list(read_jsonl(flaky))), 24)


if __name__ == "__main__":
    unittest.main()
