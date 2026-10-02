import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decision_eval.checkpoint import Checkpoint, score_records, transient
from decision_eval.predictions import read_predictions
from decision_eval.records import question_of, questions_of, record_id, write_jsonl
from tests.synthetic import muse_records


class Status(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status_code = status


class FlakyScorer:
    """Uniform logits; raises the queued errors for a record before answering it."""

    def __init__(self, errors=None):
        self.errors, self.calls, self.requests = dict(errors or {}), [], []

    def record_logits(self, record):
        rid = record_id(record)
        self.calls.append(rid)
        queue = self.errors.get(rid) or []
        if queue:
            raise queue.pop(0)
        self.requests.append({"id": rid, "model": "m", "input_tokens": 10, "latency_ms": 1.0})
        return [(question_of(record, q).key, question_of(record, q).keys, [0.0] * len(question_of(record, q).keys))
                for q in record["questions"]]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.records = muse_records(n_states=4, seed=7)

    def test_transient_classification(self):
        self.assertTrue(transient(Status(429)) and transient(Status(503)) and transient(TimeoutError()))
        self.assertFalse(transient(Status(400)) or transient(ValueError("bad answer")))

    def test_retries_transient_errors_and_resumes_without_rescoring(self):
        rid = record_id(self.records[1])
        scorer = FlakyScorer({rid: [Status(429), Status(503)]})
        checkpoint = Checkpoint(self.dir / "p.partial.jsonl", "sha")
        rows, requests, failures = score_records(scorer, self.records, checkpoint, 2, backoff=(0, 0), sleep=lambda s: 0)
        self.assertEqual((failures, len(requests)), ({}, 4))
        self.assertEqual(len(rows), len(questions_of(self.records)))
        again = FlakyScorer()
        rows2, _, _ = score_records(again, self.records, Checkpoint(self.dir / "p.partial.jsonl", "sha"), 2)
        self.assertEqual((again.calls, rows2), ([], rows))

    def test_persistent_rate_limit_stops_and_other_data_is_refused(self):
        rid = record_id(self.records[0])
        scorer = FlakyScorer({rid: [Status(429)] * 5})
        checkpoint = Checkpoint(self.dir / "q.partial.jsonl", "sha")
        _, _, failures = score_records(scorer, self.records, checkpoint, 1, backoff=(0,), sleep=lambda s: 0)
        self.assertEqual(set(failures), {record_id(r) for r in self.records})   # first failed, the rest not attempted
        self.assertIn("HTTP 429", failures[rid])
        with self.assertRaises(SystemExit):
            Checkpoint(self.dir / "q.partial.jsonl", "other-sha")

    @unittest.skipUnless(importlib.util.find_spec("typesafe_sdk"), "the jev extra is not installed")
    def test_cli_stops_on_unscorable_records_then_fills_uniform_on_request(self):
        from decision_eval.__main__ import main
        data, out = self.dir / "dev" / "dev.jsonl", self.dir / "dev" / "jev.jsonl"
        write_jsonl(data, self.records)
        bad = record_id(self.records[2])
        scorer = FlakyScorer({bad: [ValueError("unusable")] * 2})
        scorer.model, scorer.accounting = "jev-1.13.0", lambda requests=None: {"requests": len(requests)}
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "ts-test"}), \
                mock.patch("decision_eval.jev_scorer.JevScorer", return_value=scorer):
            with self.assertRaises(SystemExit):
                main(["predict-jev", "--data", str(data), "--out", str(out)])
            self.assertFalse(out.exists())
            main(["predict-jev", "--data", str(data), "--out", str(out), "--fill-unscorable", "uniform"])
        self.assertEqual(scorer.calls.count(record_id(self.records[0])), 1)   # resumed, not re-sent
        rows = read_predictions(out)
        self.assertEqual(len(rows), len(questions_of(self.records)))
        meta = json.loads(Path(f"{out}.meta.json").read_text())
        self.assertEqual((meta["unscorable"], meta["requests"]), ([bad], 3))


if __name__ == "__main__":
    unittest.main()
