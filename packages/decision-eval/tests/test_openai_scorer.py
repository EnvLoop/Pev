import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decision_eval.openai_scorer import OpenAIScorer, parse_answer, request_text, resolve_settings
from decision_eval.predictions import prediction_row
from decision_eval.records import questions_of
from tests.synthetic import muse_records

SETTINGS = {"OPENAI_API_KEY": "sk-test-not-real", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "RENDER_MODEL": "m"}


def uniform_answer(record):
    return {qid: {k: 1 / len(q.keys) for k in q.keys} for q in questions_of([record]) for qid in [q.qid]}


class FakeResponses:
    def __init__(self, texts):
        self.texts, self.requests = list(texts), []

    def create(self, **body):
        self.requests.append(body)
        return type("Response", (), {"output_text": self.texts.pop(0)})()


class FakeClient:
    def __init__(self, texts):
        self.responses = FakeResponses(texts)


class OpenAIScorerTests(unittest.TestCase):
    def test_request_lists_every_question_and_option_key_but_no_label(self):
        record = muse_records(n_states=3, seed=5)[2]
        text = request_text(record)
        for q in questions_of([record]):
            self.assertIn(f"Question id: {q.qid}", text)
            for key in q.keys:
                self.assertIn(f"- {key}", text)
        self.assertNotIn("label", text)
        self.assertNotIn("muse/", text)

    def test_parse_normalises_and_floors(self):
        record = muse_records(n_states=1)[0]
        answer = uniform_answer(record)
        first = next(iter(answer))
        answer[first] = {k: (2.0 if i == 0 else 0.0) for i, k in enumerate(answer[first])}
        parsed = parse_answer("```json\n" + json.dumps(answer) + "\n```", record)
        self.assertAlmostEqual(sum(parsed[first]), 1.0)
        self.assertGreater(min(parsed[first]), 0)            # floored, so log(p) is finite
        with self.assertRaises(ValueError):
            parse_answer(json.dumps({first: answer[first]}), record)     # missing questions

    def test_parse_accepts_copied_option_lines_but_not_ambiguous_ones(self):
        record = muse_records(n_states=1)[0]
        answer = uniform_answer(record)
        qid = next(iter(answer))
        answer[qid] = {f"{k}: some description": v for k, v in answer[qid].items()}
        self.assertEqual(set(parse_answer(json.dumps(answer), record)), set(answer))
        first = next(iter(answer[qid]))
        answer[qid][first.split(": ")[0]] = 0.5          # the same option twice
        with self.assertRaises(ValueError):
            parse_answer(json.dumps(answer), record)

    def test_scorer_retries_unusable_answers_and_writes_log_probs(self):
        record = muse_records(n_states=1)[0]
        client = FakeClient(["not json", json.dumps(uniform_answer(record))])
        scorer = OpenAIScorer(SETTINGS, client=client)
        rows = scorer.record_logits(record)
        self.assertEqual(scorer.calls, 2)
        self.assertFalse(client.responses.requests[0]["store"])
        key, keys, logits = rows[0]
        self.assertAlmostEqual(logits[0], math.log(1 / len(keys)))
        prediction_row(key, "openai", None, keys, logits)    # valid prediction format
        with self.assertRaises(ValueError):
            OpenAIScorer(SETTINGS, client=FakeClient(["x", "y", "z"])).record_logits(record)

    def test_settings_from_env_file_and_environment_without_exposing_the_key(self):
        env = Path(tempfile.mkdtemp()) / ".env"
        env.write_text("OPENAI_API_KEY=sk-file\nOPENAI_BASE_URL=http://api.test/v1\nRENDER_MODEL=astra\nOTHER=1\n")
        with mock.patch.dict(os.environ, {"RENDER_MODEL": "from-env"}, clear=False):
            for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
                os.environ.pop(key, None)
            settings = resolve_settings(env)
        self.assertEqual(settings["RENDER_MODEL"], "from-env")    # the process environment wins
        self.assertNotIn("OTHER", settings)
        legacy = Path(tempfile.mkdtemp()) / ".env"
        legacy.write_text("OPENAI_API_KEY=sk-file\nOPENAI_BASE_URL=http://api.test/v1\nRSI_TEACHER_MODEL=old-name\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_settings(legacy)["RENDER_MODEL"], "old-name")   # legacy variable name
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError) as caught:
            resolve_settings(Path(tempfile.mkdtemp()) / "missing.env")
        self.assertNotIn("sk-", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
