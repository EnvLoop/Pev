"""predict-jev with a mocked TypeSafe client: 1:1 request mapping, full distributions, accounting, DEV-only guard."""
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from decision_eval.jev_scorer import USD_PER_INPUT_MTOK, JevScorer, distribution, resolve_key
from decision_eval.predictions import read_predictions
from decision_eval.conventions import question_keys
from decision_eval.records import questions_of, write_jsonl
from tests.synthetic import muse_records

HAS_SDK = importlib.util.find_spec("typesafe_sdk") is not None


def answer_for(question):
    k = len(question.keys)
    if question.type == "noul":
        return SimpleNamespace(type="noul", noul=0.8)
    if question.type == "choice":
        return SimpleNamespace(type="choice", probabilities={key: 1 / k for key in question.keys})
    return SimpleNamespace(type="score", probabilities={i: (1.0 if i == 0 else 0.0) for i in range(k)})


class FakeClient:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        answers = {qid: answer_for(_question(q)) for qid, q in questions.items()}
        return SimpleNamespace(model="jev-1.13.0", request_id="req-1", answers=answers,
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=20))


def _question(q):
    return SimpleNamespace(type=q.type, keys=tuple(question_keys(q.type, q.criteria)))


class DistributionTests(unittest.TestCase):
    def test_noul_choice_and_score_distributions(self):
        noul = SimpleNamespace(type="noul", keys=("false", "true"))
        self.assertEqual(distribution(noul, SimpleNamespace(type="noul", noul=0.25)), [0.75, 0.25])
        choice = SimpleNamespace(type="choice", keys=("b", "a"))
        answer = SimpleNamespace(type="choice", probabilities={"a": 0.9, "b": 0.1})
        self.assertAlmostEqual(distribution(choice, answer)[0], 0.1)
        score = SimpleNamespace(type="score", keys=("0", "1", "2"))
        p = distribution(score, SimpleNamespace(type="score", probabilities={0: 0.0, 1: 0.5, 2: 0.5}))
        self.assertGreater(p[0], 0)                                  # floored, so log(p) is finite
        self.assertAlmostEqual(sum(p), 1.0)
        with self.assertRaises(ValueError):
            distribution(choice, SimpleNamespace(type="choice", probabilities={"a": 1.0}))
        with self.assertRaises(ValueError):
            distribution(noul, SimpleNamespace(type="choice", probabilities={}))

    def test_key_from_env_file_or_environment_never_echoed(self):
        env = Path(tempfile.mkdtemp()) / ".env"
        env.write_text("TYPESAFE_API_KEY=ts-file-key\nOTHER=1\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_key(env), "ts-file-key")
            with self.assertRaises(ValueError) as caught:
                resolve_key(Path(tempfile.mkdtemp()) / "missing.env")
        self.assertNotIn("ts-", str(caught.exception))
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "ts-env"}):
            self.assertEqual(resolve_key(env), "ts-env")


@unittest.skipUnless(HAS_SDK, "the jev extra (typesafe-sdk) is not installed")
class JevScorerTests(unittest.TestCase):
    def test_request_maps_records_one_to_one_and_writes_log_probs(self):
        record = muse_records(n_states=6, seed=3)[5]
        client = FakeClient()
        scorer = JevScorer(client=client)
        rows = scorer.record_logits(record)
        call = client.calls[0]
        self.assertEqual(call["state"], record["state"])
        self.assertEqual(call["model"], "jev-1.13.0")
        for qid, q in record["questions"].items():
            sent = call["questions"][qid]
            self.assertEqual((sent.type, sent.instructions), (q["type"], q["instructions"]))
            self.assertEqual(sent.model_dump()["criteria"], q.get("criteria"))
        self.assertEqual([key for key, _, _ in rows], [q.key for q in questions_of([record])])
        for _, keys, logits in rows:
            self.assertAlmostEqual(sum(math.exp(z) for z in logits), 1.0)
        request = scorer.requests[0]
        self.assertEqual((request["input_tokens"], request["model"]), (1000, "jev-1.13.0"))
        self.assertAlmostEqual(scorer.accounting()["usd_estimated"], 1000 * USD_PER_INPUT_MTOK / 1e6)

    def test_cli_writes_predictions_requests_and_meta(self):
        from decision_eval.__main__ import main
        directory = Path(tempfile.mkdtemp()) / "dev"
        data, out = directory / "dev.jsonl", directory / "jev.jsonl"
        records = muse_records(n_states=3, seed=4)
        write_jsonl(data, records)
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "ts-test"}), \
                mock.patch("typesafe_sdk.TypeSafeClient", return_value=FakeClient()):
            main(["predict-jev", "--data", str(data), "--out", str(out), "--concurrency", "2"])
        rows = read_predictions(out)
        self.assertEqual(len(rows), len(questions_of(records)))
        self.assertTrue(all(r["predictor"] == "jev" and r["template"] is None for r in rows))
        meta = json.loads(Path(f"{out}.meta.json").read_text())
        self.assertEqual((meta["requests"], meta["served_models"]), (3, ["jev-1.13.0"]))
        self.assertEqual(len(Path(f"{out}.requests.jsonl").read_text().splitlines()), 3)
        self.assertNotIn("ts-test", Path(f"{out}.meta.json").read_text() + Path(f"{out}.requests.jsonl").read_text())

    def test_refuses_sealed_or_hidden_data(self):
        from decision_eval.__main__ import main
        sealed = Path(tempfile.mkdtemp()) / "sealed" / "hidden.jsonl"
        write_jsonl(sealed, muse_records(n_states=1))
        with self.assertRaises(SystemExit):
            main(["predict-jev", "--data", str(sealed), "--out", str(sealed.with_name("x.jsonl"))])


if __name__ == "__main__":
    unittest.main()
