"""predict-base (B0 and, with --adapter, C) and predict-kev end to end on CPU with a tiny random Qwen3.5-architecture
hybrid (the 27B paths minus the size): padding invariance of the batched label log-probabilities, a LoRA adapter scored
through the same path, and kev's loader returning raw head logits (kev extra only)."""
import importlib.util
import math
import json
import tempfile
import unittest
from pathlib import Path

import torch

from decision_eval.__main__ import main
from decision_eval.base_scorer import BaseScorer
from decision_eval.predictions import read_predictions
from decision_eval.prompts import prompt_rows
from decision_eval.records import questions_of, write_jsonl
from tests.synthetic import kev_records, muse_records
from tests.tiny import cached_tokenizer_dir, tiny_model_dir, tokenizer


HAS_KEV = importlib.util.find_spec("kev") is not None


def make_adapter(directory):
    """A tiny LoRA saved the way TRL/PEFT save one from AutoModelForCausalLM, with non-zero B so it changes outputs."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    torch.manual_seed(2)
    lm = AutoModelForCausalLM.from_pretrained(tiny_model_dir())
    config = LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "v_proj", "in_proj_qkv", "gate_proj", "down_proj"])
    peft_model = get_peft_model(lm, config)
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)
    peft_model.save_pretrained(directory)
    return directory


def make_kev_run(directory, temperature):
    from kev.checkpoint import Meta, write_meta
    from kev.model import DecisionModel
    torch.manual_seed(1)
    model = DecisionModel(tiny_model_dir(), tokenizer(), "cpu", lora=2)
    for name, parameter in model.lm.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.02)   # a non-trivial adapter
    model.lm.save_pretrained(directory)
    write_meta(directory, Meta(base=tiny_model_dir(), head=model.head.state_dict(), lora=2, temperature=temperature))
    return directory


@unittest.skipUnless(cached_tokenizer_dir(), "Qwen3.8-27B tokenizer not in the local Hugging Face cache")
class ModelPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.records = muse_records(n_states=4, seed=9) + kev_records(3)
        cls.records[0]["state"] = cls.records[0]["state"] * 20          # uneven lengths -> real padding
        cls.data = cls.dir / "data.jsonl"
        write_jsonl(cls.data, cls.records)

    def test_batched_log_probs_match_single_rows_and_the_full_lm(self):
        tok = tokenizer()
        rows = prompt_rows(self.records, "direct", tok)
        scorer = BaseScorer.load(tiny_model_dir(), None, "cpu", tok, dtype=torch.float32)
        batched = scorer.label_logprobs(rows, batch=8)
        single = scorer.label_logprobs(rows, batch=1)
        for a, b in zip(batched, single):
            self.assertTrue(all(abs(x - y) < 1e-4 for x, y in zip(a, b)))
        with torch.no_grad():
            full = torch.log_softmax(scorer.lm(input_ids=torch.tensor([rows[0].ids])).logits[0, -1].float(), -1)
        self.assertTrue(all(abs(full[i].item() - x) < 1e-4 for i, x in zip(rows[0].label_ids, single[0])))
        self.assertTrue(all(x <= 0 for lp in batched for x in lp))

    def test_predict_base_cli(self):
        out = self.dir / "base.jsonl"
        main(["predict-base", "--model", tiny_model_dir(), "--revision", "main", "--data", str(self.data),
              "--template", "assistant", "--out", str(out), "--device", "cpu", "--dtype", "fp32", "--batch", "3"])
        rows = read_predictions(out)
        questions = questions_of(self.records)
        self.assertEqual([(r["id"], r["qid"]) for r in rows], [q.key for q in questions])
        self.assertTrue(all(r["predictor"] == "base" and r["template"] == "assistant" for r in rows))
        self.assertTrue(all(tuple(r["options"]) == q.keys for r, q in zip(rows, questions)))
        for r in rows:
            self.assertAlmostEqual(sum(r["probs"]), 1.0, places=6)
            self.assertLess(sum(math.exp(z) for z in r["logits"]), 1.0 + 1e-6)   # log-probs of a subset of the vocab
        self.assertTrue((self.dir / "base.jsonl.meta.json").exists())

    def test_predict_base_with_adapter_scores_c_through_the_same_path(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        adapter = make_adapter(self.dir / "adapter")
        common = ["--model", tiny_model_dir(), "--revision", "main", "--data", str(self.data), "--template", "evidence",
                  "--device", "cpu", "--dtype", "fp32", "--batch", "4"]
        main(["predict-base", *common, "--out", str(self.dir / "b0.jsonl")])
        main(["predict-base", *common, "--adapter", str(adapter), "--out", str(self.dir / "c.jsonl")])
        b0, c = read_predictions(self.dir / "b0.jsonl"), read_predictions(self.dir / "c.jsonl")
        self.assertTrue(all(r["predictor"] == "sft" and r["template"] == "evidence" for r in c))
        identity = lambda rows: [(r["id"], r["qid"], r["options"]) for r in rows]
        self.assertEqual(identity(b0), identity(c))
        self.assertGreater(max(abs(x - y) for r, s in zip(b0, c) for x, y in zip(r["logits"], s["logits"])), 1e-3)
        rows = prompt_rows(self.records, "evidence", tokenizer())
        reference = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tiny_model_dir()), adapter).eval()
        with torch.no_grad():
            full = torch.log_softmax(reference(input_ids=torch.tensor([rows[2].ids])).logits[0, -1].float(), -1)
        self.assertTrue(all(abs(full[i].item() - x) < 1e-4 for i, x in zip(rows[2].label_ids, c[2]["logits"])))
        meta = json.loads((self.dir / "c.jsonl.meta.json").read_text())
        self.assertIn("adapter_model.safetensors", meta["adapter_files_sha256"])

    @unittest.skipUnless(HAS_KEV, "the kev extra is not installed")
    def test_predict_kev_cli_writes_raw_head_logits(self):
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor
        from kev.suite import SERVING_CONTEXT
        from decision_eval.records import kev_request
        run = make_kev_run(self.dir / "run", temperature=2.0)
        out = self.dir / "kev.jsonl"
        main(["predict-kev", "--run", str(run), "--data", str(self.data), "--out", str(out), "--device", "cpu"])
        rows = {(r["id"], r["qid"]): r for r in read_predictions(out)}
        self.assertEqual(len(rows), len(questions_of(self.records)))
        shipped = LocalPredictor(str(run), "cpu", LoadOptions(), context=SERVING_CONTEXT)   # at head.pt's T = 2.0
        self.assertEqual(shipped.temperature, 2.0)
        record = self.records[1]
        served = shipped(kev_request(record))["probabilities"]
        for qid in record["questions"]:
            row = rows[(record["id"], qid)]
            self.assertEqual(row["predictor"], "kev")
            self.assertIsNone(row["template"])
            z = torch.tensor(row["logits"]) / 2.0
            expected = torch.softmax(z, -1).tolist()
            self.assertTrue(all(abs(a - served[qid][k]) < 1e-5 for a, k in zip(expected, row["options"])))
        # the generic reference-loader signature gives the same predictions for the same run
        ref_out = self.dir / "kev-ref.jsonl"
        main(["predict-kev-ref", "--model", str(run), "--revision", "0" * 40, "--template", "direct",
              "--data", str(self.data), "--out", str(ref_out), "--device", "cpu"])
        self.assertEqual([r["logits"] for r in read_predictions(ref_out)], [r["logits"] for r in read_predictions(out)])
        meta = json.loads(Path(f"{ref_out}.meta.json").read_text())
        self.assertEqual((meta["template"], meta["template_requested"]), (None, "direct"))

    @unittest.skipUnless(HAS_KEV, "the kev extra is not installed")
    def test_predict_kev_ref_resolves_a_hub_run_at_its_revision(self):
        from unittest import mock
        seen = {}

        def fake_load(run, device, dtype):
            seen["run"] = run
            raise RuntimeError("stop before any download")
        with mock.patch("decision_eval.kev_scorer.load_predictor", fake_load), self.assertRaises(RuntimeError):
            main(["predict-kev-ref", "--model", "jaredpalmer/kev-27b",
                  "--revision", "01b81998019be550f0ae858727df49bac9511195", "--template", "direct",
                  "--data", str(self.data), "--out", str(self.dir / "never.jsonl"), "--device", "cpu"])
        self.assertEqual(seen["run"], "jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195")


if __name__ == "__main__":
    unittest.main()
