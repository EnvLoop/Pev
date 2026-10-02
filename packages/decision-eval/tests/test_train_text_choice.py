"""training/train_text_choice.py on CPU with the tiny random Qwen3.5-architecture model: row tokenization and its
drift checks, the label-token loss against the model's own full-vocabulary masked loss, gradient-accumulation
normalisation, the recipe's trainer arguments, and a two-step run whose adapter predict-base then scores as C."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from decision_eval.__main__ import main
from decision_eval.predictions import read_predictions
from decision_eval.records import write_jsonl
from decision_eval.sft import sft_rows
from tests.synthetic import muse_records
from tests.tiny import cached_tokenizer_dir, tiny_model_dir, tokenizer

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "training" / "train_text_choice.py"
EVALUATE = ROOT / "training" / "evaluate_adapter.py"
CONFIG = ROOT / "training" / "configs" / "qwen3.8-27b-r2-mix-s20260930.json"


def load_trainer(path=SCRIPT):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(cached_tokenizer_dir(), "Qwen3.8-27B tokenizer not in the local Hugging Face cache")
class TrainTextChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = load_trainer()
        cls.dir = Path(tempfile.mkdtemp())
        cls.records = muse_records(n_states=6, seed=4, per_state=2)
        cls.records[0]["state"] *= 15  # uneven lengths -> real padding in a batch
        rows, _ = sft_rows(cls.records, "direct", tokenizer())
        cls.rows = rows
        cls.rows_path = cls.dir / "rows.jsonl"
        cls.rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_rows_mask_the_prompt_and_refuse_drift(self):
        data = self.t.tokenize(tokenizer(), self.rows, 4096)
        for row, item in zip(self.rows, data):
            n = row["meta"]["prompt_tokens"]
            self.assertEqual(len(item["input_ids"]), n + 1)
            self.assertEqual(item["labels"][:n], [-100] * n)
            self.assertEqual(item["labels"][n:], [row["meta"]["label_token_id"]])
        drifted = [dict(self.rows[0], meta={**self.rows[0]["meta"], "prompt_tokens": 3})]
        with self.assertRaises(SystemExit):
            self.t.tokenize(tokenizer(), drifted, 4096)
        with self.assertRaises(SystemExit):
            self.t.tokenize(tokenizer(), self.rows[:1], 8)
        with self.assertRaises(SystemExit):
            self.t.read_rows(self.rows_path, "assistant")

    def test_label_token_loss_equals_the_full_masked_loss(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(tiny_model_dir()).eval()
        data = self.t.tokenize(tokenizer(), self.rows[:3], 4096)
        batch = self.t.Collator(tokenizer().pad_token_id)(data)
        self.assertEqual(batch["input_ids"].shape[0], 3)
        with torch.no_grad():
            ours, _ = self.t.label_token_loss(model, batch)
            reference = model(**batch).loss
            scaled, _ = self.t.label_token_loss(model, batch, num_items_in_batch=6)
        self.assertAlmostEqual(ours.item(), reference.item(), places=5)
        self.assertAlmostEqual(scaled.item(), ours.item() * 3 / 6, places=5)

    def test_trainer_arguments_follow_the_recipe(self):
        config = self.t.load_config(CONFIG, {})
        args = self.t.training_arguments(config, self.dir / "x", bf16=True, gradient_checkpointing=True)
        self.assertEqual((args["learning_rate"], args["lr_scheduler_type"], args["warmup_steps"], args["max_steps"]),
                         (5e-05, "cosine", 24, 776))
        self.assertEqual((args["per_device_train_batch_size"], args["gradient_accumulation_steps"]), (1, 8))
        self.assertEqual((args["optim"], args["bf16"], args["seed"]), ("adamw_torch", True, 20260930))
        lora = self.t.lora_config(config)
        self.assertEqual((lora.r, lora.lora_alpha, lora.lora_dropout, lora.bias), (16, 32, 0.05, "none"))

    def test_two_step_cpu_run_produces_an_adapter_predict_base_scores(self):
        config = {**self.t.load_config(CONFIG, {"model": tiny_model_dir(), "revision": "main", "max_steps": 2}),
                  "gradient_accumulation_steps": 2, "save_steps": 100, "learning_rate": 1e-2, "warmup_steps": 0}
        out = self.dir / "run"
        manifest = self.t.train(config, self.rows_path, out, device="cpu")
        self.assertEqual(manifest["global_step"], 2)
        weights = out / "adapter" / "adapter_model.safetensors"
        self.assertTrue(weights.exists())
        from safetensors import safe_open
        with safe_open(weights, framework="pt") as saved:
            keys = list(saved.keys())
            self.assertTrue(any("linear_attn.in_proj_qkv" in key for key in keys))
            self.assertTrue(any("self_attn.q_proj" in key for key in keys))
            self.assertTrue(any("mlp.down_proj" in key for key in keys))
            self.assertTrue(any(saved.get_tensor(k).abs().sum() > 0 for k in keys if "lora_B" in k))
        data = self.dir / "records.jsonl"
        write_jsonl(data, self.records)
        common = ["--model", tiny_model_dir(), "--revision", "main", "--data", str(data), "--template", "direct",
                  "--device", "cpu", "--dtype", "fp32"]
        main(["predict-base", *common, "--adapter", str(out / "adapter"), "--out", str(self.dir / "c.jsonl")])
        self.assertTrue(all(row["predictor"] == "sft" for row in read_predictions(self.dir / "c.jsonl")))
        with self.assertRaises(SystemExit):  # never overwrite a run
            self.t.train(config, self.rows_path, out, device="cpu")
        evaluate = load_trainer(EVALUATE)
        argv = ["--adapter", str(out / "adapter"), "--val", str(data), "--data", str(data), "--out",
                str(self.dir / "eval"), "--model", tiny_model_dir(), "--revision", "main", "--device", "cpu",
                "--dtype", "fp32"]
        evaluate.main(argv)
        report = json.loads((self.dir / "eval" / "report.json").read_text())
        self.assertIn("macro", report)
        self.assertTrue((self.dir / "eval" / "reference.sft.json").exists())
        before = {p.name: p.stat().st_mtime_ns for p in (self.dir / "eval").iterdir()}
        evaluate.main(argv)  # resumes: every step's output exists, nothing reruns
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in (self.dir / "eval").iterdir()})


if __name__ == "__main__":
    unittest.main()
