"""Standalone text-choice LoRA SFT: the open re-implementation of the trainer that produced the released adapter.

Input rows are `to-sft` output (packages/decision-eval): `{"prompt", "completion", "meta": {..., "label_token_id",
"prompt_tokens", "template"}}`, where `prompt` is exactly the chat-templated text `predict-base` scores and
`completion` is the single option-label token. Each row is tokenized here, not by a chat template:

    input_ids = tokenizer(prompt, add_special_tokens=False) + [label_token]
    labels    = [-100] * len(prompt ids)                     + [label_token]

so the loss is the cross-entropy of the one label token given the prompt (no EOS is appended, the prompt is never
truncated; a row longer than `max_length` is refused). Rows whose tokens differ from the evaluator's (`prompt_tokens`,
`label_token_id`) are refused too, so training and scoring can never drift apart.

Model: `AutoModelForCausalLM` (the text model of the Qwen3.5/3.8 checkpoint, the class predict-base loads the adapter
onto) in bf16, LoRA (bias none, adapters in fp32, PEFT's default) on the target-module regex of the config, gradient
checkpointing (non-reentrant) with input gradients enabled, AdamW (torch), linear warmup + cosine decay, bf16
autocast, batch 1 x gradient accumulation 8, loss = sum of label-token NLL / label tokens in the accumulated batch.
Only the label positions are projected through the LM head (`logits_to_keep`), the same math as a full-vocabulary
loss on every position with the prompt masked, at a fraction of the memory.

    uv run --project packages/decision-eval python training/train_text_choice.py \
        --config training/configs/qwen3.8-27b-r2-mix-s20260930.json \
        --train work/release/train.r2-mix.sft.direct.jsonl --out runs/r2-mix

Bitwise-identical weights are not expected (GPU kernels, LoRA initialisation RNG); the recipe, data, loss and
optimisation are the same as the run reported in docs/RESULTS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

CONFIG_KEYS = {"model", "revision", "template", "lora", "learning_rate", "lr_scheduler", "warmup_steps", "max_steps",
               "save_steps", "per_device_batch_size", "gradient_accumulation_steps", "max_length", "seed"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, overrides: dict) -> dict:
    config = json.loads(Path(path).read_text())
    missing = CONFIG_KEYS - set(config)
    if missing:
        raise SystemExit(f"config {path} lacks {sorted(missing)}")
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def read_rows(path: Path, template: str) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        meta = row.get("meta") or {}
        if (not isinstance(row.get("prompt"), str) or not row["prompt"] or not isinstance(row.get("completion"), str)
                or meta.get("template") != template or type(meta.get("label_token_id")) is not int
                or type(meta.get("prompt_tokens")) is not int):
            raise SystemExit(f"{path}:{number} is not a to-sft row for template {template!r}")
        rows.append(row)
    return rows


def tokenize(tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    """input_ids = prompt ids + label token, labels mask the prompt; refuses any drift from the evaluator."""
    out = []
    for number, row in enumerate(rows, 1):
        ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        label = tokenizer.encode(row["completion"], add_special_tokens=False)
        meta = row["meta"]
        if len(ids) != meta["prompt_tokens"] or label != [meta["label_token_id"]]:
            raise SystemExit(f"row {number}: tokens differ from the evaluator's rendering (wrong tokenizer?)")
        if tokenizer(row["prompt"] + row["completion"], add_special_tokens=False).input_ids != ids + label:
            raise SystemExit(f"row {number}: the label does not tokenize alone after the prompt")
        if len(ids) + 1 > max_length:
            raise SystemExit(f"row {number}: {len(ids) + 1} tokens > max_length {max_length} (never truncated)")
        out.append({"input_ids": ids + label, "labels": [-100] * len(ids) + label})
    return out


class Collator:
    """Right padding; padded positions are masked out of attention and loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, examples: list[dict]) -> dict:
        import torch
        width = max(len(example["input_ids"]) for example in examples)
        ids, labels, mask = [], [], []
        for example in examples:
            pad = width - len(example["input_ids"])
            ids.append(example["input_ids"] + [self.pad_id] * pad)
            labels.append(example["labels"] + [-100] * pad)
            mask.append([1] * len(example["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def label_token_loss(model, inputs: dict, num_items_in_batch=None):
    """Sum of the label tokens' NLL over the micro-batch, divided by the label tokens of the whole accumulated batch
    (or of this micro-batch when that count is unknown). Logits are computed at the label positions only."""
    import torch
    import torch.nn.functional as F
    labels = inputs["labels"]
    shifted = labels[:, 1:]  # position i predicts token i + 1
    positions = (shifted != -100).any(dim=0).nonzero().squeeze(-1)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=positions)
    logits = outputs.logits.float()
    targets = shifted[:, positions]
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
                            reduction="sum")
    count = num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum()
    return total / torch.as_tensor(count, device=total.device).clamp(min=1), outputs


def make_trainer_class():
    from transformers import Trainer

    class LabelTokenTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True  # our loss already divides by the accumulated label count

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = label_token_loss(model, inputs, num_items_in_batch)
            return (loss, outputs) if return_outputs else loss

    return LabelTokenTrainer


def training_arguments(config: dict, out: Path, bf16: bool, gradient_checkpointing: bool) -> dict:
    return {"output_dir": str(out), "per_device_train_batch_size": config["per_device_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False} if gradient_checkpointing else None,
            "learning_rate": config["learning_rate"], "lr_scheduler_type": config["lr_scheduler"],
            "warmup_steps": config["warmup_steps"], "max_steps": config["max_steps"],
            "save_steps": config["save_steps"], "save_strategy": "steps", "logging_steps": 1, "report_to": "none",
            "seed": config["seed"], "bf16": bf16, "optim": "adamw_torch", "remove_unused_columns": False,
            "include_num_input_tokens_seen": "non_padding", "dataloader_drop_last": False, "use_cpu": not bf16}


def lora_config(config: dict):
    from peft import LoraConfig
    lora = config["lora"]
    return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                      target_modules=lora["target_modules"], task_type="CAUSAL_LM")


def train(config: dict, train_path: Path, out: Path, validation_path: Path | None = None, device: str = "cuda",
          resume: Path | None = None) -> dict:
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    if out.exists() and resume is None and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; pass --resume CHECKPOINT or choose a new directory")
    cuda = device == "cuda"
    if cuda and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise SystemExit("the registered recipe needs a bf16-capable CUDA GPU (use --device cpu only for tests)")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    data = tokenize(tokenizer, read_rows(train_path, config["template"]), config["max_length"])
    if validation_path:  # checked like the training rows (the run never evaluates on it)
        tokenize(tokenizer, read_rows(validation_path, config["template"]), config["max_length"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], dtype=torch.bfloat16 if cuda else torch.float32,
        device_map={"": 0} if cuda else None)
    model.config.use_cache = False
    set_seed(config["seed"])
    model = get_peft_model(model, lora_config(config))
    model.enable_input_require_grads()
    args = TrainingArguments(**training_arguments(config, out, bf16=cuda, gradient_checkpointing=True))
    trainer = make_trainer_class()(model=model, args=args, train_dataset=data,
                                   data_collator=Collator(tokenizer.pad_token_id))
    started = time.time()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if not math.isfinite(float(result.training_loss)):
        raise SystemExit("training loss is not finite")
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    manifest = {"config": config, "train_sha256": sha256_file(train_path), "train_rows": len(data),
                "validation_sha256": sha256_file(validation_path) if validation_path else None,
                "global_step": result.global_step, "training_loss": float(result.training_loss),
                "seconds": round(time.time() - started, 1),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "adapter_files_sha256": {p.name: sha256_file(p) for p in sorted(adapter.iterdir()) if p.is_file()}}
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True, help="to-sft rows of the training records")
    ap.add_argument("--validation", type=Path, help="to-sft rows of VAL (token-checked only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, help="a checkpoint-N directory of this run")
    ap.add_argument("--model", help="override the config's model (e.g. a local snapshot)")
    ap.add_argument("--revision", help="override the config's revision")
    ap.add_argument("--max-steps", type=int, help="override (tests only; the recipe is the config)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cpu: fp32 smoke tests only")
    a = ap.parse_args(argv)
    config = load_config(a.config, {"model": a.model, "revision": a.revision, "max_steps": a.max_steps})
    print(json.dumps(train(config, a.train, a.out, a.validation, a.device, a.resume), sort_keys=True))


if __name__ == "__main__":
    main()
