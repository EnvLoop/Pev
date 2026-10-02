"""predict-base (B0, or C with --adapter) and predict-kev: one prediction line per question (decision_eval.predictions).

Each also writes `<out>.meta.json`: what produced the file (model, revision or run, template and its SHA-256, the
data's SHA-256), wall time and, on CUDA, peak GPU memory.
"""
import argparse
import sys
import time

from ..predictions import prediction_row, write_predictions
from ..prompts import TEMPLATES, template_sha256
from ..records import read_jsonl, sha256_file
from .common import write_json


def _peak_memory(device):
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return None


def _progress(label):
    def report(done, total):
        if done == total or done % 50 == 0:
            print(f"{label}: {done}/{total}", file=sys.stderr, flush=True)
    return report


def finish(out, rows, meta, started, device=None):
    write_predictions(out, rows)
    write_json(f"{out}.meta.json", {**meta, "questions": len(rows), "seconds": round(time.time() - started, 1),
                                    "peak_gpu_memory_gib": _peak_memory(device) if device else None})
    print(f"wrote {len(rows)} predictions to {out}", file=sys.stderr)


def _adapter_meta(adapter):
    if adapter is None:
        return {"adapter": None}
    from pathlib import Path
    files = sorted(p for p in Path(adapter).iterdir() if p.name.startswith("adapter_"))
    return {"adapter": str(adapter), "adapter_files_sha256": {p.name: sha256_file(p) for p in files}}


def predict_base_main(argv=None):
    ap = argparse.ArgumentParser(prog="predict-base",
                                 description="B0: zero-shot label log-probabilities of a base model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--template", required=True, choices=TEMPLATES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-batch-tokens", type=int, default=32_768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"), help="bf16 is the registered path")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter directory: scores C (predictor 'sft')")
    a = ap.parse_args(argv)
    import torch
    from ..base_scorer import BaseScorer, load_tokenizer
    from ..prompts import prompt_rows
    started = time.time()
    records = read_jsonl(a.data)
    tok = load_tokenizer(a.model, a.revision)
    rows = prompt_rows(records, a.template, tok)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    scorer = BaseScorer.load(a.model, a.revision, a.device, tok, dtype=dtype, adapter=a.adapter)
    predictor = "sft" if a.adapter else "base"
    log_probs = scorer.label_logprobs(rows, batch=a.batch, max_tokens=a.max_batch_tokens,
                                      progress=_progress("predict-base"))
    out = [prediction_row(row.key, predictor, a.template, row.options, lp) for row, lp in zip(rows, log_probs)]
    meta = {"predictor": predictor, "model": a.model, **_adapter_meta(a.adapter), "revision": a.revision,
            "template": a.template, "template_sha256": template_sha256(a.template), "dtype": a.dtype,
            "data_sha256": sha256_file(a.data),
            "prompt_tokens": sum(len(r.ids) for r in rows), "max_prompt_tokens": max(len(r.ids) for r in rows)}
    finish(a.out, out, meta, started, a.device)


def _predict_kev(run, data, out, device, dtype, extra_meta):
    try:
        from ..kev_scorer import load_predictor, record_logits
        import kev  # noqa: F401
    except ImportError:
        raise SystemExit("the Kev reference needs the kev extra: uv sync --extra kev") from None
    started = time.time()
    records = read_jsonl(data)
    predictor = load_predictor(run, device, dtype)
    rows, report = [], _progress("predict-kev")
    for i, record in enumerate(records, 1):
        rows += [prediction_row(key, "kev", None, keys, z) for key, keys, z in record_logits(predictor, record)]
        report(i, len(records))
    meta = {"predictor": "kev", "run": str(run), "base": predictor.checkpoint.meta.base,
            "base_revision": predictor.checkpoint.meta.base_revision,
            "head_sha256": sha256_file(predictor.checkpoint.file("head.pt")),
            "shipped_temperature_ignored": predictor.checkpoint.meta.temperature, "data_sha256": sha256_file(data),
            **extra_meta}
    finish(out, rows, meta, started, device)


def _kev_device_args(ap):
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, choices=("bf16", "fp32"), help="default: the dtype the run was trained in")


def predict_kev_main(argv=None):
    ap = argparse.ArgumentParser(prog="predict-kev", description="Kev reference: a trained Kev run's raw head logits")
    ap.add_argument("--run", required=True, help="run directory (adapter + head.pt) or kev Hub id[@revision]")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    _kev_device_args(ap)
    a = ap.parse_args(argv)
    _predict_kev(a.run, a.data, a.out, a.device, a.dtype, {})


def predict_kev_ref_main(argv=None):
    """The generic reference-loader signature (predict-base's arguments): --model/--revision name a released Kev
    run on the Hub (e.g. jaredpalmer/kev-27b at a 40-hex revision); --template is accepted and ignored (Kev has its own
    encoder), recorded as null. A --model that is a local run directory is used as is (tests)."""
    ap = argparse.ArgumentParser(prog="predict-kev-ref", description="released Kev run, predict-base's arguments")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--template", default=None, help="ignored: Kev encodes records itself")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    _kev_device_args(ap)
    a = ap.parse_args(argv)
    from pathlib import Path
    run = a.model if Path(a.model).is_dir() else f"{a.model}@{a.revision}"
    _predict_kev(run, a.data, a.out, a.device, a.dtype,
                 {"model": a.model, "revision": a.revision, "template": None, "template_requested": a.template})
