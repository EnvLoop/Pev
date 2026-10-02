"""to-sft (alias sft-rows): SFT prompt-completion rows for C, byte-identical to predict-base's prompts."""
import argparse
import sys

from ..prompts import MODEL, REVISION, TEMPLATES, pinned_tokenizer, template_sha256
from ..records import read_jsonl, sha256_file, write_jsonl
from ..sft import sft_rows
from .common import write_json


def to_sft_main(argv=None):
    ap = argparse.ArgumentParser(prog="to-sft", description="one prompt-completion row per question (ties skipped)")
    ap.add_argument("--data", required=True, help="TRAIN (or VAL) records")
    ap.add_argument("--template", required=True, choices=TEMPLATES, help="the template B0 selected on VAL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL, help="tokenizer / chat template source")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--max-length", type=int, default=None, help="refuse rows longer than this (prompt + label)")
    a = ap.parse_args(argv)
    records = read_jsonl(a.data)
    rows, skipped = sft_rows(records, a.template, pinned_tokenizer(a.model, a.revision), a.max_length)
    if not rows:
        raise SystemExit("no trainable questions")
    write_jsonl(a.out, rows)
    lengths = [row["meta"]["prompt_tokens"] for row in rows]
    meta = {"sha256": sha256_file(a.out), "rows": len(rows), "ambiguous_excluded": skipped,
            "records": len(records), "questions": len(rows) + skipped, "template": a.template,
            "template_sha256": template_sha256(a.template), "model": a.model, "revision": a.revision,
            "data_sha256": sha256_file(a.data), "prompt_tokens_total": sum(lengths), "max_prompt_tokens": max(lengths)}
    write_json(f"{a.out}.meta.json", meta)
    print(f"wrote {len(rows)} rows ({skipped} ambiguous skipped) to {a.out}", file=sys.stderr)
