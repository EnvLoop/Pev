"""B0 vs C decision evaluation (docs/PREREGISTRATION.md). The importable API for the SFT trainer:

    render_prompt(record, qid, template, tok=None) -> (prompt text, option labels)
    render_messages(record, qid, template, tok=None) -> (chat messages, option labels)
    label_token(record, qid, template, tok=None) -> (label text, token id)
    sft_rows(records, template, tok, max_length=None) -> (rows, ambiguous questions skipped)

`tok` defaults to the pinned Qwen3.8-27B tokenizer. None of this imports torch or kev.
"""
from .prompts import render_messages, render_prompt
from .sft import label_token, sft_rows

__all__ = ["label_token", "render_messages", "render_prompt", "sft_rows"]
