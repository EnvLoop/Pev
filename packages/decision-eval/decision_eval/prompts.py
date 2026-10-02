"""Decision prompts shared by B0, C (base + SFT LoRA) and the SFT training rows: three frozen templates, single-token
option labels. render_prompt is the one place a question becomes prompt text; predict-base and to-sft both call it.

Each question becomes one chat turn (the template filled with the state, the instructions and the labelled options),
rendered with the model's chat template with thinking disabled (enable_thinking=False), so the assistant turn opens
with an empty think block and the next token is the answer. Labels: A, B, C, ... for choice (two-letter single-token
labels AA, AB, ... when a question has more than 26 options); no / yes for noul (in kev's key order false, true);
0, 1, 2, ... for score levels (at most 10). Every label must be one token that the tokenizer also produces after the
generation prompt; verify_labels checks that.
"""
import functools
import hashlib
import re
import string
from dataclasses import dataclass
from pathlib import Path

from .conventions import option_text, question_keys, render
from .records import question_of, record_id

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TEMPLATES = ("direct", "assistant", "evidence")
TEMPLATE_DIR = Path(__file__).parent / "templates"
LETTERS = string.ascii_uppercase
MAX_SCORE_LEVELS = 10
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")   # kev.model._SPECIAL_RE


def template_text(name):
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; one of {', '.join(TEMPLATES)}")
    return (TEMPLATE_DIR / f"{name}.txt").read_text(encoding="utf-8")


def template_sha256(name):
    return hashlib.sha256(template_text(name).encode("utf-8")).hexdigest()


def single_token(tok, text):
    return len(tok.encode(text, add_special_tokens=False)) == 1


def option_labels(qtype, n, tok=None):
    if qtype == "noul":
        return ["no", "yes"]
    if qtype == "score":
        if n > MAX_SCORE_LEVELS:
            raise ValueError(f"score questions with more than {MAX_SCORE_LEVELS} levels have no single-digit labels")
        return [str(i) for i in range(n)]
    if n <= len(LETTERS):
        return list(LETTERS[:n])
    pairs = [a + b for a in LETTERS for b in LETTERS]
    usable = [p for p in pairs if tok is None or single_token(tok, p)]
    if n > len(usable):
        raise ValueError(f"no {n} single-token option labels")
    return usable[:n]


def options_block(q, labels):
    if q["type"] == "choice":
        pairs = zip(labels, q["criteria"].items())
        return "\n".join(f"{label}. {option_text(name, desc)}" for label, (name, desc) in pairs)
    if q["type"] == "noul":
        criteria = q.get("criteria") or {}
        lines = []
        for label, key in (("yes", "true"), ("no", "false")):
            desc = render(criteria.get(key))
            lines.append(f"{label}: {desc}" if desc else label)
        return "\n".join(lines)
    return "\n".join(f"{label}: {render(level)}" for label, level in zip(labels, q["criteria"]))


def answer_phrase(qtype, labels):
    if qtype == "noul":
        return "yes or no"
    kind = "option" if qtype == "choice" else "level"
    return f"the label of exactly one {kind} ({', '.join(labels)})"


def defang(text):
    """Caller text can never produce control tokens: `<|name|>` becomes `<¦name¦>` (kev.model.user_tokens' rule)."""
    return SPECIAL.sub(r"<¦\1¦>", text)


def user_prompt(record, qid, template, labels):
    q = record["questions"][qid]
    text = template_text(template).format(state=render(record["state"]), instructions=render(q.get("instructions")),
                                          options=options_block(q, labels), answer=answer_phrase(q["type"], labels))
    return defang(text)


def messages(text):
    return [{"role": "user", "content": text}]


def chat_prompt(tok, text):
    return tok.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def question_labels(record, qid, tok):
    q = record["questions"][qid]
    return option_labels(q["type"], len(question_keys(q["type"], q.get("criteria"))), tok)


@functools.cache
def pinned_tokenizer(model=MODEL, revision=REVISION):
    """The pre-registered base's tokenizer and chat template (a few MB; never the weights)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, revision=revision)


def render_messages(record, qid, template, tok=None):
    """-> (chat messages, option labels): the single user turn before the chat template is applied. `tok` defaults to
    the pinned Qwen3.8-27B tokenizer (labels past 26 choices depend on its vocabulary)."""
    tok = tok if tok is not None else pinned_tokenizer()
    labels = question_labels(record, qid, tok)
    return messages(user_prompt(record, qid, template, labels)), labels


def render_prompt(record, qid, template, tok=None):
    """-> (prompt text, option labels): the chat-templated text (thinking disabled) whose next token is scored by
    predict-base and trained as the completion by SFT. Labels are in option-key order."""
    tok = tok if tok is not None else pinned_tokenizer()
    turn, labels = render_messages(record, qid, template, tok)
    return chat_prompt(tok, turn[0]["content"]), labels


def verify_labels(tok, labels):
    """-> label token ids. Each label is one token, the labels are distinct, and the token is what the tokenizer yields
    for the label written right after the generation prompt (so its log-prob is the answer's first-token log-prob)."""
    ids = []
    prompt = chat_prompt(tok, "x")
    head = tok(prompt, add_special_tokens=False).input_ids
    for label in labels:
        own = tok.encode(label, add_special_tokens=False)
        if len(own) != 1:
            raise ValueError(f"option label {label!r} is {len(own)} tokens for this tokenizer")
        if tok(prompt + label, add_special_tokens=False).input_ids != head + own:
            raise ValueError(f"option label {label!r} does not tokenize alone after the generation prompt")
        ids.append(own[0])
    if len(set(ids)) != len(ids):
        raise ValueError("option labels share a token")
    return ids


@dataclass(frozen=True)
class PromptRow:
    key: tuple[str, str]
    options: tuple[str, ...]
    labels: tuple[str, ...]
    ids: tuple[int, ...]
    label_ids: tuple[int, ...]


def prompt_rows(records, template, tok):
    rows, verified = [], {}
    for record in records:
        for qid, q in record["questions"].items():
            question_of(record, qid)   # validates the label layout before any model time is spent
            keys = question_keys(q["type"], q.get("criteria"))
            text, labels = render_prompt(record, qid, template, tok)
            labels = tuple(labels)
            if labels not in verified:
                verified[labels] = tuple(verify_labels(tok, labels))
            ids = tuple(tok(text, add_special_tokens=False).input_ids)
            rows.append(PromptRow((record_id(record), str(qid)), tuple(keys), labels, ids, verified[labels]))
    return rows
