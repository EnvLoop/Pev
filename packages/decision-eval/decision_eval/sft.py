"""SFT rows for C (docs/PREREGISTRATION.md amendment 2), byte-identical to what predict-base scores.

Each question becomes one prompt-completion row: the prompt is render_prompt's chat-templated text (the exact string
predict-base tokenizes), the completion is the label text of the question's label (one token; the unique argmax of a
soft label). Questions whose soft label has no unique argmax are not trained on. Standard prompt-completion rows: TRL's
SFTTrainer computes the loss on the completion only, and appends the tokenizer's EOS to a completion that does not end
with it, so a trainer that wants the loss on the label token alone must turn that off (or build token rows from
`label_token`).
"""
from .prompts import pinned_tokenizer, render_prompt, template_sha256, verify_labels
from .records import question_of


def label_token(record, qid, template, tok=None):
    """-> (label text, token id) the question is trained towards; ValueError for a question without a unique label.
    `template` does not change the label, it is checked so a caller cannot pair a label with an unknown template."""
    tok = tok if tok is not None else pinned_tokenizer()
    question = question_of(record, qid)
    if question.ambiguous:
        raise ValueError("a soft label without a unique argmax has no training label")
    _, labels = render_prompt(record, qid, template, tok)
    ids = verify_labels(tok, labels)
    return labels[question.label], ids[question.label]


def sft_rows(records, template, tok, max_length=None):
    """-> (rows, number of ambiguous questions skipped). Rows are in data order."""
    rows, skipped, verified = [], 0, {}
    digest = template_sha256(template)
    for record in records:
        for qid in record["questions"]:
            question = question_of(record, qid)
            if question.ambiguous:
                skipped += 1
                continue
            prompt, labels = render_prompt(record, qid, template, tok)
            if tuple(labels) not in verified:
                verified[tuple(labels)] = verify_labels(tok, labels)
            prompt_tokens = len(tok(prompt, add_special_tokens=False).input_ids)
            if max_length is not None and prompt_tokens + 1 > max_length:
                raise ValueError(f"a row needs {prompt_tokens + 1} tokens, over --max-length {max_length}")
            rows.append({"prompt": prompt, "completion": labels[question.label],
                         "meta": {"id": question.record_id, "qid": question.qid, "family": question.family,
                                  "state_id": question.state_id, "template": template, "template_sha256": digest,
                                  "label_key": question.keys[question.label],
                                  "label_token_id": verified[tuple(labels)][question.label],
                                  "prompt_tokens": prompt_tokens}})
    return rows, skipped
