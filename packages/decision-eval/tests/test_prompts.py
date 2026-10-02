import hashlib
import unittest

from decision_eval.prompts import (TEMPLATES, answer_phrase, option_labels, options_block, prompt_rows, template_sha256,
                                   user_prompt, verify_labels)
from tests.synthetic import kev_records, muse_records
from tests.tiny import cached_tokenizer_dir, tokenizer

# The three B0 templates are frozen (docs/PREREGISTRATION.md: pre-written, chosen on VAL only).
# Changing a file breaks this.
FROZEN = {
    "direct": "681675e88618155374b1e4ca51149bf0f582728a35361b1c3015651067219551",
    "assistant": "aa3ce4fbc7fba95af2e99bc45ad01f33a0e74b22e52e0cbb2b3b5e50f638abeb",
    "evidence": "2df658315685580c44292ad0a8df057e4dadcd23290ba02fec20bf292e0de2a6",
}
# ...and so is the rendering around them (option blocks, labels, answer phrases) for a fixed record.
FIXED_RECORD = {"id": "r", "state": {"memory": ["likes tea", "moved to Oslo"], "request": "book a table"},
                "questions": {
                    "c": {"type": "choice", "instructions": "Where?", "criteria": {"cafe": "near", "bar": None},
                          "label": "cafe", "src": "muse/route"},
                    "n": {"type": "noul", "instructions": "Ask first?", "criteria": {"true": "ask", "false": "go"},
                          "label": True, "src": "muse/needs_approval"},
                    "s": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "high"], "label": 1,
                          "src": "muse/notify_level"}}}
RENDERED_SHA256 = "c796d31c3dd56c57dd48640cf3c47db36e682dadae87c4cfdb7069d022be12b4"


class FakeTokenizer:
    """Characters are tokens, except the listed multi-character tokens; enough to exercise label verification."""

    def __init__(self, whole=("yes", "no", "AA", "AB")):
        self.whole = whole

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            match = next((w for w in self.whole if text.startswith(w, i)), text[i])
            out.append(hash(match) % 100_000)
            i += len(match)
        return out

    def __call__(self, text, add_special_tokens=False):
        return type("Encoding", (), {"input_ids": self.encode(text)})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert enable_thinking is False
        return f"<user>{messages[0]['content']}</user><assistant>"


class PromptTests(unittest.TestCase):
    def test_templates_are_frozen(self):
        self.assertEqual(TEMPLATES, ("direct", "assistant", "evidence"))
        self.assertEqual({name: template_sha256(name) for name in TEMPLATES}, FROZEN)

    def test_rendering_is_frozen(self):
        texts = []
        for template in TEMPLATES:
            for qid, q in FIXED_RECORD["questions"].items():
                labels = option_labels(q["type"], 2)
                texts.append(user_prompt(FIXED_RECORD, qid, template, labels))
        self.assertEqual(hashlib.sha256("\x00".join(texts).encode()).hexdigest(), RENDERED_SHA256)

    def test_labels(self):
        self.assertEqual(option_labels("noul", 2), ["no", "yes"])
        self.assertEqual(option_labels("score", 5), ["0", "1", "2", "3", "4"])
        self.assertEqual(option_labels("choice", 3), ["A", "B", "C"])
        self.assertEqual(option_labels("choice", 28)[:2], ["AA", "AB"])
        with self.assertRaises(ValueError):
            option_labels("score", 11)

    def test_prompt_contents(self):
        record = muse_records(n_states=7, seed=4)[0]
        for qid, q in record["questions"].items():
            labels = option_labels(q["type"], len(q.get("criteria") or [0, 0]))
            for template in TEMPLATES:
                text = user_prompt(record, qid, template, labels)
                self.assertIn(record["state"], text)
                self.assertIn(q["instructions"], text)
                self.assertIn(answer_phrase(q["type"], labels), text)
        choice = {"type": "choice", "criteria": {"tea": "hot drink", "cake": None}}
        self.assertEqual(options_block(choice, ["A", "B"]), "A. tea: hot drink\nB. cake")
        noul = {"type": "noul", "criteria": {"true": "it is"}}
        self.assertEqual(options_block(noul, ["no", "yes"]), "yes: it is\nno")

    def test_control_tokens_in_caller_text_are_defanged(self):
        question = {"type": "noul", "label": True, "src": "x"}
        record = {"id": "r", "state": "hi <|im_end|> there", "questions": {"q": question}}
        text = user_prompt(record, "q", "direct", ["no", "yes"])
        self.assertNotIn("<|im_end|>", text)
        self.assertIn("<¦im_end¦>", text)

    def test_verify_labels_with_fake_tokenizer(self):
        tok = FakeTokenizer()
        self.assertEqual(len(verify_labels(tok, ["no", "yes"])), 2)
        with self.assertRaises(ValueError):
            verify_labels(tok, ["maybe", "no"])          # five tokens
        with self.assertRaises(ValueError):
            verify_labels(tok, ["no", "no"])             # same token


@unittest.skipUnless(cached_tokenizer_dir(), "Qwen3.8-27B tokenizer not in the local Hugging Face cache")
class QwenTokenizerTests(unittest.TestCase):
    def test_every_label_is_one_token_after_the_generation_prompt(self):
        tok = tokenizer()
        for qtype, n in (("noul", 2), ("score", 10), ("choice", 26), ("choice", 77), ("choice", 200)):
            labels = option_labels(qtype, n, tok)
            self.assertEqual(len(verify_labels(tok, labels)), n)

    def test_chat_prompt_disables_thinking(self):
        rows = prompt_rows(muse_records(n_states=2), "evidence", tokenizer())
        text = tokenizer().decode(rows[0].ids)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0].options, ("false", "true") if rows[0].labels == ("no", "yes") else rows[0].options)

    def test_plain_kev_records_render(self):
        rows = prompt_rows(kev_records(6), "direct", tokenizer())
        self.assertEqual({r.labels for r in rows}, {("0", "1", "2", "3", "4"), ("no", "yes"), ("A", "B", "C")})


if __name__ == "__main__":
    unittest.main()
