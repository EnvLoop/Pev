"""SFT rows are byte-identical to what predict-base scores; ties are not trained; default paths never import kev."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import decision_eval
from decision_eval.__main__ import main
from decision_eval.prompts import prompt_rows
from decision_eval.records import questions_of, read_jsonl, sha256_file, write_jsonl
from tests.synthetic import kev_records, muse_records
from tests.tiny import cached_tokenizer_dir, tokenizer

NO_KEV = """
import sys
import decision_eval, decision_eval.__main__, decision_eval.base_scorer, decision_eval.report
from decision_eval.cli import fit, predict, score, sft
assert not [m for m in sys.modules if m == "kev" or m.startswith("kev.")], "kev imported"
"""


def with_soft_labels(records):
    """First question of record 0: a uniform (tied) soft label; of record 1: a soft label with a unique argmax."""
    first = next(iter(records[0]["questions"]))
    q = records[0]["questions"][first]
    keys = list(q["criteria"]) if q["type"] == "choice" else (["false", "true"] if q["type"] == "noul"
                                                                 else [str(i) for i in range(len(q["criteria"]))])
    q["soft_label"] = {k: 1 / len(keys) for k in keys}
    second = records[1]["questions"]["q0"]
    second["type"], second["criteria"], second["label"] = "noul", None, True
    second["soft_label"] = {"true": 0.3, "false": 0.7}
    return records


class ImportTests(unittest.TestCase):
    def test_default_paths_do_not_import_kev(self):
        subprocess.run([sys.executable, "-c", NO_KEV], check=True)

    def test_public_api(self):
        self.assertEqual(set(decision_eval.__all__), {"label_token", "render_messages", "render_prompt", "sft_rows"})


@unittest.skipUnless(cached_tokenizer_dir(), "Qwen3.8-27B tokenizer not in the local Hugging Face cache")
class SftRowTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.records = with_soft_labels(muse_records(n_states=6, seed=11) + kev_records(3))
        self.data = self.dir / "train.jsonl"
        write_jsonl(self.data, self.records)

    def test_rows_are_byte_identical_to_the_scored_prompts(self):
        tok = tokenizer()
        rows, skipped = decision_eval.sft_rows(self.records, "assistant", tok)
        scored = {row.key: row for row in prompt_rows(self.records, "assistant", tok)}
        self.assertEqual(skipped, 1)
        self.assertEqual(len(rows), len(scored) - 1)
        for row in rows:
            key = (row["meta"]["id"], row["meta"]["qid"])
            prompt_ids = tok(row["prompt"], add_special_tokens=False).input_ids
            self.assertEqual(tuple(prompt_ids), scored[key].ids)
            full = tok(row["prompt"] + row["completion"], add_special_tokens=False).input_ids
            self.assertEqual(full, prompt_ids + [row["meta"]["label_token_id"]])
            self.assertIn(row["meta"]["label_token_id"], scored[key].label_ids)
            record = next(r for r in self.records if (r.get("id") or r["_meta"]["id"]) == key[0])
            text, labels = decision_eval.render_prompt(record, key[1], "assistant", tok)
            self.assertEqual(text, row["prompt"])
            self.assertEqual(decision_eval.label_token(record, key[1], "assistant", tok),
                             (row["completion"], row["meta"]["label_token_id"]))
            self.assertEqual(row["meta"]["prompt_tokens"], len(prompt_ids))

    def test_labels_follow_the_question_label(self):
        tok = tokenizer()
        rows, _ = decision_eval.sft_rows(self.records, "direct", tok)
        by_key = {(r["meta"]["id"], r["meta"]["qid"]): r for r in rows}
        for q in questions_of(self.records):
            if q.ambiguous:
                self.assertNotIn(q.key, by_key)
                with self.assertRaises(ValueError):
                    decision_eval.label_token(self.records[0], q.qid, "direct", tok)
                continue
            self.assertEqual(by_key[q.key]["meta"]["label_key"], q.keys[q.label])
        soft = by_key[(self.records[1]["id"], "q0")]
        self.assertEqual((soft["completion"], soft["meta"]["label_key"]), ("no", "false"))   # the argmax, not `label`
        messages, labels = decision_eval.render_messages(self.records[1], "q0", "direct", tok)
        self.assertEqual((messages[0]["role"], labels), ("user", ["no", "yes"]))

    def test_to_sft_cli_and_sidecar(self):
        out = self.dir / "rows.jsonl"
        main(["to-sft", "--data", str(self.data), "--template", "evidence", "--out", str(out),
              "--model", cached_tokenizer_dir(), "--revision", "main"])
        rows = read_jsonl(out)
        meta = json.loads(Path(f"{out}.meta.json").read_text())
        self.assertEqual(meta["sha256"], sha256_file(out))
        self.assertEqual((meta["rows"], meta["ambiguous_excluded"], meta["questions"]), (len(rows), 1, len(rows) + 1))
        self.assertEqual(meta["data_sha256"], sha256_file(self.data))
        self.assertEqual(set(rows[0]), {"prompt", "completion", "meta"})
        self.assertEqual(meta["max_prompt_tokens"], max(r["meta"]["prompt_tokens"] for r in rows))
        with self.assertRaises(ValueError):
            main(["sft-rows", "--data", str(self.data), "--template", "evidence", "--out", str(out),
                  "--model", cached_tokenizer_dir(), "--revision", "main", "--max-length", "20"])


if __name__ == "__main__":
    unittest.main()
