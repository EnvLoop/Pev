import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions import leakage
from personal_decisions.external import export, ingest, instructions_markdown, write_batches
from personal_decisions.generate import Options, run
from personal_decisions.jsonl import read_jsonl
from personal_decisions.llm import Reply
from personal_decisions.render.astra import INSTRUCTIONS, AstraRenderer
from personal_decisions.render.fixture import template
from personal_decisions.split import email_shard, partition_kb

from .support import workspace

STYLES = Path(__file__).resolve().parents[1] / "styles" / "public"
MODEL = "external-test-model"


def render_text(user_message: str) -> str:
    snapshot = user_message.removeprefix("SNAPSHOT (JSON):\n").split("\n\nVERBATIM:\n")[0]
    return template(json.loads(snapshot))


class FakeClient:
    model = MODEL

    def complete(self, instructions, text, max_output_tokens=4000):
        return Reply(text=render_text(text), usage={"input_tokens": 1, "output_tokens": 1}, model=MODEL)


class ExternalRenderTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.styles = sorted(STYLES.glob("*.json"))[:3]
        self.options = Options(n=12, seed=5, family_weights={"pick_option": 3.0})

    def export_and_ingest(self, edit=None, drop_missing=False, margin=0):
        out, requests = self.folder / "ext.jsonl", self.folder / "requests.jsonl"
        receipt = export(workspace() / "dev", out, self.options, self.styles, requests, margin=margin)
        rows = list(read_jsonl(requests))
        texts = [{"id": row["id"], "text": render_text(row["user_message"])} for row in rows]
        if edit:
            texts = edit(texts)
        output = self.folder / "output.jsonl"
        output.write_text("".join(json.dumps(row) + "\n" for row in texts))
        return receipt, ingest(workspace() / "dev", out, self.options, self.styles, requests, [output], MODEL,
                               drop_missing=drop_missing)

    def test_same_records_as_generate_with_the_same_texts(self):
        reference = self.folder / "astra.jsonl"
        run(workspace() / "dev", reference, self.options, AstraRenderer(FakeClient()), self.styles)
        exported, ingested = self.export_and_ingest()
        self.assertEqual(exported["exported"], 12)
        self.assertEqual(ingested["ingested"], {"kept": 12})
        self.assertEqual((self.folder / "ext.jsonl").read_bytes(), reference.read_bytes())

    def test_drops_follow_the_anchor_rule_and_missing_stays_pending(self):
        def edit(texts):
            texts[0]["text"] = ""
            texts[1]["text"] = "Current time: unknown"
            return texts[:-1] + [texts[2]]  # last one missing; a duplicate id is not used at all
        _, ingested = self.export_and_ingest(edit, margin=2)
        self.assertEqual(ingested["ingested"]["pending"], 2)
        self.assertEqual(ingested["ingested"]["dropped:empty_render"], 1)
        self.assertEqual(ingested["ingested"]["dropped:missing_required_anchor"], 1)
        self.assertNotIn("states", ingested)  # not finalized while ids are pending
        self.assertEqual(list(ingested["files"].values())[0]["duplicate_id"], 2)

    def test_drop_missing_finalizes(self):
        _, ingested = self.export_and_ingest(lambda texts: texts[:-1], drop_missing=True)
        self.assertEqual(ingested["ingested"], {"kept": 11, "dropped:missing_output": 1})
        self.assertEqual(ingested["states"], 11)

    def test_refill_exports_only_new_states(self):
        self.export_and_ingest(lambda texts: [{**texts[0], "text": ""}, *texts[1:]])
        again = export(workspace() / "dev", self.folder / "ext.jsonl", self.options, self.styles,
                       self.folder / "requests.jsonl")
        self.assertEqual((again["exported"], again["kept_before"]), (1, 11))

    def test_batches_and_instructions(self):
        rows = [{"id": str(number)} for number in range(10)]
        report = write_batches(rows, self.folder / "batches", 4, instructions_markdown("Claude"))
        self.assertEqual(report["batches"], {"batch-01": 3, "batch-02": 3, "batch-03": 3, "batch-04": 1})
        text = (self.folder / "batches" / "batch-02" / "INSTRUCTIONS.md").read_text()
        self.assertIn(INSTRUCTIONS.replace("{style}", "{STYLE_BLOCK}"), text)
        self.assertEqual(text, (self.folder / "batches" / "INSTRUCTIONS.md").read_text())


class PartitionTest(unittest.TestCase):
    def test_equal_disjoint_parts_with_named_shards(self):
        folder = Path(tempfile.mkdtemp())
        manifest = partition_kb(workspace() / "kb", {"test-a": folder / "a", "test-b": folder / "b"},
                                email_pool="dev")
        users = {name: {row["user_id"] for row in read_jsonl(folder / name[-1] / "users.jsonl")}
                 for name in ("test-a", "test-b")}
        everyone = {row["user_id"] for row in read_jsonl(workspace() / "kb" / "users.jsonl")}
        self.assertFalse(users["test-a"] & users["test-b"])
        self.assertEqual(users["test-a"] | users["test-b"], everyone)
        for domain, count in manifest["parts"]["test-a"]["per_domain"].items():
            self.assertLessEqual(abs(count - manifest["parts"]["test-b"]["per_domain"][domain]), 1)
        self.assertEqual(json.loads((folder / "a" / "manifest.json").read_text())["shard"], "test-a")
        self.assertTrue(all(email_shard(row["email_id"]) == "dev" for row in read_jsonl(folder / "a" / "emails.jsonl")))
        self.assertEqual(manifest["leakage"]["test-a_vs_test-b"]["users"], 0)

    def test_leakage_puts_test_first(self):
        users = list(read_jsonl(workspace() / "kb" / "users.jsonl"))
        report = leakage.audit({"train": users[:5], "test": users[3:8]})
        self.assertEqual(list(report), ["test_vs_train"])
        self.assertEqual(report["test_vs_train"]["users"], 2)


if __name__ == "__main__":
    unittest.main()
