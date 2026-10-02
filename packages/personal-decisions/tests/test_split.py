import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions.jsonl import file_sha256, read_jsonl
from personal_decisions.split import assign_users, split_kb

from .support import SPLIT_USERS, workspace

SHARDS = ("train", "val", "dev", "hidden")


class SplitTest(unittest.TestCase):
    def ids(self, shard: str, table: str = "users.jsonl", key: str = "user_id") -> set[str]:
        return {row[key] for row in read_jsonl(workspace() / shard / table)}

    def test_users_are_disjoint_and_complete(self):
        everyone = {row["user_id"] for row in read_jsonl(workspace() / "kb" / "users.jsonl")}
        seen = set()
        for shard in SHARDS:
            ids = self.ids(shard)
            self.assertFalse(ids & seen, shard)
            seen |= ids
        self.assertEqual(seen, everyone)
        for name, count in SPLIT_USERS.items():
            self.assertEqual(len(self.ids(name)), count)

    def test_emails_are_disjoint(self):
        seen = set()
        for shard in SHARDS:
            ids = self.ids(shard, "emails.jsonl", "email_id")
            self.assertFalse(ids & seen)
            seen |= ids

    def test_manifest_has_hashes_and_counts_only(self):
        manifest = json.loads((workspace() / "split.json").read_text())
        for shard in SHARDS:
            entry = manifest["shards"][shard]
            self.assertEqual(entry["files"]["users.jsonl"]["sha256"], file_sha256(workspace() / shard / "users.jsonl"))
            self.assertEqual(entry["users"], len(self.ids(shard)))
        self.assertNotIn("u0", json.dumps(manifest))  # no user ids or contents

    def test_assignment_is_deterministic_and_order_free(self):
        ids = [f"user{index}" for index in range(50)]
        first = assign_users(ids, SPLIT_USERS)
        self.assertEqual(first, assign_users(list(reversed(ids)), SPLIT_USERS))
        self.assertEqual(len(first["train"]), 50 - sum(SPLIT_USERS.values()))

    def test_only_requested_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = split_kb(workspace() / "kb", {"dev": Path(folder) / "dev"}, None, SPLIT_USERS)
            self.assertEqual(list(manifest["shards"]), ["dev"])
            self.assertEqual(file_sha256(Path(folder) / "dev" / "users.jsonl"),
                             file_sha256(workspace() / "dev" / "users.jsonl"))


if __name__ == "__main__":
    unittest.main()
