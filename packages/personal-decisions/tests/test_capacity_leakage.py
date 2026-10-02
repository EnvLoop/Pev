import copy
import json
import unittest

from personal_decisions.capacity import shard_capacity
from personal_decisions.families import FAMILY_NAMES
from personal_decisions.jsonl import read_jsonl
from personal_decisions.leakage import audit

from .support import shard, workspace

CHECKS = ("users", "user_target_pairs", "target_review_text", "target_events")


class LeakageTest(unittest.TestCase):
    def shards(self) -> dict[str, list[dict]]:
        return {name: list(read_jsonl(workspace() / name / "users.jsonl"))
                for name in ("train", "val", "dev", "hidden")}

    def test_split_manifest_reports_zero_collisions(self):
        manifest = json.loads((workspace() / "split.json").read_text())
        self.assertEqual(len(manifest["leakage"]), 6)
        for entry in manifest["leakage"].values():
            self.assertEqual(entry["users"], 0)
            self.assertEqual(entry["user_target_pairs"], 0)
        self.assertNotIn("u0", json.dumps(manifest["leakage"]))

    def test_a_duplicated_account_is_detected(self):
        shards = self.shards()
        clone = copy.deepcopy(shards["dev"][0])
        clone["user_id"] = "someone-else"
        shards["train"].append(clone)
        report = audit(shards)["dev_vs_train"]
        self.assertEqual(report["users"], 0)
        self.assertGreater(report["target_events"], 0)
        self.assertGreater(report["target_review_text"], 0)
        shards["train"].append(shards["hidden"][0])
        report = audit(shards)["hidden_vs_train"]
        self.assertEqual(report["users"], 1)
        self.assertGreater(report["user_target_pairs"], 0)
        self.assertEqual(set(report), set(CHECKS))


class CapacityTest(unittest.TestCase):
    def test_capacity_counts_are_consistent(self):
        report = shard_capacity(shard("train"), trials=2)
        self.assertEqual(report["users"], len(shard("train").users))
        self.assertLessEqual(report["pick_feasible_targets"], report["decision_times"])
        self.assertEqual(set(report["attempt_success_rate"]), set(FAMILY_NAMES))
        self.assertEqual(report["balanced_questions"]["total"], report["pick_feasible_targets"] * len(FAMILY_NAMES))
        self.assertEqual(report, shard_capacity(shard("train"), trials=2))  # deterministic


if __name__ == "__main__":
    unittest.main()
