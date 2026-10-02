import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from personal_decisions import PersonalDecisionsError
from personal_decisions.jsonl import read_jsonl, write_jsonl
from personal_decisions.kb.rules import FIRST_NAMES as SYNTHETIC_FIRST
from personal_decisions.release.anonymize import (KEY_VARIABLE, PSEUDO_FIRST, PSEUDO_LAST, Pseudonymizer,
                                                  anonymize_record, is_pseudonym, load_key)
from personal_decisions.release.export import export_release
from personal_decisions.release.pii import FIRST_NAMES, brand_names, find_pii, redact_example
from personal_decisions.release.scan import privacy_scan

KEY = bytes.fromhex("ab" * 32)
EMAIL_TEXT = ("---- Forwarded by Benjamin Rogers/HOU/ECT on 09/26/2000\nFrom: Stephen Bilby\n"
              "Hello Mr. Kean, call 713-853-1234 or jane.doe@enron.com, see www.enron.com/x, 1400 Smith Street.\n"
              "Sara Shackleton agreed. Avery Okafor <avery.okafor@example.com> too.")


def record(state: str, required: list[str], criteria=None, label=True, user="u1") -> dict:
    criteria = criteria or {"true": "Allowed", "false": "Not allowed"}
    return {"id": "muse/val/1/1", "state": state,
            "questions": {"a": {"type": "noul", "instructions": "May the assistant share it with Sara Shackleton?",
                                "criteria": criteria, "label": label, "src": "muse/share_ok"}},
            "meta": {"user_id": user, "state_id": "val/1/1", "families": {"a": "share_ok"},
                     "anchors": {"a": {"required": required, "forbidden": []}}, "variants": {"a": "buried+clean"}}}


class PiiTest(unittest.TestCase):
    def test_kinds_found_and_synthetic_contacts_skipped(self):
        found = {(hit.kind, hit.text) for hit in find_pii(EMAIL_TEXT)}
        for expected in [("person_name", "Benjamin Rogers"), ("person_name", "Stephen Bilby"),
                         ("person_name", "Mr. Kean"), ("phone", "713-853-1234"), ("email", "jane.doe@enron.com"),
                         ("url", "www.enron.com/x"), ("person_name", "Sara Shackleton")]:
            self.assertIn(expected, found)
        self.assertFalse(any("Okafor" in text or "example.com" in text for _, text in found))
        self.assertTrue(any(kind == "street_address" for kind, _ in found))

    def test_brands_options_and_product_slashes_are_not_people(self):
        allow = brand_names(["Bob Evans Restaurant", "Dr. Noy's Teddy Bear", "Multipet Mr. Bill Dog Toy"])
        text = "Bob Evans menu. Dr. Noy's toy. Mr. Bill\nSize Small/Medium/Large. Paul Smith is an option."
        found = [hit.text for hit in find_pii(text, protected=("Paul Smith is an option",), allow=allow)]
        self.assertEqual(found, [])

    def test_redacted_examples_keep_only_initials(self):
        self.assertEqual(redact_example("Sara Shackleton"), "S*** S*********")


class AnonymizeTest(unittest.TestCase):
    def test_pseudonyms_are_keyed_shaped_and_disjoint(self):
        pseudo, other = Pseudonymizer(KEY), Pseudonymizer(bytes.fromhex("cd" * 32))
        self.assertEqual(pseudo.name("Steve Kean").split()[1], pseudo.name("Mr. Kean").split()[1])
        self.assertTrue(pseudo.name("Mr. Kean").startswith("Mr. "))
        self.assertEqual(pseudo.user_id("u1"), Pseudonymizer(KEY).user_id("u1"))
        self.assertNotEqual(pseudo.user_id("u1"), other.user_id("u1"))
        self.assertFalse(set(PSEUDO_FIRST + PSEUDO_LAST) & (FIRST_NAMES | set(SYNTHETIC_FIRST)))
        self.assertTrue(is_pseudonym(pseudo.name("Sara Shackleton")))

    def test_record_text_and_anchors_change_together_options_never(self):
        anchor = "Sara Shackleton agreed."
        original = record(EMAIL_TEXT, [anchor])
        released, counts = anonymize_record(original, Pseudonymizer(KEY))
        self.assertNotIn("Shackleton", released["state"] + released["questions"]["a"]["instructions"])
        new_anchor = released["meta"]["anchors"]["a"]["required"][0]
        self.assertIn(new_anchor, released["state"])
        self.assertNotEqual(new_anchor, anchor)
        self.assertEqual(released["questions"]["a"]["criteria"], original["questions"]["a"]["criteria"])
        self.assertEqual(released["questions"]["a"]["label"], True)
        self.assertNotEqual(released["meta"]["user_id"], "u1")
        self.assertGreaterEqual(counts["person_name"], 4)
        self.assertIn("[PHONE]", released["state"])
        self.assertEqual(original["state"], EMAIL_TEXT)  # input untouched

    def test_names_inside_options_are_kept_everywhere(self):
        criteria = {"Sara Shackleton Bakery": "Bakery, $", "Joe's Diner": "Diner, $"}
        original = record("Sara Shackleton wrote. Options: Sara Shackleton Bakery", [], criteria, "Joe's Diner")
        released, _ = anonymize_record(original, Pseudonymizer(KEY))
        self.assertEqual(released["state"], original["state"])
        self.assertEqual(list(released["questions"]["a"]["criteria"]), list(criteria))

    def test_key_comes_from_environment_and_is_validated(self):
        with mock.patch.dict(os.environ, {KEY_VARIABLE: "ab" * 32}):
            self.assertEqual(load_key(), KEY)
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {KEY_VARIABLE: ""}):
            env = Path(folder) / ".env"
            env.write_text(f"{KEY_VARIABLE}=short\n")
            with self.assertRaises(PersonalDecisionsError):
                load_key(env)
            key_file = Path(folder) / "pseudonym_key.txt"
            key_file.write_text("cd" * 32 + "\n")
            with mock.patch.dict(os.environ, {KEY_VARIABLE: "ab" * 32}):
                self.assertEqual(load_key(env, key_file), bytes.fromhex("cd" * 32))


class ExportTest(unittest.TestCase):
    def test_export_is_deterministic_and_drops_foreign_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [record(EMAIL_TEXT, ["Sara Shackleton agreed."]), {"state": "x", "questions": {}, "_meta": {}}]
            write_jsonl(root / "in.jsonl", rows, sort_keys=False)
            first = export_release(root / "in.jsonl", root / "a.jsonl", KEY)
            second = export_release(root / "in.jsonl", root / "b.jsonl", KEY)
            self.assertEqual(first["out_sha256"], second["out_sha256"])
            self.assertEqual((first["records_written"], first["foreign_records"]), (1, 1))
            self.assertEqual(first["residual_detections"], {})
            kept = export_release(root / "in.jsonl", root / "c.jsonl", KEY, keep_foreign=True)
            self.assertEqual(kept["records_written"], 2)
            with self.assertRaises(ValueError):
                export_release(root / "in.jsonl", root / "in.jsonl", KEY)
            self.assertNotIn("Shackleton", json.dumps(list(read_jsonl(root / "a.jsonl"))))

    def test_scan_counts_without_examples_and_refuses_sealed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "dev.jsonl", [record(EMAIL_TEXT, [])], sort_keys=False)
            report = privacy_scan([root / "dev.jsonl"], [], None, 5, [root / "dev.jsonl"])
            kinds = report["records"]["dev.jsonl"]["kinds"]
            self.assertGreater(kinds["person_name"]["hits"], 0)
            self.assertNotIn("redacted_examples", kinds["person_name"])
            with self.assertRaises(ValueError):
                privacy_scan([root / "sealed" / "x.jsonl"], [], None, 0, [])


if __name__ == "__main__":
    unittest.main()
