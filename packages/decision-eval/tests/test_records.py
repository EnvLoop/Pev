import unittest

from decision_eval.records import kev_request, question_of, questions_of
from tests.synthetic import kev_records, muse_records


def record(questions, meta=None):
    return {"id": "r1", "state": "s", "questions": questions,
            "meta": meta or {"user_id": "u", "state_id": "st1", "families": {qid: "fam" for qid in questions}}}


class RecordTests(unittest.TestCase):
    def test_muse_layout_uses_meta_family_and_state(self):
        qs = questions_of(muse_records(n_states=4))
        self.assertEqual(len(qs), 12)
        self.assertTrue(all(q.state_id.startswith("state-") for q in qs))
        self.assertIn("needs_approval", {q.family for q in qs})

    def test_plain_kev_layout_uses_src_and_group(self):
        qs = questions_of(kev_records(n=6))
        self.assertEqual({q.family for q in qs}, {"sst5", "boolq", "agnews"})
        self.assertEqual(qs[0].record_id, "sst5/test/0")
        self.assertEqual(qs[0].state_id, "sst5/test/0")
        noul = next(q for q in qs if q.type == "noul")
        self.assertEqual(noul.keys, ("false", "true"))   # kev key order

    def test_family_falls_back_to_src_without_muse_prefix(self):
        rec = {"id": "r", "state": "s", "questions": {"a": {"type": "noul", "label": True, "src": "muse/share_ok"}}}
        self.assertEqual(question_of(rec, "a").family, "share_ok")
        self.assertEqual(question_of(rec, "a").state_id, "r")

    def test_labels_by_type(self):
        rec = record({
            "c": {"type": "choice", "criteria": {"x": None, "y": "why"}, "label": "y"},
            "n": {"type": "noul", "label": False},
            "n2": {"type": "noul", "label": "true"},
            "s": {"type": "score", "criteria": ["lo", "mid", "hi"], "label": 2}})
        got = {qid: question_of(rec, qid) for qid in rec["questions"]}
        self.assertEqual(got["c"].label, 1)
        self.assertEqual(got["n"].label, 0)
        self.assertEqual(got["n2"].label, 1)
        self.assertEqual(got["s"].label, 2)
        self.assertEqual(got["s"].target, (0.0, 0.0, 1.0))

    def test_soft_label_is_target_and_argmax_is_label(self):
        rec = record({"n": {"type": "noul", "label": True, "soft_label": {"true": 1, "false": 3}},
                      "s": {"type": "score", "criteria": ["a", "b"], "soft_label": {"0": 0.5, "1": 0.5}}})
        n, s = question_of(rec, "n"), question_of(rec, "s")
        self.assertEqual(n.target, (0.75, 0.25))
        self.assertEqual(n.label, 0)                      # soft label wins over the hard one
        self.assertEqual(s.label, 0)                      # ties -> first option

    def test_invalid_labels_raise(self):
        bad = [{"type": "choice", "criteria": {"x": None}, "label": "z"}, {"type": "noul", "label": "maybe"},
               {"type": "score", "criteria": ["a"], "label": 3}, {"type": "score", "criteria": ["a"], "label": True},
               {"type": "noul", "label": True, "soft_label": {"perhaps": 1}}]
        for q in bad:
            with self.assertRaises(ValueError):
                question_of(record({"q": q}), "q")

    def test_duplicates_and_missing_ids_raise(self):
        recs = muse_records(n_states=2)
        with self.assertRaises(ValueError):
            questions_of(recs + recs[:1])
        with self.assertRaises(ValueError):
            questions_of([{"state": "s", "questions": {"a": {"type": "noul", "label": True, "src": "x"}}}])

    def test_kev_request_carries_types_src_and_kev_labels_only(self):
        rec = record({"n": {"type": "noul", "soft_label": {"true": 0.9, "false": 0.1}, "src": "muse/share_ok"},
                      "c": {"type": "choice", "criteria": {"x": None, "y": None}, "label": "y", "src": "muse/route"}})
        req = kev_request(rec)
        self.assertEqual(req["questions"]["n"]["label"], True)
        self.assertEqual(req["questions"]["c"]["label"], "y")
        self.assertNotIn("soft_label", req["questions"]["n"])
        self.assertNotIn("meta", req)


if __name__ == "__main__":
    unittest.main()
