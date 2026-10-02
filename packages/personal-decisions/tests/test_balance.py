from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import unittest

from personal_decisions.balance import LabelSchedule, place_label
from personal_decisions.families.base import Fragment, Question
from personal_decisions.generate import Options, run
from personal_decisions.jsonl import file_sha256, read_jsonl
from personal_decisions.render.fixture import FixtureRenderer

from .support import workspace


def label_tables(path: Path) -> dict[str, Counter]:
    """family -> Counter of hard labels (noul, score) or of the label's position per option count (choice)."""
    tables: dict[str, Counter] = defaultdict(Counter)
    for record in read_jsonl(path):
        for qid, question in record["questions"].items():
            if "soft_label" in question:
                continue
            family = record["meta"]["families"][qid]
            if question["type"] == "choice":
                names = list(question["criteria"])
                tables[family][f"{len(names)}:{names.index(question['label'])}"] += 1
            else:
                tables[family][str(question["label"])] += 1
    return tables


class ScheduleTest(unittest.TestCase):
    def test_blocks_are_exactly_balanced(self):
        schedule = LabelSchedule("train", 1)
        key = ("apply_memory", "noul", 2)
        values = []
        for _ in range(10):
            values.append(schedule.target(key))
            schedule.used[key] += 1
        self.assertEqual(values.count(True), 5)
        self.assertTrue(all(values[i] != values[i + 1] or i % 2 for i in range(0, 10, 2)))

    def test_place_label_moves_the_answer_and_the_option_lines(self):
        question = Question("pick_option", "choice", "i", {"Alpha": "a", "Beta": "b", "Gamma": "c"}, "Gamma", [])
        fragment = Fragment("pick_option", question, {"label": "Request A", "text": "t",
                                                      "options": ["Alpha (a)", "Beta (b)", "Gamma (c)"]})
        place_label(fragment, 0)
        self.assertEqual(list(question.criteria), ["Gamma", "Alpha", "Beta"])
        self.assertEqual(fragment.pending["options"], ["Gamma (c)", "Alpha (a)", "Beta (b)"])


class BalancedGenerateTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.style = self.folder / "style.json"
        self.style.write_text(json.dumps({"style_id": "test-style", "voice": "plain"}))

    def generate(self, name: str, n: int) -> Path:
        out = self.folder / name
        run(workspace() / "train", out, Options(n=n, seed=9, concurrency=1), FixtureRenderer(), self.style)
        return out

    def test_every_family_is_balanced_by_construction(self):
        tables = label_tables(self.generate("balanced.jsonl", 90))
        self.assertEqual(len(tables), 7)
        for family, table in tables.items():
            groups: dict[str, list[int]] = defaultdict(list)
            for key, count in table.items():
                groups[key.split(":")[0] if ":" in key else "labels"].append(count)
            for size, counts in groups.items():
                expected = int(size) if size != "labels" else (2 if family != "notify_level" else 4)
                if size != "labels" or family != "notify_level":
                    counts = counts + [0] * (expected - len(counts))
                self.assertLessEqual(max(counts) - min(counts), 2, (family, dict(table)))

    def test_balancing_does_not_inflate_the_soft_share(self):
        def soft_share(balance: bool) -> float:
            out = self.folder / f"soft-{balance}.jsonl"
            run(workspace() / "train", out, Options(n=120, seed=4, concurrency=1, balance=balance), FixtureRenderer(),
                self.style)
            questions = [question for record in read_jsonl(out) for question in record["questions"].values()]
            return sum("soft_label" in question for question in questions) / len(questions)
        self.assertLess(abs(soft_share(True) - soft_share(False)), 0.05)

    def test_resume_replays_the_schedule(self):
        whole = self.generate("whole.jsonl", 40)
        part = self.folder / "part.jsonl"
        run(workspace() / "train", part, Options(n=20, seed=9, concurrency=1), FixtureRenderer(), self.style)
        run(workspace() / "train", part, Options(n=40, seed=9, concurrency=1), FixtureRenderer(), self.style)
        self.assertEqual(file_sha256(part), file_sha256(whole))


if __name__ == "__main__":
    unittest.main()
