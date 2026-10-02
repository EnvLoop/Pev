from collections import Counter
import unittest

from personal_decisions import anchors
from personal_decisions.compose import compose, state_text
from personal_decisions.families import FAMILIES, FAMILY_NAMES, approval, notify, sharing
from personal_decisions.families.base import Context, touched_items
from personal_decisions.jsonl import seeded_rng
from personal_decisions.kb.rules import parse_time

from .support import shard

WEIGHTS = {"clean": 1.0, "override": 1.0, "removed": 1.0}


def fragments(family: str, variant: str, count: int = 120):
    data = shard("train")
    for index in range(count):
        user = data.users[index % len(data.users)]
        liked = [position for position, row in enumerate(user.later_interactions) if row.rating >= 4]
        target = liked[index % len(liked)]
        ctx = Context(user=user, shard=data, rng=seeded_rng("test", family, variant, index),
                      now=parse_time(user.later_interactions[target].timestamp), target_index=target, letter="A",
                      variant=variant)
        fragment = FAMILIES[family][0](ctx)
        if fragment is not None:
            yield ctx, fragment


class FamilyContractTest(unittest.TestCase):
    def test_every_family_builds_every_supported_variant(self):
        for family, (_, variants) in FAMILIES.items():
            for variant in variants:
                built = Counter(fragment.question.variant for _, fragment in fragments(family, variant))
                self.assertGreater(sum(built.values()), 20, (family, variant))
                self.assertGreater(built[variant], 0, (family, variant))

    def test_labels_have_the_kev_types(self):
        for family, (_, variants) in FAMILIES.items():
            for _, fragment in fragments(family, variants[-1], 40):
                question = fragment.question
                if question.type == "choice":
                    self.assertIn(question.label, question.criteria)
                elif question.type == "noul":
                    self.assertIsInstance(question.label, bool)
                    self.assertEqual(set(question.criteria), {"true", "false"})
                else:
                    self.assertIn(question.label, range(len(question.criteria)))
                if question.soft_label:
                    self.assertAlmostEqual(sum(question.soft_label.values()), 1.0, places=3)

    def test_same_seed_same_fragment(self):
        for family in FAMILY_NAMES:
            first = [(f.question.label, f.pending["text"]) for _, f in fragments(family, "clean", 20)]
            again = [(f.question.label, f.pending["text"]) for _, f in fragments(family, "clean", 20)]
            self.assertEqual(first, again)


class PickOptionTest(unittest.TestCase):
    def test_label_is_the_real_later_liked_item_and_distractors_are_untouched(self):
        for ctx, fragment in fragments("pick_option", "clean"):
            row = ctx.user.later_interactions[ctx.target_index]
            target = ctx.shard.items[row.item_id]
            self.assertEqual(fragment.question.label, target.name)
            names = {item.name: item for item in ctx.shard.items.values()}
            touched = touched_items(ctx.user)
            low = {ctx.shard.items[r.item_id].name for r in ctx.user.later_interactions if r.rating <= 2}
            for option in fragment.question.criteria:
                if option != target.name and option not in low:
                    self.assertNotIn(names[option].item_id, touched)
                    self.assertEqual(names[option].category, target.category)

    def test_target_is_not_systematically_the_most_popular(self):
        ranks = Counter()
        for _, fragment in fragments("pick_option", "clean", 300):
            features = fragment.question.option_features
            target = features[fragment.question.label]["popularity"]
            ranks[sum(1 for stats in features.values() if stats["popularity"] > target)] += 1
        total = sum(ranks.values())
        self.assertGreater(total, 50)
        self.assertLess(ranks[0] / total, 0.5)
        self.assertGreater(ranks[0] + ranks[1], 0)
        self.assertGreater(sum(count for rank, count in ranks.items() if rank >= 2), 0)

    def test_removed_evidence_is_soft_and_withheld(self):
        removed = [(ctx, f) for ctx, f in fragments("pick_option", "removed") if f.question.variant == "removed"]
        self.assertTrue(removed)
        for _, fragment in removed:
            question = fragment.question
            self.assertEqual(set(question.soft_label), set(question.criteria))
            self.assertTrue(fragment.excluded_memory_ids)


class RuleFamiliesTest(unittest.TestCase):
    def test_approval_purchase_labels_follow_the_threshold_in_force(self):
        checked = 0
        for ctx, fragment in fragments("needs_approval", "override"):
            text = fragment.pending["text"]
            if not text.startswith("Buy ") or fragment.question.soft_label:
                continue
            amount = float(text.rsplit("$", 1)[1].split(" ")[0].replace(",", ""))
            threshold, _ = approval.effective_threshold(ctx)
            self.assertEqual(fragment.question.label, amount > threshold)
            checked += 1
        self.assertGreater(checked, 5)

    def test_share_ok_matches_clearance(self):
        for ctx, fragment in fragments("share_ok", "clean"):
            fact = next(f for f in ctx.user.memory_facts if f.fact_id == fragment.memory_ids[0])
            name = fragment.pending["text"]
            kind = next((p.relationship for p in ctx.user.contacts if p.name in name), "public")
            if "public social feed" in name:
                kind = "public"
            clearance = ctx.user.privacy.clearance[kind]
            self.assertEqual(fragment.question.label, sharing.shareable(clearance, fact.privacy))

    def test_notify_levels_are_balanced(self):
        counts = Counter(f.question.label for _, f in fragments("notify_level", "clean", 200))
        self.assertEqual(set(counts), {0, 1, 2, 3})
        self.assertLess(max(counts.values()), 0.45 * sum(counts.values()))

    def test_binary_labels_are_balanced_by_construction(self):
        for family in ("share_ok", "forgotten_violation"):
            labels = [f.question.label for variant in ("clean", "removed") for _, f in fragments(family, variant, 200)]
            self.assertGreater(len(labels), 100, family)
            self.assertTrue(0.35 < sum(labels) / len(labels) < 0.65, family)

    def test_notify_quiet_hours_window(self):
        self.assertTrue(notify.in_window(23, 22, 7))
        self.assertTrue(notify.in_window(3, 22, 7))
        self.assertFalse(notify.in_window(12, 22, 7))
        self.assertTrue(notify.in_window(3, 0, 6))

    def test_forgotten_contrast_pair(self):
        for _, fragment in fragments("forgotten_violation", "removed"):
            if fragment.question.variant == "removed":
                self.assertFalse(fragment.question.label)
                self.assertNotIn("forget_requests", fragment.lines)


class ComposeTest(unittest.TestCase):
    def test_states_are_consistent_with_their_anchors(self):
        data = shard("train")
        made = 0
        for index in range(60):
            user = data.users[index % len(data.users)]
            families = [FAMILY_NAMES[index % 7], FAMILY_NAMES[(index + 3) % 7], FAMILY_NAMES[(index + 5) % 7]]
            draft = compose(data, user, seed=3, index=index, visit=index // len(data.users), families=families,
                            variant_weights=WEIGHTS)
            if draft is None:
                continue
            made += 1
            text = state_text(draft.structured)
            for question in draft.questions.values():
                self.assertIsNone(anchors.check(text, question.required, question.forbidden))
            self.assertEqual(len(draft.structured["pending"]), len(draft.questions))
        self.assertGreater(made, 45)


if __name__ == "__main__":
    unittest.main()
