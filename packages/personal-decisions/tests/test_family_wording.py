"""Generator fixes from the TRAIN label spot-read: unambiguous apply_memory negatives, notify_level's time
reference, and bookings only at restaurants."""
import unittest

from personal_decisions.families import approval
from personal_decisions.kb.memory import category_label

from .support import shard
from .test_families import fragments


class FamilyWordingTest(unittest.TestCase):
    def test_restaurant_fact_negatives_never_name_another_restaurant_category(self):
        data = shard("train")
        restaurant_labels = {category_label(item.category) for item in data.by_domain.get("restaurant", [])}
        checked = 0
        for ctx, fragment in fragments("apply_memory", "clean", 300):
            fact = next(f for f in ctx.user.memory_facts if f.fact_id == fragment.memory_ids[0])
            text = fragment.pending["text"]
            if fact.domain != "restaurant" or fragment.question.label:
                continue
            checked += 1
            self.assertFalse(text.startswith("Find me a "), text)
            self.assertFalse(any(f"a {label} in" in text for label in restaurant_labels), text)
        self.assertGreater(checked, 0)

    def test_relevant_requests_name_the_fact_category(self):
        for ctx, fragment in fragments("apply_memory", "clean", 200):
            if not fragment.question.label:
                continue
            fact = next(f for f in ctx.user.memory_facts if f.fact_id == fragment.memory_ids[0])
            self.assertIn(category_label(fact.category), fragment.pending["text"])

    def test_notify_question_is_judged_when_the_event_arrives(self):
        for _, fragment in fragments("notify_level", "clean", 30):
            self.assertTrue(fragment.question.instructions.endswith("about this when it arrives?"))

    def test_bookings_are_cancelled_only_at_restaurants(self):
        data = shard("train")
        restaurants = {item.name for item in data.by_domain.get("restaurant", [])}
        seen = 0
        for _, fragment in fragments("needs_approval", "clean", 400):
            text = fragment.pending["text"]
            if text.startswith("Cancel your booking at "):
                seen += 1
                venue = text.removeprefix("Cancel your booking at ").split("; the deposit")[0]
                self.assertIn(venue, restaurants)
        self.assertTrue(any("{venue}" in action for action in approval.IRREVERSIBLE_ACTIONS))


if __name__ == "__main__":
    unittest.main()
