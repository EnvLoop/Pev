import json
from pathlib import Path
import random
import tempfile
import unittest

from personal_decisions.baselines import by_geometry, shortcut_baselines
from personal_decisions.families.candidates import FEATURES, member_classes, sample_options
from personal_decisions.kb.schema import Item


def catalog(count: int = 300, seed: int = 5) -> list[Item]:
    rng = random.Random(seed)
    return [Item(item_id=f"i{index}", domain="product", name=f"Item {index}", category="Books",
                 price=round(rng.lognormvariate(3, 0.6), 2), popularity=int(rng.lognormvariate(5, 1.2)))
            for index in range(count)]


class CandidateSamplingTest(unittest.TestCase):
    def rates(self, target: Item, fixed: list[Item], pool: list[Item], size: int, runs: int = 300) -> dict:
        totals = {name: {"first": 0.0, "last": 0.0} for name in FEATURES}
        built = 0
        for seed in range(runs):
            options = sample_options(target, fixed, pool, size, random.Random(seed))
            if options is None:
                continue
            built += 1
            order = random.Random(seed + 10_000).sample(options, len(options))  # what a predictor sees
            classes = member_classes(order)[order.index(target)]
            for name, share in zip(FEATURES, classes):
                for position in ("first", "last"):
                    totals[name][position] += share[position]
        self.assertGreater(built, runs // 4)
        return {name: {position: value / built for position, value in shares.items()}
                for name, shares in totals.items()}

    def test_a_popular_target_is_not_first_or_central_more_than_chance(self):
        pool = sorted(catalog(), key=lambda item: item.popularity)
        target = pool[240]  # popular (80th percentile), like real later choices
        pool = [item for item in pool if item is not target]
        for name, shares in self.rates(target, [], pool, 4).items():
            for position, value in shares.items():
                self.assertAlmostEqual(value, 0.25, delta=0.07, msg=(name, position))

    def test_a_fixed_real_item_joins_only_when_the_target_can_still_be_balanced(self):
        pool = sorted(catalog(), key=lambda item: item.popularity)
        target = pool[150]
        rest = pool[:150] + pool[151:]
        # A real item just below the target blocks it from ever being least popular, although sampled options often
        # are (they can fall below the real item): the target cannot sit where a sampled option sits, so skipped.
        blocker = target.model_copy(update={"item_id": "low", "name": "Low", "popularity": target.popularity - 1})
        self.assertTrue(all(sample_options(target, [blocker], rest, 4, random.Random(seed)) is None
                            for seed in range(5)))
        # A real item below every option keeps that place; the target shares the other positions: it joins.
        floor = target.model_copy(update={"item_id": "floor", "name": "Floor", "popularity": 1, "price": 1.0})
        options = next(found for seed in range(20)
                       if (found := sample_options(target, [floor], rest, 4, random.Random(seed))))
        self.assertEqual(options[:2], [target, floor])
        self.assertEqual(len({item.name for item in options}), 4)

    def test_price_availability_matches_the_target(self):
        pool = catalog()
        unpriced = [item.model_copy(update={"price": None}) for item in pool[:50]]
        options = sample_options(pool[60], [], unpriced + pool[61:], 5, random.Random(3))
        self.assertTrue(all(item.price is not None for item in options))


class GeometryBaselinesTest(unittest.TestCase):
    def test_medoid_and_centroid_pick_the_central_option(self):
        features = {"a": {"price": 10.0, "popularity": 100}, "b": {"price": 20.0, "popularity": 200},
                    "c": {"price": 40.0, "popularity": 400}}
        self.assertEqual(by_geometry(features, "medoid"), "b")
        self.assertEqual(by_geometry(features, "centroid"), "b")

    def test_report_has_a_predictor_per_sampling_feature(self):
        record = {"state": "s", "questions": {"pick_option": {
            "type": "choice", "instructions": "i", "criteria": {"a": "", "b": "", "c": ""}, "label": "b",
            "src": "muse/pick_option"}},
            "meta": {"families": {"pick_option": "pick_option"}, "option_features": {"pick_option": {
                "a": {"price": 10.0, "popularity": 100}, "b": {"price": 20.0, "popularity": 200},
                "c": {"price": 40.0, "popularity": 400}}}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "r.jsonl"
            path.write_text(json.dumps(record) + "\n")
            entry = shortcut_baselines(path)["per_family"]["pick_option"]
        for predictor in ("most_popular", "least_popular", "cheapest", "priciest", "medoid", "centroid"):
            self.assertIn(predictor, entry)
        self.assertEqual(entry["medoid"], 1.0)
        self.assertEqual(entry["cheapest"], 0.0)


if __name__ == "__main__":
    unittest.main()
