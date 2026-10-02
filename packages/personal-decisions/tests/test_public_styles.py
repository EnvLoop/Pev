"""The shared public rendering styles load as valid style specs with unique ids."""
import unittest
from pathlib import Path

from personal_decisions.render.style import load_style

PUBLIC = Path(__file__).resolve().parents[1] / "styles" / "public"


class PublicStylesTest(unittest.TestCase):
    def test_all_public_styles_load_with_unique_ids(self):
        paths = sorted(PUBLIC.glob("*.json"))
        self.assertGreaterEqual(len(paths), 6)
        ids = [load_style(path)["style_id"] for path in paths]
        self.assertEqual(len(ids), len(set(ids)))
        for path, style_id in zip(paths, ids):
            self.assertEqual(path.stem, style_id)


if __name__ == "__main__":
    unittest.main()
