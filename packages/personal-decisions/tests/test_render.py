import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from personal_decisions import PersonalDecisionsError, anchors
from personal_decisions.llm import LlmSettings, TextClient, resolve_settings
from personal_decisions.render.astra import AstraRenderer
from personal_decisions.render.style import load_style

FAKE_KEY = "sk-test-not-a-real-key"


class AnchorTest(unittest.TestCase):
    def test_boundaries(self):
        self.assertTrue(anchors.contains("Ask me before any purchase over $50.", "$50"))
        self.assertFalse(anchors.contains("It costs $500 today", "$50"))
        self.assertFalse(anchors.contains("It costs $50.99 today", "$50"))
        self.assertTrue(anchors.contains("It costs $50.99 today", "$50.99"))
        self.assertFalse(anchors.contains("Anna came", "Ann"))
        self.assertTrue(anchors.contains("(urgency: high).", "(urgency: high)"))

    def test_typography_and_whitespace_are_normalized(self):
        self.assertTrue(anchors.contains("it’s   fine\nhere", "it's fine here"))

    def test_check_reasons(self):
        self.assertIsNone(anchors.check("a b c", ["b"], ["d"]))
        self.assertEqual(anchors.check("a b c", ["x"], []), "missing_required_anchor")
        self.assertEqual(anchors.check("a b c", [], ["c"]), "forbidden_anchor_present")


class StyleTest(unittest.TestCase):
    def test_style_id_defaults_to_content_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "style.json"
            path.write_text(json.dumps({"voice": "warm", "verbosity": "terse"}))
            style = load_style(path)
            self.assertTrue(style["style_id"].startswith("sha256:"))
            path.write_text(json.dumps({"voice": "warm", "colour": "red"}))
            with self.assertRaises(ValueError):
                load_style(path)


class FakeResponses:
    def __init__(self, failures: int = 0):
        self.failures, self.calls = failures, []

    def create(self, **body):
        import httpx2 as httpx  # the transport openai 3.x ships with
        import openai
        self.calls.append(body)
        if len(self.calls) <= self.failures:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.test/v1/responses"))
        usage = SimpleNamespace(input_tokens=120, output_tokens=40)
        return SimpleNamespace(output_text="Current time: now\n\nRequest A: pick", usage=usage, model="gpt-6-astra")


class AstraTest(unittest.TestCase):
    def client(self, failures: int = 0) -> TextClient:
        responses = FakeResponses(failures)
        client = TextClient(LlmSettings(base_url="https://api.test/v1", model="gpt-6-astra"),
                            client=SimpleNamespace(responses=responses), sleep=lambda _: None)
        client.fake = responses
        return client

    def test_request_lists_verbatim_anchors_and_hashes_prompt(self):
        client = self.client()
        rendered = AstraRenderer(client).render({"now": "2020-01-01 10:00"}, ["Request A", "$50"],
                                                {"style_id": "s", "voice": "brisk"})
        body = client.fake.calls[0]
        self.assertIn("VERBATIM:\n- Request A\n- $50", body["input"])
        self.assertIn("voice: brisk", body["instructions"])
        self.assertFalse(body["store"])
        self.assertEqual(rendered.usage, {"input_tokens": 120, "output_tokens": 40})
        self.assertEqual(len(rendered.prompt_sha256), 64)

    def test_transient_failures_are_retried(self):
        client = self.client(failures=2)
        self.assertEqual(client.complete("i", "t").text.split("\n")[0], "Current time: now")
        self.assertEqual(len(client.fake.calls), 3)

    def test_settings_come_from_env_file_without_leaking_the_key(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=True):
            env = Path(folder) / ".env"
            env.write_text(f"OPENAI_API_KEY={FAKE_KEY}\nOPENAI_BASE_URL=https://api.test/v1/\n"
                           "RENDER_MODEL=gpt-6-astra\n")
            settings, key = resolve_settings(env)
            self.assertEqual((settings.base_url, settings.model, key), ("https://api.test/v1", "gpt-6-astra",
                                                                        FAKE_KEY))
            self.assertNotIn(FAKE_KEY, repr(settings))
            env.write_text(f"OPENAI_API_KEY={FAKE_KEY}\nOPENAI_BASE_URL=https://api.test/v1\nRSI_TEACHER_MODEL=old\n")
            self.assertEqual(resolve_settings(env)[0].model, "old")   # legacy variable name
            env.write_text("OPENAI_BASE_URL=https://api.test/v1\nRSI_TEACHER_MODEL=gpt-6-astra\n")
            with self.assertRaises(PersonalDecisionsError) as raised:
                resolve_settings(env)
            self.assertNotIn(FAKE_KEY, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
