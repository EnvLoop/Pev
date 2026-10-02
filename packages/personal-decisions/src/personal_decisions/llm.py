"""OpenAI-compatible text client (Responses API) for gpt-6-astra rendering and optional fact extraction.

Settings: OPENAI_API_KEY, OPENAI_BASE_URL and RENDER_MODEL (legacy name RSI_TEACHER_MODEL), from the process
environment first, then the repo .env read with python-dotenv. The key only lives inside the SDK
client; it is never printed, logged, written or returned.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
import time

from . import PersonalDecisionsError

SETTING_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "RENDER_MODEL", "RSI_TEACHER_MODEL")   # last: legacy name


def default_env_file() -> Path | None:
    """The repository .env: the nearest ancestor of this file (or the working directory) that is the repository root
    (has CITATION.cff) and has a .env."""
    for start in (Path.cwd(), Path(__file__).resolve()):
        for folder in (start, *start.parents):
            if (folder / ".env").is_file() and (folder / "CITATION.cff").is_file():
                return folder / ".env"
    return None


@dataclass(frozen=True)
class LlmSettings:
    base_url: str
    model: str
    timeout_s: float = 180.0
    max_attempts: int = 5
    backoff_s: float = 2.0
    max_backoff_s: float = 60.0


def resolve_settings(env_file: Path | None = None, model: str | None = None) -> tuple[LlmSettings, str]:
    """(settings, api_key). The process environment wins over the .env file; `model` wins over both."""
    values: dict = {}
    env_file = env_file or default_env_file()
    if env_file is not None and Path(env_file).is_file():
        from dotenv import dotenv_values
        values = {key: value for key, value in dotenv_values(env_file).items() if key in SETTING_KEYS and value}
    values.update({key: os.environ[key] for key in SETTING_KEYS if os.environ.get(key)})
    api_key, base_url = values.get("OPENAI_API_KEY"), values.get("OPENAI_BASE_URL")
    model = model or values.get("RENDER_MODEL") or values.get("RSI_TEACHER_MODEL")
    if not api_key:
        raise PersonalDecisionsError("OPENAI_API_KEY is not set in the environment or .env; no request was sent")
    if not base_url or not re.match(r"^https?://", base_url):
        raise PersonalDecisionsError("OPENAI_BASE_URL must be an http(s) URL")
    if not model:
        raise PersonalDecisionsError("RENDER_MODEL is not set")
    return LlmSettings(base_url=base_url.rstrip("/"), model=model), api_key


@dataclass
class Reply:
    text: str
    usage: dict
    model: str


class TextClient:
    """One Responses API call per completion, store=False, bounded jittered retries on transient failures."""

    def __init__(self, settings: LlmSettings, api_key: str | None = None, *, client=None, sleep=time.sleep):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=settings.base_url, max_retries=0, timeout=settings.timeout_s)
        self.settings, self.client, self.sleep = settings, client, sleep
        self.model = settings.model

    def complete(self, instructions: str, text: str, max_output_tokens: int = 4000) -> Reply:
        import openai
        retryable = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)
        retries = {"rate_limited": 0, "server_error": 0, "connection": 0, "backoff_s": 0.0}
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                response = self.client.responses.create(model=self.model, instructions=instructions, input=text,
                                                        max_output_tokens=max_output_tokens, store=False)
            except retryable as error:
                if attempt == self.settings.max_attempts:
                    message = f"LLM API failed {attempt} times: {type(error).__name__}"
                    raise PersonalDecisionsError(message) from None
                kind = "rate_limited" if isinstance(error, openai.RateLimitError) else \
                    "server_error" if isinstance(error, openai.InternalServerError) else "connection"
                retries[kind] += 1
                delay = min(self.settings.max_backoff_s, self.settings.backoff_s * 2 ** (attempt - 1))
                delay *= 0.5 + random.random()
                retries["backoff_s"] = round(retries["backoff_s"] + delay, 1)
                self.sleep(delay)
                continue
            except openai.APIStatusError as error:
                raise PersonalDecisionsError(f"LLM API refused the request: HTTP {error.status_code}") from None
            usage = getattr(response, "usage", None)
            return Reply(text=getattr(response, "output_text", "") or "",
                         usage={"input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                                **({"retries": retries} if any(retries.values()) else {})},
                         model=getattr(response, "model", None) or self.model)
        raise PersonalDecisionsError("LLM retries exhausted")


def open_client(env_file: Path | None = None, model: str | None = None, max_attempts: int | None = None) -> TextClient:
    settings, api_key = resolve_settings(env_file, model)
    if max_attempts:
        from dataclasses import replace
        settings = replace(settings, max_attempts=max_attempts)
    return TextClient(settings, api_key)
