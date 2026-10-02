"""R: the gpt-6-astra reference (DEV only, never a gate input) through the OpenAI-compatible API.

Settings: OPENAI_API_KEY, OPENAI_BASE_URL and RENDER_MODEL (legacy name RSI_TEACHER_MODEL) from the process
environment, else the repository
`.env` (python-dotenv). The key stays inside the SDK client: never printed, logged or written. One Responses API request
per record (store=False) asks for a JSON object {question id: {option key: probability}}; answers are validated,
floored at 1e-9, renormalised, and written with logits = log(p).
"""
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from .conventions import render
from .records import question_of, record_id

SETTING_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "RENDER_MODEL")
LEGACY_KEYS = {"RENDER_MODEL": "RSI_TEACHER_MODEL"}   # read when the new name is unset
READ_KEYS = SETTING_KEYS + tuple(LEGACY_KEYS.values())
REPO_ENV = Path(__file__).resolve().parents[3] / ".env"
FLOOR = 1e-9
FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)
INSTRUCTIONS = ("You estimate calibrated probabilities for decisions a personal assistant makes for its user. Use only "
                "the given context. For every question, give a probability for every option key; each question's "
                "probabilities sum to 1. Return only a JSON object mapping each question id to an object mapping each "
                "option key to its probability.")


def resolve_settings(env_file=REPO_ENV):
    values = {}
    if env_file is not None and Path(env_file).is_file():
        from dotenv import dotenv_values
        values = {k: v for k, v in dotenv_values(env_file).items() if k in READ_KEYS and v}
    values.update({k: os.environ[k] for k in READ_KEYS if os.environ.get(k)})
    for key, legacy in LEGACY_KEYS.items():
        if not values.get(key) and values.get(legacy):
            values[key] = values[legacy]
        values.pop(legacy, None)
    missing = [k for k in SETTING_KEYS if not values.get(k)]
    if missing:
        raise ValueError(f"missing API settings: {', '.join(missing)} (environment or {env_file})")
    return values


def question_text(qid, q, keys):
    lines = [f"Question id: {qid}", f"Question: {render(q.get('instructions'))}", "Option keys:"]
    criteria = q.get("criteria")
    if q["type"] == "noul":
        criteria = criteria or {}
        descriptions = [criteria.get("false") or "no", criteria.get("true") or "yes"]
    elif q["type"] == "choice":
        descriptions = [render(v) for v in criteria.values()]
    else:
        descriptions = [render(level) for level in criteria]
    for key, desc in zip(keys, map(render, descriptions)):
        lines.append(f"- {key}: {desc}" if desc and desc != key else f"- {key}")
    return "\n".join(lines)


def request_text(record):
    parts = [f"Context:\n{render(record['state'])}"]
    for qid, q in record["questions"].items():
        parts.append(question_text(qid, q, question_of(record, qid).keys))
    return "\n\n".join(parts)


def option_probabilities(answer, keys):
    """{option key: p}: an answer key is the option key itself, or the whole "key: description" line copied from the
    request when that names exactly one option; anything else, or an option missing / given twice, is a ValueError."""
    if not isinstance(answer, dict):
        raise ValueError("answer does not give every option of every question")
    out = {}
    for given, value in answer.items():
        given = str(given)
        matches = [given] if given in keys else [k for k in keys if given.startswith(f"{k}: ")]
        if len(matches) != 1 or matches[0] in out:
            raise ValueError("answer does not give every option of every question")
        out[matches[0]] = value
    if set(out) != set(keys):
        raise ValueError("answer does not give every option of every question")
    return out


def parse_answer(text, record):
    """-> {qid: [p per option key]} or ValueError."""
    text = (text or "").strip()
    fenced = FENCE.match(text)
    data = json.loads(fenced.group(1) if fenced else text)
    out = {}
    for qid in record["questions"]:
        keys = question_of(record, qid).keys
        answer = option_probabilities(data.get(qid) if isinstance(data, dict) else None, keys)
        p = [float(answer[key]) for key in keys]
        if not all(math.isfinite(x) and x >= 0 for x in p) or sum(p) <= 0:
            raise ValueError("answer probabilities must be finite, non-negative, with positive mass")
        floored = [max(x / sum(p), FLOOR) for x in p]
        out[qid] = [x / sum(floored) for x in floored]
    return out


class OpenAIScorer:
    def __init__(self, settings=None, client=None, attempts=3, max_output_tokens=2048):
        settings = settings or resolve_settings()
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=settings["OPENAI_API_KEY"], base_url=settings["OPENAI_BASE_URL"], max_retries=2,
                            timeout=180)
        self.client, self.model = client, settings["RENDER_MODEL"]
        self.attempts, self.max_output_tokens, self.calls = attempts, max_output_tokens, 0
        self.requests, self._lock = [], threading.Lock()

    def _note(self, rid, responses, started):
        """Usage summed over every attempt of one record (unusable answers are paid for too)."""
        def total(field, sub=None):
            values = []
            for response in responses:
                usage = getattr(response, "usage", None)
                value = getattr(getattr(usage, sub, None) if sub else usage, field, None)
                values.append(value)
            return None if all(v is None for v in values) else sum(v or 0 for v in values)
        entry = {"id": rid, "calls": len(responses), "model": getattr(responses[-1], "model", None),
                 "latency_ms": round(1000 * (time.perf_counter() - started), 1),
                 "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
                 "reasoning_tokens": total("reasoning_tokens", "output_tokens_details")}
        with self._lock:
            self.requests.append(entry)

    def accounting(self, requests=None):
        """Token totals over `requests` (default: this process's); multiply by the API's prices for the cost."""
        requests = self.requests if requests is None else requests
        total = lambda field: sum(r.get(field) or 0 for r in requests)   # noqa: E731
        return {"requests": len(requests), "calls": sum(r.get("calls") or 1 for r in requests),
                "served_models": sorted({r["model"] for r in requests if r.get("model")}),
                "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
                "reasoning_tokens": total("reasoning_tokens")}

    def record_logits(self, record):
        """-> [(question key, option keys, logits)] for one record; appends its usage to `requests`."""
        last, started, responses = None, time.perf_counter(), []
        for _ in range(self.attempts):
            self.calls += 1
            response = self.client.responses.create(model=self.model, instructions=INSTRUCTIONS,
                                                    input=request_text(record), store=False,
                                                    max_output_tokens=self.max_output_tokens)
            responses.append(response)
            try:
                probs = parse_answer(getattr(response, "output_text", ""), record)
                break
            except (ValueError, json.JSONDecodeError) as error:
                last = error
        else:
            self._note(record_id(record), responses, started)
            raise ValueError(f"API answer unusable after {self.attempts} attempts: {last}")
        self._note(record_id(record), responses, started)
        return [((record_id(record), str(qid)), question_of(record, qid).keys, [math.log(x) for x in probs[qid]])
                for qid in record["questions"]]
