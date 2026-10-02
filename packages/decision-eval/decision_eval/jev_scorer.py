"""Jev (TypeSafe's hosted System One model): a DEV-only reference predictor, never a gate input, never run on HIDDEN.

Records map 1:1 onto a System One request (docs.typesafe.ai/api): the record's `state` as the state, each question's
`type`, `instructions` and `criteria` as a Noul / Choice / Score, keyed by its question id (ids are not sent to the
model). One request per record through the official `typesafe-sdk` (extra `jev`). The full distribution is written in
kev's option-key order: noul [1 - p(yes), p(yes)], choice probabilities by option name, score probabilities by level;
floored at 1e-9, renormalised, logits = log(p). The key is TYPESAFE_API_KEY from the environment or the repository
`.env`; it stays inside the SDK client, never printed or written. Per request, latency and token usage are recorded,
with the cost at the listed input price.
"""
import math
import os
import time
from pathlib import Path

from .records import question_of, record_id

MODEL = "jev-1.13.0"                 # pinned version, not the moving `jev-latest` alias (docs.typesafe.ai/models)
USD_PER_INPUT_MTOK = 0.042           # listed price for jev-1.13.0; output tokens are free
KEY = "TYPESAFE_API_KEY"
REPO_ENV = Path(__file__).resolve().parents[3] / ".env"
FLOOR = 1e-9


def resolve_key(env_file=REPO_ENV):
    if os.environ.get(KEY, "").strip():
        return os.environ[KEY].strip()
    if env_file is not None and Path(env_file).is_file():
        from dotenv import dotenv_values
        value = (dotenv_values(env_file).get(KEY) or "").strip()
        if value:
            return value
    raise ValueError(f"{KEY} is not set in the environment or {env_file}; no request was sent")


def request_questions(record):
    """{qid: typesafe_sdk question} for one record, 1:1 from its typed questions."""
    from typesafe_sdk import Choice, Noul, Score
    out = {}
    for qid, q in record["questions"].items():
        if q.get("instructions") in (None, ""):
            raise ValueError("System One questions need instructions")
        if q["type"] == "noul":
            out[qid] = Noul(instructions=q["instructions"], criteria=q.get("criteria"))
        elif q["type"] == "choice":
            out[qid] = Choice(instructions=q["instructions"], criteria=q["criteria"])
        else:
            out[qid] = Score(instructions=q["instructions"], criteria=q["criteria"])
    return out


def distribution(question, answer):
    """The answer's probabilities in the question's option-key order, floored and renormalised."""
    if answer.type != question.type:
        raise ValueError("answer type differs from the question type")
    if answer.type == "noul":
        p = [1.0 - float(answer.noul), float(answer.noul)]
    elif answer.type == "choice":
        if set(answer.probabilities) != set(question.keys):
            raise ValueError("choice answer does not cover exactly the question's options")
        p = [float(answer.probabilities[k]) for k in question.keys]
    else:
        levels = {str(k): float(v) for k, v in answer.probabilities.items()}
        if set(levels) != set(question.keys):
            raise ValueError("score answer does not cover exactly the question's levels")
        p = [levels[k] for k in question.keys]
    if not all(math.isfinite(x) and x >= 0 for x in p) or sum(p) <= 0:
        raise ValueError("answer probabilities must be finite and non-negative with positive mass")
    floored = [max(x / sum(p), FLOOR) for x in p]
    return [x / sum(floored) for x in floored]


class JevScorer:
    def __init__(self, api_key=None, model=MODEL, client=None):
        if client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient
            client = TypeSafeClient(api_key=api_key or resolve_key(), model=model, timeout=120.0,
                                    retry=RetryPolicy(max_retries=4, backoff_initial=2.0, backoff_max=30.0))
        self.client, self.model, self.requests = client, model, []

    def record_logits(self, record):
        """-> [(question key, option keys, logits)] for one record; appends the request's usage to `requests`."""
        started = time.perf_counter()
        response = self.client.system_one(state=record["state"], questions=request_questions(record), model=self.model)
        latency_ms = 1000 * (time.perf_counter() - started)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        self.requests.append({"id": record_id(record), "latency_ms": round(latency_ms, 1), "model": response.model,
                              "request_id": getattr(response, "request_id", None), "input_tokens": input_tokens,
                              "output_tokens": getattr(usage, "output_tokens", None),
                              "usd": None if input_tokens is None else input_tokens * USD_PER_INPUT_MTOK / 1e6})
        rows = []
        for qid in record["questions"]:
            question = question_of(record, qid)
            p = distribution(question, response.answers[qid])
            rows.append((question.key, question.keys, [math.log(x) for x in p]))
        return rows

    def accounting(self, requests=None):
        """Totals over `requests` (default: this process's requests; a resumed run passes the checkpointed ones)."""
        requests = self.requests if requests is None else requests
        tokens = [r["input_tokens"] for r in requests if r["input_tokens"] is not None]
        latency = sorted(r["latency_ms"] for r in requests)
        return {"requests": len(requests), "served_models": sorted({r["model"] for r in requests}),
                "input_tokens": sum(tokens), "usd_estimated": sum(tokens) * USD_PER_INPUT_MTOK / 1e6,
                "usd_per_input_mtok": USD_PER_INPUT_MTOK,
                "latency_ms_median": latency[len(latency) // 2] if latency else None,
                "latency_ms_max": latency[-1] if latency else None}
