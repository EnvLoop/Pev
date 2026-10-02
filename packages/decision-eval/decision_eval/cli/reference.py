"""predict-openai and predict-jev: DEV-only reference predictors over hosted APIs (never gate inputs, never HIDDEN).

Both refuse a data path that names the sealed or hidden set, send one request per record, checkpoint every finished
record to `<out>.partial.jsonl` (a rerun resumes from it), and write the prediction file plus `<out>.meta.json` and
`<out>.requests.jsonl` (latency, tokens, cost per request) once every record is scored. Records that still fail can be
written as uniform predictions with `--fill-unscorable uniform` (listed in the meta as `unscorable`).
"""
import argparse
import math
import sys
import time
from pathlib import Path

from ..checkpoint import Checkpoint, score_records
from ..predictions import prediction_row
from ..records import question_of, read_jsonl, record_id, sha256_file, write_jsonl
from .common import refuse_sealed
from .predict import finish


def _args(prog, description):
    ap = argparse.ArgumentParser(prog=prog, description=description)
    ap.add_argument("--data", required=True, help="DEV records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-file", default=None, help="default: the repository .env")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N records (smoke tests)")
    ap.add_argument("--fill-unscorable", choices=("none", "uniform"), default="none",
                    help="records that still fail: stop without predictions (none) or write uniform ones")
    return ap


def _uniform(record):
    out = []
    for qid in record["questions"]:
        q = question_of(record, qid)
        out.append((q.key, q.keys, [-math.log(len(q.keys))] * len(q.keys)))
    return out


def _run(scorer, records, predictor, a, data_sha256):
    """-> (prediction rows, requests, unscorable record ids); exits non-zero while records remain unscored."""
    checkpoint = Checkpoint(f"{a.out}.partial.jsonl", data_sha256)
    if checkpoint.done:
        print(f"resuming: {len(checkpoint.done)} records already scored", file=sys.stderr)
    log = lambda message: print(message, file=sys.stderr, flush=True)   # noqa: E731
    triples, requests, failures = score_records(scorer, records, checkpoint, a.concurrency, log=log)
    if failures and a.fill_unscorable == "none":
        reasons = sorted(set(failures.values()))
        raise SystemExit(f"{len(failures)} of {len(records)} records unscored ({'; '.join(reasons)}); "
                         "rerun to resume, or pass --fill-unscorable uniform")
    if failures:
        triples = [t for r in records for t in (_uniform(r) if record_id(r) in failures else [])] + triples
    rows = [prediction_row(key, predictor, None, keys, z) for key, keys, z in triples]
    return rows, requests, sorted(failures)


def predict_openai_main(argv=None):
    a = _args("predict-openai", "gpt-6-astra reference probabilities from the OpenAI-compatible API (DEV)")
    a = a.parse_args(argv)
    refuse_sealed(a.data)
    from ..openai_scorer import REPO_ENV, OpenAIScorer, resolve_settings
    started = time.time()
    records, data_sha256 = read_jsonl(a.data)[: a.limit], sha256_file(a.data)
    scorer = OpenAIScorer(resolve_settings(a.env_file or REPO_ENV))
    rows, requests, unscorable = _run(scorer, records, "openai", a, data_sha256)
    write_jsonl(f"{a.out}.requests.jsonl", requests)
    meta = {"predictor": "openai", "model": scorer.model, "records": len(records), "data_sha256": data_sha256,
            "unscorable": unscorable, **scorer.accounting(requests)}
    finish(a.out, rows, meta, started)


def predict_jev_main(argv=None):
    ap = _args("predict-jev", "TypeSafe Jev reference probabilities (DEV; extra `jev`)")
    ap.add_argument("--model", default=None, help="default: the pinned jev-1.13.0")
    a = ap.parse_args(argv)
    refuse_sealed(a.data)
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        raise SystemExit("predict-jev needs the jev extra: uv sync --extra jev") from None
    from ..jev_scorer import MODEL, REPO_ENV, JevScorer, resolve_key
    started = time.time()
    records, data_sha256 = read_jsonl(a.data)[: a.limit], sha256_file(a.data)
    scorer = JevScorer(resolve_key(Path(a.env_file) if a.env_file else REPO_ENV), model=a.model or MODEL)
    rows, requests, unscorable = _run(scorer, records, "jev", a, data_sha256)
    write_jsonl(f"{a.out}.requests.jsonl", requests)
    meta = {"predictor": "jev", "model": scorer.model, "records": len(records), "data_sha256": data_sha256,
            "unscorable": unscorable, **scorer.accounting(requests)}
    finish(a.out, rows, meta, started)
