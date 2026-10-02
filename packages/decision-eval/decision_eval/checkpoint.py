"""Checkpointed, resumable per-record scoring for the hosted-API references (predict-openai, predict-jev).

Every finished record is appended at once to `<out>.partial.jsonl` (first line: the data file's SHA-256), so an
interrupted run resumes where it stopped and never pays twice for a record. Transient failures (HTTP 408/409/429/5xx,
connection errors, timeouts) are retried with exponential backoff after the SDK's own retries; a record that still
fails is reported and left out of the checkpoint, and the run stops without writing final predictions until a rerun
completes every record.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .records import record_id

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
BACKOFF_SECONDS = (5, 15, 30, 60, 120)


def transient(error):
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUS
    name = type(error).__name__
    return isinstance(error, (ConnectionError, TimeoutError)) or "Timeout" in name or "Connection" in name


def describe(error):
    """A failure description without request bodies or credentials: the type and HTTP status only."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    return f"{type(error).__name__}" + (f" (HTTP {status})" if isinstance(status, int) else "")


class Checkpoint:
    def __init__(self, path, data_sha256):
        self.path, self.lock, self.done = Path(path), threading.Lock(), {}
        if self.path.is_file():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            header = json.loads(lines[0]) if lines else {}
            if header.get("data_sha256") != data_sha256:
                raise SystemExit(f"{self.path} was written for different data; move it away to start over")
            for line in lines[1:]:
                if line.strip():
                    entry = json.loads(line)
                    self.done[entry["id"]] = entry
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"data_sha256": data_sha256}) + "\n", encoding="utf-8")

    def add(self, rid, rows, request):
        entry = {"id": rid, "rows": [[list(key), list(keys), list(z)] for key, keys, z in rows], "request": request}
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self.done[rid] = entry


def _request_of(scorer, rid):
    for request in reversed(getattr(scorer, "requests", [])):
        if request.get("id") == rid:
            return request
    return None


def score_records(scorer, records, checkpoint, concurrency, backoff=BACKOFF_SECONDS, sleep=time.sleep, log=None):
    """-> (rows [(key, keys, logits)] in record order, requests of every scored record, failures {id: reason})."""
    failures, stop = {}, threading.Event()

    def one(record):
        rid = record_id(record)
        if rid in checkpoint.done or stop.is_set():
            return
        for attempt in range(len(backoff) + 1):
            try:
                rows = scorer.record_logits(record)
                break
            except Exception as error:      # noqa: BLE001 - classified below; nothing else escapes a worker
                if not transient(error) or attempt == len(backoff):
                    failures[rid] = describe(error)
                    if transient(error):
                        stop.set()          # persistent rate limiting / outage: stop, resume later
                    return
                if log:
                    log(f"{rid}: {describe(error)}; retry in {backoff[attempt]} s")
                sleep(backoff[attempt])
        checkpoint.add(rid, rows, _request_of(scorer, rid))
        if log and len(checkpoint.done) % 25 == 0:
            log(f"{len(checkpoint.done)}/{len(records)} records scored")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one, records))
    rows, requests = [], []
    for record in records:
        entry = checkpoint.done.get(record_id(record))
        if entry is None:
            continue
        rows += [(tuple(key), tuple(keys), z) for key, keys, z in entry["rows"]]
        if entry.get("request") is not None:
            requests.append(entry["request"])
    missing = [record_id(r) for r in records if record_id(r) not in checkpoint.done and record_id(r) not in failures]
    failures.update({rid: "not attempted (stopped after a persistent transient failure)" for rid in missing})
    return rows, requests, failures
