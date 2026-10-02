"""Sample ~20k Enron emails (seeded, deterministic) as scrubbed distractor text -> emails.jsonl."""
import hashlib

import pyarrow.parquet as pq

from common import DL, EMAIL_BODY_MAX, RAW, clip, scrub, seeded_key, write_jsonl

TARGET = 20_000


def main() -> None:
    files = sorted((DL / "enron").glob("*.parquet"))
    cands, seen = [], set()
    for f in files:
        t = pq.read_table(f, columns=["message_id", "subject", "date", "body"]).to_pylist()
        for r in t:
            body = (r["body"] or "").strip()
            if len(body) < 80:
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()
            if digest in seen:  # the corpus holds many copies of the same message across folders
                continue
            seen.add(digest)
            cands.append((seeded_key(r["message_id"]), r))
    cands.sort(key=lambda x: x[0])
    out = []
    for i, (_, r) in enumerate(cands[:TARGET]):
        out.append({
            "email_id": f"enron-{hashlib.sha256(r['message_id'].encode()).hexdigest()[:16]}",
            "date": r["date"].strftime("%Y-%m-%dT%H:%M:%SZ") if r["date"] else None,
            "subject": clip(scrub(r["subject"] or ""), 300),
            "body": clip(scrub(r["body"]), EMAIL_BODY_MAX),
        })
    out.sort(key=lambda r: r["email_id"])
    print("unique candidates", len(cands), "written", write_jsonl(RAW / "emails.jsonl", out))


if __name__ == "__main__":
    main()
