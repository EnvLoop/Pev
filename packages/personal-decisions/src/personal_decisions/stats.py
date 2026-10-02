"""Finalize a generation run: sorted output, SHA-256 sidecar, stats/cost sidecar (from the attempt log)."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .jsonl import file_sha256, read_jsonl, write_json, write_jsonl


def index_of(record_id: str) -> int:
    return int(record_id.rsplit("/", 1)[1])


def label_key(label) -> str:
    return str(label).lower() if isinstance(label, bool) else str(label)


def record_stats(records: list[dict]) -> dict:
    per_family = Counter()
    variants: dict[str, Counter] = defaultdict(Counter)
    labels: dict[str, Counter] = defaultdict(Counter)
    soft, styles = Counter(), Counter()
    for record in records:
        styles[record["meta"].get("render", {}).get("style_id")] += 1
        for qid, question in record["questions"].items():
            family = record["meta"]["families"][qid]
            per_family[family] += 1
            variants[family][record["meta"].get("variants", {}).get(qid, "clean")] += 1
            if question["type"] != "choice":
                labels[family][label_key(question["label"])] += 1
            if "soft_label" in question:
                soft[family] += 1
    return {"states": len(records), "questions": sum(per_family.values()), "per_family": dict(per_family),
            "variants": {family: dict(counts) for family, counts in variants.items()},
            "label_counts": {family: dict(counts) for family, counts in labels.items()},
            "soft_labelled": dict(soft), "style_counts": dict(styles)}


def log_stats(log: Path) -> dict:
    tokens = Counter()
    status, drops = Counter(), Counter()
    requests = 0
    for row in read_jsonl(log) if log.is_file() else []:
        status[row["status"]] += 1
        if row["status"] == "dropped":
            drops[row.get("reason") or "unknown"] += 1
        if row["status"] == "skipped":
            drops["skipped:" + (row.get("reason") or "unknown")] += 1
        usage = row.get("usage") or {}
        if row["status"] != "skipped":
            requests += 1
        for key in ("input_tokens", "output_tokens"):
            tokens[key] += int(usage.get(key, 0) or 0)
    return {"attempts": dict(status), "drops_by_reason": dict(drops), "render_requests": requests,
            "tokens": dict(tokens)}


def style_fields(styles) -> dict:
    styles = [styles] if isinstance(styles, dict) else list(styles)
    if len(styles) == 1:
        return {"style_id": styles[0]["style_id"]}
    return {"style_ids": [style["style_id"] for style in styles]}


def finalize(out: Path, log: Path, options, shard_name: str, styles, renderer_name: str) -> dict:
    records = sorted(read_jsonl(out), key=lambda record: index_of(record["id"])) if out.is_file() else []
    unique = list({record["id"]: record for record in records}.values())[:options.n]
    write_jsonl(out, unique, sort_keys=False)
    sha = file_sha256(out)
    out.with_name(out.name + ".sha256").write_text(f"{sha}  {out.name}\n", encoding="utf-8")
    logged = log_stats(log)
    stats = {"shard": shard_name, "seed": options.seed, "n_requested": options.n, "renderer": renderer_name,
             **style_fields(styles), "families": list(options.families), "p_buried": options.p_buried,
             "family_weights": getattr(options, "family_weights", None),
             "output_sha256": sha, **record_stats(unique), **logged}
    if options.price_in_per_mtok is not None and options.price_out_per_mtok is not None:
        tokens = logged["tokens"]
        stats["cost_usd"] = round(tokens.get("input_tokens", 0) / 1e6 * options.price_in_per_mtok
                                  + tokens.get("output_tokens", 0) / 1e6 * options.price_out_per_mtok, 4)
    write_json(out.with_name(out.name + ".stats.json"), stats)
    return stats
