"""Round 2 data: mix general decision records from Kev's public training material into TRAIN (anti-forgetting).

Source: kev@0fe8fc97 `evals/decision-v2/train.jsonl`, the train split of the suite whose test split is the decision-v7
guard (`work/muse/eval-dev/guard.jsonl` is byte-identical to `evals/decision-v2/test.jsonl`). No test, development or
calibration file is read as a source. Steps:

1. leakage audit (counts only): every candidate state against the guard, DEV and VAL states by exact text, normalized
   text, and word 8-gram overlap >= 0.8 (shared 8-grams over either side's 8-grams); flagged candidates are dropped;
2. source balance: the general rows (one per question) are split evenly over Kev's families (`src`), water-filling
   when a family has fewer questions, sized so general rows are GENERAL_SHARE of the final SFT rows;
3. prompts over MAX_PROMPT_TOKENS under the `direct` template are dropped; labels are checked single-token by to-sft;
4. TRAIN + general mix, shuffled with SEED, plus the to-sft rows, their sidecar and a receipt.

    uv run --project packages/decision-eval python scripts/general_mix.py
"""
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from decision_eval.cli.sft import to_sft_main
from decision_eval.conventions import render
from decision_eval.prompts import pinned_tokenizer
from decision_eval.records import question_of, read_jsonl, record_id, write_jsonl
from decision_eval.sft import sft_rows

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/muse"
KEV = ROOT / "work/upstream/kev"
SOURCE = KEV / "evals/decision-v2/train.jsonl"
TRAIN = WORK / "data/train.v2-5k.jsonl"
TARGETS = {"guard": WORK / "eval-dev/guard.jsonl", "dev": WORK / "eval-dev/dev.jsonl", "val": WORK / "data/val.jsonl"}
GENERAL, MIX = WORK / "data/general-mix.jsonl", WORK / "data/train.r2-mix.jsonl"
SEED, GENERAL_SHARE, MAX_PROMPT_TOKENS, NGRAM, OVERLAP = 20261004, 0.30, 12288, 8, 0.8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def shingles(normalized):
    words = normalized.split()
    if len(words) < NGRAM:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


class TargetIndex:
    """Exact, normalized and 8-gram views of one target set's states."""

    def __init__(self, states):
        self.exact = set(states)
        normalized = [normalize(s) for s in states]
        self.normalized = set(normalized)
        self.sizes, self.postings = [], defaultdict(set)
        for i, text in enumerate(normalized):
            grams = shingles(text)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings[gram].add(i)

    def match(self, state):
        """-> the strongest kind of match ("exact" > "normalized" > "ngram") or None."""
        if state in self.exact:
            return "exact"
        text = normalize(state)
        if text in self.normalized:
            return "normalized"
        grams = shingles(text)
        shared = defaultdict(int)
        for gram in grams:
            for i in self.postings.get(gram, ()):
                shared[i] += 1
        for i, count in shared.items():
            if count / max(len(grams), 1) >= OVERLAP or count / max(self.sizes[i], 1) >= OVERLAP:
                return "ngram"
        return None


def audit(records, indexes):
    """-> ({target: {kind: count}}, set of flagged record ids)."""
    counts = {name: {"exact": 0, "normalized": 0, "ngram": 0} for name in indexes}
    flagged = set()
    for record in records:
        state = render(record["state"])
        for name, index in indexes.items():
            kind = index.match(state)
            if kind:
                counts[name][kind] += 1
                flagged.add(record_id(record))
    return counts, flagged


def quotas(available, total):
    """Even split of `total` over families, water-filling families with fewer than their share."""
    out, left, families = {}, total, sorted(available, key=lambda f: (available[f], f))
    for i, family in enumerate(families):
        share = left // (len(families) - i)
        out[family] = min(available[family], share)
        left -= out[family]
    return out


def select(records, flagged, total, rng):
    """-> (general records holding only the chosen questions, quotas, available per family)."""
    units = defaultdict(list)
    for record in records:
        if record_id(record) in flagged:
            continue
        for qid in record["questions"]:
            units[question_of(record, qid).family].append((record_id(record), qid))
    available = {family: len(items) for family, items in units.items()}
    chosen_quota, chosen = quotas(available, total), set()
    for family in sorted(units):
        items = sorted(units[family])
        rng.shuffle(items)
        chosen |= set(items[:chosen_quota[family]])
    out = []
    for record in records:
        keep = {qid: q for qid, q in record["questions"].items() if (record_id(record), qid) in chosen}
        if keep:
            out.append({**record, "questions": keep})
    return out, chosen_quota, available


def drop_long(records, tok):
    """Drop questions whose direct-template prompt exceeds MAX_PROMPT_TOKENS (labels verified single-token here)."""
    rows, _ = sft_rows(records, "direct", tok)
    too_long = {(r["meta"]["id"], r["meta"]["qid"]) for r in rows if r["meta"]["prompt_tokens"] > MAX_PROMPT_TOKENS}
    out = []
    for record in records:
        keep = {qid: q for qid, q in record["questions"].items() if (record_id(record), qid) not in too_long}
        if keep:
            out.append({**record, "questions": keep})
    return out, len(too_long), max(r["meta"]["prompt_tokens"] for r in rows)


def main():
    rng, tok = random.Random(SEED), pinned_tokenizer()
    source, train = read_jsonl(SOURCE), read_jsonl(TRAIN)
    indexes = {name: TargetIndex([render(r["state"]) for r in read_jsonl(path)]) for name, path in TARGETS.items()}
    before, flagged = audit(source, indexes)
    train_rows = json.loads((WORK / "data/train.v2-5k.sft.direct.jsonl.meta.json").read_text())["rows"]
    total = round(GENERAL_SHARE / (1 - GENERAL_SHARE) * train_rows)
    general, quota, available = select(source, flagged, total, rng)
    general, dropped_long, max_tokens = drop_long(general, tok)
    after, _ = audit(general, indexes)
    if any(v for counts in after.values() for v in counts.values()):
        raise SystemExit("general mix still collides with a target set")
    write_jsonl(GENERAL, general)
    mix = train + general
    random.Random(SEED).shuffle(mix)
    write_jsonl(MIX, mix)
    sft_out = WORK / "data/train.r2-mix.sft.direct.jsonl"
    to_sft_main(["--data", str(MIX), "--template", "direct", "--out", str(sft_out)])
    sft_meta = json.loads(Path(f"{sft_out}.meta.json").read_text())
    general_questions = sum(len(r["questions"]) for r in general)
    receipt = {
        "schema": "muse-general-mix-receipt/1", "seed": SEED, "general_share_target": GENERAL_SHARE,
        "source": {"repo": "jaredpalmer/kev", "commit": "0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91",
                   "file": "evals/decision-v2/train.jsonl", "sha256": sha256(SOURCE), "records": len(source),
                   "note": "train split of the suite whose test split is the decision-v7 guard"},
        "leakage_audit": {"method": f"exact state, normalized text, word {NGRAM}-gram overlap >= {OVERLAP} either side",
                          "targets_sha256": {name: sha256(path) for name, path in TARGETS.items()},
                          "source_matches": before, "source_records_dropped": len(flagged),
                          "general_mix_matches": after},
        "families": {f: {"available": available[f], "selected": quota[f]} for f in sorted(quota)},
        "dropped_over_max_prompt_tokens": dropped_long, "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "general_max_prompt_tokens": max_tokens,
        "general_mix": {"path": str(GENERAL.relative_to(ROOT)), "sha256": sha256(GENERAL), "records": len(general),
                        "questions": general_questions},
        "train": {"path": str(TRAIN.relative_to(ROOT)), "sha256": sha256(TRAIN), "records": len(train),
                  "sft_rows": train_rows},
        "mix": {"path": str(MIX.relative_to(ROOT)), "sha256": sha256(MIX), "records": len(mix),
                "sft": {"path": str(sft_out.relative_to(ROOT)), **{k: sft_meta[k] for k in
                        ("sha256", "rows", "ambiguous_excluded", "max_prompt_tokens", "template_sha256")}},
                "general_row_share": general_questions / sft_meta["rows"]},
    }
    (WORK / "data/GEN_RECEIPT.r2-mix.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("leakage_audit", "families", "general_mix", "mix")}, indent=2))


if __name__ == "__main__":
    main()
