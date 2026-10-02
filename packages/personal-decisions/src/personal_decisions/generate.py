"""`generate`: one shard -> rendered, anchor-verified records in the Kev labelled-request format (plus `meta`).

State i of (shard, seed) is deterministic: its user (a seeded order over the shard's users), primary family
(round robin, or a seeded weighted draw with `family_weights`), extra families (uniform, or weighted without
replacement), render style (seeded shuffled blocks over the style files), variants, distractors and memory are drawn
from RNGs seeded by (shard, seed, i), and
with `balance` (default) its hard labels follow the per-family label schedule of balance.py, consumed in state order.
Resumable: records are appended to OUT and every attempt is logged to OUT.log.jsonl; a rerun skips ids that are
already in OUT or were dropped/skipped before. Rendering failures (API down) stop the run without marking the
ids, so a rerun retries them. At the end OUT is rewritten sorted by index (first N kept), and OUT.stats.json and
OUT.sha256 are written.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from . import anchors
from .balance import LabelSchedule
from .compose import Draft, compose
from .distractors import insert_documents, pick_documents
from .families import FAMILY_NAMES
from .families.shard import ShardData, load_shard
from .jsonl import dumps, read_jsonl, seeded_rng
from .render.style import load_style
from .stats import finalize


@dataclass
class Options:
    n: int
    seed: int
    families: tuple[str, ...] = FAMILY_NAMES
    questions_per_state: int = 3
    p_buried: float = 0.25
    concurrency: int = 4
    max_attempts_factor: int = 4
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    balance: bool = True  # exact per-family label / position balance (balance.py)
    # Oversampling, e.g. {"pick_option": 3.0}: a family that builds on few attempts is drawn more often, both as the
    # primary family and among the extras. None keeps the plain round robin and uniform extras.
    family_weights: dict[str, float] | None = None


def record_id(shard: str, seed: int, index: int) -> str:
    return f"muse/{shard}/{seed}/{index}"


def plan(shard: ShardData, options: Options, index: int) -> tuple:
    """(user, visit, family list) for state `index`."""
    order = sorted(shard.users, key=lambda user: user.user_id)
    seeded_rng("order", shard.name, options.seed).shuffle(order)
    user = order[index % len(order)]
    visit = index // len(order)
    count = min(len(options.families) - 1, max(0, options.questions_per_state - 1))
    rng = seeded_rng("extras", shard.name, options.seed, index)
    if options.family_weights:
        weights = [options.family_weights.get(family, 1.0) for family in options.families]
        primary = seeded_rng("primary", shard.name, options.seed, index).choices(options.families, weights)[0]
        # Weighted sampling without replacement (Efraimidis-Spirakis keys u ** (1 / w), largest first).
        keyed = sorted(((rng.random() ** (1 / weight), family) for family, weight in zip(options.families, weights)
                        if family != primary and weight > 0), reverse=True)
        return user, visit, [primary, *(family for _, family in keyed[:count])]
    offset = seeded_rng("families", shard.name, options.seed).randrange(len(options.families))
    primary = options.families[(index + offset) % len(options.families)]
    others = [family for family in options.families if family != primary]
    return user, visit, [primary, *rng.sample(others, count)]


def style_for(styles: list[dict], shard_name: str, seed: int, index: int) -> dict:
    """Seeded shuffled blocks over the styles: every block of len(styles) consecutive states uses each style once."""
    block = list(range(len(styles)))
    seeded_rng("style", shard_name, seed, index // len(styles)).shuffle(block)
    return styles[block[index % len(styles)]]


def draft_state(shard: ShardData, options: Options, index: int, schedule: LabelSchedule | None = None) -> Draft | None:
    user, visit, families = plan(shard, options, index)
    draft = compose(shard, user, seed=options.seed, index=index, visit=visit, families=families, schedule=schedule)
    if draft is None:
        return None
    rng = seeded_rng("buried", shard.name, options.seed, index)
    if rng.random() < options.p_buried:
        draft.documents = pick_documents(shard, draft, rng)
    return draft


def to_record(shard: ShardData, options: Options, index: int, draft: Draft, text: str, rendered, style: dict) -> dict:
    questions, anchor_meta, option_features = {}, {}, {}
    for qid, question in draft.questions.items():
        entry = {"type": question.type, "instructions": question.instructions, "criteria": question.criteria,
                 "label": question.label, "src": f"muse/{question.family}"}
        if question.soft_label:
            entry["soft_label"] = question.soft_label
        questions[qid] = entry
        anchor_meta[qid] = {"required": list(question.required), "forbidden": list(question.forbidden)}
        if question.option_features:
            option_features[qid] = question.option_features
    meta = {"user_id": draft.user_id, "shard": shard.name, "state_id": draft.state_id,
            "families": {qid: question.family for qid, question in draft.questions.items()},
            "anchors": anchor_meta,
            "render": {"model": rendered.model, "style_id": style["style_id"], "prompt_sha256": rendered.prompt_sha256},
            "variants": draft.variants()}
    if option_features:
        meta["option_features"] = option_features
    return {"id": record_id(shard.name, options.seed, index), "state": text, "questions": questions, "meta": meta}


def render_and_verify(renderer, draft: Draft, style: dict, index: int, options: Options, shard_name: str):
    required = [anchor for question in draft.questions.values() for anchor in question.required]
    forbidden = [anchor for question in draft.questions.values() for anchor in question.forbidden]
    rendered = renderer.render(draft.structured, required, style)
    text = rendered.text
    if draft.documents:
        text = insert_documents(text, draft.documents, seeded_rng("insert", shard_name, options.seed, index))
    reason = anchors.check(text, required, forbidden) if text.strip() else "empty_render"
    return rendered, text, reason


def previous_attempts(out: Path, log: Path) -> tuple[set[str], set[str]]:
    kept = {row["id"] for row in read_jsonl(out)} if out.is_file() else set()
    attempted = {row["id"] for row in read_jsonl(log) if row["status"] != "kept"} if log.is_file() else set()
    return kept, attempted


def run(shard_dir: str | Path, out: str | Path, options: Options, renderer, style_file=None) -> dict:
    """`style_file`: one style spec path, or a list of them rotated over the states (see style_for)."""
    shard = load_shard(shard_dir)
    if not shard.users:
        raise ValueError(f"{shard_dir}: shard has no users")
    unknown = set(options.families) - set(FAMILY_NAMES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}")
    files = list(style_file) if isinstance(style_file, (list, tuple)) else [style_file]
    styles = [load_style(path) for path in files]
    if len({style["style_id"] for style in styles}) != len(styles):
        raise ValueError("style files must have distinct style ids")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_name(out.name + ".log.jsonl")
    kept_ids, attempted = previous_attempts(out, log)
    kept, index = len(kept_ids), 0
    limit = options.n * options.max_attempts_factor
    schedule = LabelSchedule(shard.name, options.seed) if options.balance else None
    window = max(1, options.concurrency) * 2
    in_flight: dict = {}  # future -> (index, draft, style); a sliding window, so one slow render never stalls others

    def next_draft():
        nonlocal index
        while index < limit:
            current, index = index, index + 1
            rid = record_id(shard.name, options.seed, current)
            if rid in kept_ids or rid in attempted:
                if schedule is not None:
                    draft_state(shard, options, current, schedule)  # replay: keeps later states identical
                continue
            draft = draft_state(shard, options, current, schedule)
            if draft is None:
                append(log, {"id": rid, "status": "skipped", "reason": "no_compatible_fragment"})
                continue
            return current, draft, style_for(styles, shard.name, options.seed, current)
        return None

    def settle(future) -> None:
        nonlocal kept
        current, draft, style = in_flight.pop(future)
        rendered, text, reason = future.result()
        kept += settle_attempt(out, log, shard, options, current, draft, style, rendered, text, reason)

    with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool:
        try:
            while True:
                # Only start new states while the kept records plus the ones in flight can still fall short of N.
                while len(in_flight) < window and kept + len(in_flight) < options.n:
                    item = next_draft()
                    if item is None:
                        break
                    current, draft, style = item
                    future = pool.submit(render_and_verify, renderer, draft, style, current, options, shard.name)
                    in_flight[future] = item
                if not in_flight:
                    break
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: in_flight[f][0]):
                    settle(future)
        except BaseException:
            for future in in_flight:
                future.cancel()  # an API failure stops the run; unfinished ids are retried on the next run
            raise
    return finalize(out, log, options, shard.name, styles, renderer_name=getattr(renderer, "model", "?"))


def settle_attempt(out: Path, log: Path, shard: ShardData, options: Options, index: int, draft: Draft, style: dict,
                   rendered, text: str, reason: str | None) -> bool:
    """Log one verified render; append its record when it passed. Shared by `run` and external.ingest."""
    entry = {"id": record_id(shard.name, options.seed, index), "status": "dropped" if reason else "kept",
             "reason": reason, "usage": rendered.usage, "families": sorted(draft.questions), "model": rendered.model,
             "style_id": style["style_id"]}
    if reason is None:
        append(out, to_record(shard, options, index, draft, text, rendered, style))
    append(log, entry)
    return reason is None


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(row, sort_keys=False) + "\n")  # keeps the criteria (option) order
