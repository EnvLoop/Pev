"""personal-decisions CLI: population, build, split, capacity, generate, shortcut-baselines, to-kev (plus partition,
leakage, render-export and render-ingest from cli_external).

Every command prints one JSON receipt as its last stdout line (no record contents, never credentials).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import PersonalDecisionsError
from .families import FAMILY_NAMES


def parse_users(text: str) -> dict[str, int]:
    counts = {}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        if name not in ("hidden", "dev", "val") or not value.isdigit():
            raise argparse.ArgumentTypeError(f"bad --users entry {part!r} (expected hidden=N,dev=N,val=N)")
        counts[name] = int(value)
    return counts


def parse_families(values: list[str] | None) -> tuple[str, ...]:
    names = tuple(name for value in values or [] for name in value.split(",") if name) or FAMILY_NAMES
    unknown = set(names) - set(FAMILY_NAMES)
    if unknown:
        raise PersonalDecisionsError(f"unknown families {sorted(unknown)}; choose from {', '.join(FAMILY_NAMES)}")
    return tuple(dict.fromkeys(names))


def parse_weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    weights = {}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        try:
            weights[name] = float(value)
        except ValueError:
            raise PersonalDecisionsError(f"bad --family-weights entry {part!r} (expected family=weight)") from None
        if name not in FAMILY_NAMES or weights[name] <= 0:
            raise PersonalDecisionsError(f"bad --family-weights entry {part!r} (known family, weight > 0)")
    return weights


def parse_domains(text: str) -> dict[str, int]:
    counts = {}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        if name not in ("product", "restaurant") or not value.isdigit():
            raise argparse.ArgumentTypeError(f"bad --per-domain entry {part!r} (expected product=N,restaurant=N)")
        counts[name] = int(value)
    return counts


def cmd_population(args) -> dict:
    from .jsonl import write_json
    from .kb.population import PopulationRule, select_population, write_population
    rule = PopulationRule(seed=args.seed, per_domain=args.per_domain, min_items=args.min_items,
                          max_items=args.max_items, min_text_share=args.min_text_share,
                          require_low_later=not args.allow_no_low_later)
    user_ids, receipt = select_population(args.raw, rule)
    write_population(args.out, user_ids)
    if args.receipt:
        write_json(args.receipt, receipt)
    return {"command": "population", "out": str(args.out), **receipt}


def cmd_build(args) -> dict:
    from .kb.build import build_kb
    prefetcher, prefetch_stats = None, {}
    if args.extractor == "llm":
        from .kb.llm_extract import prefetch
        from .llm import open_client
        client = open_client(args.env_file)
        cache = args.llm_cache or Path(args.out) / "llm_extract_cache.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        deadline = args.llm_deadline_min * 60 if args.llm_deadline_min else None

        def prefetcher(jobs):
            extractor, stats = prefetch(jobs, client, concurrency=args.concurrency, cache_path=cache,
                                        deadline_s=deadline)
            prefetch_stats.update(stats.as_dict(), model=client.model)
            return extractor
    user_ids = None
    if args.users_file:
        from .kb.population import read_population
        user_ids = read_population(args.users_file)
    manifest = build_kb(args.raw, args.out, min_history=args.min_history, memory_fraction=args.memory_fraction,
                        max_users=args.max_users, user_ids=user_ids, prefetcher=prefetcher)
    receipt = {"command": "build", "out": str(args.out), "stats": manifest["stats"],
               "files": {name: entry["count"] for name, entry in manifest["files"].items()}}
    if prefetch_stats:
        receipt["llm_usage"] = prefetch_stats
        manifest["llm_usage"] = prefetch_stats
        from .jsonl import write_json
        write_json(Path(args.out) / "manifest.json", manifest)
    return receipt


def cmd_split(args) -> dict:
    from .split import split_kb
    outputs = {name: getattr(args, name) for name in ("train", "val", "dev", "hidden") if getattr(args, name)}
    manifest = split_kb(args.kb, outputs, args.manifest, args.users)
    return {"command": "split", "users": {name: entry["users"] for name, entry in manifest["shards"].items()},
            "warnings": manifest["warnings"], "leakage": manifest["leakage"],
            "manifest": str(args.manifest) if args.manifest else None}


def cmd_capacity(args) -> dict:
    from .capacity import shard_capacity
    from .families.shard import load_shard
    report = shard_capacity(load_shard(args.shard), trials=args.trials, families=parse_families(args.families))
    if args.out:
        from .jsonl import write_json
        write_json(args.out, report)
    return {"command": "capacity", "shard": str(args.shard), **report}


def cmd_generate(args) -> dict:
    from .generate import Options, run
    from .render import make_renderer
    options = Options(n=args.n, seed=args.seed, families=parse_families(args.families),
                      questions_per_state=args.questions_per_state, p_buried=args.p_buried,
                      concurrency=args.concurrency, price_in_per_mtok=args.price_in_per_mtok,
                      price_out_per_mtok=args.price_out_per_mtok, balance=not args.no_balance,
                      family_weights=parse_weights(args.family_weights))
    renderer = make_renderer(args.renderer, args.env_file, args.model, args.max_retries)
    style_files = args.style_file if len(args.style_file) > 1 else args.style_file[0]
    stats = run(args.shard, args.out, options, renderer, style_files)
    keys = ("states", "questions", "per_family", "drops_by_reason", "tokens", "output_sha256", "cost_usd")
    return {"command": "generate", "out": str(args.out), **{key: stats[key] for key in keys if key in stats}}


def cmd_baselines(args) -> dict:
    from .baselines import shortcut_baselines
    report = shortcut_baselines(args.records)
    if args.out:
        from .jsonl import write_json
        write_json(args.out, report)
    return {"command": "shortcut-baselines", **report}


def cmd_to_kev(args) -> dict:
    from .to_kev import convert
    return {"command": "to-kev", **convert(args.records, args.out, keep_soft=not args.no_soft)}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="personal-decisions", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="raw tables -> knowledge base")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--min-history", type=int, default=5)
    build.add_argument("--memory-fraction", type=float, default=0.6)
    build.add_argument("--max-users", type=int)
    build.add_argument("--extractor", choices=("rule", "llm"), default="rule")
    build.add_argument("--env-file", type=Path)
    build.add_argument("--users-file", type=Path, help="restrict to these user ids (output of `population`)")
    build.add_argument("--concurrency", type=int, default=8, help="parallel LLM extraction requests")
    build.add_argument("--llm-cache", type=Path,
                       help="resumable extraction cache (default OUT/llm_extract_cache.jsonl)")
    build.add_argument("--llm-deadline-min", type=float, help="after this, remaining users use the rule extractor")
    build.set_defaults(handler=cmd_build)
    pop = sub.add_parser("population", help="raw tables -> deterministic list of KB user ids")
    pop.add_argument("--raw", type=Path, required=True)
    pop.add_argument("--out", type=Path, required=True, help="user id list, one per line")
    pop.add_argument("--receipt", type=Path, help="write the rule and counts (no ids) here")
    pop.add_argument("--seed", type=int, required=True)
    pop.add_argument("--per-domain", type=parse_domains, required=True, help="product=N,restaurant=N")
    pop.add_argument("--min-items", type=int, default=8)
    pop.add_argument("--max-items", type=int, default=30)
    pop.add_argument("--min-text-share", type=float, default=0.8)
    pop.add_argument("--allow-no-low-later", action="store_true", help="do not require a later rating <= 2")
    pop.set_defaults(handler=cmd_population)
    split = sub.add_parser("split", help="knowledge base -> disjoint shards by user id")
    split.add_argument("--kb", type=Path, required=True)
    for name in ("train", "val", "dev", "hidden"):
        split.add_argument(f"--{name}", type=Path, help=f"output directory of the {name} shard")
    split.add_argument("--manifest", type=Path, help="write the contents-free split manifest here")
    split.add_argument("--users", type=parse_users, default={}, help="hidden=200,dev=100,val=150 (train = rest)")
    split.set_defaults(handler=cmd_split)
    gen = sub.add_parser("generate", help="shard -> rendered, anchor-verified records")
    gen.add_argument("--shard", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--n", type=int, required=True, help="number of states (records) to keep")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--style-file", type=Path, nargs="+", required=True,
                     help="one style spec, or several rotated over the states in seeded shuffled blocks")
    gen.add_argument("--family-weights", help="oversampling, e.g. pick_option=3 (others default to 1)")
    gen.add_argument("--families", nargs="+", help=f"subset of: {' '.join(FAMILY_NAMES)}")
    gen.add_argument("--renderer", choices=("astra", "fixture"), default="astra")
    gen.add_argument("--concurrency", type=int, default=4)
    gen.add_argument("--questions-per-state", type=int, default=3)
    gen.add_argument("--p-buried", type=float, default=0.25)
    gen.add_argument("--env-file", type=Path, help="default: the repository .env")
    gen.add_argument("--model", help="override RENDER_MODEL")
    gen.add_argument("--max-retries", type=int, help="attempts per render on rate limits / 5xx (default 5; "
                     "jittered exponential backoff capped at 60 s); raise it on a shared, rate-limited API endpoint")
    gen.add_argument("--price-in-per-mtok", type=float)
    gen.add_argument("--price-out-per-mtok", type=float)
    gen.add_argument("--no-balance", action="store_true", help="do not stratify labels / answer positions")
    gen.set_defaults(handler=cmd_generate)
    cap = sub.add_parser("capacity", help="questions each family can supply from a shard (counts only)")
    cap.add_argument("--shard", type=Path, required=True)
    cap.add_argument("--trials", type=int, default=6)
    cap.add_argument("--families", nargs="+", help=f"subset of: {' '.join(FAMILY_NAMES)}")
    cap.add_argument("--out", type=Path)
    cap.set_defaults(handler=cmd_capacity)
    base = sub.add_parser("shortcut-baselines", help="accuracy of trivial predictors per family")
    base.add_argument("--records", type=Path, required=True)
    base.add_argument("--out", type=Path)
    base.set_defaults(handler=cmd_baselines)
    kev = sub.add_parser("to-kev", help="records -> Kev labelled requests (soft_label -> target, meta -> _meta)")
    kev.add_argument("--records", type=Path, required=True)
    kev.add_argument("--out", type=Path, required=True)
    kev.add_argument("--no-soft", action="store_true", help="drop soft labels instead of converting to target")
    kev.set_defaults(handler=cmd_to_kev)
    from .cli_external import add_commands
    add_commands(sub)
    from .cli_release import add_commands as add_release_commands
    add_release_commands(sub)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        receipt = args.handler(args)
    except (PersonalDecisionsError, ValueError, FileNotFoundError) as error:
        print(json.dumps({"command": args.command, "ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **receipt}, sort_keys=True))
    return 0
