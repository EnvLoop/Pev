"""CLI commands for a fresh evaluation population and external renderers: partition, leakage, render-export,
render-ingest. Receipts hold counts, paths and hashes only (never record contents or credentials)."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_named_dirs(values: list[str]) -> dict[str, Path]:
    named = {}
    for value in values:
        name, sep, folder = value.partition("=")
        if not sep or not name or not folder or name in named:
            raise ValueError(f"bad entry {value!r} (expected distinct NAME=DIR)")
        named[name] = Path(folder)
    return named


def generation_options(args):
    from .cli import parse_families, parse_weights
    from .generate import Options
    return Options(n=args.n, seed=args.seed, families=parse_families(args.families),
                   questions_per_state=args.questions_per_state, p_buried=args.p_buried,
                   balance=not args.no_balance, family_weights=parse_weights(args.family_weights))


def cmd_partition(args) -> dict:
    from .jsonl import write_json
    from .split import partition_kb
    manifest = partition_kb(args.kb, parse_named_dirs(args.part), email_pool=args.email_pool)
    if args.manifest:
        write_json(args.manifest, manifest)
    return {"command": "partition", "users": {name: entry["users"] for name, entry in manifest["parts"].items()},
            "per_domain": {name: entry["per_domain"] for name, entry in manifest["parts"].items()},
            "emails": {name: entry["files"].get("emails.jsonl", {}).get("count")
                       for name, entry in manifest["parts"].items()}, "leakage": manifest["leakage"]}


def cmd_leakage(args) -> dict:
    """Collision counts between shards; each shard's users are read in-process and never printed."""
    from . import leakage
    from .jsonl import read_jsonl
    shards = {name: list(read_jsonl(folder / "users.jsonl")) for name, folder in parse_named_dirs(args.shard).items()}
    return {"command": "leakage", "users": {name: len(users) for name, users in shards.items()},
            "leakage": leakage.audit(shards)}


def cmd_render_export(args) -> dict:
    from .external import export, instructions_markdown, write_batches
    from .jsonl import read_jsonl
    receipt = export(args.shard, args.out, generation_options(args), args.style_file, args.requests, args.margin)
    if args.batches:
        exported = list(read_jsonl(args.requests))[-receipt["exported"]:] if receipt["exported"] else []
        receipt.update(write_batches(exported, args.batches, args.n_batches, instructions_markdown(args.renderer_name)))
    return {"command": "render-export", **receipt}


def cmd_render_ingest(args) -> dict:
    from .external import ingest
    receipt = ingest(args.shard, args.out, generation_options(args), args.style_file, args.requests, args.outputs,
                     args.model, drop_missing=args.drop_missing)
    return {"command": "render-ingest", **receipt}


def add_generation_arguments(command: argparse.ArgumentParser) -> None:
    """The arguments that fix the drafts: they must equal those of the export (and of a `generate` run)."""
    from .families import FAMILY_NAMES
    command.add_argument("--shard", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True, help="the records file, as for `generate --out`")
    command.add_argument("--requests", type=Path, required=True, help="the exported requests (JSONL)")
    command.add_argument("--n", type=int, required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--style-file", type=Path, nargs="+", required=True)
    command.add_argument("--family-weights")
    command.add_argument("--families", nargs="+", help=f"subset of: {' '.join(FAMILY_NAMES)}")
    command.add_argument("--questions-per-state", type=int, default=3)
    command.add_argument("--p-buried", type=float, default=0.25)
    command.add_argument("--no-balance", action="store_true")


def add_commands(sub) -> None:
    part = sub.add_parser("partition", help="whole KB -> equal named parts by user id (domain-stratified)")
    part.add_argument("--kb", type=Path, required=True)
    part.add_argument("--part", nargs="+", required=True, help="NAME=DIR, one per part (NAME is the shard name)")
    part.add_argument("--email-pool", help="keep only the emails of this split email shard (train/val/dev/hidden)")
    part.add_argument("--manifest", type=Path)
    part.set_defaults(handler=cmd_partition)
    leak = sub.add_parser("leakage", help="cross-shard collision counts (users, targets, review text, events)")
    leak.add_argument("--shard", nargs="+", required=True, help="NAME=DIR of a KB or shard with users.jsonl")
    leak.set_defaults(handler=cmd_leakage)
    export = sub.add_parser("render-export", help="draft states as `generate` would; write astra render requests")
    add_generation_arguments(export)
    export.add_argument("--margin", type=int, default=0, help="export this many states beyond N")
    export.add_argument("--batches", type=Path, help="also write the new requests as batch-NN/input.jsonl here")
    export.add_argument("--n-batches", type=int, default=8)
    export.add_argument("--renderer-name", default="the renderer", help="who the INSTRUCTIONS.md addresses")
    export.set_defaults(handler=cmd_render_export)
    ingest = sub.add_parser("render-ingest", help="verify externally rendered texts exactly like astra renders")
    add_generation_arguments(ingest)
    ingest.add_argument("--outputs", type=Path, nargs="+", required=True, help="output.jsonl files ({id, text})")
    ingest.add_argument("--model", required=True, help="the renderer model recorded in meta.render.model")
    ingest.add_argument("--drop-missing", action="store_true",
                        help="log ids without a valid output as dropped (missing_output) instead of pending")
    ingest.set_defaults(handler=cmd_render_ingest)
