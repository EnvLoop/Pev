"""CLI commands for the public release: privacy-scan (aggregate PII counts) and export-release (anonymization).
Receipts hold counts, paths and hashes only; examples, when asked for, are redacted (first letter of each word)."""
from __future__ import annotations

from pathlib import Path


def cmd_privacy_scan(args) -> dict:
    from .jsonl import write_json
    from .release.scan import privacy_scan
    report = privacy_scan(args.records or [], args.kb or [], args.items, args.examples, args.counts_only or [])
    if args.out:
        write_json(args.out, report)
    return {"command": "privacy-scan", "out": str(args.out) if args.out else None,
            "files": len(report["records"]) + len(report["kb"])}


def cmd_export_release(args) -> dict:
    from .release.anonymize import load_key
    from .release.export import export_release
    key = load_key(args.env_file, args.key_file)
    receipt = export_release(args.records, args.out, key, args.items, args.keep_foreign)
    if args.receipt:
        from .jsonl import write_json
        write_json(args.receipt, receipt)
    return {"command": "export-release", **receipt}


def add_commands(sub) -> None:
    scan = sub.add_parser("privacy-scan", help="PII counts over records / KB shards (release audit)")
    scan.add_argument("--records", type=Path, nargs="+", help="records JSONL files")
    scan.add_argument("--kb", type=Path, nargs="+", help="KB or shard directories (users.jsonl, emails.jsonl)")
    scan.add_argument("--items", type=Path, help="item catalog: names in it are brands, not people")
    scan.add_argument("--examples", type=int, default=0, help="redacted examples per kind (0 = counts only)")
    scan.add_argument("--counts-only", type=Path, nargs="+", help="inputs that never get examples (DEV, TEST)")
    scan.add_argument("--out", type=Path)
    scan.set_defaults(handler=cmd_privacy_scan)
    export = sub.add_parser("export-release", help="records -> pseudonymized public records (keyed, deterministic)")
    export.add_argument("--records", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--items", type=Path, help="item catalog (brand names are not pseudonymized)")
    export.add_argument("--key-file", type=Path, help="the published pseudonym key (release/pseudonym_key.txt)")
    export.add_argument("--env-file", type=Path, help="else RELEASE_PSEUDONYM_KEY from here (default: repository .env)")
    export.add_argument("--keep-foreign", action="store_true", help="pass Kev-format rows (no meta) through unchanged")
    export.add_argument("--receipt", type=Path, help="also write the receipt here")
    export.set_defaults(handler=cmd_export_release)
