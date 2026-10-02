"""JSON lines I/O, content hashes and stable seeds shared by every step."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
import hashlib
import json
import os
from pathlib import Path
import random


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON ({error.msg})") from None


def dumps(value, sort_keys: bool = True) -> str:
    """Compact JSON. Records use sort_keys=False: the order of a choice question's criteria is its option order."""
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def write_jsonl(path: str | Path, rows: Iterable[dict], sort_keys: bool = True) -> int:
    """Write rows atomically (temp file + rename); returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps(row, sort_keys) + "\n")
            count += 1
    os.replace(temp, path)
    return count


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_int(*parts) -> int:
    """A 64-bit integer from the parts' text, identical across runs and platforms (unlike hash())."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def seeded_rng(*parts) -> random.Random:
    return random.Random(stable_int(*parts))
