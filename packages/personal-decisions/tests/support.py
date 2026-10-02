"""Shared test setup: synthetic raw tables -> KB -> shards in a temporary directory (built once per process)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile

from personal_decisions.families.shard import load_shard
from personal_decisions.kb.build import build_kb
from personal_decisions.split import split_kb

from .fixture_tables import write_raw

SPLIT_USERS = {"hidden": 4, "dev": 4, "val": 4}


@lru_cache(maxsize=1)
def workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="personal-decisions-test-"))
    write_raw(root / "raw", users=40)
    build_kb(root / "raw", root / "kb")
    split_kb(root / "kb", {name: root / name for name in ("train", "val", "dev", "hidden")}, root / "split.json",
             SPLIT_USERS)
    return root


@lru_cache(maxsize=4)
def shard(name: str = "train"):
    return load_shard(workspace() / name)
