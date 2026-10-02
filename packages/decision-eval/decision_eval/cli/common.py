"""Shared command-line helpers: JSON output, input hashes, temperature / threshold arguments."""
import json
import math
from pathlib import Path

from ..records import sha256_file


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inputs(**paths):
    """{role: {"path", "sha256"}} for every given input file."""
    return {role: {"path": str(path), "sha256": sha256_file(path)} for role, path in paths.items() if path is not None}


def temperature_arg(value):
    """A literal temperature, or a temperature.json written by `calibrate` -> (temperature, source sha256 or None)."""
    path = Path(value)
    if path.is_file():
        temperature = float(read_json(path)["temperature"])
        source = sha256_file(path)
    else:
        try:
            temperature, source = float(value), None
        except ValueError:
            raise SystemExit(f"--temperature must be a number or a temperature.json path, got {value!r}") from None
    if not math.isfinite(temperature) or temperature <= 0:
        raise SystemExit("temperature must be finite and positive")
    return temperature, source


def threshold_file(path, temperature):
    """The threshold from a `thresholds` output, refusing one fitted at a different temperature."""
    data = read_json(path)
    if abs(float(data["temperature"]) - temperature) > 1e-9:
        raise SystemExit(f"{path} was fitted at temperature {data['temperature']}, not {temperature}")
    return None if data["threshold"] is None else float(data["threshold"])


SEALED_PARTS = ("sealed", "hidden")


def refuse_sealed(*paths):
    """DEV-only commands: refuse any path with a component naming the sealed / hidden set."""
    for path in paths:
        if path is not None and any(part in name.lower() for name in Path(path).resolve().parts
                                    for part in SEALED_PARTS):
            raise SystemExit("this command runs on VAL/DEV only; a path names the sealed/hidden set")


def print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
