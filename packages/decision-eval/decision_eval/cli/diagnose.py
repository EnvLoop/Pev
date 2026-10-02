"""noise-floor, validity and export-hillclimb: the eval-design checks and hillclimb plumbing of amendment 4."""
import argparse

from ..hillclimb import export
from ..noise import noise_floor, seed_spread
from ..predictions import load_aligned
from ..records import load_questions, read_jsonl, sha256_file
from ..stats import SEED
from ..validity import PERMUTATIONS, validity
from .common import inputs, print_json, refuse_sealed, temperature_arg, write_json


def _finish(result, out):
    if out:
        write_json(out, result)
    print_json(result)


def noise_floor_main(argv=None):
    ap = argparse.ArgumentParser(prog="noise-floor", description="CI half-widths at this n, or build variance")
    ap.add_argument("--data", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preds", help="one predictor: per-family and macro noise floor next to headroom")
    mode.add_argument("--seed-spread", nargs=2, metavar=("FIRST", "SECOND"),
                      help="two candidates from identically configured trainings: their DEV deltas")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    refuse_sealed(a.data)
    questions = load_questions(a.data)
    if a.preds:
        result = {"mode": "noise-floor", **noise_floor(questions, load_aligned(questions, a.preds)[0], a.reps, a.seed),
                  "inputs": inputs(data=a.data, preds=a.preds)}
    else:
        first, second = (load_aligned(questions, path)[0] for path in a.seed_spread)
        result = {"mode": "seed-spread", **seed_spread(questions, first, second, a.reps, a.seed),
                  "inputs": inputs(data=a.data, first=a.seed_spread[0], second=a.seed_spread[1])}
    _finish(result, a.out)


def _ladder_arg(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("expected name=PREDICTIONS")
    return name, path


def validity_main(argv=None):
    ap = argparse.ArgumentParser(prog="validity", description="oracle, null baselines and capability ladder checks")
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds-ladder", nargs="*", type=_ladder_arg, default=[], metavar="NAME=P",
                    help="predictors from weakest to strongest")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS, help="shuffled-label permutations (>= 20)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.permutations < 20:
        raise SystemExit("--permutations must be at least 20")
    refuse_sealed(a.data, *(path for _, path in a.preds_ladder))
    questions = load_questions(a.data)
    named = [(name, load_aligned(questions, path)[0]) for name, path in a.preds_ladder]
    result = {**validity(questions, named, a.samples, a.seed, a.permutations),
              "inputs": inputs(data=a.data, **{f"ladder_{name}": path for name, path in a.preds_ladder})}
    _finish(result, a.out)


def export_hillclimb_main(argv=None):
    ap = argparse.ArgumentParser(prog="export-hillclimb", description="one round in the claude-api hillclimb layout")
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--variant", required=True, help="baseline or v<N>")
    ap.add_argument("--split", default="val", choices=("val", "test"),
                    help="val: readable set (prompts exported); test: DEV (prompts withheld)")
    ap.add_argument("--temperature", default="1.0", help="a number or a temperature.json (Brier)")
    ap.add_argument("--description", default=None)
    ap.add_argument("--model", default=None, help="model label for the rows (default: predictor/template)")
    ap.add_argument("--out", required=True, help="the hillclimb flow directory")
    a = ap.parse_args(argv)
    refuse_sealed(a.data, a.preds, a.out)
    questions = load_questions(a.data)
    logits, identity = load_aligned(questions, a.preds)
    temperature, _ = temperature_arg(a.temperature)
    summary = export(read_jsonl(a.data), questions, logits, identity, a.variant, a.split, a.out, temperature,
                     a.description, a.model, {"data": sha256_file(a.data), "preds": sha256_file(a.preds)})
    print_json(summary)
