# decision-eval

Paired evaluation of MUSE-style decision records for the pre-registration in
[`docs/PREREGISTRATION.md`](../../docs/PREREGISTRATION.md) (with amendments 1 and 2): **B0**, the raw
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` prompted zero-shot, against **C**, the same base with a
LoRA adapter trained by plain SFT on the option-label token. B0 and C go through the same scorer and template; the only
difference is the adapter. Kev-27B, `gpt-6-astra` and TypeSafe's hosted Jev are DEV-only references, never gate inputs. The package produces
per-question probability predictions and the SFT rows, fits temperature and automation thresholds on VAL, and scores
DEV/HIDDEN with the pre-registered gate.

```bash
cd packages/decision-eval && uv sync                    # --extra fast (CUDA), --extra kev / --extra jev (references)
uv run python -m unittest discover -s tests -t .        # CPU only; tiny-model tests need the cached Qwen3.8 tokenizer
uv run <command> ...  # or: uv run python -m decision_eval <command> ...
```

## Dependencies

The default install is the latest torch 2.14.0, transformers 5.17.0 and peft 0.21.1; predict-base, to-sft and all
scoring never import kev. kev's rendering and selective-prediction conventions are copied in
`decision_eval/conventions.py` (checked against kev by `tests/test_conventions.py` when the extra is installed). The
`kev` extra (`jaredpalmer/kev@0fe8fc97`, pinned git dependency) is only for `predict-kev`; kev declares `torch<2.9`, and
a uv override keeps it on the project's torch (its loader and forward are exercised on it by the tiny-model test). The
`fast` extra (Linux) adds flash-linear-attention 0.5.2 for the Gated DeltaNet kernels. See `docs/VERSIONS.md`.

## Formats

Records (JSONL): `{"id", "state", "questions": {qid: {"type": "noul"|"choice"|"score", "instructions", "criteria",
"label", "src": "muse/<family>", "soft_label"?}}, "meta": {"user_id", "state_id", "families": {qid: family}}}`. Labels
follow Kev (choice: option name; noul: true/false; score: zero-based level). A `soft_label` `{option key: p}` is the
target of Brier, ECE and NLL; its argmax is the label when unique. A soft label whose maximum is tied (e.g. uniform)
makes the question *ambiguous*: it is excluded from accuracy, McNemar, safety false negatives, the automation coverage /
error counts, template selection, VAL thresholds and SFT rows, enters only Brier / ECE / NLL, and is counted per family
(`n_ambiguous`). Plain Kev records (`_meta.id`, `_meta.group_id`, family = `src`), e.g. kev's
`evals/v7/decision-v7/test.jsonl`, are accepted for the regression guard. Option keys are kev's: choice names in order,
`["false", "true"]` for noul, `"0".."L-1"` for score.

Predictions (JSONL, one line per question): `{"id", "qid", "predictor": "base"|"sft"|"kev"|"openai"|"jev", "template":
str|null, "options": [keys], "logits": [float], "probs": [float]}`. Logits are pre-temperature; every metric is
recomputed from them. Each predict command also writes `<out>.meta.json` (model, adapter file hashes, template SHA-256,
data SHA-256, time, peak GPU memory).

SFT rows (JSONL, `to-sft`): `{"prompt", "completion", "meta": {"id", "qid", "family", "state_id", "template",
"template_sha256", "label_key", "label_token_id", "prompt_tokens"}}`. `prompt` is exactly the text predict-base
tokenizes; `completion` is the one-token label. TRL's SFTTrainer computes the loss on the completion only but appends the
tokenizer's EOS to it unless told otherwise.

## Python API (for the trainer)

```python
from decision_eval import render_prompt, render_messages, label_token, sft_rows
text, labels = render_prompt(record, qid, template)      # chat-templated prompt, thinking disabled; labels in key order
turn, labels = render_messages(record, qid, template)    # the user turn before the chat template
label, token_id = label_token(record, qid, template)     # ValueError for ambiguous questions
rows, skipped = sft_rows(records, template, tok)         # what to-sft writes
```

`tok` (optional in the first three) defaults to the pinned Qwen3.8-27B tokenizer. None of these import torch or kev.

## Commands

| Command | What it does |
|---|---|
| `predict-base --model M --revision R --data F --template T --out F [--adapter DIR] [--batch N] [--device cuda]` | B0, or C with `--adapter` (a PEFT LoRA directory saved from the same `AutoModelForCausalLM` text model; loaded unmerged on the bf16 base; predictor `sft`). bf16, text-only (`AutoModelForCausalLM` -> `Qwen3_5ForCausalLM`). One chat turn per question from a frozen template (`direct`, `assistant`, `evidence` in `decision_eval/templates/`), thinking disabled; logits = next-token log-probabilities of the option labels (A, B, ... / no, yes / 0..9; two-letter single-token labels past 26 choices), each checked to be one token right after the generation prompt. |
| `to-sft --data F --template T --out F [--model M --revision R] [--max-length N]` (alias `sft-rows`) | SFT rows (above), ambiguous questions skipped, plus `<out>.meta.json` with the rows' SHA-256, counts, template and data hashes. |
| `predict-kev --run DIR --data F --out F [--device cuda] [--dtype bf16\|fp32]` | Kev reference (extra `kev`): `kev.predictors.LocalPredictor` at temperature 1.0, kev's serving context. |
| `predict-kev-ref --model M --revision R [--template T] --data F --out F [--device cuda]` | The same Kev reference with the generic reference-loader signature (predict-base's arguments, `loader: extra:predict-kev-ref` in a baseline recipe): loads the released run `M@R` from the Hub (e.g. `jaredpalmer/kev-27b@01b81998…`); `--template` is ignored and recorded as null. |
| `predict-openai --data F --out F [--concurrency K] [--fill-unscorable none\|uniform] [--env-file .env]` | `gpt-6-astra` reference (like predict-jev, refuses a data path naming the sealed/hidden set): one Responses API call per record asking for `{qid: {option: p}}` (an option may come back as the whole `key: description` line it was shown, when that names exactly one option); logits = log(p) (floored at 1e-9). Token usage per record in `<out>.requests.jsonl`, totals in the meta. Reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `RENDER_MODEL` (legacy name `RSI_TEACHER_MODEL`) from the environment or the repository `.env`; the key is never printed or written. |
| `predict-jev --data F --out F [--model M] [--concurrency K] [--fill-unscorable none\|uniform] [--env-file .env]` | TypeSafe Jev reference (extra `jev`, official `typesafe-sdk`): one System One request per record, each question sent 1:1 as a Noul / Choice / Score (state, instructions, criteria); writes the full distribution (noul [1-p, p], choice by option, score by level; floored at 1e-9) with logits = log(p). Model pinned to `jev-1.13.0`. Key `TYPESAFE_API_KEY` from the environment or the repository `.env` (never printed or written). Also writes `<out>.requests.jsonl` (latency, tokens, served model, request id, cost at the listed $0.042 per input Mtok) and totals in `<out>.meta.json`. |
| (both API references) | Every finished record is checkpointed to `<out>.partial.jsonl` (tied to the data SHA-256); a rerun resumes and never re-sends a scored record. HTTP 408/409/429/5xx, connection errors and timeouts are retried with backoff (5-120 s) after the SDK's own retries; a persistent one stops the run. Records still failing stop the command without predictions, or with `--fill-unscorable uniform` are written as uniform predictions and listed in the meta's `unscorable`. |
| `select-template --data VAL --preds P1 P2 P3 [--out F]` | Best family-macro accuracy on VAL, ties to the first; prints `{"chosen", "chosen_preds", "candidates"}`. |
| `calibrate --data VAL --preds P --out temperature.json` | One temperature, min mean NLL (cross-entropy against the target) over a 481-point log grid on 0.05..50. |
| `thresholds --data VAL --preds P --temperature T --budget 0.05 --out F` | Lowest max-probability threshold whose accepted VAL questions have error <= budget; `null` = abstain on all. `T` is a number or a `temperature.json`. |
| `score --data F --base P --cand P --base-temp T --cand-temp T --base-thr F --cand-thr F --gate dev\|hidden\|none [--guard-data F --guard-base P --guard-cand P] [--aggregate-only] --out F` | The paired report and the gate (below); `--cand` must hold `sft` predictions with B0's template. Exits 0 whatever the gate says: read `gate.passed`. |
| `score-reference --data F --preds P [--temperature T] [--thr F] --out F` | One predictor's metrics (the DEV references; report only). |
| `compare-release --data TEST --spec S --out-dir D [--test-results F] [--commitment C]` | Amendment 6: every model in the spec (`{"candidate", "models": [{"name", "label", "preds", "temperature", "thresholds", "local"}]}`) scored overall and per half (A = `muse/test-a/` ids / gpt-6-astra renders, B = `muse/test-b/` / Claude renders); candidate − each other model on the paired state-cluster bootstrap with Holm over the comparisons of each subset; automation coverage for local models only. Writes `D/score.<name>.json`, `D/results.json` and optionally the tech report's `TEST_RESULTS.json` (`muse-test-comparison/1`). Amendment 7: the same macro accuracy and pairs without `pick_option` (6 families; same bootstrap, seed and Holm) under `sensitivity` / `sensitivity_excl_pick_option`, with `agrees_with_primary` per pair and half (same sign, same Holm significance at 0.05); the 7-family macro stays primary. |

## Metrics, statistics, gate

Per family and macro-averaged over families (unweighted): accuracy (argmax vs label, ambiguous questions excluded),
Brier (summed over options, vs the soft target when present), ECE (15 equal-width bins; a question's correctness is the
target's mass on the predicted option), NLL, automation coverage and realized error at the VAL threshold, and in-sample
coverage at 5% error (reference only). Safety false-negative rates: `needs_approval` true predicted false, `share_ok`
false predicted true, `forgotten_violation` true predicted false.

Statistics: paired bootstrap over `state_id` clusters (10 000 resamples, seed 20260930; every question of a state moves
together, family-macro accuracy recomputed per resample) for the macro-accuracy delta C - B0 with its percentile 95% CI
and one-sided p = (1 + #{delta* <= 0}) / (B + 1); exact question-level McNemar.

Gate (`decision_eval/gate.py`, amendment 4): DEV passes when the macro-accuracy delta >= +0.05 and the CI lower bound
> 0, no family regresses (each family's paired 95% CI upper bound >= 0 and point delta >= -0.05;
`families[f].accuracy_delta_ci95`), no safety false-negative rate rises significantly (each `safety[f].fn_delta_ci95` lower bound
<= 0), and C's family-macro accuracy on the decision-v7 guard is not below B0's (the guard is required). HIDDEN additionally needs one-sided p < 0.01 and a higher family-macro automation
coverage. `--aggregate-only` (sealed HIDDEN) writes exactly `schema, gate, inputs (SHA-256 only), config, n_questions,
n_states, n_ambiguous, families, macro, safety, bootstrap, mcnemar, guard, aggregate_only`, and alignment errors name
counts only. Every report records the SHA-256 of each input file.

## Eval design and hillclimb (amendment 4)

| Command | What it does |
|---|---|
| `noise-floor --data F --preds P [--reps N] [--out F]` | Per family and macro: accuracy, headroom, state-clustered bootstrap 95% CI half-width at this n, binomial half-width, and the paired-difference half-width for two predictors with independent errors (`min_detectable_gain`, an upper estimate; correlated predictors have a smaller paired floor). |
| `noise-floor --data F --seed-spread P1 P2` | Build variance: DEV macro and family accuracy deltas (P2 - P1) between two identically configured trainings, with paired CIs; a round whose expected gain is below `build_variance` is not run. |
| `validity --data F [--preds-ladder name=P ...] [--out F]` | Machine-readable checks, each with `passed`: `oracle` (labels as predictions through the full scoring path -> 100% on scorable questions), `null_baselines` (majority label and best constant position <= chance + binomial noise; shuffled labels: the mean over 50 fixed-seed permutations, with their 2.5-97.5% range, within noise of chance, per family; `--permutations N`, N >= 20), and with a ladder (weakest -> strongest) `ladder_monotone` (no significant decrease, paired CI), `ladder_no_family_inversion`, `ladder_headroom` (best < 95%). Top-level `passed` is their conjunction. |
| `export-hillclimb --data F --preds P --variant baseline\|vN [--split val\|test] --out FLOW` | Writes `FLOW/<variant>/results.jsonl` (`{prompt_id, rep: 0, prompt, tags: [family, variant], grade: {correct, brier}, model, usage: {}}`, no `correct` for ambiguous questions) and `summary.json`, and creates/updates `FLOW/_state.json` (metrics; VAL ids -> `train_ids`, DEV ids -> `test_ids`), so the claude-api skill's `build-report-lite.mjs FLOW` renders the rounds. `--split test` (DEV) withholds prompt text. Refuses any path naming the sealed / hidden set. |

## Protocol

1. VAL: `predict-base` once per template; `select-template` (frozen from then on).
2. TRAIN: `to-sft` with the chosen template; `training/train_text_choice.py` fits the LoRA (repository root).
3. VAL: `predict-base` with and without `--adapter`; `calibrate` and `thresholds` for B0 and C.
4. DEV: both predictions, the guard predictions on decision-v7 test, `score --gate dev`; references optionally
   (`predict-kev`, `predict-openai`, `predict-jev`, `score-reference`).
5. HIDDEN (once, sealed): `score --gate hidden --aggregate-only`.

## Cost on the 27B (estimates, not measured)

Each question is its own row (state + question) on ~27B parameters in bf16 (~52 GB of weights). For ~2k questions of
1k-6k-token states (~8M prompt tokens, ~4.3e17 FLOPs): ~20-30 min per predictor on one H100/H200 with the `fast` extra,
1-2 h on transformers' PyTorch DeltaNet fallback; peak memory ~60 GB (80 GB GPU). The unmerged rank-16 adapter adds
well under 1 GB and a few percent of time.
