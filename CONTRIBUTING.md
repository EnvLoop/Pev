# Contributing

Pev is a personal-agent decision model and benchmark: a knowledge base built from real public data, isolated
generation of TRAIN / VAL / DEV / HIDDEN / TEST decision records, and a LoRA SFT of `Qwen/Qwen3.8-27B` with a
pre-registered, paired evaluation. The pre-registration is [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md); it is
amended only by appending dated sections, never by editing earlier text.

Issues and pull requests are welcome: bug reports with a minimal reproduction, fixes, and documentation improvements.
Changes that would alter released results (data generation, labels, scoring, gates) need a clear rationale and must
not rewrite the recorded results or the pre-registration.

## Layout

| Path | What lives there |
|---|---|
| `packages/personal-decisions/` | KB builder, user-id split, 7 question families, LLM renderer, anchor checks, privacy scan, `export-release` |
| `packages/decision-eval/` | B0 / SFT / reference predictors, calibration, thresholds, paired scoring, gates, `compare-release` |
| `training/` | Standalone LoRA trainer and the adapter evaluation entrypoint |
| `scripts/acquire/` | Download + normalisation of the public sources into `work/muse/raw/` |
| `scripts/general_mix.py` | Rebuilds the Kev general-decision mix (not redistributed) |
| `docs/` | Pre-registration, results, data notes, versions, TEST commitment and results, technical report |
| `work/` | **Ignored.** Raw data, KB, shards, generated records, sealed sets, receipts |

## Isolation rules (keep them when extending the pipeline)

- `work/muse/sealed/` is readable only by the hidden-set builder; everyone else sees only its commitment file
  (hashes and counts).
- Training-data generators read only `work/muse/shards/{train,val}/`; they never read DEV/HIDDEN/TEST shards or
  evaluation-only styles.
- A sealed evaluation set is rendered once by its builder and never sent to an external service (LLM API, hosted
  reference model) for scoring, auditing or references.
- Rendering styles: `packages/personal-decisions/styles/public/` is shared (TRAIN, VAL, DEV, TEST, half of HIDDEN);
  a hidden builder's private styles live only under `work/muse/sealed/`.
- Leakage audits return collision counts only.
- No question is ever filtered on whether a model answers it correctly.

## Development

Python 3.12 [uv](https://docs.astral.sh/uv/) projects. Files <= 300 lines (hard limit 400), lines <= 120
characters, `unittest` tests next to each package, names that describe purpose, pinned dependency versions recorded
in [docs/VERSIONS.md](docs/VERSIONS.md). Both test suites run in CI on every push and pull request
([.github/workflows/tests.yml](.github/workflows/tests.yml)); run them locally before opening a pull request:

```bash
(cd packages/personal-decisions && uv run python -m unittest discover -s tests -t .)
(cd packages/decision-eval && uv run python -m unittest discover -s tests -t .)
```

## Secrets

API credentials (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `RENDER_MODEL`, `TYPESAFE_API_KEY`) live only in an ignored
`.env` at the repository root; never print, log or commit them. The pseudonymization key in
`release/pseudonym_key.txt` is public by design (see [docs/DATA.md](docs/DATA.md)).

## Data and removal requests

Questions and removal requests concerning quoted third-party text: research@envloop.ai. Security issues: see
[SECURITY.md](SECURITY.md).
