# Dependency versions

2026-09-30: `packages/personal-decisions` pins `openai` 3.22.1, `pydantic` 2.13.5 and `python-dotenv` 1.2.3 (PyPI
latest at that date) with a `uv_build` >=0.12.19 backend. openai 3.x ships its own `httpx2` transport.

2026-09-30: `packages/decision-eval` pins the PyPI latest at that date: torch 2.14.0, transformers 5.17.0, peft 0.21.1,
numpy 2.5.3, openai 3.22.1, python-dotenv 1.2.3 (`uv_build` >=0.12.20 backend). Optional extras: `fast` (Linux:
flash-linear-attention / fla-core 0.5.2 and einops 0.8.2, the pins of the environment the released adapter was trained in; fla-core needs torch >= 2.7 and
Triton >= 3.3), `jev` (the official TypeSafe `typesafe-sdk` 0.7.2, DEV-only Jev reference) and `kev` (git dependency
jaredpalmer/kev@0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91, DEV-only Kev reference). kev declares `torch<2.9`; a uv
`override-dependencies` keeps it on torch 2.14.0 so one environment serves every predictor, and kev's loader and
forward pass the package's tiny-model test on it. The default install never imports kev: its rendering and
selective-prediction conventions are copied in `decision_eval/conventions.py` and checked against kev when the extra is
installed.
