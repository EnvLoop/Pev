# Security policy

Please report security issues privately to **research@envloop.ai** rather than in a public issue. Include a
description of the issue, the affected files or commands, and steps to reproduce. We will acknowledge the report and
follow up with a fix or mitigation.

In scope: the code in this repository (`packages/`, `training/`, `scripts/`). For problems with the released data —
for example personal information that should have been pseudonymized — use the same address; we remove affected
records on request and publish a new revision of the dataset.

The pseudonymization key in `release/pseudonym_key.txt` is public by design (see [docs/DATA.md](docs/DATA.md)) and is
not a secret.
