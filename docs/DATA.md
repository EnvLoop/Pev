# Data: sources, licences and pseudonymization

This page summarises what the released records contain and how they were made public. The records themselves are on
Hugging Face (`EnvLoop/Pev-Bench`, gated, research only); the dataset card is
[`cards/DATASET_CARD.md`](../cards/DATASET_CARD.md).

## Sources

| Source | Used for | Terms |
|---|---|---|
| Amazon Reviews 2023 (McAuley Lab, UC San Diego), 5 categories | users' product histories (memory, real later choices); review spans quoted as evidence | no explicit licence; released for research; cite Hou et al. 2024 (arXiv:2403.03952) |
| Google Local Data 2021 (UC San Diego), food businesses in DC, DE, RI | users' restaurant histories; review spans quoted as evidence | no explicit licence; cite Li, Shang and McAuley (ACL 2022) and Yan et al. (SIGIR 2023) |
| OpenFlights airports, airlines, routes | travel routing | ODbL 1.0 (database), DbCL 1.0 (contents); records are a Produced Work |
| Enron e-mail corpus (CMU release, via a Hugging Face mirror) | buried distractor text only | research release, no licence; addresses, phone numbers and URLs scrubbed at acquisition |
| gpt-6-astra | rendering TRAIN, VAL, DEV, TEST-A; memory-fact extraction | — |
| Claude Opus 5.5 | rendering TEST-B (evaluation only; never used for training) | — |
| Kev `evals/decision-v2/train.jsonl` @ `0fe8fc97` | general decision rows mixed into the adapter's training set | **not redistributed** (third-party text under its own terms); rebuilt by `scripts/general_mix.py`, ids and hashes in `general_mix_ids.jsonl` |

`scripts/acquire/` downloads and normalises the sources; `personal-decisions build` turns them into the knowledge base.
The knowledge base and the raw sources are not distributed (the OpenFlights-derived tables would be a Derivative
Database under the ODbL); the scripts rebuild them.

Everything about a user other than their public reviews is **synthetic**: approval thresholds, privacy settings,
contacts (fictional names at `example.com`), calendars, notification rules, forget requests and connected services
are generated from a per-user seed. Nothing in a record says anything true about a real reviewer beyond what their
public reviews say.

## Pseudonymization (`export-release`)

Every released record goes through `personal-decisions export-release` (code in
`packages/personal-decisions/src/personal_decisions/release/`, unchanged since it pseudonymized TEST):

1. `meta.user_id` (an unsalted hash of the public source user id) becomes `HMAC-SHA256(key, "user:" + id)[:16]`.
2. Person names found by the detector in the state or the question instructions become keyed pseudonyms of the same
   shape (one pseudonym per real first name and per real surname, so "Mr. X" and "First X" stay one person); e-mail
   addresses, phone numbers, URLs and street addresses become `[EMAIL]` / `[PHONE]` / `[URL]` / `[ADDRESS]`.
3. Replacements apply identically to the state, the instructions and the anchors, and never to text that occurs in an
   option, so options, labels and soft labels are unchanged; every anchor is re-checked on the new text.
4. Records without `meta` (Kev-format general-mix rows) are dropped from the export.

This is **reversible pseudonymization for readability and consistency, not anonymization.** The key is published in
[`release/pseudonym_key.txt`](../release/pseudonym_key.txt) so that the export is exactly reproducible from the public
sources; anyone can therefore recompute the mapping. All source data were already public. (The per-record flag
`meta.release.anonymized: true` predates this wording; it means "went through the pseudonymization pass".)

Export receipts (same key, frozen exporter):

| File | Records | Pseudonymous users | Names replaced | Street addresses | URLs | Residual detections | SHA-256 |
|---|---|---|---|---|---|---|---|
| `train.jsonl` (Pev-Bench part) | 1,970 | 1,331 | 7,054 | 247 | 7 | 10 names, 1 address | `62252b09…265f` |
| `val.jsonl` | 357 | 251 | 1,218 | 38 | 0 | 4 names | `e3a228ce…1ed2` |
| `dev.jsonl` | 630 | 428 | 2,071 | 96 | 6 | 2 names | `1ee1bd7c…b9c8` |
| `test_a.jsonl` | 490 | 325 | 2,081 | 83 | 0 | 4 names | `b2fade97…ef26` |
| `test_b.jsonl` | 490 | 329 | 1,635 | 67 | 0 | 3 names | `d7907b2a…1d8a` |

Reproduce an export (inputs are the generator's records, rebuilt from the public sources):

```bash
cd packages/personal-decisions
uv run personal-decisions export-release --records RECORDS.jsonl --out RELEASED.jsonl \
    --items SHARD/items.jsonl --key-file ../../release/pseudonym_key.txt --receipt RECEIPT.json
```

Re-running gives byte-identical output; the released TEST halves were re-exported from their pre-release records with
the published key and match the committed hashes (`docs/TEST_COMMITMENT.json`).

### Residual detections and known gaps

"Residual detections" are strings the detector still flags after the pass. All 7 in TEST were reviewed: none is an
unreplaced name of a real person (6 are a pseudonym followed by an adjacent capitalised word, such as a company name;
1 is a product name). Across TRAIN, VAL and DEV most residuals are of the same boundary kind; a few keep the second
part of a two-part real surname next to a pseudonymized first part, and one TRAIN hit is a restaurant name inside an
option (kept by design).

The detector is rule-based and its recall is limited. Names written "Last, First" (frequent in the distribution lists
of Enron e-mail headers), all-capital names and names outside its first-name list are not replaced; nearly all of
them occur in Enron distractor text, which is a public corpus. Removal requests: research@envloop.ai.

## What the pseudonymization changes for reproducibility

- **TEST** was pseudonymized before any model was scored: the scored file (SHA-256 `131d0927…79a3`) is the byte
  concatenation of `test_a.jsonl` and `test_b.jsonl`.
- **VAL / DEV** numbers in the results and the report were computed on the pre-pseudonymization text; the released
  files differ only in person names (mostly inside distractor e-mails), contact details and user ids, and were not
  re-scored.
- **TRAIN**: the adapter was trained on the pre-pseudonymization rows plus the Kev mix (SFT rows SHA-256
  `046a2ddb…`). Retraining on the released TRAIN with the rebuilt mix will not hash-match those rows.
