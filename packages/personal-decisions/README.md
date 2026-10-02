# packages/personal-decisions: MUSE-style personal decision records

This project turns the real normalized tables in `work/muse/raw/` into a knowledge base (KB). It then splits the KB
by user id into shards and generates Kev-format decision records for the seven question families of
[docs/PREREGISTRATION.md](../../docs/PREREGISTRATION.md). It contains code only. Each agent runs it on its own isolated
shard, and no data is checked in.

```bash
uv sync
uv run personal-decisions population --raw work/muse/raw --out work/muse/kb/population.txt --seed S \
    --per-domain product=N,restaurant=N [--receipt FILE] [--min-items 8] [--max-items 30] [--min-text-share 0.8]
uv run personal-decisions build --raw work/muse/raw --out work/muse/kb [--extractor rule|llm] [--max-users N] \
    [--users-file work/muse/kb/population.txt] [--concurrency 8] [--llm-cache FILE] [--llm-deadline-min M]
uv run personal-decisions split --kb work/muse/kb --train work/muse/shards/train --val work/muse/shards/val \
    --dev work/muse/eval-dev/kb --hidden work/muse/sealed/hidden/kb --manifest work/muse/kb/split_manifest.json \
    [--users hidden=200,dev=100,val=150]
uv run personal-decisions generate --shard DIR --out FILE.jsonl --n N --seed S \
    --style-file STYLE.json [STYLE2.json ...] [--families pick_option needs_approval ...] \
    [--family-weights pick_option=3.5] [--renderer astra|fixture] [--concurrency K] \
    [--p-buried 0.25] [--questions-per-state 3] [--env-file .env] [--price-in-per-mtok X --price-out-per-mtok Y]
uv run personal-decisions capacity --shard DIR [--trials 6] [--out report.json]
uv run personal-decisions shortcut-baselines --records FILE.jsonl [--out report.json]
uv run personal-decisions to-kev --records FILE.jsonl --out FILE.kev.jsonl [--no-soft]
uv run personal-decisions partition --kb KB --part test-a=DIR test-b=DIR [--email-pool dev] [--manifest FILE]
uv run personal-decisions leakage --shard test=DIR train=DIR ...
uv run personal-decisions render-export --shard DIR --out FILE.jsonl --requests REQ.jsonl --n N --seed S \
    --style-file STYLE.json ... [generate's draft options] [--margin M] [--batches DIR --n-batches 8]
uv run personal-decisions render-ingest --shard DIR --out FILE.jsonl --requests REQ.jsonl --n N --seed S \
    --style-file STYLE.json ... [same draft options] --outputs DIR/batch-*/output.jsonl --model NAME [--drop-missing]
uv run personal-decisions privacy-scan [--records F.jsonl ...] [--kb DIR ...] [--items ITEMS.jsonl] \
    [--examples K] [--counts-only DEV.jsonl ...] --out report.json
uv run personal-decisions export-release --records F.jsonl --out F.release.jsonl --items ITEMS.jsonl \
    [--key-file release/pseudonym_key.txt | --env-file .env] [--keep-foreign] [--receipt receipt.json]
uv run python -m unittest discover -s tests -t .
```

Every command prints a one-line JSON receipt. Receipts contain no record contents and never any credentials.

## Layout

| Module | Role |
|---|---|
| `kb/schema.py` | The pydantic KB schema (`Item`, `Interaction`, `UserProfile`, rules, facts). |
| `kb/raw_tables.py` | Readers for items, interactions, emails and OpenFlights. Popularity is the catalog review count (`rating_number` / `num_of_reviews`); when that is missing it falls back to the interaction count. |
| `kb/population.py` | `population`. Deterministic KB population: users with `min_items`..`max_items` distinct items, review text on most memory-side interactions, and later ratings both >= 4 and (by default) <= 2; ordered per domain by `sha256(muse-population:seed:user_id)`. |
| `kb/llm_extract.py` | The `--extractor llm` path of `build`: batched per user, concurrent, resumable through a JSONL cache, with a deadline and a consecutive-failure abort after which the remaining users use the rule extractor. |
| `kb/memory.py` | Extracts memory facts from early reviews. The rule extractor quotes a verbatim sentence. The optional LLM extractor must return a verbatim evidence span; if it does not, the fact is rejected and the rule extractor is used instead. |
| `kb/rules.py` | Synthesized personal rules seeded by `sha256(salt:user_id)`: approval, privacy, forget requests, notification rules, contacts, calendar and connected services. |
| `kb/build.py` | `build`. Each user's history is sorted by time and deduplicated per item. The first 60% becomes memory and the rest are real choices. A user is kept only with at least 5 items and at least one later rating of 4 or more. |
| `split.py` | `split`. Users are ordered by `sha256(salt:user_id)`: hidden, dev and val take fixed counts and train takes the rest. Emails are partitioned by hash bucket. The item catalog and OpenFlights data are copied to every shard. The manifest holds per-file SHA-256 and counts only. |
| `leakage.py` | Cross-shard audit run by `split`: user, (user, target), target review text and target event collisions, counts only, stored in the manifest. |
| `split.py` (`partition`) | `partition`. A whole KB (a fresh evaluation population) into equal named parts, domain-stratified, ordered by `sha256(muse-partition-v1:user_id)`; emails optionally limited to one `split` email pool. |
| `external.py`, `cli_external.py` | `render-export` / `render-ingest`: an external renderer (e.g. agents) gets exactly the gpt-6-astra request of each state (system prompt + style block + input text, as batch `input.jsonl` files with a generated `INSTRUCTIONS.md`); the returned texts go through the same distractor insertion, anchor check, record format, log and stats as `generate`. `leakage`: the cross-shard audit for any named shards. |
| `release/` (`cli_release.py`) | `privacy-scan`: aggregate PII counts (emails, phones, URLs, street addresses, person names) over records and KB shards, optional redacted examples. `export-release`: the (reversible, keyed) pseudonymization pass of every public record: `meta.user_id` -> keyed HMAC pseudonym, detected names -> keyed same-shape pseudonyms and contacts -> `[EMAIL]`/`[PHONE]`/`[URL]`/`[ADDRESS]` consistently in state, instructions and anchors; option texts and labels never change; anchors re-checked; deterministic given the key, which is published with the release (`release/pseudonym_key.txt`) so the export is reproducible; read from `--key-file` or `RELEASE_PSEUDONYM_KEY`, never printed. |
| `capacity.py` | `capacity`. Per-family feasibility of a shard (counts only, nothing rendered): feasible pick_option (user, target) pairs and per-family build success. |
| `families/` | One pure function per family: `build(Context) -> Fragment`. A fragment carries its state lines, one pending item and one question (label, anchors, variant). |
| `balance.py` | Exact label balance by construction (route also by label name: its 5 services and `ask_user` in equal shares; soft labels must be tied, i.e. have no unique argmax, or the fragment is rebuilt): per family a seeded, shuffled-block schedule of hard labels (noul 50/50, score levels uniform) or of the correct option's position (choice, per option count). Mismatching noul/score fragments are rebuilt with fresh seeded RNGs; choice options are reordered. On by default (`generate --no-balance` turns it off); resume replays the schedule. |
| `compose.py` | Merges up to 3 compatible fragments into one structured state. Everything is seeded by `(shard, seed, index)`. |
| `distractors.py` | The buried variant: 1k to 6k tokens of shard emails and other shard users' reviews, inserted after rendering. |
| `render/` | `FixtureRenderer` (offline template) and `AstraRenderer` (gpt-6-astra). A style is a JSON spec file. |
| `anchors.py` | The deterministic anchor verifier. |
| `llm.py` | An OpenAI-compatible Responses API client. It reads `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `RENDER_MODEL` (legacy name `RSI_TEACHER_MODEL`) from the environment or the repo `.env`, and the key stays inside the SDK client. |
| `generate.py`, `stats.py` | `generate` plus its sidecars. |
| `baselines.py` | `shortcut-baselines`. |
| `to_kev.py` | `to-kev`. |

## Knowledge base

`build` writes the following files. Every shard has the same layout.

- `users.jsonl`: one `UserProfile` per line.
- `items.jsonl`: the catalog with popularity.
- `emails.jsonl`
- `airports.jsonl`
- `routes.jsonl`
- `airlines.json`
- `manifest.json`: counts, SHA-256 per file and build stats.

Real data determines what a user likes: the time-split history, memory facts, and real preference changes, which are
categories rated 4 or more early and 2 or less later, or the reverse. Seeded rules determine what the assistant may
do. The same user id always gets the same rules.

## Families

| Family | Type | Label | Variants |
|---|---|---|---|
| `pick_option` | choice | The item from a real later interaction rated 4 or more. Candidates are:<br>- the target<br>- the user's real later same-category item rated 2 or less, when the target can still be balanced around it<br>- untouched items of the same category (for restaurants: same city, category among the Google `categories`; for products: the same subcategory when there are enough).<br>The option set is drawn so that the target sits where a sampled distractor would on price, popularity, medoid and centroid distance (`families/candidates.py`); a target that cannot be balanced is skipped. | override, removed (soft) |
| `needs_approval` | noul | Spend threshold in force at that time, new-recipient rule, irreversible-action rule. | override (threshold change flips it), removed (soft 0.5) |
| `apply_memory` | noul | Whether the fact's category matches the request. Negatives are other product departments (never another restaurant category: dining facts bear on any restaurant search) or unrelated tasks. | clean |
| `forgotten_violation` | noul | Whether the action relies on a fact the user asked to forget. | removed (contrast: forget request withheld, label false) |
| `share_ok` | noul | Recipient clearance is at least the fact's privacy level. | override (re-marked), removed (soft 0.5) |
| `notify_level` | score | Urgency capped by the type maximum, quiet hours and meeting cap (judged at the event's arrival time, as the question says), with VIP bypass. Labels are balanced by construction. | override (quiet hours change), removed (urgency withheld; soft label exactly uniform over the 2+ levels the urgency could reach, so tied and never scored) |
| `route` | choice | The needed service if it is connected. Otherwise `ask_user`, which also covers ambiguous OpenFlights city names across countries. The label name follows a balanced schedule (each of the 6 classes 1/6). | override (dated (dis)connections, balanced) |

The **buried** variant applies to a whole state (`--p-buried`). **Removed** evidence has forbidden anchors that must
not appear anywhere in the state. The pick_option **override** lifts an older budget so that it covers every
candidate. With equal odds, the stale budget excluded either the target or only a pricier candidate, so the update
does not single out an answer.

## Record contract

Each record is one JSON object per line:

```json
{"id": "muse/<shard>/<seed>/<index>", "state": "...",
 "questions": {"<qid>": {"type": "noul|choice|score", "instructions": "...", "criteria": {...} | [...],
                          "label": "<option>|true|false|<level index>", "src": "muse/<family>",
                          "soft_label": {"<option>": 0.5}}},
 "meta": {"user_id": "...", "shard": "...", "state_id": "<shard>/<seed>/<index>", "families": {"<qid>": "<family>"},
          "anchors": {"<qid>": {"required": [...], "forbidden": [...]}},
          "render": {"model": "...", "style_id": "...", "prompt_sha256": "..."},
          "variants": {"<qid>": "clean|override|removed|buried+..."},
          "option_features": {"<qid>": {"<option>": {"price": 12.5, "popularity": 340}}}}}
```

- The qid is the family name, and a state never holds two questions of the same family.
- Records keep the criteria in option order (written without key sorting): for a choice question the criteria
  order is the option order, and with `balance` the correct option's position is stratified.
- `soft_label` is present only on ambiguous questions. Its keys follow Kev's convention: option names for choice,
  `"true"`/`"false"` for noul, and level indices as strings for score.
- On a soft question, `label` is the arg-max of `soft_label`. For uniform soft labels it is the real or rule answer.
- `meta.variants` and `meta.option_features` go beyond the contract and are additive: `option_features` exists only
  for pick_option, where the shortcut baselines use it.

## Kev compatibility (kev @ 0fe8fc97)

- **Labels.** `kev.data.load_records` takes one labelled request per line: `{"state", "questions": {id: {type,
  instructions, criteria, label, src}}}`.
  - choice: `label` is the option name.
  - noul: `label` is `true`/`false`.
  - score: `label` is the zero-based level index.
- **Extra keys.** Unknown top-level keys (`meta`) and question keys (`soft_label`) are kept but ignored.
  `kev.data.api_request` sends only `type`, `instructions` and `criteria` to the model, so our records load unchanged.
- **Soft targets.** Kev reads them only from the question field **`target`**, as `{option key: weight}`:
  - option names for choice
  - `"false"`/`"true"` for noul
  - `"0"`, `"1"`, ... for score

  Kev normalizes the weights, gives 0 to any option not named, and still requires `label`
  (`kev.data.materialize`, `kev.train` loss). It ignores `soft_label`.
- **Conversion.** `to-kev` drops `meta`, renames `soft_label` to `target` (or drops it with `--no-soft`) and sets
  `_meta = {id, group_id: state_id, source: "muse"}`. Both files load through `load_records` and `materialize`.

## Rendering and verification

- **The model's role.** `AstraRenderer` sends the structured state as JSON, plus a VERBATIM list of every required
  anchor and the style spec, with `store=False`. The model only renders: it never sees questions or labels.
- **Anchor matching.** After rendering (and after distractor insertion), the verifier checks exact, case-sensitive
  matches. It normalizes only whitespace and typographic quotes and dashes.
  - An anchor may not extend into another word or number, so `$50` does not match `$500` or `$50.99`.
  - A record is dropped if any required anchor is missing or any forbidden anchor appears.
  - No question is filtered on whether a model answers it correctly.
- **Public styles.** `styles/public/` holds the shared rendering styles (TRAIN, VAL, DEV, half of HIDDEN). Passing
  several `--style-file`s rotates them over the states in seeded shuffled blocks (each block of k states uses each of
  the k styles once); `meta.render.style_id`, the attempt log and `OUT.stats.json` (`style_counts`) record which.
- **Family weights.** `--family-weights pick_option=3.5` draws that family more often as the primary family and among
  the extras (weighted, without replacement), for families that build on few attempts. Without it the primary
  family is a plain round robin.
- **Style files.** A style spec is `{"style_id"?, "voice", "format", "language_mix", "verbosity", "notes"?}`. Only
  `style_id` enters records; when it is omitted, the id is a SHA-256 of the spec.
- **Outputs of `generate`.** Each run writes:
  - `OUT`: the records, sorted by index, first N.
  - `OUT.log.jsonl`: every attempt, with status, drop reason and token usage.
  - `OUT.stats.json`: tokens, drops by reason, per-family counts, label and variant counts, optional cost, and the
    output SHA-256.
  - `OUT.sha256`
- **Resuming.** A rerun skips ids already kept, dropped or skipped. An API failure stops the run without marking
  the ids, so they are retried on the next run.

Smoke test (2026-09-30): 2 gpt-6-astra calls on fixture states used 1,444 input and 590 output tokens. Both passed
anchor verification.

## Shortcut baselines

`shortcut-baselines` reports per-family accuracy of trivial predictors on hard-labelled questions:

- majority label
- first option
- pick_option, one per cheap feature the sampler uses: most_popular, least_popular, cheapest, priciest, medoid
  (least summed log-price + log-popularity distance to the others) and centroid (nearest to the mean point)

It also reports chance and the best shortcut per family. The evaluation must beat all of these.

Binary families draw their label first, so they are balanced by construction. notify_level draws its level first.

**Validation on the real tables (2026-09-30).** The run used the fixture renderer and no model calls. It took the
first 600 users, 300 of them in a train shard, and generated 1,000 states with 2,960 questions. The best shortcut was
within 0.02 of chance for every family except route:

- route: `ask_user` was 0.26 of labels against a 1/6 chance (fixed 2026-09-30: route labels are now scheduled by name).
- pick_option (superseded, see below): most_popular 0.26, cheapest 0.27 and first_option 0.29, against a chance of
  0.27; the nearest-neighbour distractors still left a medoid shortcut (0.35 against 0.26).

**Balanced pick_option (2026-09-30).** With `families/candidates.py`, 2,000 pick_option-only fixture states from the
1,500-user TRAIN dry run (two seeds, ~1,675 hard-labelled each): every pick_option predictor was within 0.03 of
chance on each seed (medoid 0.297 / 0.268, centroid 0.281 / 0.279, most_popular 0.250 / 0.268, priciest 0.244 /
0.295 against chance 0.270 / 0.272), averaging within 0.013. About 31% of pick_option attempts succeed, and ~12% of
questions carry the user's real low-rated item (it often cannot be balanced around).

The full build loads all 881k interactions and needs about 2.2 GB of RAM.

**Null-baseline fixes (2026-09-30).** The DEV null-baseline check found two shortcuts: notify_level's removed variant
had soft labels with a unique argmax (always level 1 or 2) that bypassed the schedule, and route's `ask_user` was 26%
of labels. Now every soft label is tied (the schedule rebuilds any that is not) and route's label names are scheduled.
On a 1,500-state TRAIN fixture sample (seed 11) every family's best shortcut is within 0.002 of chance except
pick_option (centroid +0.016), and decision-eval `validity` (oracle, majority, constant position, shuffled) passes for
every family.
