# Pev pre-registration (Kev method, Qwen3.8-27B)

> This project was released as Pev, a calibrated fast-decision model for personal agents; the pre-registration calls it a "MUSE-style personal decision model".

> **About this file.** Written on 2026-09-30 and extended only by dated amendments (1–7) and a release note recorded
> after the fact. Paths under `work/` refer to the pipeline's ignored working tree.
>
> This public copy omits internal infrastructure details; the data-file hashes are unchanged.

Registered 2026-09-30, before any DEV/HIDDEN data was generated. From then on it may only be extended by dated
"amendments" appended at the end; the original text is not changed.

## Goal

Train a "fast-thinking" decision model for a MUSE-style personal agent (long-term memory, execution across services,
approval required for sensitive actions, able to forget): a `state` (user memory + current request + candidate
actions) plus several typed questions, producing a calibrated probability for each question.

- Method: Kev ([jaredpalmer/kev](https://github.com/jaredpalmer/kev) @ `0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91`):
  rank-16 LoRA + pointer head, temperature fitted on VAL.
- Starting point: the original `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (no `--init_from`).
- Rendering / reference model: `gpt-6-astra` (OpenAI-compatible API, model name configured in `.env`).

## Data: the knowledge base is the only ground truth

1. Real sources (licences in `work/muse/raw/LICENSES.json`): product/restaurant reviews (the real history of the same
   user), flight routes, a public e-mail corpus as distractor text.
2. Knowledge base `work/muse/kb/`: user profiles (real history + seeded synthetic approval rules, privacy levels,
   items to forget, preference changes), a world catalogue, and deterministic question rules for the 7 families.
3. Split by **user ID**: TRAIN / VAL / DEV / HIDDEN are mutually disjoint; a real person appears in only one shard.
4. `gpt-6-astra` only renders the structured state into natural language; the decisive evidence for each question
   carries anchors that are checked verbatim after rendering, and a render that fails is discarded. Questions are never
   filtered on whether any model answers them correctly.

### Question families

| Family | Type | Label source |
|---|---|---|
| `pick_option` | choice | Real behaviour: an item the user rated highly **after** the time cut-off (candidates include the user's low-rated item and similar items they did not interact with) |
| `needs_approval` | noul | The user's approval rules + action attributes |
| `apply_memory` | noul | Whether the request and the memory are related (guards against over-personalisation) |
| `forgotten_violation` | noul | Whether the candidate action relies on a fact the user asked to forget |
| `share_ok` | noul | Privacy level × recipient |
| `notify_level` | score | Event urgency × the user's do-not-disturb rules |
| `route` | choice | Request → service / ask the user |

Hard cases: evidence buried in 1k–6k tokens of distractor text; preferences overridden by later updates; evidence
removed (the model should be uncertain; soft labels).

## Information isolation

| Directory | Who may read it |
|---|---|
| `work/muse/raw/`, `work/muse/kb/` (complete) | acquisition, KB build (the splitter), audits |
| `work/muse/shards/{train,val}/`, `work/muse/data/` | training-data generation |
| `work/muse/eval-dev/` | DEV construction, evaluation |
| `work/muse/sealed/` | only the HIDDEN builder; everyone else sees only `work/muse/HIDDEN_COMMITMENT.json` (hashes and counts) |

Training-data generators do not read the DEV/HIDDEN shards or their rendering-style descriptions; the orchestrator
does not read HIDDEN contents. Leakage audits return collision counts only.

## Evaluation

- **B0**: the original Qwen3.8-27B zero-shot; the log-probabilities of the option labels are normalised into a
  distribution; of 3 pre-written templates, the best on VAL is chosen and then frozen.
- **C**: Qwen3.8-27B + the Kev structure, trained from the base.
- **R**: `gpt-6-astra`, DEV only, reference only, not part of any gate.
- B0 and C have their temperatures fitted on VAL with the same procedure.

Metrics (question level, macro-averaged over families): accuracy (primary), Brier, ECE (15 equal-width bins),
automation rate at a 5% error budget (threshold set on VAL and used unchanged on DEV/HIDDEN), safety false-negative
rates (`needs_approval`: should need approval but predicted no; `share_ok`: must not share but predicted may;
`forgotten_violation`: violation predicted as no violation). Statistics: paired bootstrap clustered by state
(10,000 resamples, 95% CI); question-level McNemar as a secondary test. Regression guard: on Kev's `decision-v7` test,
C must not be below B0.

### Gates

- **DEV** (about 600 questions): macro accuracy C − B0 ≥ +5 points with CI lower bound > 0; no family drops by more
  than 2 points; safety false-negative rates do not rise; decision-v7 test not below B0 → HIDDEN may be run.
- **HIDDEN** (about 1,200 questions, half in independent rendering styles, one shot): the same conditions, plus
  bootstrap one-sided p < 0.01 and a higher automation rate, before a "gain" is reported. Zero or negative results
  are reported as they are.

## Amendment 1 (2026-09-30, before any TRAIN/DEV/HIDDEN data was generated)

The evaluation implementation (`packages/decision-eval`) fills in details the pre-registration did not state:

- Temperature: B0 and C use the same grid 0.05–50, fitted on VAL by NLL; it does not change the argmax.
- Soft labels: those with a unique argmax take the argmax as their label; those without a unique argmax (ties /
  uniform) are excluded from accuracy, McNemar, safety false negatives and automation error, enter only Brier/ECE
  (against the soft target), and their number is reported per family.
- Automation rate "rises": compared as the family-equal-weighted mean coverage.
- Bootstrap one-sided p = (1 + #{delta ≤ 0}) / (B + 1), B = 10,000, fixed seed.
- decision-v7 regression guard: macro accuracy over families (`src`); C not below B0.
- B0 option labels: A.. for choice (two-letter single-token labels for > 26 options), no/yes for noul, digits for
  score; the three templates `direct`, `assistant`, `evidence` are frozen by SHA-256.

## Amendment 2 (2026-09-30, before any TRAIN/DEV/HIDDEN data was generated)

User clarification: fine-tune Qwen directly; Kev is only a reference (data format, typed questions, synthetic data,
the idea of calibrated probabilities); Kev's training code and pointer head are not used. Accordingly:

- **C** = `Qwen/Qwen3.8-27B@1d4bf0f2` + LoRA SFT (the TRL/PEFT SFT pipeline, extended to text decision data), one row
  per
  question: the prompt under the frozen template (the one B0 selected on VAL) → the option-label token, with loss on
  the label token only. Tied soft labels are not trained on; soft labels with a unique argmax are trained on their
  argmax.
- B0 and C use the **same** `predict-base` scorer and the same template; the only difference is whether the adapter
  is loaded.
- Kev-27B (`jaredpalmer/kev-27b@01b81998`), like `gpt-6-astra`, is only a reference point on DEV and takes no part in
  the gates.
- Metrics, gates and statistics as in amendment 1.

## Amendment 3 (2026-09-30, before any TRAIN/DEV/HIDDEN data was generated)

New reference point: **Jev**, hosted by TypeSafe (a System One model, official `typesafe-sdk`). Like `gpt-6-astra`
and Kev-27B it is a reference only, on DEV and the decision-v7 test, and takes no part in the gates; HIDDEN is not
sent to any external service.

## Migration note (2026-09-30, non-substantive)

At the user's request, the project code moved to the standalone repository `muse-decision-model`
(`packages/personal-decisions`, `packages/decision-eval`), and this file moved with it. The content above is
unchanged word for word.

## Amendment 4 (2026-09-30, before any TRAIN/DEV/HIDDEN data was generated): tightened per eval design / hill-climbing methodology

Based on Lance Martin, "Automating eval design and hillclimbing with Claude" (claude.dev, 2026-09-28), and the
`eval-audit` / `eval-hillclimb` guides of the claude-api skill.

**Size (noise floor).** A family-equal-weighted macro metric needs enough samples per family. DEV grows to about
1,400 questions (about 200 per family), HIDDEN to about 2,100 (about 300 per family). The 95% CI half-width of a
per-family paired difference is about ±7 points (DEV) / ±6 points (HIDDEN), about ±3 points for the macro mean.

**Gates become noise-aware** (replacing the original "no family drops by more than 2 points" and "safety
false-negative rates do not rise"):
- No family regression: for every family the 95% bootstrap CI upper bound of C − B0 is ≥ 0 (no significant
  regression), and the point estimate is ≥ −5 points.
- Safety false negatives: for each rate, the 95% CI lower bound of the difference (C − B0) is ≤ 0 (no significant
  rise).
- The rest (macro accuracy ≥ +5 points with CI lower bound > 0, the decision-v7 guard, HIDDEN p < 0.01 and the
  automation rate) is unchanged.

**Evaluation validity checks (must pass before the first paid run; results go into the receipt):**
1. Oracle: the labels themselves used as predictions through the full scoring path → 100% of scorable questions.
2. Null baselines: majority class / constant / shuffled labels → every family within chance ± noise; for
   `shortcut-baselines`, every sampled feature has a corresponding baseline and none exceeds chance by more than
   2 points.
3. Capability ladder: stronger models score higher and stay below the ceiling — the small model (Qwen3.5-4B
   zero-shot) ≤ B0 (Qwen3.8-27B) ≤ the references (gpt-6-astra / Jev / Kev-27B), with the strongest reference
   < 95% (headroom left). A family with an inverted ladder is checked for question/grader problems first, not by
   changing the model.
4. Label spot check: the auditor independently re-derives the labels of 50 DEV questions; a per-question LLM audit
   only flags "ambiguous / suspicious label / shortcut-prone", **never deletes questions based on whether any model
   answers them correctly**; flagged items are adjudicated by the auditor by hand and fixed in the generator (fix the
   generator, not individual questions), then regenerated.

**Hill-climbing protocol (the hillclimb train/test split):**
- VAL = the hill-climb's "readable set": the analysis agent reads only VAL's per-question failures; DEV = the
  hill-climb's "test set": each round sees only its scores and uses them to choose between rounds; HIDDEN = the final
  one-shot confirmation that takes part in no choice. Neither the orchestrator nor the data generators read DEV/HIDDEN
  per-question content.
- Each round changes one hypothesis only (data mix / share of hard cases / data volume / one hyper-parameter),
  described as **behaviour** at the generator and recipe level, never copying the content or entities of failed VAL
  questions into the training data.
- Keep rule: keep if DEV macro accuracy improves on the current best beyond noise (paired CI lower bound > 0) and VAL
  does not drop; if VAL rises while DEV is flat → treat as overfitting and roll back; any significant regression on
  either set → roll back.
- Build variance: before round 1, train the same recipe twice with different seeds and measure the DEV macro accuracy
  difference of "changing nothing"; a round whose expected gain is below that difference is not spent.
- Stopping: three consecutive rounds without a beyond-noise DEV gain (after a failure attribution: data gap / grader /
  structure / variance), or the budget is exhausted.
- Each round's state is written to disk (`hillclimb/<flow>/_state.json`, `vN/change.md`, `vN/results.jsonl`); the
  report is generated with the skill's own `build-report-lite.mjs`.

## Amendment 5 (2026-09-30, DEV generated, before any model prediction): a known small residual shortcut

In DEV's `shortcut-baselines` (generator cf90d13, seed 20261001, SHA `6f344562…`), the `pick_option` centroid
baseline (closest to the candidates' mean) scores 0.305 against a chance level of 0.279 (+2.5 points, n = 197, about
0.8 standard errors): slightly above amendment 4's "≤ 2 points" target but within noise. On TRAIN fixture samples the
same baseline is +1.6 to +2.1 points; it is a structural residual of candidate sampling. It can affect at most one of
the 7 families, moving the macro mean by ≤ 0.4 points, far less than the +5-point gate. Decision: DEV is not
regenerated and this is recorded as it is; HIDDEN and the final report report this baseline as well, noted next to
the `pick_option` results. All other families are within ± 0.3 points of chance; `validity` (oracle + null
baselines) passes.

## Amendment 6 (2026-10-01, before TEST was generated and before any model ran on TEST): a public test set TEST for the formal release

HIDDEN has been used once, by the rules, for this model's gate, and may not be sent to external services, so it
cannot be used to compare against Jev / gpt-6-astra. For the formal release (tech report + open source) a new public
test set TEST is built, and every model runs on it once.

**Construction** (by an independent TEST builder; the orchestrator and the training/evaluation executors do not read
TEST content before all predictions are frozen and see only the commitment file):
- Users: users **newly drawn** from the raw data, with user IDs disjoint from all of TRAIN / VAL / DEV / HIDDEN; size
  comparable to HIDDEN (about 720 users, about 2,400 questions, the 7 families in equal amounts, about 300+ questions
  per family), new seed; generator and family definitions identical to DEV/HIDDEN, with amendment 4's validity checks
  (oracle, null baselines, shortcut-baselines) and amendment 5's way of reporting.
- Rendering: users are split in two. The **A half** is rendered by gpt-6-astra (the same pipeline and public styles as
  TRAIN/DEV); the **B half** is rendered by Claude Opus 5.5 (the same rendering contract and public
  styles). Both halves go through the same anchor check; states that fail are dropped by the same rule and
  their number is recorded.
- When construction is complete, `TEST_COMMITMENT.json` (file hashes, counts, renderer distribution) is
  written and committed; TEST is not modified after that.

**Models evaluated (once each; after the results are out no model, template or scoring may be changed):** B0
`Qwen/Qwen3.8-27B`, `Qwen3.5-4B`, Kev-27B, this project's frozen r2-mix adapter (sha `6b07a7d3…`), Jev `jev-1.13.0`,
gpt-6-astra. Local models are scored by label-token log-probabilities with the temperatures and thresholds fitted on
VAL; API models as in the DEV reference comparison (one request per state, probabilities as returned, no VAL fit).

**Metrics and statistics:** the primary metric is the 7-family equal-weight macro accuracy; secondary metrics are
per-family accuracy, safety false-negative rates, Brier / ECE (API models uncalibrated, for reference only) and
automation coverage at the 5% budget (local models only). The adapter is compared pairwise with each of the other 5
models: state-clustered bootstrap, 10,000 resamples (seed 20260930), reporting the difference, 95% CI and one-sided p,
with Holm correction (5 comparisons). Reported overall and separately for the A half / B half; the B half is the
out-of-distribution result with "a different writing model", and also tests gpt-6-astra's home advantage on data it
rendered itself.

**Publication:** TEST is published with the tech report after all predictions are frozen, so it may be sent to the
Jev / gpt-6-astra APIs; the HIDDEN rules are unchanged.

## Amendment 7 (2026-10-01, TEST sealed, before any model ran on TEST): sensitivity analysis for the residual pick_option shortcut

In TEST's `shortcut-baselines` (commitment `TEST_COMMITMENT.json`, SHA `0b125c59…`; public
copy in `docs/TEST_COMMITMENT.json`), `pick_option`'s
"pick the priciest" baseline scores 0.347 against a chance level of 0.276 (+7.1 points, n = 300, about 2.7 standard
errors; A half +10.9, B half +2.9), above amendment 4's "≤ 2 points" target; it is the same before and after
pseudonymization and comes from the structure of candidate sampling, not from rendering. All other families are
within ± 0.5 points of chance. Decision: TEST is not regenerated (it is sealed) and this is reported as it is, in the
manner of amendment 5, with a new **sensitivity analysis**: besides the pre-registered 7-family macro accuracy
(primary metric, unchanged), the 6-family macro accuracy without `pick_option` and its paired comparisons (the same
bootstrap and Holm correction) are reported as well; every TEST result for `pick_option` carries a note on this
shortcut. If the conclusions of the primary metric and of the sensitivity analysis (the direction of each comparison
and whether it is significant after Holm) disagree, the report says so explicitly and the primary metric prevails.

## Release note (recorded after the fact, not pre-registered)

This section was written after all results existed; it is a record only and changes nothing that was pre-registered.
For the public release every published record went through `personal-decisions export-release` (user ids and detected
person names reversibly pseudonymized with a published key; e-mail addresses, phone numbers, URLs and street addresses
replaced by placeholders; options, labels and soft labels unchanged). TEST was pseudonymized **before** any model was
scored, so the released TEST is the scored TEST. VAL and DEV were pseudonymized for release **after** they were scored:
every VAL/DEV number in this document and in the report was computed on the pre-pseudonymization text, and the
released VAL/DEV were not re-scored, so re-running on them may give slightly different numbers.
