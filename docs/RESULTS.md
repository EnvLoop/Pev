# Results

Public summary of every scored run, in the order they happened. It replaces the internal run log, which also holds
infrastructure operations (machine provisioning, provider accounts, incident handling) that are not part of the study.
Every number here also appears in the technical report ([`tech-report/report.md`](tech-report/report.md)), which gives
the methods, figures and sources; the GPU cost of the whole project is itemized in its Appendix D.

All accuracies are family-macro accuracy over the 7 question families (equal weight; questions with tied soft labels
excluded). Paired comparisons use the state-clustered bootstrap (10,000 resamples, seed 20260930).

> DEV and VAL numbers were computed on the **pre-pseudonymization** text. The released VAL/DEV are pseudonymized
> (person names, contact details and user ids; options and labels unchanged) and were not re-scored, so re-running on
> them may differ slightly. TEST was pseudonymized **before** any model was scored (released TEST == scored TEST).

## 1. Baseline (DEV, 1,409 scorable questions)

- Prompt template chosen on VAL and frozen: `direct` (VAL macro 0.759; `assistant` 0.748, `evidence` 0.742).
- DEV macro accuracy: Qwen3.5-4B **0.638** < B0 Qwen3.8-27B **0.766** < Kev-27B **0.824**. `validity` passed
  (oracle, null baselines, monotone capability ladder, no family inversion, strongest model below 95%).
  Noise floor (minimum detectable paired gain) about 2.7 points.
- B0 per family: apply_memory 0.850, forgotten_violation 0.966, needs_approval 0.892, notify_level 0.515,
  pick_option 0.376 (chance about 0.28; centroid shortcut 0.305, amendment 5), route 0.919, share_ok 0.847.
- B0 calibration: Brier 0.261, ECE 0.074, automation coverage at a 5% error budget 0.57. Guard (Kev `decision-v7`
  test) macro 0.834.
- Gate target for round 1: DEV macro >= 0.816 (+5 points) with CI lower bound > 0.

## 2. HIDDEN built (sealed; aggregates only)

- 960 states / 2,456 questions from 720 users, 2,106 scorable (275–322 per family). Commitment SHA-256
  `caf14a06…b773`. 514 states use public rendering styles and 446 private ones.
- Balance and `validity` (oracle + null baselines) pass. Known residual: `pick_option` medoid shortcut +3.3 points
  (about 1.3 SE, n = 299), the same structural residual as on DEV (amendment 5).

## 3. Round 1 (LoRA SFT on decision TRAIN; 4,341 rows, 543 steps, two seeds)

| Training | DEV macro | Δ vs B0 | Guard | Decision |
|---|---|---|---|---|
| Seed A (20260930) | 0.906 | +13.9 | 0.834 -> 0.829 | rollback (guard) |
| Seed B (20261003) | 0.903 | +13.6 | 0.834 -> 0.829 | rollback (guard) |

- Every pre-registered DEV check passed except "guard not lower" (−0.5 points: within noise, but the rule is "not
  lower" and was not relaxed after the fact).
- Build variance (seed A vs seed B): 0.3 points macro, far below the noise floor.
- Families (B0 -> A / B): apply_memory 0.85 -> 1.00 / 1.00, forgotten_violation 0.97 -> 1.00 / 1.00, needs_approval
  0.89 -> 0.995 / 1.00, share_ok 0.85 -> 0.995 / 0.995, route 0.92 -> 0.985 / 0.99, notify_level 0.52 -> 0.91 / 0.95,
  pick_option 0.38 -> 0.46 / 0.38. Brier 0.261 -> 0.134 / 0.133; coverage 0.57 -> 0.92 / 0.93.
- Reading: the rule-derived families saturate, `pick_option` (real behaviour) barely moves, and general decision
  ability slips slightly. Round 2 tests one change each: a general-decision data mix, and more decision data.

## 4. Round 2

| Training | Rows / steps | DEV macro | Δ vs B0 [95% CI] | Guard | Decision |
|---|---|---|---|---|---|
| Mix (+30% Kev general decision rows) | 6,201 / 776 | **0.907** | +14.1 [+12.0, +16.1] | 0.834 -> **0.848** | **advance to HIDDEN** |
| More decision data (2,729 records) | 6,013 / 752 | 0.908 | +14.2 [+12.1, +16.2] | 0.8339 -> 0.8336 | rollback (guard) |

- Mix, per family (B0 -> mix): apply_memory 0.85 -> 1.00, forgotten_violation 0.97 -> 1.00, needs_approval 0.89 ->
  0.995, share_ok 0.85 -> 0.995, route 0.92 -> 0.99, notify_level 0.52 -> 0.91, pick_option 0.38 -> 0.46. Brier
  0.261 -> 0.131, ECE 0.074 -> 0.066, coverage 0.57 -> 0.93.
- Caveat: the general mix comes from Kev `decision-v2/train`, the train side of the suite whose test side is the guard
  (disjoint items, n-gram audited), so the guard gain is partly in-distribution: it shows the forgetting is stopped,
  not that general decision ability improved.

**Decision (plateau).** All four DEV macros lie within 0.5 points of each other (0.903–0.908), below the noise floor;
`pick_option` ranges 0.38–0.46 with no consistent gain from more data. The only change that moved a gate was the
general mix (guard −0.5 -> +1.4). The round-2 mix adapter (archive SHA-256 `6b07a7d3…`) was frozen as the final
candidate and climbing stopped.

## 5. HIDDEN one-shot gate (final)

- 960 states from 720 users, 2,456 questions, 2,106 scorable; 934 state clusters with at least one scorable question.
  The run returned aggregate-only outputs; the sealed set was deleted afterwards and is not released.
- Macro accuracy 0.754 -> **0.905** (+15.1 points, CI95 [+13.5, +16.8], one-sided p = 1.0 × 10⁻⁴, the floor for
  10,000 resamples); McNemar 359 vs 48 (p = 5.7 × 10⁻⁶⁰). **All 7 pre-registered gate checks pass.**
- Families (B0 -> adapter): apply_memory 0.84 -> 1.00; forgotten_violation 0.96 -> 1.00; needs_approval 0.83 -> 0.99;
  share_ok 0.81 -> 0.97; route 0.90 -> 0.98; notify_level 0.55 -> 0.97; pick_option 0.40 -> 0.43 (CI [−3.1, +9.6],
  not significant).
- Safety false negatives fall: forgotten_violation 8.8% -> 0%, share_ok 11.0% -> 3.9%, needs_approval 1.3% -> 0%.
- Automation coverage at the 5% budget 0.54 -> 0.93. Guard 0.834 -> 0.848 (+1.4; partly in-distribution, above).
- HIDDEN matches DEV (0.907) within 0.2 points, so the DEV hill-climb did not overfit. `pick_option` is unsolved and
  its VAL threshold over-automates it (realized error 44.7% on HIDDEN and 43.8% on DEV against the 5% budget):
  `pick_option` must stay "ask the user" in deployment.

## 6. Reference comparison (DEV, after the HIDDEN gate)

Same DEV set (SHA-256 `6f344562…`, 630 states, 1,409 scorable questions), same scorer (`decision-eval
score-reference`). Aggregates: [`reference-dev.json`](reference-dev.json).

| Predictor | Macro | apply_memory | forgotten_violation | needs_approval | notify_level | pick_option | route | share_ok |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 0.638 | 0.737 | 0.737 | 0.686 | 0.598 | 0.340 | 0.731 | 0.636 |
| B0 Qwen3.8-27B | 0.766 | 0.850 | 0.966 | 0.892 | 0.515 | 0.376 | 0.919 | 0.847 |
| Jev `jev-1.13.0` | 0.785 | 0.944 | 0.995 | 0.923 | 0.552 | 0.376 | 0.868 | 0.837 |
| Kev-27B | 0.824 | 0.972 | 1.000 | 0.954 | 0.577 | 0.396 | 0.909 | 0.962 |
| gpt-6-astra | 0.881 | 0.977 | 0.995 | 0.995 | 0.830 | 0.442 | 0.954 | 0.971 |
| **Adapter r2-mix** | **0.907** | 1.000 | 1.000 | 0.995 | 0.912 | 0.457 | 0.990 | 0.995 |

- Paired: adapter − gpt-6-astra **+2.7 points** [+1.1, +4.2], p = 7 × 10⁻⁴; adapter − Jev **+12.2** [+10.4, +14.0],
  p < 10⁻⁴. Versus gpt-6-astra the adapter is ahead on notify_level (+8.2), route (+3.6), share_ok (+2.4) and
  apply_memory (+2.3), tied on needs_approval and forgotten_violation, and pick_option +1.5 [−6.3, +9.4] (n.s.).
- Safety false-negative rates (forgotten_violation / needs_approval / share_ok): adapter 0 / 0 / 1.0%; gpt-6-astra
  1.0% / 0 / 1.0%; Kev-27B 0 / 0 / 7.7%; Jev 0 / 0 / 27.9%; B0 6.9% / 2.1% / 4.8%.
- Calibration (macro Brier / ECE): adapter 0.131 / 0.066; gpt-6-astra 0.178 / 0.075; Kev-27B 0.213 / 0.075;
  Jev 0.279 / 0.120 (API models uncalibrated: probabilities as returned, T = 1, no VAL fit, no automation threshold).
- Caveats: the API models see each state once with all its questions, the local models one question per prompt;
  gpt-6-astra rendered DEV and so scores text it wrote; DEV was the adapter's hill-climb test set (selection bias,
  which HIDDEN bounds). No question was unscorable (three gpt-6-astra answers needed the parser to accept an option
  echoed as its full `key: description` line). The references were run on DEV only, never on HIDDEN.

## 7. TEST (amendments 6 and 7)

**Reproduction check before TEST.** The four local models were re-run on DEV on a different GPU type (A100 instead of
H100) with the same scorer, template, adapter and temperatures: argmax agreement with the frozen DEV predictions
1,616/1,626 for B0 (macro 0.768 vs 0.766) and 1,619/1,626 for the adapter (0.909 vs 0.907; Brier and coverage
identical to three decimals). Every flip is a near-tie in the frozen predictions (top-2 gap <= 0.054), caused by bf16
kernels on a different GPU. All local TEST predictions were therefore produced on a single machine.
`decision-eval compare-release` reproduces the DEV reference comparison of §6 exactly.

**TEST.** 980 states / 2,506 questions / 2,156 scorable from 720 new users (commitment
[`TEST_COMMITMENT.json`](TEST_COMMITMENT.json), a public copy without internal infrastructure details of the original
commitment SHA-256 `0b125c59…0fbb`; data-file hashes unchanged; scored file SHA-256 `131d0927…79a3`, the byte
concatenation of the released `test_a.jsonl` and `test_b.jsonl`). Half A (gpt-6-astra-rendered) 490 states / 1,073
scorable; half B (Claude Opus 5.5-rendered) 490 / 1,083. Every model ran once; all predictions were frozen
([`TEST_PREDICTIONS.sha256`](TEST_PREDICTIONS.sha256)) before any scoring. Scored with `decision-eval compare-release`
(spec [`../scripts/test_eval/release-spec.test.json`](../scripts/test_eval/release-spec.test.json); VAL-fitted
temperatures and thresholds for the local models, API models at T = 1). Aggregates:
[`TEST_RESULTS.json`](TEST_RESULTS.json).

| Model | Macro (7 fam.) | A | B | Macro excl. pick_option | A | B | Brier | ECE | Coverage@5% |
|---|---|---|---|---|---|---|---|---|---|
| **Adapter r2-mix** | **0.915** | 0.913 | 0.917 | **0.987** | 0.987 | 0.987 | 0.125 | 0.065 | 0.94 |
| gpt-6-astra | 0.873 | 0.876 | 0.871 | 0.945 | 0.946 | 0.944 | 0.180\* | 0.074\* | — |
| Kev-27B | 0.823 | 0.825 | 0.821 | 0.888 | 0.885 | 0.891 | 0.218 | 0.074 | 0.71 |
| Jev `jev-1.13.0` | 0.788 | 0.779 | 0.797 | 0.851 | 0.837 | 0.864 | 0.266\* | 0.123\* | — |
| B0 Qwen3.8-27B | 0.762 | 0.759 | 0.766 | 0.818 | 0.811 | 0.824 | 0.264 | 0.067 | 0.56 |
| Qwen3.5-4B | 0.652 | 0.627 | 0.677 | 0.686 | 0.660 | 0.711 | 0.389 | 0.074 | 0.11 |

\* Uncalibrated (API probabilities as returned).

Adapter − model, family-macro accuracy in points [95% CI]; Holm over the 5 comparisons per column; every one-sided
p <= 3 × 10⁻⁴ and every Holm-adjusted p = 5 × 10⁻⁴:

| vs | All (7 fam.) | A | B | All (6 fam.) | A (6) | B (6) |
|---|---|---|---|---|---|---|
| gpt-6-astra | +4.1 [+2.8, +5.5] | +3.7 [+1.9, +5.5] | +4.6 [+2.6, +6.6] | +4.2 [+3.1, +5.3] | +4.1 | +4.3 |
| Kev-27B | +9.2 [+7.8, +10.5] | +8.8 | +9.6 | +9.9 [+8.7, +11.2] | +10.3 | +9.6 |
| Jev | +12.7 [+11.2, +14.1] | +13.3 | +12.0 | +13.7 [+12.2, +15.1] | +15.0 | +12.3 |
| B0 | +15.2 [+13.6, +16.8] | +15.4 | +15.1 | +16.9 [+15.4, +18.6] | +17.6 | +16.3 |
| Qwen3.5-4B | +26.3 [+24.1, +28.4] | +28.5 | +24.1 | +30.1 [+27.9, +32.3] | +32.8 | +27.6 |

- Amendment 7: the 6-family sensitivity analysis agrees with the primary 7-family result in every comparison and half
  (same sign, all Holm-significant).
- `pick_option` (residual "priciest" shortcut on TEST: 0.347 vs chance 0.276; A +10.9, B +2.9 points): adapter 0.480,
  Qwen3.5-4B 0.450, gpt-6-astra 0.443, Kev-27B 0.433, B0 0.430, Jev 0.413; adapter − gpt-6-astra +3.7 [−3.1, +10.3],
  adapter − B0 +5.0 [−1.2, +11.2]: not significant. The adapter's VAL threshold again over-automates `pick_option`
  (realized error 37% against the 5% budget).
- Versus gpt-6-astra by family: notify_level +15.7, route +8.0, apply_memory +2.3; needs_approval and
  forgotten_violation tied; share_ok −1.0 [−2.3, 0.0].
- Safety false negatives (forgotten_violation / needs_approval / share_ok, about 150 at risk each): adapter 0 / 1.3% /
  2.0%; gpt-6-astra 0.7% / 0 / 0; Kev-27B 0 / 2.0% / 4.6%; Jev 0 / 4.0% / 21.9%; B0 6.0% / 4.7% / 4.6%.
- The halves agree: the adapter's lead over gpt-6-astra is +3.7 points on gpt-6-astra's own renders (A) and +4.6 on
  Claude Opus 5.5 renders (B): no visible home advantage for gpt-6-astra and no drop for the adapter on an unseen
  writer.
- Process: one adapter launch failed with a CUDA out-of-memory error while loading weights (the GPU was still busy
  with the B0 run), before any question was scored; it produced no output and was discarded. Every model then produced
  exactly one prediction set. gpt-6-astra needed two extra calls within its 980 record requests (parser retries);
  neither API model had an unscorable record.

## Errata

- An earlier running cost total was overstated by an arithmetic error; the itemized figures in the technical report
  (Appendix D) are authoritative. Per-run GPU costs are estimates from hourly price × runtime.
- HIDDEN: "960 HIDDEN users" in an early summary should read 960 HIDDEN **states** from 720 users; the bootstrap used
  934 clusters (states with at least one scorable question).
- The "7k" variant trained on 6,013 rows (6,946 questions), not 7k rows.
- `pick_option` over-automation was already visible on DEV (realized error 43.8% at the VAL threshold), not only on
  HIDDEN.

## Protocol deviations (no result was changed because of them)

- Amendment 4's stop rule says stop after 3 rounds without a beyond-noise gain; climbing stopped after 2 rounds (all
  four DEV candidates within 0.5 points of each other), to spend the remaining effort on the one-shot gate.
- r2-mix was chosen because it was the only candidate passing every pre-registered DEV gate check (including the
  guard), not by amendment 4's keep rule (its DEV gain over round 1 was within noise).
- Amendment 3 lists Jev as a reference on DEV and on the `decision-v7` guard; Jev and gpt-6-astra were run on DEV
  only (and later on TEST).
- Amendment 5 covers DEV only; on VAL the best `pick_option` shortcut ("cheapest") is +4.2 points over chance (target
  <= 2). The split audit also found 2 target review texts shared between VAL and TRAIN (different users).
- Amendment 7: the TEST `pick_option` "priciest" shortcut (+7.1 points) exceeds amendment 4's 2-point target. TEST
  was already sealed and was not regenerated; the 6-family sensitivity analysis was added before any model ran on TEST.
