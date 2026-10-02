# Pev

**A calibrated fast-decision model for personal agents with long-term memory, and the pre-registered benchmark it
was evaluated on.**

[Paper (PDF)](docs/tech-report/pev.pdf) ·
[Model](https://huggingface.co/EnvLoop/Pev-27B-LoRA) ·
[Dataset](https://huggingface.co/datasets/EnvLoop/Pev-Bench) ·
[Leaderboard](https://huggingface.co/spaces/EnvLoop/Pev-Leaderboard) ·
[Collection](https://huggingface.co/collections/EnvLoop/pev-6abf32b6b1940477ad4c52c3) ·
[License](#license)

[![Tests](https://github.com/EnvLoop/Pev/actions/workflows/tests.yml/badge.svg)](https://github.com/EnvLoop/Pev/actions/workflows/tests.yml)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Model and data: CC BY-NC 4.0, gated](https://img.shields.io/badge/model%20%26%20data-CC%20BY--NC%204.0%2C%20gated-lightgrey.svg)](#license)

Personalized agents (assistants that remember one user and act for them across shopping, travel, email and calendar)
are becoming a mainstream direction. Their hard part is not generating text but **personal judgement**: doing what
*this* user would want. Asked to "re-order the dog food", an agent has to know that the user switched brands after an
allergy, that the new $54 bag now exceeds the user's $50 approval rule, that an address the user asked it to forget
must not be reused, and whether the request is worth interrupting the user for. An agent that asks about everything is
useless; one that acts on a wrong guess is unsafe.

[Pev-27B](https://huggingface.co/EnvLoop/Pev-27B-LoRA) answers these personal decisions as typed
questions in a single forward pass with calibrated probabilities, so an agent can automate the confident ones under an
error budget and ask about the rest. It is a LoRA adapter on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
evaluated on [Pev-Bench](https://huggingface.co/datasets/EnvLoop/Pev-Bench), a pre-registered benchmark
built from real public user behaviour. On a public test set of 720 new users it reaches 0.915 family-macro accuracy,
ahead of gpt-6-astra (0.873), Kev-27B (0.823), Jev (0.788) and its base model (0.762).

## Highlights

- **Benchmark from real public behaviour.** Users' histories come from Amazon Reviews 2023 and Google Local 2021;
  personal rules (approvals, privacy, contacts, notifications, forget requests) are synthetic and seeded per user.
  Labels are computed from a knowledge base, never by an LLM; an LLM only renders the state into natural text, and
  the evidence for every label must appear verbatim.
- **Pre-registered evaluation.** The [pre-registration](docs/PREREGISTRATION.md) was written before any evaluation
  data existed and was only extended by dated amendments. The model was selected on DEV, passed a one-shot HIDDEN
  gate, and was then compared once with five other models on a TEST set built after it was frozen and committed by
  hash ([docs/TEST_COMMITMENT.json](docs/TEST_COMMITMENT.json)).
- **Simple method.** Supervised fine-tuning on the single option-label token; prediction is a temperature-scaled
  softmax over the label tokens' log-probabilities.
- **Results.** Family-macro accuracy 0.915 on TEST, ahead of every compared model in both renderer halves, with
  automation coverage 0.94 at a 5% error budget.

## Task

Each record is one `state` (what the agent remembers, the current request and candidate actions, sometimes with
thousands of tokens of distractor text) with one to three typed questions:

| Family | Type | Question |
|---|---|---|
| `apply_memory` | yes/no | Is this remembered fact relevant to the request (or would using it over-personalise)? |
| `forgotten_violation` | yes/no | Does the action rely on something the user asked to forget? |
| `needs_approval` | yes/no | Must the agent ask before acting (spend threshold, new recipient, irreversible action)? |
| `share_ok` | yes/no | May this memory be shared with this recipient under the user's privacy settings? |
| `route` | choice | Which connected service handles this, or should the agent ask the user? |
| `notify_level` | score 0–3 | Silent, digest, notify or interrupt, given urgency, quiet hours, meetings and VIPs |
| `pick_option` | choice | Which option will this user actually choose and rate highly? (labels from real later behaviour) |

## Results

**TEST** (public; 720 new users, 980 states, 2,156 scorable questions; half A rendered by gpt-6-astra, half B by
Claude Opus 5.5; pseudonymized before any model was scored; every model run once). Family-macro accuracy and
Pev-27B's paired lead in points with a 95% state-clustered bootstrap CI (Holm-adjusted p = 5 × 10⁻⁴ for all
five comparisons):

| Model | All | A half (gpt-6-astra) | B half (Claude Opus 5.5) | Pev-27B − model, all |
|---|---|---|---|---|
| Qwen3.5-4B (zero-shot) | 0.652 | 0.627 | 0.677 | +26.3 [+24.1, +28.4] |
| Qwen3.8-27B (base, zero-shot, same prompt) | 0.762 | 0.759 | 0.766 | +15.2 [+13.6, +16.8] |
| Jev (`jev-1.13.0`) | 0.788 | 0.779 | 0.797 | +12.7 [+11.2, +14.1] |
| Kev-27B | 0.823 | 0.825 | 0.821 | +9.2 [+7.8, +10.5] |
| gpt-6-astra | 0.873 | 0.876 | 0.871 | +4.1 [+2.8, +5.5] |
| **Pev-27B** | **0.915** | **0.913** | **0.917** | — |

The pre-registered sensitivity analysis without `pick_option` (amendment 7) agrees in every comparison and half
(Pev-27B 0.987 on the other six families). `pick_option` stays hard for every model (0.41–0.48, against a
"priciest" shortcut of 0.347 and chance 0.276), so no progress is claimed there. Safety false negatives
(forgotten_violation / needs_approval / share_ok): Pev-27B 0% / 1.3% / 2.0%, gpt-6-astra 0.7% / 0% / 0%.
Full tables: [docs/RESULTS.md](docs/RESULTS.md) and [docs/TEST_RESULTS.json](docs/TEST_RESULTS.json).

![TEST family-macro accuracy per model, overall and per renderer half](docs/tech-report/figures/fig6_test_macro.png)

*TEST family-macro accuracy per model, overall and per renderer half (A: gpt-6-astra, B: Claude Opus 5.5).*

![Pev-27B minus each model on TEST with paired 95% CIs](docs/tech-report/figures/fig7_test_deltas.png)

*Pev-27B minus each model on TEST, family-macro accuracy with paired 95% CIs, overall and per half.*

**HIDDEN** (one-shot pre-registered gate; 960 states from 720 users, 2,106 scorable questions, half in rendering
styles never seen in training; not released): family-macro accuracy **0.754 → 0.905** (+15.1 points, 95% CI
[+13.5, +16.8], one-sided p = 1 × 10⁻⁴); all seven pre-registered checks pass. Automation coverage at a 5% error
budget rises from 0.54 to 0.93. `pick_option` does not improve significantly (0.40 → 0.43) and should remain "ask the
user" in deployment.

![Per-family accuracy on HIDDEN for the base model and Pev-27B](docs/tech-report/figures/fig1_hidden_accuracy.png)

*Per-family accuracy on HIDDEN for the base model (orange) and Pev-27B (blue), with chance marks.*

**DEV** (630 states, 1,409 scorable questions; the model-selection set): Qwen3.5-4B 0.638, Qwen3.8-27B (base) 0.766,
Jev 0.785, Kev-27B 0.824, gpt-6-astra 0.881, Pev-27B **0.907**. DEV and VAL numbers were computed on the
pre-pseudonymization text; the released VAL and DEV are pseudonymized (options and labels unchanged), so re-running on
them may differ slightly. TEST was pseudonymized before scoring (released TEST = scored TEST).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12, and one 80 GB GPU (H100/H200 class) for the model. The
model and the dataset are gated: accept their terms on the Hugging Face Hub first.

```bash
git clone https://github.com/EnvLoop/Pev && cd Pev
uv sync --project packages/decision-eval --extra fast          # --extra fast: Linux/CUDA kernels for Qwen3.8
alias hf="uv run --project packages/decision-eval hf"           # the Hugging Face CLI from the same environment
hf auth login
hf download EnvLoop/Pev-27B-LoRA --local-dir adapter
hf download EnvLoop/Pev-Bench --repo-type dataset --local-dir data
```

**Load the model and score one record** (run with `uv run --project packages/decision-eval python`; the same code as
[`usage_example.py`](https://huggingface.co/EnvLoop/Pev-27B-LoRA/blob/main/usage_example.py) on the Hub):

```python
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from decision_eval.conventions import question_keys
from decision_eval.prompts import MODEL, REVISION, render_prompt, verify_labels   # Qwen/Qwen3.8-27B @ 1d4bf0f2

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda"})
model = PeftModel.from_pretrained(base, "adapter", is_trainable=False).eval()
T = json.load(open("adapter/temperature.json"))["temperature"]               # 2.772, fitted on VAL

record = json.loads(open("data/test_a.jsonl").readline())
for qid, q in record["questions"].items():
    prompt, labels = render_prompt(record, qid, "direct", tok)               # frozen template, thinking disabled
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    probs = torch.softmax(torch.log_softmax(logits, -1)[verify_labels(tok, labels)] / T, -1)
    print(qid, dict(zip(question_keys(q["type"], q.get("criteria")), probs.tolist())), "label:", q.get("label"))
```

**Evaluate** the adapter against its base model on TEST (calibration and thresholds are refit on the released VAL;
roughly 20–30 minutes per predictor pass with `--extra fast`):

```bash
cat data/test_a.jsonl data/test_b.jsonl > data/test.jsonl     # byte-identical to the scored TEST file
uv run --project packages/decision-eval python training/evaluate_adapter.py \
    --adapter adapter --val data/val.jsonl --data data/test.jsonl --out eval/test
```

**Retrain** (the released TRAIN holds only Pev-Bench records; see below for the general mix):

```bash
uv run --project packages/decision-eval to-sft --data data/train.jsonl --template direct --out rows/train.jsonl
uv run --project packages/decision-eval python training/train_text_choice.py \
    --config training/configs/qwen3.8-27b-r2-mix-s20260930.json --train rows/train.jsonl --out runs/pev
```

The released adapter was also trained on 1,860 general decision questions from
[Kev](https://github.com/jaredpalmer/kev)'s `evals/decision-v2/train.jsonl`, which are not redistributed (third-party
licences). `scripts/general_mix.py` rebuilds that mix from Kev at the pinned commit (clone Kev into
`work/upstream/kev`, then `uv run --project packages/decision-eval python scripts/general_mix.py`), and the dataset's
`general_mix_ids.jsonl` lists the selected rows to check a rebuild against. `evaluate_adapter.py` refits the
temperature and thresholds on the released (pseudonymized) VAL, so they can differ slightly from the shipped values
(T = 2.772, threshold 0.369).

### Rebuilding the data from the public sources

```bash
python scripts/acquire/download.py && python scripts/acquire/{amazon,google_local,openflights,enron,finalize}.py
cd packages/personal-decisions
uv run personal-decisions population ... && uv run personal-decisions build ... && uv run personal-decisions split ...
uv run personal-decisions generate --shard ... --renderer astra ...      # needs API access to gpt-6-astra
uv run personal-decisions export-release --records F.jsonl --out F.release.jsonl \
    --items SHARD/items.jsonl --key-file ../../release/pseudonym_key.txt
```

`export-release` is a consistent, **reversible** pseudonymization, not anonymization: the key is published
([release/pseudonym_key.txt](release/pseudonym_key.txt)) so the export is exactly reproducible from the public source
datasets. See [docs/DATA.md](docs/DATA.md) and the package READMEs for every command.

## Repository structure

| Path | Contents |
|---|---|
| [`packages/personal-decisions/`](packages/personal-decisions/) | Knowledge base from public data, user-disjoint splits, the 7 question families, LLM rendering with verbatim anchor checks, privacy scan and release export |
| [`packages/decision-eval/`](packages/decision-eval/) | Base / adapter / reference predictors, VAL calibration and thresholds, paired scoring with a state-clustered bootstrap, gates, validity checks, TEST comparison |
| [`training/`](training/) | Standalone LoRA trainer (transformers + PEFT), the training config and the evaluation entry point |
| [`scripts/`](scripts/) | Source acquisition, Kev general-mix rebuild, TEST reproduction helpers |
| [`docs/`](docs/) | Pre-registration, results, data notes, dependency versions, TEST commitment and results |
| [`docs/tech-report/`](docs/tech-report/) | Technical report (Markdown, PDF, LaTeX source), figures, data snapshots and the scripts that generate them |
| [`cards/`](cards/) | Model and dataset cards and their licences (copies of the Hugging Face READMEs) |
| [`release/`](release/) | The published pseudonymization key |

Development: see [CONTRIBUTING.md](CONTRIBUTING.md). Both packages' unit tests run in CI
([.github/workflows/tests.yml](.github/workflows/tests.yml)).

## Data and licensing

- Pev-Bench quotes real public product and restaurant reviews (Amazon Reviews 2023, Google Local 2021) and
  Enron emails as distractor text; everything about a user other than their reviews is synthetic. Names and user ids
  are pseudonymized with a published key. Do not attempt to identify the people behind reviews or emails.
- The B half of TEST (`test_b.jsonl`, rendered by Claude Opus 5.5) is for evaluation only and must not be used for
  training.
- Removal requests and questions: research@envloop.ai.

## License

- Code: [Apache-2.0](LICENSE) (see also [NOTICE](NOTICE)).
- Model weights ([EnvLoop/Pev-27B-LoRA](https://huggingface.co/EnvLoop/Pev-27B-LoRA)) and dataset
  ([EnvLoop/Pev-Bench](https://huggingface.co/datasets/EnvLoop/Pev-Bench)): gated, non-commercial
  research use only — CC-BY-NC-4.0 for our contributions and the weights, with the additional conditions in
  [cards/MODEL_LICENSE.md](cards/MODEL_LICENSE.md) and [cards/DATASET_LICENSE.md](cards/DATASET_LICENSE.md); quoted
  third-party text stays under its owners' terms.
- Technical report (text, figures, data snapshots): [CC-BY-4.0](docs/tech-report/LICENSE.md).
- The base model [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is Apache-2.0 and is obtained under its own
  licence.

## Citation

```bibtex
@techreport{envloop_pev,
  title       = {Pev: A Calibrated Fast-Decision Model for Personal Agents},
  author      = {{EnvLoop Research}},
  institution = {EnvLoop},
  url         = {https://github.com/EnvLoop/Pev}
}
```

Please also cite the data sources and models below ([CITATION.cff](CITATION.cff) lists them).

## References

- Qwen Team. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Hugging Face model card.
- Qwen Team. [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Hugging Face model card.
- E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen.
  [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9). ICLR 2022.
- S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, B. Bossan, M. Tietz.
  [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods](https://github.com/huggingface/peft). 2022.
- J. Palmer. [Kev: Small Jev-like decision models you can train and run yourself](https://github.com/jaredpalmer/kev)
  and [Kev-27B](https://huggingface.co/jaredpalmer/kev-27b).
- TypeSafe AI. [System One](https://docs.typesafe.ai/concepts/system-one) (Jev).
- Meta. [Introducing Muse: The World's First Personal AI Agent Built for Everyone](https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/).
- Y. Hou, J. Li, Z. He, A. Yan, X. Chen, J. McAuley.
  [Bridging Language and Items for Retrieval and Recommendation](https://amazon-reviews-2023.github.io/)
  (Amazon Reviews 2023). 2024.
- J. Li, J. Shang, J. McAuley.
  [UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining](https://aclanthology.org/2022.acl-long.426/).
  ACL 2022 (Google Local).
- A. Yan, Z. He, J. Li, T. Zhang, J. McAuley.
  [Personalized Showcases: Generating Multi-Modal Explanations for Recommendations](https://dl.acm.org/doi/10.1145/3539618.3592036).
  SIGIR 2023 (Google Local).
- B. Klimt, Y. Yang.
  [The Enron Corpus: A New Dataset for Email Classification Research](https://doi.org/10.1007/978-3-540-30115-8_22).
  ECML 2004.
- OpenFlights. [Airport, airline and route data](https://openflights.org/data.php). ODbL 1.0.
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger.
  [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML 2017.
