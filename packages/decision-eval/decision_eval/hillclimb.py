"""Export one scored round in the claude-api hillclimb layout (`eval-hillclimb.md`, `report/SCHEMA.md`), so the
skill's `build-report-lite.mjs <flow>` renders the rounds:

    <flow>/<variant>/results.jsonl   one row per question: {"prompt_id", "rep": 0, "prompt", "tags": [family, variant],
                                     "grade": {"correct", "brier"}, "model", "usage": {}, "meta"}
    <flow>/<variant>/summary.json    the variant header ({"description", "label", "target"}) and this package's totals
    <flow>/_state.json               created or updated: metrics (correct: binary; brier: float, lower is better) and
                                     the question ids of the split (VAL -> train_ids, DEV -> test_ids)

Split roles follow amendment 4: VAL is the hillclimb's readable set (prompts exported), DEV its test set (scores only;
the prompt is withheld so no DEV content reaches the session proposing changes). Ambiguous questions carry no
`correct` grade. Paths naming the sealed / hidden set are refused.
"""
import json
import re
from pathlib import Path

import numpy as np

from .conventions import render
from .metrics import family_accuracy, score_questions
from .records import record_id

VARIANT = re.compile(r"^(baseline|v\d+)$")
SPLITS = {"val": "train_ids", "test": "test_ids"}
METRICS = [{"id": "correct", "kind": "binary", "label": "Correct"},
           {"id": "brier", "kind": "float", "label": "Brier", "scale": 2, "better": "lower"}]


def readable_prompt(record, qid):
    q = record["questions"][qid]
    options = q.get("criteria")
    if q["type"] == "noul":
        options = {"true": (options or {}).get("true"), "false": (options or {}).get("false")}
    listed = "\n".join(f"- {k}: {render(v)}" if render(v) else f"- {k}" for k, v in
                       (options.items() if isinstance(options, dict) else enumerate(options)))
    return f"{render(record['state'])}\n\nQuestion ({q['type']}): {render(q.get('instructions'))}\nOptions:\n{listed}"


def result_rows(records, questions, scored, variant, split, model, identity):
    by_key = {(record_id(r), str(qid)): r for r in records for qid in r["questions"]}
    rows = []
    for i, q in enumerate(questions):
        grade = {"brier": float(scored.brier[i])}
        if not q.ambiguous:
            grade["correct"] = int(scored.correct[i])
        prompt = (readable_prompt(by_key[q.key], q.qid) if split == "val"
                  else f"[DEV question withheld; family {q.family}]")
        rows.append({"prompt_id": f"{q.record_id}/{q.qid}", "rep": 0, "prompt": prompt, "tags": [q.family, variant],
                     "grade": grade, "model": model, "usage": {},
                     "meta": {"split": split, "ambiguous": q.ambiguous, **identity}})
    return rows


def update_state(flow, ids, split):
    path = Path(flow) / "_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state.setdefault("metrics", METRICS)
    key = SPLITS[split]
    state[key] = sorted(set(state.get(key, [])) | set(ids))
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def export(records, questions, logits, identity, variant, split, flow, temperature=1.0, description=None,
           model=None, sha256=None):
    if not VARIANT.match(variant):
        raise ValueError("variant must be `baseline` or `v<N>`")
    if split not in SPLITS:
        raise ValueError("split must be val (readable) or test (DEV, prompts withheld)")
    scored = score_questions(questions, logits, temperature)
    model = model or "/".join(x for x in (identity["predictor"], identity["template"]) if x)
    rows = result_rows(records, questions, scored, variant, split, model, identity)
    directory = Path(flow) / variant
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "results.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    families = family_accuracy(questions, scored)
    summary = {"description": description or f"{model} predictions", "label": variant, "target": "code",
               "decision_eval": {"split": split, "n": len(rows), "n_ambiguous": sum(q.ambiguous for q in questions),
                                 "macro_accuracy": float(np.mean(list(families.values()))),
                                 "mean_brier": float(scored.brier.mean()), "temperature": temperature,
                                 "families": families, "inputs_sha256": sha256 or {}}}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_state(flow, [row["prompt_id"] for row in rows], split)
    return summary
