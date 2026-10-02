"""The Kev DEV reference (amendment 2; needs the `kev` extra): a trained Kev run directory (LoRA adapter + pointer
head) scored through kev's own loader and forward.

kev.predictors.LocalPredictor loads the run with kev.checkpoint (the base and revision head.pt names, the adapter, the
pointer head; bf16 when the run was trained on a bf16 backbone), encodes each record with kev's encoder and runs the
row form a hybrid Qwen3.5/3.8 backbone needs. It is asked for temperature 1.0, so the logits written are the pointer
head's raw (pre-temperature) logits; the temperature is refit on VAL like B0's. Records are admitted under kev's
serving context (states up to 64k tokens, never truncated): a record that does not fit fails loudly rather than being
dropped from the paired comparison.
"""
from .records import kev_request, question_of, record_id


def load_predictor(run, device, dtype=None):
    import torch
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT
    dtypes = {None: None, "bf16": torch.bfloat16, "fp32": torch.float32}
    return LocalPredictor(run, device, LoadOptions(temperature=1.0, dtype=dtypes[dtype]), context=SERVING_CONTEXT)


def record_logits(predictor, record):
    """-> [(question key, option keys, logits)] for one record, in question order."""
    out = predictor(kev_request(record))
    if out.get("inference_temperature", 1.0) != 1.0:
        raise ValueError("kev returned tempered logits; predict-kev needs the raw ones")
    rows = []
    for qid in record["questions"]:
        question = question_of(record, qid)
        logits = out["logits"][qid]
        if list(logits) != list(question.keys):
            raise ValueError("kev's option keys differ from the question's")
        rows.append(((record_id(record), str(qid)), question.keys, [float(logits[k]) for k in question.keys]))
    return rows
