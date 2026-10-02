"""B0 and C: next-token log-probabilities of the option labels, from the base model alone (B0) or with a PEFT LoRA
adapter on it (C, docs/PREREGISTRATION.md amendment 2). The two differ only in the adapter.

The text-only causal LM is loaded in bf16 (AutoModelForCausalLM maps the Qwen3.5/3.8 `Qwen3_5ForConditionalGeneration`
checkpoint to its `Qwen3_5ForCausalLM` text model, as kev.model.DecisionModel does for its backbone). Prompts run
right-padded in length-sorted batches: padding sits after every real token, so causal attention and the Gated DeltaNet
recurrence never see it. Only the hidden state of each prompt's last token goes through the LM head (a [batch, vocab]
matrix instead of [batch, length, vocab]); the log-softmax over the whole vocabulary is taken in fp32 and each option's
logit is its label token's log-probability. An adapter (saved by PEFT/TRL from the same AutoModelForCausalLM text model)
is loaded unmerged on the bf16 base, as it was trained; its LoRA layers sit inside the backbone modules.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_BATCH_TOKENS = 32_768


def load_tokenizer(model_id, revision=None):
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


class BaseScorer:
    def __init__(self, lm, pad_id, device):
        self.lm, self.pad_id, self.device = lm.eval(), pad_id, device
        core = lm.get_base_model() if hasattr(lm, "get_base_model") else lm   # a PeftModel wraps the causal LM
        self.backbone = core.model if hasattr(core, "model") else core.get_decoder()
        self.head = core.get_output_embeddings()

    @classmethod
    def load(cls, model_id, revision, device, tok, dtype=torch.bfloat16, attn=None, adapter=None):
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        options = {"dtype": dtype, "attn_implementation": attn}
        if str(device).startswith("cuda"):
            options["device_map"] = {"": device}   # straight onto the GPU; no 54 GB staging copy in host memory
        lm = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **options)
        if not str(device).startswith("cuda"):
            lm.to(device)
        if adapter is not None:
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, adapter, is_trainable=False)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        return cls(lm, pad, device)

    def batches(self, rows, batch, max_tokens):
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i].ids))
        current = []
        for i in order:
            longest = len(rows[current[0]].ids) if current else len(rows[i].ids)
            if current and (len(current) >= batch or longest * (len(current) + 1) > max_tokens):
                yield current
                current = []
            current.append(i)
        if current:
            yield current

    @torch.no_grad()
    def label_logprobs(self, rows, batch=4, max_tokens=MAX_BATCH_TOKENS, progress=None):
        """-> one list of label log-probabilities per row, in row order."""
        if batch < 1:
            raise ValueError("batch must be >= 1")
        out = [None] * len(rows)
        done = 0
        for part in self.batches(rows, batch, max_tokens):
            length = max(len(rows[i].ids) for i in part)
            ids = torch.full((len(part), length), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(part), length), dtype=torch.long)
            for j, i in enumerate(part):
                ids[j, : len(rows[i].ids)] = torch.tensor(rows[i].ids)
                mask[j, : len(rows[i].ids)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            last = hidden[torch.arange(len(part), device=hidden.device), mask.sum(1) - 1]
            log_probs = torch.log_softmax(self.head(last).float(), -1).cpu()
            for j, i in enumerate(part):
                out[i] = log_probs[j, list(rows[i].label_ids)].tolist()
            done += len(part)
            if progress:
                progress(done, len(rows))
        return out
