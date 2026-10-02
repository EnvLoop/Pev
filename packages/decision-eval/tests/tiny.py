"""A tiny random Qwen3.5-architecture (Gated DeltaNet hybrid) text model with the real Qwen3.8-27B tokenizer, for CPU
tests of the GPU paths. The tokenizer comes from the local Hugging Face cache only (a few MB, never the weights); tests
skip when it is not cached."""
import atexit
import functools
import shutil
import tempfile
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]


@functools.cache
def cached_tokenizer_dir():
    """The cached tokenizer snapshot, or None when it is not in the local cache."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL, revision=REVISION, allow_patterns=TOKENIZER_FILES, local_files_only=True)
    except Exception:
        return None
    return path if all((Path(path) / name).exists() for name in TOKENIZER_FILES) else None


@functools.cache
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cached_tokenizer_dir())


@functools.cache
def tiny_model_dir():
    """A saved random 2-layer hybrid (one DeltaNet layer, one attention layer) with the real vocabulary."""
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(vocab_size=248320, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                               num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                               max_position_embeddings=4096, tie_word_embeddings=False, pad_token_id=248044)
    model = Qwen3_5ForCausalLM(config)
    directory = Path(tempfile.mkdtemp(prefix="tiny-qwen35-"))
    atexit.register(shutil.rmtree, directory, True)
    model.save_pretrained(directory)
    for name in TOKENIZER_FILES:
        shutil.copy(Path(cached_tokenizer_dir()) / name, directory / name)
    return str(directory)
