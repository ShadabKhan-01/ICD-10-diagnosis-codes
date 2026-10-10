"""Regression tests for the LLaMA experiment layer (CPU only, no model weights).

Covers the chat-template BOS handling for the LLaMA-3.2 entry, that existing
HF models keep their previous tokenization behaviour, and that the shared
phase1 config is consistent across arms.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backends import HFBackend, create_backend  # noqa: E402

LLAMA32_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MESSAGES = [
    {"role": "system", "content": "You are an experienced clinical coder."},
    {"role": "user", "content": "Read the record. ANSWER:"},
]


class _FakeTokenizer:
    """Records the kwargs each encode call receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        self.calls.append(add_special_tokens)
        return {"input_ids": [[1, 2, 3]]}


def _cached_llama32_tokenizer():
    """Load the tokenizer from the local HF cache only; None if not cached."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(LLAMA32_ID, local_files_only=True)
    except Exception:
        return None


def test_llama32_entry_in_shared_config_uses_intended_model_and_no_extra_bos():
    cfg = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8"))
    entry = cfg["models"]["llama3.2-3b"]
    assert entry["model_id"] == LLAMA32_ID
    assert entry["template_mode"] == "system"
    assert entry["tokenize_special_tokens"] is False
    assert entry["stop_tokens"] == ["<|eot_id|>"]
    # Decoding budget is shared by all arms, not per model.
    assert cfg["decoding"]["max_new_tokens"] == 256


def test_llama3_8b_uses_no_extra_bos_and_others_keep_default():
    # LLaMA-3 family (8B and 3.2-3B): the template emits BOS, so no extra special tokens.
    cfg = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8"))
    assert cfg["models"]["llama3"]["tokenize_special_tokens"] is False
    assert cfg["models"]["llama3.2-3b"]["tokenize_special_tokens"] is False
    # BioMistral and Qwen: template not verified here, so their behaviour is unchanged.
    for key in ("biomistral", "qwen2.5-7b"):
        assert "tokenize_special_tokens" not in cfg["models"][key]
    backend = HFBackend("some/model")
    assert backend.tokenize_special_tokens is True


def test_encode_formatted_passes_flag_to_tokenizer():
    fake = _FakeTokenizer()
    b = HFBackend("x", tokenize_special_tokens=False)
    b.tokenizer = fake
    b.encode_formatted("<|begin_of_text|>hello")
    b.tokenizer.calls.clear()
    b2 = HFBackend("x")
    b2.tokenizer = fake
    b2.encode_formatted("hello")
    assert fake.calls == [True]


def test_create_backend_forwards_flag_to_hf_only():
    b = create_backend(LLAMA32_ID, backend_type="hf", tokenize_special_tokens=False)
    assert isinstance(b, HFBackend)
    assert b.tokenize_special_tokens is False
    # API backends must not receive the HF-only flag (it is consumed, not forwarded).
    g = create_backend("gemini-1.5-flash", api_key="fake_key")
    assert not hasattr(g, "tokenize_special_tokens")


def test_llama32_chat_template_has_single_bos_and_matches_count_path():
    tok = _cached_llama32_tokenizer()
    if tok is None:
        pytest.skip("unsloth/Llama-3.2-3B-Instruct tokenizer not in local HF cache")

    formatted = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    bos = tok.bos_token_id

    # Path used by the fixed generate(): no extra special tokens.
    fixed = HFBackend(LLAMA32_ID, tokenize_special_tokens=False)
    fixed.tokenizer = tok
    ids_fixed = fixed.encode_formatted(formatted)["input_ids"][0].tolist()

    # Path used by count_tokens(): add_special_tokens=False.
    count_ids = tok.encode(formatted, add_special_tokens=False)

    assert ids_fixed[0] == bos
    assert not (len(ids_fixed) > 1 and ids_fixed[1] == bos), "duplicate BOS"
    assert ids_fixed == count_ids, "prompt_tokens would disagree with the overflow check"

    # Default path (the previous behaviour for other HF models) does duplicate BOS
    # for this template. Documents why the flag exists.
    default = HFBackend(LLAMA32_ID)
    default.tokenizer = tok
    ids_default = default.encode_formatted(formatted)["input_ids"][0].tolist()
    assert ids_default[0] == bos and ids_default[1] == bos
