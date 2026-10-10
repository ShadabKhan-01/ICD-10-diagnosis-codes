"""Regression tests: generation-limit truncation detection and cut-off parsing.

Covers the LLaMA debugging findings:
- `truncated` only reflected prompt overflow; generation hitting max_new_tokens
  was never recorded (now `generation_truncated`).
- A code cut off at the token limit (e.g. "E11" from "E11.9") was extracted as a
  prediction by the regex fallback.
- gpu_info() used a non-existent torch attribute (total_mem).
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backends import HFBackend, detect_generation_truncated  # noqa: E402
from parser import parse_output  # noqa: E402
import utils  # noqa: E402

EOT = 128009
STOP = [128009]


# ── detect_generation_truncated ─────────────────────────────────────────────

def test_hitting_token_limit_without_stop_is_truncated():
    assert detect_generation_truncated([5, 6, 7, 8], max_new_tokens=4, stop_ids=STOP) is True


def test_stop_token_at_limit_is_complete():
    assert detect_generation_truncated([5, 6, 7, EOT], max_new_tokens=4, stop_ids=STOP) is False


def test_short_output_is_complete():
    assert detect_generation_truncated([5, 6, EOT], max_new_tokens=256, stop_ids=STOP) is False


def test_empty_output_is_not_flagged():
    assert detect_generation_truncated([], max_new_tokens=256, stop_ids=STOP) is False


# ── HFBackend.generate uses the detector (fake model, real code path) ──────

class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = EOT
    unk_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "formatted prompt"

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return _FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def convert_tokens_to_ids(self, name):
        return EOT if name == "<|eot_id|>" else 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class _FakeModel:
    device = "cpu"

    def __init__(self, new_ids):
        self._new_ids = new_ids

    def generate(self, input_ids=None, max_new_tokens=None, **kwargs):
        new = torch.tensor([self._new_ids[:max_new_tokens]])
        return torch.cat([input_ids, new], dim=1)


def _backend_with(new_ids):
    b = HFBackend("fake/model", stop_tokens=["<|eot_id|>"])
    b.tokenizer = _FakeTokenizer()
    b.model = _FakeModel(new_ids)
    return b


def test_generate_flags_truncation_when_limit_reached():
    b = _backend_with([10, 11, 12, 13])  # 4 tokens, none is EOT
    res = b.generate([{"role": "user", "content": "x"}], max_new_tokens=4)
    assert res.new_tokens == 4
    assert res.generation_truncated is True


def test_generate_not_flagged_when_model_stops_on_its_own():
    b = _backend_with([10, 11, EOT])
    res = b.generate([{"role": "user", "content": "x"}], max_new_tokens=256)
    assert res.generation_truncated is False


def test_generate_not_flagged_when_stop_token_is_last_at_limit():
    b = _backend_with([10, 11, 12, EOT])
    res = b.generate([{"role": "user", "content": "x"}], max_new_tokens=4)
    assert res.generation_truncated is False


# ── parser: cut-off final token ────────────────────────────────────────────

def test_cut_off_partial_code_is_dropped_when_truncated():
    raw = '{"codes": ["E11.9", "I10", "E11'
    codes, mode = parse_output(raw, generation_truncated=True)
    assert mode == "regex"
    assert codes == ["E11.9", "I10"]


def test_cut_off_after_dot_is_dropped_when_truncated():
    codes, _ = parse_output('"E11.9", "I10", "E11.', generation_truncated=True)
    assert codes == ["E11.9", "I10"]


def test_without_truncation_flag_partial_code_is_kept_as_before():
    # Unchanged behaviour for complete outputs and for existing rows (default False).
    codes, _ = parse_output('"E11.9", "I10", "E11')
    assert codes == ["E11.9", "I10", "E11"]


def test_complete_regex_output_unchanged_when_truncated_flag_set():
    # Last code is followed by a closing quote, so it is complete and kept.
    codes, _ = parse_output('"E11.9", "I10"', generation_truncated=True)
    assert codes == ["E11.9", "I10"]


def test_complete_json_unchanged_by_truncation_flag():
    codes, mode = parse_output('{"codes": ["E11.9", "I10"]}', generation_truncated=True)
    assert mode == "json"
    assert codes == ["E11.9", "I10"]


# ── utils.gpu_info uses the real torch attribute ───────────────────────────

class _Props:
    total_memory = 15 * 1024 ** 3
    name = "Tesla T4"


def test_gpu_info_uses_total_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _Props())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    info = utils.gpu_info()
    assert info["gpu_vram_mb"] == 15 * 1024
    assert info["gpu_name"] == "Tesla T4"
