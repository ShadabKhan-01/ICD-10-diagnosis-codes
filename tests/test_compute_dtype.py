"""Regression tests: the requested bitsandbytes compute dtype is the one in use.

The checkpoint's own quantization_config (bfloat16) used to override the dtype we
requested. These tests cover the helpers, and a real CPU load when the weights are
in the local HF cache (skipped otherwise). The float16-on-CUDA path cannot be tested
here; it is verified in Colab by the logged dtype.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backends import (  # noqa: E402
    HFBackend,
    actual_bnb_compute_dtype,
    select_bnb_compute_dtype,
    set_bnb_compute_dtype_in_config,
)

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"


def test_set_dtype_in_dict_config_overrides_checkpoint_value():
    cfg = SimpleNamespace(quantization_config={"bnb_4bit_compute_dtype": "bfloat16",
                                               "quant_method": "bitsandbytes"})
    assert set_bnb_compute_dtype_in_config(cfg, "float16") is True
    assert cfg.quantization_config["bnb_4bit_compute_dtype"] == "float16"
    assert cfg.quantization_config["quant_method"] == "bitsandbytes"  # other keys kept


def test_set_dtype_in_object_config_overrides_checkpoint_value():
    qc = SimpleNamespace(bnb_4bit_compute_dtype="bfloat16")
    cfg = SimpleNamespace(quantization_config=qc)
    assert set_bnb_compute_dtype_in_config(cfg, "float16") is True
    assert qc.bnb_4bit_compute_dtype == "float16"


def test_config_without_quantization_is_left_alone():
    cfg = SimpleNamespace()
    assert set_bnb_compute_dtype_in_config(cfg, "float16") is False


class _Linear4bit:
    def __init__(self, compute_dtype):
        self.compute_dtype = compute_dtype


class _FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


def test_actual_dtype_reads_linear4bit_layers():
    import torch
    Linear4bit = type("Linear4bit", (), {})
    layer = Linear4bit()
    layer.compute_dtype = torch.float16
    assert actual_bnb_compute_dtype(_FakeModel([object(), layer])) == "float16"


def test_actual_dtype_none_without_4bit_layers():
    assert actual_bnb_compute_dtype(_FakeModel([object()])) is None


def test_backend_default_dtype_is_paper_setting_before_load():
    # Before load() there is no model; the default is the paper's float16 setting.
    # load() replaces it with the dtype actually in use.
    assert HFBackend(MODEL_ID).info["dtype"] == "float16"


def _weights_cached() -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return False
    snap = Path(HF_HUB_CACHE) / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    if not snap.exists():
        return False
    return any((s / "model.safetensors").exists() for s in snap.iterdir())


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_dtype_selection_is_float16_on_cuda_and_float32_without():
    # The decision is made on CUDA availability, not on allow_cpu.
    assert select_bnb_compute_dtype(True) == "float16"
    assert select_bnb_compute_dtype(False) == "float32"


@pytest.mark.skipif(_cuda_available(), reason="CPU path only runs on machines without CUDA")
@pytest.mark.skipif(not _weights_cached(), reason="LLaMA-3.2-3B weights not in local HF cache")
def test_cpu_load_uses_requested_float32_not_checkpoint_bfloat16():
    # No CUDA here, so the CPU smoke path must load in float32 (not the checkpoint's bf16).
    b = HFBackend(MODEL_ID, allow_cpu=True, tokenize_special_tokens=False)
    b.load()
    assert b.info["device"] == "cpu (smoke only)"
    assert b.info["dtype"] == "float32"
    assert actual_bnb_compute_dtype(b.model) == "float32"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA path needs a GPU (e.g. Colab T4)")
@pytest.mark.skipif(not _weights_cached(), reason="LLaMA-3.2-3B weights not in local HF cache")
def test_cuda_load_uses_float16_compute():
    # Research setting (configs/phase1.yaml): NF4 with float16 compute on the GPU,
    # overriding the checkpoint's bfloat16 config.
    b = HFBackend(MODEL_ID, tokenize_special_tokens=False)
    b.load()
    assert b.info["device"] != "cpu (smoke only)"
    assert b.info["dtype"] == "float16"
    assert actual_bnb_compute_dtype(b.model) == "float16"
