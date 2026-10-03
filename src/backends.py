"""LLM backend abstraction: HFBackend (NF4 4-bit) and MockBackend.

Paper setting: NF4 quantisation, greedy decoding, max_new_tokens=256, batch_size=1.
"""

import gc
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenResult:
    """Result of a single LLM generation."""
    text: str
    prompt_tokens: int
    new_tokens: int
    latency_s: float


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer into memory."""
        ...

    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tokens in the formatted prompt."""
        ...

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        """Generate text from messages. Returns GenResult."""
        ...

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Return model metadata: model_id, revision, quantisation, device, etc."""
        ...

    @property
    @abstractmethod
    def context_limit(self) -> int:
        """Maximum context length in tokens."""
        ...

    @property
    def model_name(self) -> str:
        """Short name for display."""
        return self.info.get("model_id", "unknown")


class HFBackend(LLMBackend):
    """HuggingFace Transformers backend with NF4 4-bit quantisation.

    Paper setting:
    - BitsAndBytesConfig: load_in_4bit, nf4, float16 compute, double quant
    - Greedy decoding (do_sample=False)
    - Batch size 1
    - Deterministic seeding before each generation
    """

    def __init__(self, model_id: str, context_limit_tokens: int = 8192,
                 seed: int = 42, stop_tokens: Optional[List[str]] = None,
                 template_mode: str = "system"):
        self.model_id = model_id
        self._context_limit = context_limit_tokens
        self.seed = seed
        self.stop_token_names = stop_tokens or []
        self.template_mode = template_mode
        self.model = None
        self.tokenizer = None
        self.model_revision: Optional[str] = None
        self._device_name: Optional[str] = None

    def load(self) -> None:
        """Load model with NF4 4-bit quantisation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from huggingface_hub import model_info

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. NF4 quantisation requires CUDA.\n"
                "Use --backend mock for CPU-only testing.\n"
                "See README.md for hardware requirements."
            )

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set. Gated models (e.g., LLaMA-3) will fail.\n"
                "Accept the licence at the model page, then: export HF_TOKEN=<your_token>"
            )

        # Resolve exact commit SHA
        try:
            info = model_info(self.model_id, token=hf_token)
            self.model_revision = info.sha
            logger.info(f"Resolved {self.model_id} → commit {self.model_revision}")
        except Exception as e:
            logger.warning(f"Could not resolve model revision: {e}")
            self.model_revision = "unknown"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        logger.info(f"Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=hf_token, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model {self.model_id} with NF4 4-bit...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=False,
        )
        self.model.eval()

        self._device_name = torch.cuda.get_device_name(0)
        logger.info(f"Model loaded on {self._device_name}")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens using the model's chat template."""
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        return len(token_ids)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        """Greedy generation. Seeds torch before each call for determinism."""
        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Format and tokenize
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Build stop token IDs
        stop_ids = [self.tokenizer.eos_token_id]
        for tok_name in self.stop_token_names:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok_name)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                stop_ids.append(tok_id)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
            )
        latency = time.time() - start

        # Decode ONLY the new tokens (strip prompt)
        new_token_ids = outputs[0][prompt_tokens:]
        new_tokens = len(new_token_ids)
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=latency,
        )

    @property
    def info(self) -> Dict[str, Any]:
        info = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantisation": "nf4_4bit",
            "dtype": "float16",
            "device": self._device_name or "unknown",
            "backend": "hf",
            "template_mode": self.template_mode,
        }
        try:
            import torch
            import transformers
            info["torch_version"] = torch.__version__
            info["transformers_version"] = transformers.__version__
        except ImportError:
            pass
        return info

    @property
    def context_limit(self) -> int:
        return self._context_limit

    def free_memory(self) -> None:
        """Release GPU memory between models."""
        import torch
        logger.info(f"Freeing GPU memory for {self.model_id}")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MockBackend(LLMBackend):
    """Deterministic mock backend for CPU-only testing.

    - Returns fixed codes: 2 common valid codes + 1 invalid code (Z99.999)
    - All results carry backend='mock'
    - Goes to results_mock/ directory
    """

    def __init__(self, model_id: str = "mock_model",
                 context_limit_tokens: int = 8192,
                 seed: int = 42,
                 template_mode: str = "system"):
        self.model_id = model_id
        self._context_limit = context_limit_tokens
        self.seed = seed
        self.template_mode = template_mode
        self.model_revision = "mock_revision_000"

    def load(self) -> None:
        logger.info(f"MockBackend loaded for {self.model_id}")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Approximate token count: ~1.3 tokens per word."""
        total_words = sum(len(m.get("content", "").split()) for m in messages)
        return int(total_words * 1.3)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        """Return deterministic mock output with known valid and invalid codes."""
        import random
        rng = random.Random(self.seed)

        # Fixed mock codes: 2 valid + 1 invalid
        mock_codes = ["E11.9", "I10", "Z99.999"]
        output = json.dumps({"codes": mock_codes})

        prompt_tokens = self.count_tokens(messages)
        new_tokens = len(output.split())

        # Simulate small latency
        time.sleep(0.05)

        return GenResult(
            text=output,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=0.05,
        )

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantisation": "none",
            "dtype": "none",
            "device": "cpu",
            "backend": "mock",
            "template_mode": self.template_mode,
        }

    @property
    def context_limit(self) -> int:
        return self._context_limit

