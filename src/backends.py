"""LLM backend abstraction: HFBackend (NF4 4-bit), MockBackend, and Commercial APIs.

Supported backends:
- HFBackend: HuggingFace Transformers with NF4 4-bit quantization (open-weight models)
- GeminiBackend: Google Gemini API (gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, etc.)
- OpenAIBackend: OpenAI API (gpt-4o, gpt-4o-mini, o1, etc.) & OpenAI-compatible endpoints (Ollama, vLLM)
- AnthropicBackend: Anthropic Claude API (claude-3-5-sonnet, claude-3-haiku, etc.)
- MockBackend: Deterministic CPU mock backend for testing without GPU/APIs

Paper setting:
- Greedy decoding (do_sample=False / temperature=0.0)
- Deterministic evaluation
- Batch size 1
"""

import gc
import json
import logging
import os
import random
import ssl
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── HTTP Helpers for REST API Backends ─────────────────────────────────────

def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Execute HTTP POST with JSON body using standard library."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "ICD10-Research-Harness/1.0",
    }
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        resp_bytes = response.read()
        return json.loads(resp_bytes.decode("utf-8"))


def _retry_post(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 5,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Execute HTTP POST with exponential backoff on rate limits (429) and server errors."""
    clean_url = url.split("?")[0]
    for attempt in range(1, max_retries + 1):
        try:
            return _http_post_json(url, payload, headers=headers, timeout=timeout)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            # Retry on 429 (rate limit) or 5xx (server error)
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_time = min(60.0, (2.0 ** attempt) + random.uniform(0.5, 1.5))
                logger.warning(
                    f"HTTP {e.code} from {clean_url}. "
                    f"Retrying in {wait_time:.1f}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait_time)
                continue
            raise RuntimeError(
                f"HTTP {e.code} Error calling {clean_url}: {err_body}"
            ) from e
        except urllib.error.URLError as e:
            if attempt < max_retries:
                wait_time = min(30.0, 2.0 ** attempt)
                logger.warning(
                    f"Network error calling {clean_url}: {e.reason}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue
            raise RuntimeError(
                f"Network error calling {clean_url}: {e.reason}"
            ) from e


@dataclass
class GenResult:
    """Result of a single LLM generation.

    generation_truncated: True if generation stopped because max_new_tokens was
    reached (output may be incomplete). None when the backend cannot tell.
    """
    text: str
    prompt_tokens: int
    new_tokens: int
    latency_s: float
    generation_truncated: Optional[bool] = None


def detect_generation_truncated(new_token_ids: List[int], max_new_tokens: int,
                                stop_ids: List[int]) -> bool:
    """True if the generation ran to the token limit without emitting a stop token.

    A response that is exactly max_new_tokens long but ends with a stop token is complete.
    """
    if len(new_token_ids) < max_new_tokens:
        return False
    if not new_token_ids:
        return False
    return int(new_token_ids[-1]) not in set(int(s) for s in stop_ids)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model, tokenizer, or verify API keys."""
        ...

    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tokens in the formatted prompt."""
        ...

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        """Generate text from messages with greedy decoding. Returns GenResult."""
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


# ── HuggingFace Transformers Backend ───────────────────────────────────────


def select_bnb_compute_dtype(on_cuda: bool) -> str:
    """Requested bitsandbytes compute dtype for a device.

    CUDA (the research setting, incl. Tesla T4): float16, as in configs/phase1.yaml.
    No CUDA (allow_cpu smoke path only): float32, because float16 matmul is not usable
    on CPU. Whether CUDA is present decides this, not allow_cpu.
    """
    return "float16" if on_cuda else "float32"


def set_bnb_compute_dtype_in_config(config, dtype_name: str) -> bool:
    """Set bnb_4bit_compute_dtype in a model config's quantization_config.

    Returns True if the config carried a quantization_config (and was updated).
    Works for both the dict form (as loaded from config.json) and the object form.
    """
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    if isinstance(qc, dict):
        qc["bnb_4bit_compute_dtype"] = dtype_name
    else:
        setattr(qc, "bnb_4bit_compute_dtype", dtype_name)
    return True


def actual_bnb_compute_dtype(model) -> Optional[str]:
    """Compute dtype (e.g. 'float16') used by the model's bitsandbytes 4-bit layers.

    Returns None if the model has no Linear4bit layers.
    """
    for module in model.modules():
        if type(module).__name__ == "Linear4bit" and hasattr(module, "compute_dtype"):
            return str(module.compute_dtype).replace("torch.", "")
    return None

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
                 template_mode: str = "system",
                 tokenize_special_tokens: bool = True,
                 allow_cpu: bool = False):
        self.model_id = model_id
        # Explicit opt-in (run_experiment --allow-cpu) for smoke tests only. The default
        # refuses to run without CUDA, so a CPU run can never happen silently.
        self.allow_cpu = allow_cpu
        self._compute_dtype_name = "float16"
        self._context_limit = context_limit_tokens
        self.seed = seed
        self.stop_token_names = stop_tokens or []
        self.template_mode = template_mode
        # True (default, unchanged behaviour for existing models) lets the tokenizer
        # add BOS/special tokens on top of the chat-template string. Set False for
        # models whose chat template already emits BOS, to avoid a duplicate BOS.
        self.tokenize_special_tokens = tokenize_special_tokens
        self.model = None
        self.tokenizer = None
        self.model_revision: Optional[str] = None
        self._device_name: Optional[str] = None

    def load(self) -> None:
        """Load model with NF4 4-bit quantisation."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from huggingface_hub import model_info

        on_cuda = torch.cuda.is_available()
        if not on_cuda and not self.allow_cpu:
            raise RuntimeError(
                "No CUDA GPU detected. NF4 quantisation requires CUDA.\n"
                "To test on CPU without a GPU:\n"
                "  1. Use --backend mock for fast local verification\n"
                "  2. Or use API models: --models gemini-1.5-flash (with GEMINI_API_KEY)\n"
                "  3. Or, for a SMOKE TEST ONLY, add --allow-cpu (very slow; not a research run)\n"
                "See README.md for details."
            )
        if not on_cuda:
            logger.warning(
                "CPU smoke mode (--allow-cpu): NF4 weights on CPU, float32 compute. "
                "Outputs are NOT comparable to the CUDA float16 setting."
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

        # CUDA (research setting): float16 compute. CPU smoke mode: float32 compute
        # (float16 matmul is not usable on CPU).
        self._compute_dtype_name = select_bnb_compute_dtype(on_cuda)
        compute_dtype = getattr(torch, self._compute_dtype_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
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
        # A pre-quantised checkpoint ships its own quantization_config (bnb_4bit_compute_dtype
        # bfloat16), which transformers uses in preference to the bnb_config passed above.
        # Set the compute dtype in the config too, so the requested dtype is the one in use.
        model_config = AutoConfig.from_pretrained(
            self.model_id, token=hf_token, trust_remote_code=False
        )
        set_bnb_compute_dtype_in_config(model_config, self._compute_dtype_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto" if on_cuda else "cpu",
            token=hf_token,
            trust_remote_code=False,
        )
        self.model.eval()

        # Record the compute dtype actually in use (not the requested one) and fail loudly
        # on a mismatch: a silent fallback to bfloat16 is exactly what this guards against.
        actual = actual_bnb_compute_dtype(self.model)
        if actual is None:
            raise RuntimeError("No bitsandbytes Linear4bit layers found; NF4 load failed.")
        if actual != self._compute_dtype_name:
            raise RuntimeError(
                f"Requested bnb compute dtype {self._compute_dtype_name} but the model is "
                f"using {actual}. Refusing to run with a different numeric setting."
            )
        self._compute_dtype_name = actual

        self._device_name = torch.cuda.get_device_name(0) if on_cuda else "cpu (smoke only)"
        logger.info(f"Model loaded on {self._device_name}")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens using the model's chat template."""
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        return len(token_ids)

    def encode_formatted(self, formatted: str):
        """Tokenize a chat-template string. Honours tokenize_special_tokens."""
        return self.tokenizer(
            formatted,
            add_special_tokens=self.tokenize_special_tokens,
            return_tensors="pt",
        )

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
        inputs = self.encode_formatted(formatted).to(self.model.device)
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
        generation_truncated = detect_generation_truncated(
            new_token_ids.tolist(), max_new_tokens, stop_ids
        )

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=latency,
            generation_truncated=generation_truncated,
        )

    @property
    def info(self) -> Dict[str, Any]:
        info = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantisation": "nf4_4bit",
            "dtype": self._compute_dtype_name,
            "device": self._device_name or "unknown",
            "backend": "hf",
            "template_mode": self.template_mode,
            "tokenize_special_tokens": self.tokenize_special_tokens,
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


# ── Google Gemini API Backend ──────────────────────────────────────────────

class GeminiBackend(LLMBackend):
    """Google Gemini API backend (supports gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, etc.).

    Paper setting:
    - Greedy decoding (temperature=0.0)
    - Deterministic evaluation
    - Runs on any environment (no GPU required)
    - Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    """

    def __init__(
        self,
        model_id: str = "gemini-1.5-flash",
        context_limit_tokens: int = 1048576,
        seed: int = 42,
        api_key: Optional[str] = None,
        template_mode: str = "system",
        max_retries: int = 5,
        timeout: int = 60,
    ):
        raw_id = model_id.replace("models/", "")
        # Automatic mapping for models retired by Google API or preview limited
        if raw_id in ("gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash", "gemini"):
            self.model_id = "gemini-flash-latest"
        elif raw_id in ("gemini-1.5-pro", "gemini-2.0-pro", "gemini-2.5-pro", "gemini-pro"):
            self.model_id = "gemini-pro-latest"
        else:
            self.model_id = raw_id

        self._context_limit = context_limit_tokens
        self.seed = seed
        self.template_mode = template_mode
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_revision = "api_gemini"

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                f"Gemini API key not found for {self.model_id}.\n"
                "Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable:\n"
                "  Windows (PowerShell): $env:GEMINI_API_KEY = 'your_key'\n"
                "  Linux / Colab:        export GEMINI_API_KEY='your_key'\n"
                "Get a free API key at: https://aistudio.google.com/app/apikey"
            )
        logger.info(f"GeminiBackend initialized for {self.model_id}")

    def _convert_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert chat messages to Gemini's systemInstruction and contents."""
        system_instruction = None
        contents = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n\n" + content
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })
            else:  # user
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })

        return system_instruction, contents

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens using Gemini countTokens API or approximate fallback."""
        if self.api_key:
            sys_inst, contents = self._convert_messages(messages)
            payload: Dict[str, Any] = {"contents": contents}
            if sys_inst:
                payload["systemInstruction"] = {"parts": [{"text": sys_inst}]}

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model_id}:countTokens?key={self.api_key}"
            )
            try:
                res = _http_post_json(url, payload, timeout=10)
                if "totalTokens" in res:
                    return int(res["totalTokens"])
            except Exception as e:
                logger.debug(f"Gemini countTokens call failed ({e}). Using approximation.")

        # Approximation: ~1.3 tokens per word
        total_words = sum(len(m.get("content", "").split()) for m in messages)
        return int(total_words * 1.3)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        """Call Gemini generateContent with greedy decoding (temperature=0.0)."""
        sys_inst, contents = self._convert_messages(messages)
        gen_config: Dict[str, Any] = {
            "temperature": 0.0,
            "maxOutputTokens": max(max_new_tokens, 2048),
        }
        if "3.8" in self.model_id or "thinking" in self.model_id:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_config,
        }
        if sys_inst:
            payload["systemInstruction"] = {"parts": [{"text": sys_inst}]}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent?key={self.api_key}"
        )

        start = time.time()
        try:
            res = _retry_post(url, payload, max_retries=self.max_retries, timeout=self.timeout)
        except RuntimeError as e:
            err_str = str(e)
            if "INVALID_ARGUMENT" in err_str and "thinkingConfig" in gen_config:
                gen_config.pop("thinkingConfig", None)
                res = _retry_post(url, payload, max_retries=self.max_retries, timeout=self.timeout)
            elif "404" in err_str and self.model_id != "gemini-flash-latest":
                self.model_id = "gemini-flash-latest"
                fallback_url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model_id}:generateContent?key={self.api_key}"
                )
                res = _retry_post(fallback_url, payload, max_retries=self.max_retries, timeout=self.timeout)
            else:
                raise
        latency = time.time() - start

        # Extract output text
        text = ""
        candidates = res.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "").strip()
            else:
                finish_reason = candidates[0].get("finishReason", "UNKNOWN")
                logger.warning(f"Gemini returned empty parts. finishReason: {finish_reason}")

        usage = res.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        new_tokens = usage.get("candidatesTokenCount", len(text.split()))

        if prompt_tokens == 0:
            prompt_tokens = self.count_tokens(messages)

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=latency,
        )

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantisation": "none_api",
            "dtype": "none",
            "device": "cloud_gemini_api",
            "backend": "gemini",
            "template_mode": self.template_mode,
        }

    @property
    def context_limit(self) -> int:
        return self._context_limit


# ── OpenAI & Compatible API Backend ────────────────────────────────────────

class OpenAIBackend(LLMBackend):
    """OpenAI API backend (supports gpt-4o, gpt-4o-mini, o1, and OpenAI-compatible endpoints like Ollama/vLLM).

    Paper setting:
    - Greedy decoding (temperature=0.0)
    - Deterministic evaluation
    - Runs on any environment (no GPU required)
    - Requires OPENAI_API_KEY (or local endpoint via OPENAI_BASE_URL).
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        context_limit_tokens: int = 128000,
        seed: int = 42,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        template_mode: str = "system",
        max_retries: int = 5,
        timeout: int = 60,
    ):
        self.model_id = model_id
        self._context_limit = context_limit_tokens
        self.seed = seed
        self.template_mode = template_mode
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_revision = "api_openai"

    def load(self) -> None:
        if (
            not self.api_key
            and "localhost" not in self.base_url
            and "127.0.0.1" not in self.base_url
        ):
            raise RuntimeError(
                f"OpenAI API key not found for {self.model_id}.\n"
                "Please set OPENAI_API_KEY environment variable:\n"
                "  Windows (PowerShell): $env:OPENAI_API_KEY = 'your_key'\n"
                "  Linux / Colab:        export OPENAI_API_KEY='your_key'"
            )
        logger.info(f"OpenAIBackend initialized for {self.model_id} via {self.base_url}")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Approximate token count: ~1.3 tokens per word."""
        total_words = sum(len(m.get("content", "").split()) for m in messages)
        return int(total_words * 1.3)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        """Call OpenAI chat completions with temperature=0.0 (greedy)."""
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "seed": self.seed,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        start = time.time()
        res = _retry_post(
            url, payload, headers=headers, max_retries=self.max_retries, timeout=self.timeout
        )
        latency = time.time() - start

        text = ""
        choices = res.get("choices", [])
        if choices:
            text = (choices[0].get("message", {}).get("content", "") or "").strip()

        usage = res.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", self.count_tokens(messages))
        new_tokens = usage.get("completion_tokens", len(text.split()))

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=latency,
        )

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantisation": "none_api",
            "dtype": "none",
            "device": "cloud_openai_api",
            "backend": "openai",
            "template_mode": self.template_mode,
        }

    @property
    def context_limit(self) -> int:
        return self._context_limit


# ── Anthropic Claude API Backend ───────────────────────────────────────────

class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend (supports claude-3-5-sonnet, claude-3-haiku, etc.)."""

    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        context_limit_tokens: int = 200000,
        seed: int = 42,
        api_key: Optional[str] = None,
        template_mode: str = "system",
        max_retries: int = 5,
        timeout: int = 60,
    ):
        self.model_id = model_id
        self._context_limit = context_limit_tokens
        self.seed = seed
        self.template_mode = template_mode
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_revision = "api_anthropic"

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                f"Anthropic API key not found for {self.model_id}.\n"
                "Please set ANTHROPIC_API_KEY environment variable:\n"
                "  Windows (PowerShell): $env:ANTHROPIC_API_KEY = 'your_key'\n"
                "  Linux / Colab:        export ANTHROPIC_API_KEY='your_key'"
            )
        logger.info(f"AnthropicBackend initialized for {self.model_id}")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_words = sum(len(m.get("content", "").split()) for m in messages)
        return int(total_words * 1.3)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> GenResult:
        system_text = ""
        chat_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system_text = (system_text + "\n\n" + m.get("content", "")).strip()
            else:
                chat_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": chat_msgs,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if system_text:
            payload["system"] = system_text

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = "https://api.anthropic.com/v1/messages"
        start = time.time()
        res = _retry_post(
            url, payload, headers=headers, max_retries=self.max_retries, timeout=self.timeout
        )
        latency = time.time() - start

        text = ""
        contents = res.get("content", [])
        if contents:
            text = contents[0].get("text", "").strip()

        usage = res.get("usage", {})
        prompt_tokens = usage.get("input_tokens", self.count_tokens(messages))
        new_tokens = usage.get("output_tokens", len(text.split()))

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=latency,
        )

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantisation": "none_api",
            "dtype": "none",
            "device": "cloud_anthropic_api",
            "backend": "anthropic",
            "template_mode": self.template_mode,
        }

    @property
    def context_limit(self) -> int:
        return self._context_limit


# ── Deterministic Mock Backend for Testing ─────────────────────────────────

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
        # Fixed mock codes: 2 valid + 1 invalid
        mock_codes = ["E11.9", "I10", "Z99.999"]
        output = json.dumps({"codes": mock_codes})

        prompt_tokens = self.count_tokens(messages)
        new_tokens = len(output.split())

        # Simulate small latency
        time.sleep(0.02)

        return GenResult(
            text=output,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            latency_s=0.02,
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


# ── Unified Backend Factory ────────────────────────────────────────────────

def create_backend(
    model_id: str,
    backend_type: Optional[str] = None,
    context_limit: Optional[int] = None,
    template_mode: Optional[str] = None,
    seed: int = 42,
    stop_tokens: Optional[List[str]] = None,
    tokenize_special_tokens: bool = True,
    allow_cpu: bool = False,
    **kwargs,
) -> LLMBackend:
    """Factory function to instantiate the appropriate LLM backend.

    Supports:
    - Open source: Hugging Face models ('hf')
    - Closed source commercial APIs: Google Gemini ('gemini'), OpenAI ('openai'), Anthropic ('anthropic')
    - Testing: MockBackend ('mock')
    """
    model_lower = model_id.lower()

    # Auto-detect backend if not explicitly provided or set to 'auto'
    if not backend_type or backend_type == "auto":
        if "mock" in model_lower:
            backend_type = "mock"
        elif "gemini" in model_lower:
            backend_type = "gemini"
        elif any(k in model_lower for k in ["gpt-", "o1-", "o3-", "chatgpt"]):
            backend_type = "openai"
        elif "claude" in model_lower:
            backend_type = "anthropic"
        else:
            backend_type = "hf"

    backend_type = backend_type.lower()

    if backend_type == "mock":
        return MockBackend(
            model_id=model_id,
            context_limit_tokens=context_limit or 8192,
            seed=seed,
            template_mode=template_mode or "system",
        )
    elif backend_type == "gemini":
        return GeminiBackend(
            model_id=model_id,
            context_limit_tokens=context_limit or 1048576,
            seed=seed,
            template_mode=template_mode or "system",
            **kwargs,
        )
    elif backend_type == "openai":
        return OpenAIBackend(
            model_id=model_id,
            context_limit_tokens=context_limit or 128000,
            seed=seed,
            template_mode=template_mode or "system",
            **kwargs,
        )
    elif backend_type == "anthropic":
        return AnthropicBackend(
            model_id=model_id,
            context_limit_tokens=context_limit or 200000,
            seed=seed,
            template_mode=template_mode or "system",
            **kwargs,
        )
    elif backend_type == "hf":
        return HFBackend(
            model_id=model_id,
            context_limit_tokens=context_limit or 8192,
            seed=seed,
            stop_tokens=stop_tokens or [],
            template_mode=template_mode or "system",
            tokenize_special_tokens=tokenize_special_tokens,
            allow_cpu=allow_cpu,
        )
    else:
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. "
            "Choose from 'auto', 'hf', 'gemini', 'openai', 'anthropic', 'mock'."
        )
