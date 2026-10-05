"""Unified model caller for Gemini, OpenAI, and Mock backends."""

import json
import logging
import os
import random
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_dotenv_if_present():
    """Load key-value pairs from .env if present in current or parent directory."""
    for p in [Path(".env"), Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_dotenv_if_present()


def _post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Dict[str, Any]:
    """Execute HTTP POST with JSON payload."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_with_retry(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, max_retries: int = 5) -> Dict[str, Any]:
    """Execute HTTP POST with exponential backoff on 429 / 5xx errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return _post_json(url, payload, headers=headers)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_time = min(45.0, (2.0 ** attempt) + random.uniform(0.5, 1.5))
                logger.warning(f"HTTP {e.code}. Retrying in {wait_time:.1f}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"HTTP {e.code} Error: {err_body}") from e
        except urllib.error.URLError as e:
            if attempt < max_retries:
                wait_time = min(30.0, 2.0 ** attempt)
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Network error: {e.reason}") from e


_LAST_GEMINI_CALL_TIME = 0.0


def _pace_gemini_calls(min_interval: float = 3.5):
    """Pace calls to ensure we stay safely under Gemini free tier 15 RPM quota."""
    global _LAST_GEMINI_CALL_TIME
    now = time.time()
    elapsed = now - _LAST_GEMINI_CALL_TIME
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_GEMINI_CALL_TIME = time.time()


def call_gemini(model_id: str, system_prompt: str, user_prompt: str, api_key: Optional[str] = None) -> str:
    """Call Google Gemini API."""
    _pace_gemini_calls(3.5)
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not found. Please set $env:GEMINI_API_KEY = 'your_key' in PowerShell."
        )

    # Clean alias mapping
    clean_id = model_id.replace("models/", "")
    if clean_id in ("gemini-flash-lite", "gemini-flash-lite-latest"):
        clean_id = "gemini-flash-lite-latest"
    elif clean_id in ("gemini-3.8-flash", "gemini-3.8"):
        clean_id = "gemini-3.8-flash"
    elif clean_id in ("gemini-flash", "gemini-flash-latest", "gemini-1.5-flash"):
        clean_id = "gemini-flash-latest"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_id}:generateContent?key={key}"
    
    gen_config = {
        "temperature": 0.0,
        "maxOutputTokens": 2048,
    }
    if "3.8" in clean_id or "thinking" in clean_id:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": gen_config,
    }

    try:
        res = _post_with_retry(url, payload)
    except RuntimeError as e:
        err_msg = str(e)
        if "INVALID_ARGUMENT" in err_msg and "thinkingConfig" in gen_config:
            gen_config.pop("thinkingConfig", None)
            res = _post_with_retry(url, payload)
        elif "404" in err_msg and clean_id != "gemini-flash-lite-latest":
            clean_id = "gemini-flash-lite-latest"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_id}:generateContent?key={key}"
            res = _post_with_retry(url, payload)
        else:
            raise

    candidates = res.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "").strip()
    return ""


def call_openai(model_id: str, system_prompt: str, user_prompt: str, api_key: Optional[str] = None) -> str:
    """Call OpenAI Chat Completions API."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    url = f"{base_url}/chat/completions"
    res = _post_with_retry(url, payload, headers=headers)
    choices = res.get("choices", [])
    if choices:
        return (choices[0].get("message", {}).get("content", "") or "").strip()
    return ""


def call_mock(model_id: str, system_prompt: str, user_prompt: str) -> str:
    """Deterministic mock for instant offline testing."""
    time.sleep(0.01)
    # Return mock ICD-10 predictions
    return json.dumps({"codes": ["E11.9", "I10"]})


def call_llm(model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Unified entry point to call any model by name."""
    m_lower = model_name.lower()
    if "mock" in m_lower:
        return call_mock(model_name, system_prompt, user_prompt)
    elif "gemini" in m_lower:
        return call_gemini(model_name, system_prompt, user_prompt)
    elif any(k in m_lower for k in ["gpt", "o1", "o3", "chatgpt"]):
        return call_openai(model_name, system_prompt, user_prompt)
    else:
        # Fallback to OpenAI-compatible or Mock if unknown
        if os.environ.get("OPENAI_BASE_URL"):
            return call_openai(model_name, system_prompt, user_prompt)
        raise ValueError(
            f"Unsupported model name: '{model_name}'. "
            "Supported: 'gemini-flash-lite', 'gemini-3.8-flash', 'gpt-4o-mini', 'mock'."
        )
