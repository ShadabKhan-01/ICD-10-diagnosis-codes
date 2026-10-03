"""Shared utilities: seeding, hashing, atomic I/O, GPU info, logging.

No module in this project uses print() for runtime info — logging only.
evaluate.py's table output is the sole exception.
"""

import hashlib
import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a standard format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    """Set seeds for random, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Could not enable deterministic algorithms: {e}"
            )
    except ImportError:
        pass


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write(path: str, data: str, mode: str = "w") -> None:
    """Write data to a file atomically (temp file → os.replace).

    Safe against crashes: the file is either fully written or absent.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, mode, encoding="utf-8" if "b" not in mode else None) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_jsonl_append(path: str, obj: dict) -> None:
    """Append a single JSON line to a file with flush+fsync.

    At most one line is lost on crash.
    """
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def gpu_info() -> Dict[str, Any]:
    """Return GPU information if CUDA is available, else empty dict."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_vram_mb": props.total_mem // (1024 * 1024),
                "cuda_version": torch.version.cuda or "unknown",
                "gpu_count": torch.cuda.device_count(),
            }
    except ImportError:
        pass
    return {}


def library_versions() -> Dict[str, str]:
    """Return versions of key libraries."""
    libs = [
        "torch", "transformers", "accelerate", "bitsandbytes",
        "sentence_transformers", "numpy", "scipy", "pandas",
        "huggingface_hub", "pydantic", "yaml",
    ]
    versions = {}
    for lib in libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    return versions


def git_commit() -> Optional[str]:
    """Return the current git commit hash, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def is_path_git_tracked(path: str) -> bool:
    """Check if a path is inside a git-tracked directory."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(path)),
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str, records: list) -> None:
    """Save a list of dicts as JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

