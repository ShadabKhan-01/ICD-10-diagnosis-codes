"""Data loading utilities for simplified ICD-10 pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON lines file."""
    records = []
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_records(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load evaluation clinical notes."""
    path = ROOT_DIR / "data" / "synthetic" / "run01" / "eval.jsonl"
    records = load_jsonl(path)
    records.sort(key=lambda r: r["id"])
    if limit:
        records = records[:limit]
    return records


def load_few_shot_pool(k: int = 5) -> List[Dict[str, Any]]:
    """Load k few-shot examples from the pool."""
    path = ROOT_DIR / "data" / "synthetic" / "run01" / "fewshot_pool.jsonl"
    pool = load_jsonl(path)
    # Prefer examples with non-empty implicit codes
    with_implicit = [r for r in pool if r.get("implicit_codes")]
    without_implicit = [r for r in pool if not r.get("implicit_codes")]
    
    selected = with_implicit[:min(2, len(with_implicit))]
    remaining_needed = k - len(selected)
    pool_rem = [r for r in pool if r["id"] not in {s["id"] for s in selected}]
    selected.extend(pool_rem[:remaining_needed])
    return selected[:k]


def load_vocabulary() -> Dict[str, str]:
    """Load valid ICD-10 codes and descriptions: {code: description}."""
    path = ROOT_DIR / "data" / "vocab" / "icd10cm_2024.jsonl"
    vocab = {}
    if not path.exists():
        return vocab
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            code = item.get("code")
            desc = item.get("long_desc") or item.get("short_desc", "")
            if code:
                vocab[code] = desc
    return vocab


def get_valid_code_set() -> Set[str]:
    """Return set of all valid codes in vocabulary."""
    return set(load_vocabulary().keys())
