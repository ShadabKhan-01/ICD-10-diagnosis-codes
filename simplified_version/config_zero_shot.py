"""Configuration 1: Zero-Shot Coding Strategy."""

import logging
from typing import Any, Dict, List
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, format_zero_shot_prompt, extract_codes
from models import call_llm

logger = logging.getLogger(__name__)


def run_zero_shot(records: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    """Execute Zero-Shot configuration across given patient records."""
    results = []
    print(f"\n[Running Zero-Shot] Model: {model_name} | Instances: {len(records)}")

    for rec in tqdm(records, desc=f"Zero-Shot ({model_name})"):
        prompt = format_zero_shot_prompt(rec)
        raw_output = call_llm(model_name, SYSTEM_PROMPT, prompt)
        parsed = extract_codes(raw_output)

        results.append({
            "id": rec["id"],
            "model": model_name,
            "strategy": "Zero-Shot",
            "gold_codes": rec.get("gold_codes", []),
            "predicted_codes": parsed,
            "raw_output": raw_output,
        })

    return results
