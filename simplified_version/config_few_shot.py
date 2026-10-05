"""Configuration 2: Few-Shot (k=5) In-Context Learning Strategy."""

import logging
from typing import Any, Dict, List
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, format_few_shot_prompt, extract_codes
from inputs import load_few_shot_pool
from models import call_llm

logger = logging.getLogger(__name__)


def run_few_shot(
    records: List[Dict[str, Any]],
    model_name: str,
    k: int = 5,
    shots: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute Few-Shot configuration with k demonstrations across given patient records."""
    if shots is None:
        shots = load_few_shot_pool(k=k)

    shot_ids = [s["id"] for s in shots]
    results = []
    print(f"\n[Running Few-Shot] Model: {model_name} | k={len(shots)} ({shot_ids}) | Instances: {len(records)}")

    for rec in tqdm(records, desc=f"Few-Shot ({model_name})"):
        prompt = format_few_shot_prompt(rec, shots)
        raw_output = call_llm(model_name, SYSTEM_PROMPT, prompt)
        parsed = extract_codes(raw_output)

        results.append({
            "id": rec["id"],
            "model": model_name,
            "strategy": f"Few-Shot (k={len(shots)})",
            "gold_codes": rec.get("gold_codes", []),
            "predicted_codes": parsed,
            "raw_output": raw_output,
            "shots": shot_ids,
        })

    return results
