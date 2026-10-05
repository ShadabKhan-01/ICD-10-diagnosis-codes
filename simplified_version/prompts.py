"""Prompt templates and response extraction for simplified ICD-10 coding."""

import json
import re
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = (
    "You are an expert medical coder. Your task is to assign valid, specific "
    "ICD-10-CM diagnosis codes based ONLY on the provided clinical document.\n\n"
    "Instructions:\n"
    "- Output ONLY a JSON object in this format: {\"codes\": [\"CODE1\", \"CODE2\"]}\n"
    "- Use standard dotted ICD-10 notation (e.g. E11.9, I10, J44.9).\n"
    "- Do not include explanations, narrative, or notes outside the JSON."
)


def format_clinical_note(record: Dict[str, Any]) -> str:
    """Format the patient's text and medications."""
    text = record.get("text", "").strip()
    meds = record.get("medications", [])
    meds_str = ", ".join(meds) if meds else "None"
    return (
        f"CLINICAL NOTE:\n{text}\n\n"
        f"DISCHARGE MEDICATIONS:\n{meds_str}"
    )


def format_zero_shot_prompt(record: Dict[str, Any]) -> str:
    """Create a Zero-Shot prompt."""
    note = format_clinical_note(record)
    return (
        f"{note}\n\n"
        f"TASK:\n"
        f"Assign all applicable ICD-10-CM diagnosis codes for this patient.\n"
        f"Respond ONLY with: {{\"codes\": [\"...\"]}}"
    )


def format_few_shot_prompt(record: Dict[str, Any], shots: List[Dict[str, Any]]) -> str:
    """Create a Few-Shot prompt with k examples."""
    examples_text = "HERE ARE EXAMPLES OF CORRECT CODING:\n\n"
    for i, shot in enumerate(shots, 1):
        shot_note = format_clinical_note(shot)
        shot_codes = json.dumps({"codes": shot.get("gold_codes", [])})
        examples_text += (
            f"--- Example {i} ---\n"
            f"{shot_note}\n"
            f"Output:\n{shot_codes}\n\n"
        )

    target_note = format_clinical_note(record)
    return (
        f"{examples_text}"
        f"--- New Patient ---\n"
        f"{target_note}\n\n"
        f"TASK:\n"
        f"Assign all applicable ICD-10-CM diagnosis codes for this patient.\n"
        f"Respond ONLY with: {{\"codes\": [\"...\"]}}"
    )


def format_rag_prompt(record: Dict[str, Any], retrieved_candidates: List[Dict[str, str]]) -> str:
    """Create a RAG prompt with retrieved candidate codes and descriptions."""
    cands_text = "RETRIEVED CANDIDATE ICD-10 CODES:\n"
    if retrieved_candidates:
        for c in retrieved_candidates:
            cands_text += f"- {c['code']}: {c['desc']}\n"
    else:
        cands_text += "None available\n"

    target_note = format_clinical_note(record)
    return (
        f"{cands_text}\n"
        f"CLINICAL DOCUMENT:\n"
        f"{target_note}\n\n"
        f"TASK:\n"
        f"Select and assign all applicable ICD-10-CM diagnosis codes supported by the document.\n"
        f"You may select from the retrieved candidates or assign other supported codes.\n"
        f"Respond ONLY with: {{\"codes\": [\"...\"]}}"
    )


def extract_codes(raw_output: str) -> List[str]:
    """Robustly extract ICD-10 codes from LLM output (handles JSON, markdown, or regex)."""
    if not raw_output:
        return []

    # 1. Try finding JSON object
    json_match = re.search(r'\{[^{}]*"codes"\s*:\s*\[[^\]]*\][^{}]*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            codes = data.get("codes", [])
            if isinstance(codes, list):
                cleaned = []
                for c in codes:
                    c_clean = str(c).strip().upper()
                    # Ensure dotted format (e.g. E119 -> E11.9)
                    if len(c_clean) > 3 and "." not in c_clean:
                        c_clean = c_clean[:3] + "." + c_clean[3:]
                    if c_clean not in cleaned:
                        cleaned.append(c_clean)
                return cleaned
        except Exception:
            pass

    # 2. Regex fallback for ICD-10 patterns (e.g. A00.0, E11.9, Z99.9)
    pattern = r'\b([A-TV-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b'
    matches = re.findall(pattern, raw_output.upper())
    seen = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return seen
