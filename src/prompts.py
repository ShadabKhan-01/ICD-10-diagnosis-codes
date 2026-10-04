"""Prompt construction for the ICD-10 coding experiment.

The ONLY difference between arms is the EVIDENCE_BLOCK.
System prompt, instruction, record rendering, output format are identical.
Test §10.1 checks that removing the evidence block yields byte-identical text.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ── Fixed text (identical across all arms) ──────────────────────────────────

SYSTEM_TEXT = (
    "You are an experienced clinical coder. "
    "You assign ICD-10-CM diagnosis codes to patient records."
)

INSTRUCTION_TEXT = (
    'Read the patient record and list every ICD-10-CM diagnosis code that '
    'a professional coder would assign. Include conditions that are not named '
    'but are clearly implied by the medications or lab values. Do not code '
    'conditions that are negated, belong to family members, or that the patient '
    'does not have. Respond with ONLY a JSON object of the form '
    '{"codes": ["<ICD-10-CM code>", ...]}, most important codes first, '
    'and nothing else.'
)

# ── Evidence providers ──────────────────────────────────────────────────────


class EvidenceProvider(ABC):
    """Base class for strategy-specific evidence blocks."""

    @abstractmethod
    def build_evidence(self, record: dict) -> Optional[str]:
        """Return evidence block text, or None for zero-shot."""


class ZeroShotEvidence(EvidenceProvider):
    """No evidence — the evidence block is absent entirely."""

    def build_evidence(self, record: dict) -> None:
        return None


class FewShotEvidence(EvidenceProvider):
    """k worked examples drawn from the few-shot pool.

    The shots list is fixed once by seed (same 5 for every eval instance).
    Each shot is rendered in the exact answer format.
    """

    def __init__(self, shots: List[dict]):
        self.shots = shots

    def build_evidence(self, record: dict) -> str:
        parts = ["Worked examples:"]
        for shot in self.shots:
            shot_record = render_record_text(shot)
            answer = json.dumps({"codes": shot.get("gold_codes", [])})
            parts.append(f"{shot_record}\nAnswer: {answer}")
        return "\n\n".join(parts)


class RAGEvidence(EvidenceProvider):
    """Retrieved candidate codes for the record.

    Does NOT force selection — the model may ignore, use, or add codes.
    """

    def __init__(self, retriever: Any):
        self.retriever = retriever

    def build_evidence(self, record: dict) -> str:
        candidates = self.retriever.retrieve(record)
        header = (
            "Candidate ICD-10-CM codes retrieved for this record "
            "(you may use them, ignore them, or add another code "
            "the record clearly supports):"
        )
        lines = [header]
        for cand in candidates:
            lines.append(f"{cand.code} \u2014 {cand.desc}")
        return "\n".join(lines)


# ── Record rendering (identical across all arms) ───────────────────────────


def render_record_text(record: dict) -> str:
    """Render a record's text and medications in a fixed format.

    Used both in the prompt for the eval instance and in few-shot examples.
    """
    text = record.get("text", "")
    meds = record.get("medications", [])
    parts = [f"PATIENT RECORD:\n{text}"]
    if meds:
        parts.append(f"Medications: {'; '.join(meds)}")
    return "\n\n".join(parts)


# ── Prompt assembly ────────────────────────────────────────────────────────


def render_prompt(record: dict, evidence: Optional[str]) -> str:
    """Render the user-turn text.

    Structure:
        INSTRUCTION
        [EVIDENCE_BLOCK]   ← only if not None
        PATIENT RECORD: ...
        Medications: ...
        ANSWER:

    When evidence is None, the block is absent entirely (not an empty string).
    The remainder is byte-identical across arms.
    """
    parts = [INSTRUCTION_TEXT]
    if evidence is not None:
        parts.append(evidence)
    parts.append(render_record_text(record))
    parts.append("ANSWER:")
    return "\n\n".join(parts)


def build_messages(
    record: dict,
    evidence_provider: EvidenceProvider,
    template_mode: str,
) -> List[Dict[str, str]]:
    """Build chat messages for the model.

    Args:
        record: The clinical record dict (must have 'text', 'medications').
        evidence_provider: Strategy-specific evidence builder.
        template_mode:
            'system' — separate system and user messages (LLaMA-3)
            'prepend' — prepend system text to user message (BioMistral)

    Returns:
        List of {"role": ..., "content": ...} dicts.
    """
    evidence = evidence_provider.build_evidence(record)
    user_prompt = render_prompt(record, evidence)

    if template_mode == "system":
        return [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": user_prompt},
        ]
    elif template_mode == "prepend":
        # Mistral-style: system text prepended to user message
        combined = f"{SYSTEM_TEXT}\n\n{user_prompt}"
        return [{"role": "user", "content": combined}]
    else:
        raise ValueError(
            f"Unknown template_mode '{template_mode}'. "
            "Use 'system' (LLaMA-3) or 'prepend' (BioMistral)."
        )


def strip_evidence_block(user_content: str, evidence: Optional[str]) -> str:
    """Remove the evidence block from user content for invariance testing.

    If evidence is None, returns user_content unchanged.
    """
    if evidence is None:
        return user_content
    # The evidence block is preceded and followed by "\n\n"
    return user_content.replace(f"\n\n{evidence}", "")


def render_prompt_with_truncation(
    record: dict,
    tokenizer: Any,
    max_tokens: int = 2048,
    evidence: Optional[str] = None,
) -> Tuple[str, bool]:
    """Render prompt and truncate record text from the end if it exceeds max_tokens.

    Never truncates the instruction, evidence block, or answer slot.

    Returns:
        (prompt_text, was_truncated)
    """
    initial_prompt = render_prompt(record, evidence)
    if len(tokenizer.encode(initial_prompt)) <= max_tokens:
        return initial_prompt, False

    rec = dict(record)
    text = rec.get("text", "")
    words = text.split()

    low, high = 0, len(words)
    best_prompt = initial_prompt

    while low <= high:
        mid = (low + high) // 2
        rec["text"] = " ".join(words[:mid])
        candidate_prompt = render_prompt(rec, evidence)
        if len(tokenizer.encode(candidate_prompt)) <= max_tokens:
            best_prompt = candidate_prompt
            low = mid + 1
        else:
            high = mid - 1

    return best_prompt, True

