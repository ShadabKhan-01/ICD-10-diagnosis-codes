"""Parse LLM output into an ordered list of ICD-10-CM codes.

Identical for all arms — the parser never knows which strategy produced a row.
"""

import json
import re
import logging
from typing import Tuple, List

from codes import normalize, is_well_formed, WELL_FORMED_PATTERN

logger = logging.getLogger(__name__)

# Regex for extracting code-like tokens from free text.
# Slightly more permissive than WELL_FORMED_PATTERN to catch undotted codes.
_CODE_EXTRACTOR = re.compile(r'\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?)\b')


_CUT_OFF_TAIL = re.compile(r'[.\w]*')


def parse_output(raw_output: str, generation_truncated: bool = False) -> Tuple[List[str], str]:
    """Parse model output into (ordered_codes, parse_mode).

    Algorithm:
        1. Try to find a JSON object with a "codes" list → parse_mode='json'
        2. Fallback: regex-extract all well-formed code-like tokens → parse_mode='regex'
        3. If nothing: parse_mode='empty', codes=[]

    generation_truncated: True when generation hit max_new_tokens. The final regex
    match may then be a code cut off mid-token ("E11" from a cut "E11.9"), so it is
    dropped. Complete outputs are unaffected. None/False keeps the previous behaviour.

    After extraction:
        - normalize() each code, drop empty strings
        - De-duplicate preserving order (first occurrence wins)
        - Invalid codes are NOT dropped (they count in Invalid Code Rate)
        - No repair/snapping to nearest valid code
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return [], 'empty'

    raw = raw_output.strip()
    extracted_codes: List[str] = []
    parse_mode = 'empty'

    # --- Step 1: Try JSON ---
    # Find all {...} blocks (non-greedy within, but we try each)
    for json_match in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and 'codes' in parsed:
                codes_val = parsed['codes']
                if isinstance(codes_val, list):
                    extracted_codes = [str(c) for c in codes_val if c is not None]
                    parse_mode = 'json'
                    break
        except (json.JSONDecodeError, ValueError):
            continue

    # --- Step 2: Regex fallback ---
    if parse_mode != 'json':
        found = list(_CODE_EXTRACTOR.finditer(raw))
        if generation_truncated and found:
            last = found[-1]
            tail = raw[last.end():]
            # Only the unfinished final token is dropped: nothing after it except
            # dots/word characters, so no closing quote, comma, or bracket was emitted.
            if _CUT_OFF_TAIL.fullmatch(tail):
                found = found[:-1]
        matches = [m.group(1) for m in found]
        if matches:
            extracted_codes = matches
            parse_mode = 'regex'

    # --- Step 3: Normalize, drop empties, dedup preserving order ---
    normalized: List[str] = []
    for c in extracted_codes:
        n = normalize(c)
        if n:
            normalized.append(n)

    ordered: List[str] = []
    seen: set = set()
    for c in normalized:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    if not ordered:
        parse_mode = 'empty'

    return ordered, parse_mode


def parse_failure_rate(parse_modes: List[str]) -> float:
    """Fraction of parses that used regex fallback or failed entirely.

    parse_failure = (regex + empty) / total
    """
    if not parse_modes:
        return 0.0
    failures = sum(1 for m in parse_modes if m in ('regex', 'empty'))
    return failures / len(parse_modes)

