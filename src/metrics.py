"""Pure metric functions for the ICD-10 coding experiment.

All functions are pure (no I/O), heavily unit-tested.
Edge cases: empty predictions, empty gold, zero denominators → well-defined.

Key identity (test §10.3):
    For each emitted code, it is exactly ONE of:
    - True positive (in gold AND valid)
    - Invalid (not in vocab)
    - Unsupported (valid but not in gold)
    So: tp_share + invalid_rate + unsupported_rate ≡ 1.0
"""

import math
from typing import Dict, List, Set, Tuple


def micro_f1(pred_sets: List[Set[str]], gold_sets: List[Set[str]]) -> float:
    """Pool TP/FP/FN across all instances; F1 = 2TP / (2TP+FP+FN).

    Returns 0.0 when the denominator is 0 (no predictions and no gold).
    """
    tp = fp = fn = 0
    for p, g in zip(pred_sets, gold_sets):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def macro_f1_gold_set(pred_sets: List[Set[str]], gold_sets: List[Set[str]]) -> float:
    """Per-code F1 over codes present in gold, then unweighted mean.

    'Gold-set variant' per paper §6.3:
    - Only codes that appear in ANY gold set contribute a term.
    - Predicted-but-never-gold codes do NOT add terms (they still hurt Micro-F1).
    """
    gold_universe: Set[str] = set()
    for g in gold_sets:
        gold_universe.update(g)

    if not gold_universe:
        return 0.0

    f1_scores = []
    for code in sorted(gold_universe):  # sorted for determinism
        tp = fp = fn = 0
        for p, g in zip(pred_sets, gold_sets):
            predicted = code in p
            is_gold = code in g
            if predicted and is_gold:
                tp += 1
            elif predicted and not is_gold:
                fp += 1
            elif not predicted and is_gold:
                fn += 1
        denom = 2 * tp + fp + fn
        f1_scores.append((2 * tp / denom) if denom > 0 else 0.0)

    return sum(f1_scores) / len(f1_scores)


def precision_at_k(
    pred_lists: List[List[str]], gold_sets: List[Set[str]], k: int
) -> Tuple[float, int]:
    """P@k: precision of first k emitted codes, only for instances with ≥k codes.

    Returns:
        (mean_precision, n_contributing): n_contributing is how many instances qualified.
        If zero instances qualify, returns (float('nan'), 0).
    """
    precisions = []
    for preds, gold in zip(pred_lists, gold_sets):
        if len(preds) >= k:
            first_k = preds[:k]
            hits = sum(1 for c in first_k if c in gold)
            precisions.append(hits / k)

    if not precisions:
        return float("nan"), 0
    return sum(precisions) / len(precisions), len(precisions)


def invalid_code_rate(
    pred_lists: List[List[str]], vocab_valid_set: Set[str]
) -> float:
    """(Emitted codes not in vocabulary) / (total emitted codes).

    Pooled across instances, after de-duplication within each record.
    Returns float('nan') if zero total emitted codes.
    """
    total_invalid = 0
    total_emitted = 0
    for preds in pred_lists:
        seen: Set[str] = set()
        for c in preds:
            if c not in seen:
                seen.add(c)
                total_emitted += 1
                if c not in vocab_valid_set:
                    total_invalid += 1
    if total_emitted == 0:
        return float("nan")
    return total_invalid / total_emitted


def unsupported_code_rate(
    pred_lists: List[List[str]],
    gold_sets: List[Set[str]],
    vocab_valid_set: Set[str],
) -> float:
    """(Valid codes not in gold) / (total emitted codes).

    A code is 'unsupported' if it IS in the vocabulary but is NOT in
    that record's gold set.
    Returns float('nan') if zero total emitted codes.
    """
    total_unsupported = 0
    total_emitted = 0
    for preds, gold in zip(pred_lists, gold_sets):
        seen: Set[str] = set()
        for c in preds:
            if c not in seen:
                seen.add(c)
                total_emitted += 1
                if c in vocab_valid_set and c not in gold:
                    total_unsupported += 1
    if total_emitted == 0:
        return float("nan")
    return total_unsupported / total_emitted


def tp_share(
    pred_lists: List[List[str]],
    gold_sets: List[Set[str]],
    vocab_valid_set: Set[str],
) -> float:
    """(True positive codes) / (total emitted codes).

    A code is a TP if it is in the gold set (and implicitly valid).
    Returns float('nan') if zero total emitted codes.
    """
    total_tp = 0
    total_emitted = 0
    for preds, gold in zip(pred_lists, gold_sets):
        seen: Set[str] = set()
        for c in preds:
            if c not in seen:
                seen.add(c)
                total_emitted += 1
                if c in gold:
                    total_tp += 1
    if total_emitted == 0:
        return float("nan")
    return total_tp / total_emitted


def per_instance_tp_fp_fn(
    pred_sets: List[Set[str]], gold_sets: List[Set[str]]
) -> Tuple[List[int], List[int], List[int]]:
    """Per-instance TP, FP, FN counts. Useful for bootstrap."""
    tps, fps, fns = [], [], []
    for p, g in zip(pred_sets, gold_sets):
        tps.append(len(p & g))
        fps.append(len(p - g))
        fns.append(len(g - p))
    return tps, fps, fns


def compute_all_metrics(
    pred_lists: List[List[str]],
    gold_sets: List[Set[str]],
    vocab_valid_set: Set[str],
) -> Dict[str, object]:
    """Compute all metrics for a single configuration.

    Args:
        pred_lists: Per-instance ordered list of predicted codes.
        gold_sets: Per-instance set of gold codes.
        vocab_valid_set: Set of all valid codes from the vocabulary.

    Returns:
        Dict with keys: micro_f1, macro_f1, p5, n_p5, p8, n_p8,
        invalid_code_rate, unsupported_code_rate, tp_share,
        mean_codes_emitted.
    """
    # Convert pred_lists to sets for set-based metrics
    pred_sets = [set(lst) for lst in pred_lists]

    p5, n_p5 = precision_at_k(pred_lists, gold_sets, 5)
    p8, n_p8 = precision_at_k(pred_lists, gold_sets, 8)

    total_codes = sum(len(set(lst)) for lst in pred_lists)
    mean_emitted = total_codes / len(pred_lists) if pred_lists else 0.0

    icr = invalid_code_rate(pred_lists, vocab_valid_set)
    ucr = unsupported_code_rate(pred_lists, gold_sets, vocab_valid_set)
    tps = tp_share(pred_lists, gold_sets, vocab_valid_set)

    micro_p, micro_r = micro_precision_recall(pred_sets, gold_sets)

    return {
        "micro_f1": micro_f1(pred_sets, gold_sets),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_f1": macro_f1_gold_set(pred_sets, gold_sets),
        "p5": p5,
        "n_p5": n_p5,
        "p8": p8,
        "n_p8": n_p8,
        "invalid_code_rate": icr,
        "unsupported_code_rate": ucr,
        "tp_share": tps,
        "mean_codes_emitted": mean_emitted,
    }


def micro_precision_recall(pred_sets: List[Set[str]], gold_sets: List[Set[str]]) -> Tuple[float, float]:
    """Pooled micro precision = TP/(TP+FP) and micro recall = TP/(TP+FN).

    Returns (0.0, 0.0) for a zero denominator. Same TP/FP/FN pooling as micro_f1.
    """
    tp = fp = fn = 0
    for p, g in zip(pred_sets, gold_sets):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall

