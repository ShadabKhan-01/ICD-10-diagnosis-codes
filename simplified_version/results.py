"""Evaluation metrics, result calculation, and table formatting."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from inputs import get_valid_code_set


def compute_metrics(
    results: List[Dict[str, Any]],
    valid_codes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Compute Micro-F1, Macro-F1, Precision@k, and invalid code rates."""
    if not results:
        return {
            "n": 0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "p_at_5": "n/a",
            "p_at_8": "n/a",
            "invalid_code_rate": 0.0,
            "unsupported_code_rate": 0.0,
            "mean_emitted": 0.0,
        }

    if valid_codes is None:
        valid_codes = get_valid_code_set()

    n = len(results)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_emitted = 0
    total_invalid = 0
    total_unsupported = 0

    gold_universe: Set[str] = set()
    per_code_stats: Dict[str, Dict[str, int]] = {}

    p5_scores: List[float] = []
    p8_scores: List[float] = []

    for r in results:
        gold = set(r.get("gold_codes", []))
        preds_list = r.get("predicted_codes", [])
        pred_set = set(preds_list)

        gold_universe.update(gold)

        # Micro TP, FP, FN
        tp = len(pred_set & gold)
        fp = len(pred_set - gold)
        fn = len(gold - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Invalid and Unsupported rates
        for code in preds_list:
            total_emitted += 1
            if code not in valid_codes:
                total_invalid += 1
            elif code not in gold:
                total_unsupported += 1

        # Precision@k (only evaluated if model emitted at least k codes)
        if len(preds_list) >= 5:
            top5 = preds_list[:5]
            hits5 = sum(1 for c in top5 if c in gold)
            p5_scores.append(hits5 / 5.0)

        if len(preds_list) >= 8:
            top8 = preds_list[:8]
            hits8 = sum(1 for c in top8 if c in gold)
            p8_scores.append(hits8 / 8.0)

    # Micro-F1
    denom = 2 * total_tp + total_fp + total_fn
    micro_f1 = (2 * total_tp / denom) if denom > 0 else 0.0

    # Macro-F1 over gold code universe
    for c in gold_universe:
        per_code_stats[c] = {"tp": 0, "fp": 0, "fn": 0}

    for r in results:
        gold = set(r.get("gold_codes", []))
        pred_set = set(r.get("predicted_codes", []))
        for c in gold_universe:
            in_gold = c in gold
            in_pred = c in pred_set
            if in_gold and in_pred:
                per_code_stats[c]["tp"] += 1
            elif in_pred and not in_gold:
                per_code_stats[c]["fp"] += 1
            elif in_gold and not in_pred:
                per_code_stats[c]["fn"] += 1

    code_f1s = []
    for c, stats in per_code_stats.items():
        c_denom = 2 * stats["tp"] + stats["fp"] + stats["fn"]
        c_f1 = (2 * stats["tp"] / c_denom) if c_denom > 0 else 0.0
        code_f1s.append(c_f1)

    macro_f1 = (sum(code_f1s) / len(code_f1s)) if code_f1s else 0.0

    # P@k means
    p_at_5 = (sum(p5_scores) / len(p5_scores)) if p5_scores else None
    p_at_8 = (sum(p8_scores) / len(p8_scores)) if p8_scores else None

    # Error rates
    inv_rate = (total_invalid / total_emitted * 100.0) if total_emitted > 0 else 0.0
    unsup_rate = (total_unsupported / total_emitted * 100.0) if total_emitted > 0 else 0.0
    mean_emitted = total_emitted / n if n > 0 else 0.0

    return {
        "n": n,
        "micro_f1": round(micro_f1, 3),
        "macro_f1": round(macro_f1, 3),
        "p_at_5": f"{p_at_5:.3f}" if p_at_5 is not None else "n/a",
        "p_at_8": f"{p_at_8:.3f}" if p_at_8 is not None else "n/a",
        "invalid_code_rate": f"{inv_rate:.1f}%",
        "unsupported_code_rate": f"{unsup_rate:.1f}%",
        "mean_emitted": round(mean_emitted, 2),
    }


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format benchmark comparison Table II in GitHub markdown."""
    header = (
        "| Model | Strategy | n | Micro-F1 | Macro-F1 | P@5 | P@8 | Invalid Rate |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['strategy']} | {r['n']} | "
            f"{r['micro_f1']:.3f} | {r['macro_f1']:.3f} | "
            f"{r['p_at_5']} | {r['p_at_8']} | {r['invalid_code_rate']} |"
        )
    return "\n".join(lines)


def print_results_table(rows: List[Dict[str, Any]]) -> None:
    """Print ASCII Table II to console."""
    print("\n" + "=" * 90)
    print("                      EVALUATION RESULTS (TABLE II)")
    print("=" * 90)
    print(
        f"{'Model':<24} | {'Strategy':<18} | {'n':<4} | "
        f"{'Micro-F1':<8} | {'Macro-F1':<8} | {'P@5':<6} | {'P@8':<6} | {'Invalid Rate':<12}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['model']:<24} | {r['strategy']:<18} | {r['n']:<4} | "
            f"{r['micro_f1']:<8.3f} | {r['macro_f1']:<8.3f} | "
            f"{r['p_at_5']:<6} | {r['p_at_8']:<6} | {r['invalid_code_rate']:<12}"
        )
    print("=" * 90 + "\n")


def save_experiment_outputs(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    raw_results_by_config: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Save raw JSONL predictions and Table II CSV / Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw results JSONL per configuration
    for cfg_name, items in raw_results_by_config.items():
        jsonl_path = output_dir / f"{cfg_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    # 2. Save CSV summary
    csv_path = output_dir / "table2_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "strategy", "n", "micro_f1", "macro_f1",
                "p_at_5", "p_at_8", "invalid_code_rate", "unsupported_code_rate", "mean_emitted"
            ]
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # 3. Save Markdown summary
    md_path = output_dir / "table2_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ICD-10 LLM Coding Evaluation Summary\n\n")
        f.write(format_markdown_table(summary_rows) + "\n")

    print(f"[Results Saved] Check directory: {output_dir}")
