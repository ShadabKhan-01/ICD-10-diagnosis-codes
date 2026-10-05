"""Simplified ICD-10 Coding Experiment Runner.

Run everything with a single, clean command:
    python run.py --model gemini-flash-lite --limit 5
    python run.py --model mock --strategy zero_shot few_shot rag --limit 10
    python run.py --model gpt-4o-mini --limit 5
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import load_eval_records, get_valid_code_set
from config_zero_shot import run_zero_shot
from config_few_shot import run_few_shot
from config_rag import run_rag
from results import compute_metrics, print_results_table, save_experiment_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ICD-10 medical coding benchmark across LLMs and prompting strategies."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemini-flash-lite",
        help="Model name (e.g. gemini-flash-lite, gemini-3.8-flash, gpt-4o-mini, mock). Default: gemini-flash-lite",
    )
    parser.add_argument(
        "--strategy", "-s",
        nargs="+",
        default=["all"],
        choices=["all", "zero_shot", "few_shot", "rag"],
        help="Prompting strategy or 'all'. Default: all",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of evaluation patient records (e.g. 5, 10). Default: all available",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default="simplified_version/results",
        help="Output directory for predictions and tables. Default: simplified_version/results",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of few-shot demonstration examples. Default: 5",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of RAG candidate codes retrieved. Default: 10",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine strategies to execute
    if "all" in args.strategy:
        strategies = ["zero_shot", "few_shot", "rag"]
    else:
        strategies = args.strategy

    print("=" * 80)
    print("           ICD-10 CLINICAL CODING BENCHMARK (SIMPLIFIED)")
    print("=" * 80)
    print(f"Model:      {args.model}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Limit:      {args.limit if args.limit else 'All (200 instances)'}")
    print(f"Output:     {args.out}")
    print("=" * 80)

    # 1. Load patient records
    print("\n[1/3] Loading evaluation clinical notes...")
    records = load_eval_records(limit=args.limit)
    print(f"Loaded {len(records)} patient records.")

    # 2. Load valid code set for validation
    print("\n[2/3] Loading valid ICD-10-CM vocabulary for validation...")
    valid_codes = get_valid_code_set()
    print(f"Loaded {len(valid_codes)} valid ICD-10 codes.")

    # 3. Run selected configurations
    print("\n[3/3] Executing model evaluations...")
    summary_rows = []
    raw_results_by_config = {}

    for strat in strategies:
        if strat == "zero_shot":
            strat_results = run_zero_shot(records, args.model)
            strat_name = "Zero-Shot"
        elif strat == "few_shot":
            strat_results = run_few_shot(records, args.model, k=args.k)
            strat_name = f"Few-Shot (k={args.k})"
        elif strat == "rag":
            strat_results = run_rag(records, args.model, top_k=args.top_k)
            strat_name = f"RAG (k={args.top_k})"
        else:
            continue

        raw_results_by_config[f"{args.model}__{strat}"] = strat_results

        # Compute metrics
        metrics = compute_metrics(strat_results, valid_codes=valid_codes)
        metrics["model"] = args.model
        metrics["strategy"] = strat_name
        summary_rows.append(metrics)

    # 4. Display benchmark results table
    print_results_table(summary_rows)

    # 5. Save outputs
    out_dir = Path(args.out)
    save_experiment_outputs(out_dir, summary_rows, raw_results_by_config)
    print("\n[Done] Evaluation completed successfully!")


if __name__ == "__main__":
    main()
