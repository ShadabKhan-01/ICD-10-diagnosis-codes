import sys
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codes import normalize
# We'll use our own vectorized precomputations but import metrics per spec
try:
    from metrics import per_instance_tp_fp_fn
except ImportError:
    pass
from utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_data(gold_path, results_dir):
    """Load gold and prediction data aligned by instance ID."""
    gold_path = Path(gold_path)
    results_dir = Path(results_dir)
    
    gold_dict = {}
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            g_codes = data.get('gold_codes', data.get('codes', []))
            gold_dict[data['id']] = [normalize(c) for c in g_codes]
            
    models = ["llama3", "biomistral"]
    strategies = ["zero_shot", "few_shot", "rag"]
    
    preds = {}
    aligned_golds = {}
    
    for m in models:
        for s in strategies:
            alt_s = s.replace('_', '-')
            matching_files = list(results_dir.glob(f"*{m}*{s}*.jsonl"))
            if not matching_files:
                matching_files = list(results_dir.glob(f"*{m}*{alt_s}*.jsonl"))
            if not matching_files:
                logger.warning(f"Could not find result file for {m} {s} in {results_dir}")
                continue
                
            p_file = matching_files[0]
            inst_preds = {}
            with open(p_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    p_codes = data.get('parsed_codes', data.get('predicted_codes', data.get('codes', [])))
                    inst_preds[data['id']] = [normalize(c) for c in p_codes]
            
            # Align IDs present in this result file
            common_ids = sorted(set(inst_preds.keys()) & set(gold_dict.keys()))
            if not common_ids:
                logger.warning(f"No common IDs for {m} {s}")
                continue
                
            preds[(m, s)] = [inst_preds[i] for i in common_ids]
            aligned_golds[(m, s)] = [gold_dict[i] for i in common_ids]
            
    return aligned_golds, preds


def compute_micro_arrays(preds_list, golds):
    """Precompute tp, fp, fn per instance for micro F1."""
    n = len(golds)
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)
    fn = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        p_set = set(preds_list[i])
        g_set = set(golds[i])
        tp[i] = len(p_set & g_set)
        fp[i] = len(p_set - g_set)
        fn[i] = len(g_set - p_set)
        
    return tp, fp, fn

def compute_macro_arrays(preds_list, golds, unique_gold_codes):
    """Precompute per-code tp, fp, fn arrays for macro F1."""
    n = len(golds)
    num_codes = len(unique_gold_codes)
    code_to_idx = {c: i for i, c in enumerate(unique_gold_codes)}
    
    pred_has_c = np.zeros((n, num_codes), dtype=np.float32)
    gold_has_c = np.zeros((n, num_codes), dtype=np.float32)
    
    for i in range(n):
        for c in preds_list[i]:
            if c in code_to_idx:
                pred_has_c[i, code_to_idx[c]] = 1.0
        for c in golds[i]:
            if c in code_to_idx:
                gold_has_c[i, code_to_idx[c]] = 1.0
                
    tp_c = pred_has_c * gold_has_c
    fp_c = pred_has_c * (1 - gold_has_c)
    fn_c = (1 - pred_has_c) * gold_has_c
    
    return tp_c, fp_c, fn_c, gold_has_c

def calc_micro_f1_vectorized(tp_arr, fp_arr, fn_arr, indices):
    """Calculate micro F1 for all bootstrap samples at once."""
    tp_resampled = tp_arr[indices]
    fp_resampled = fp_arr[indices]
    fn_resampled = fn_arr[indices]
    
    TP_b = tp_resampled.sum(axis=1)
    FP_b = fp_resampled.sum(axis=1)
    FN_b = fn_resampled.sum(axis=1)
    
    denom = 2 * TP_b + FP_b + FN_b
    micro_f1_b = np.where(denom > 0, 2 * TP_b / denom, 0.0)
    return micro_f1_b

def calc_macro_f1_vectorized(tp_c, fp_c, fn_c, gold_has_c, indices):
    """Calculate macro F1 for all bootstrap samples at once."""
    # indices: shape (B, n)
    # tp_c: shape (n, num_codes)
    
    # We can compute sums over instances by dot product if we represent resampling as count matrix
    # Or sum after indexing: tp_c[indices] -> shape (B, n, num_codes) -> sum(axis=1) -> shape (B, num_codes)
    # This might use a lot of memory. If B=10000, n=1000, num_codes=1000, (10000, 1000, 1000) is 10B floats = 40GB!
    # Instead, compute per bootstrap sample iteratively or use batched approach.
    
    B, n = indices.shape
    num_codes = tp_c.shape[1]
    
    macro_f1_b = np.zeros(B, dtype=np.float32)
    
    for b in range(B):
        idx = indices[b]
        
        # shape (num_codes,)
        tp_b = tp_c[idx].sum(axis=0)
        fp_b = fp_c[idx].sum(axis=0)
        fn_b = fn_c[idx].sum(axis=0)
        g_b = gold_has_c[idx].sum(axis=0)
        
        # Only consider codes present in the resampled gold set
        valid_mask = g_b > 0
        
        tp_valid = tp_b[valid_mask]
        fp_valid = fp_b[valid_mask]
        fn_valid = fn_b[valid_mask]
        
        denom = 2 * tp_valid + fp_valid + fn_valid
        f1_valid = np.where(denom > 0, 2 * tp_valid / denom, 0.0)
        
        macro_f1_b[b] = f1_valid.mean() if len(f1_valid) > 0 else 0.0
        
    return macro_f1_b

def main():
    parser = argparse.ArgumentParser(description="Bootstrap Significance Testing")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with prediction jsonl files")
    parser.add_argument("--data", type=str, required=True, help="Path to gold eval.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--B", type=int, default=10000, help="Number of bootstrap resamples")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    logger.info(f"Loading data from {args.data} and {args.results_dir}")
    aligned_golds, preds = load_data(args.data, args.results_dir)
    
    if not preds:
        logger.error("No predictions loaded from results directory.")
        sys.exit(1)
        
    first_config = next(iter(preds.keys()))
    golds = aligned_golds[first_config]
    n = len(golds)
    if n == 0:
        logger.error("No instances loaded.")
        sys.exit(1)
        
    unique_gold_codes = sorted(list(set([c for g_list in aligned_golds.values() for g in g_list for c in g])))
    logger.info(f"Loaded {n} instances, {len(unique_gold_codes)} unique gold codes.")
    
    # Precompute arrays for all configs
    micro_data = {}
    macro_data = {}
    
    for config, p_list in preds.items():
        g = aligned_golds[config]
        micro_data[config] = compute_micro_arrays(p_list, g)
        macro_data[config] = compute_macro_arrays(p_list, g, unique_gold_codes)
        
    # Generate bootstrap indices
    logger.info(f"Generating {args.B} bootstrap resamples...")
    rng = np.random.RandomState(args.seed)
    indices = rng.randint(0, n, size=(args.B, n))
    
    # Compute point estimates (using original indices: 0 to n-1)
    orig_indices = np.arange(n).reshape(1, n)
    
    results = []
    
    def evaluate_comparison(config_a, config_b, label_a, label_b, comp_name):
        if config_a not in preds or config_b not in preds:
            logger.warning(f"Skipping {comp_name}, missing config.")
            return
            
        logger.info(f"Evaluating {comp_name} ({label_a} vs {label_b})...")
        
        # Micro F1
        tp_A, fp_A, fn_A = micro_data[config_a]
        micro_f1_a_orig = calc_micro_f1_vectorized(tp_A, fp_A, fn_A, orig_indices)[0]
        micro_f1_a_b = calc_micro_f1_vectorized(tp_A, fp_A, fn_A, indices)
        
        tp_B, fp_B, fn_B = micro_data[config_b]
        micro_f1_b_orig = calc_micro_f1_vectorized(tp_B, fp_B, fn_B, orig_indices)[0]
        micro_f1_b_b = calc_micro_f1_vectorized(tp_B, fp_B, fn_B, indices)
        
        micro_point_diff = micro_f1_a_orig - micro_f1_b_orig
        micro_diff_b = micro_f1_a_b - micro_f1_b_b
        micro_mean_diff = micro_diff_b.mean()
        micro_ci_lower = np.percentile(micro_diff_b, 2.5)
        micro_ci_upper = np.percentile(micro_diff_b, 97.5)
        micro_sig = (micro_ci_lower > 0) or (micro_ci_upper < 0)
        
        results.append({
            "Comparison": comp_name,
            "Metric": "Micro-F1",
            "Point Delta": micro_point_diff,
            "Mean Boot Delta": micro_mean_diff,
            "CI Lower": micro_ci_lower,
            "CI Upper": micro_ci_upper,
            "Significant": micro_sig
        })
        
        # Macro F1
        tp_A_c, fp_A_c, fn_A_c, gold_has_c = macro_data[config_a]
        macro_f1_a_orig = calc_macro_f1_vectorized(tp_A_c, fp_A_c, fn_A_c, gold_has_c, orig_indices)[0]
        macro_f1_a_b = calc_macro_f1_vectorized(tp_A_c, fp_A_c, fn_A_c, gold_has_c, indices)
        
        tp_B_c, fp_B_c, fn_B_c, _ = macro_data[config_b]
        macro_f1_b_orig = calc_macro_f1_vectorized(tp_B_c, fp_B_c, fn_B_c, gold_has_c, orig_indices)[0]
        macro_f1_b_b = calc_macro_f1_vectorized(tp_B_c, fp_B_c, fn_B_c, gold_has_c, indices)
        
        macro_point_diff = macro_f1_a_orig - macro_f1_b_orig
        macro_diff_b = macro_f1_a_b - macro_f1_b_b
        macro_mean_diff = macro_diff_b.mean()
        macro_ci_lower = np.percentile(macro_diff_b, 2.5)
        macro_ci_upper = np.percentile(macro_diff_b, 97.5)
        macro_sig = (macro_ci_lower > 0) or (macro_ci_upper < 0)
        
        results.append({
            "Comparison": comp_name,
            "Metric": "Macro-F1",
            "Point Delta": macro_point_diff,
            "Mean Boot Delta": macro_mean_diff,
            "CI Lower": macro_ci_lower,
            "CI Upper": macro_ci_upper,
            "Significant": macro_sig
        })

    # Comparisons
    # For each model: RAG - Zero-Shot, RAG - Few-Shot, Few-Shot - Zero-Shot
    models = ["llama3", "biomistral"]
    for m in models:
        evaluate_comparison((m, "rag"), (m, "zero_shot"), "RAG", "Zero-Shot", f"{m}: RAG vs Zero-Shot")
        evaluate_comparison((m, "rag"), (m, "few_shot"), "RAG", "Few-Shot", f"{m}: RAG vs Few-Shot")
        evaluate_comparison((m, "few_shot"), (m, "zero_shot"), "Few-Shot", "Zero-Shot", f"{m}: Few-Shot vs Zero-Shot")
        
    # Per strategy: LLaMA-3 - BioMistral
    strategies = ["zero_shot", "few_shot", "rag"]
    for s in strategies:
        evaluate_comparison(("llama3", s), ("biomistral", s), "LLaMA-3", "BioMistral", f"{s}: LLaMA-3 vs BioMistral")

    df = pd.DataFrame(results)
    
    # Print to console
    print("\n" + "="*80)
    print("Significance Testing Results")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    csv_path = reports_dir / "significance.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV report to {csv_path}")
    
    md_path = reports_dir / "significance.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Bootstrap Significance Testing Results\n\n")
        f.write(f"**Seed**: {args.seed} | **Resamples (B)**: {args.B}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    logger.info(f"Saved Markdown report to {md_path}")

if __name__ == "__main__":
    main()
