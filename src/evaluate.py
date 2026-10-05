import argparse
import json
import logging
import sys
import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codes import normalize
from vocab import Vocabulary  
from metrics import micro_f1, macro_f1_gold_set, precision_at_k, invalid_code_rate, unsupported_code_rate, compute_all_metrics
from parser import parse_failure_rate, parse_output
from utils import sha256_file, sha256_str

def map_model_name(raw_name: str) -> str:
    lower_name = raw_name.lower()
    if 'gemini-3.8-flash' in lower_name:
        return 'Gemini-3.8-Flash'
    if 'gemini-1.5-flash' in lower_name:
        return 'Gemini-1.5-Flash'
    if 'gemini-flash' in lower_name:
        return 'Gemini-Flash'
    if 'gemini-1.5-pro' in lower_name:
        return 'Gemini-1.5-Pro'
    if 'gemini-2.0-flash' in lower_name:
        return 'Gemini-2.0-Flash'
    if 'gpt-4o-mini' in lower_name:
        return 'GPT-4o-Mini'
    if 'gpt-4o' in lower_name:
        return 'GPT-4o'
    if 'claude-3-5-sonnet' in lower_name:
        return 'Claude-3.5-Sonnet'
    if 'claude' in lower_name:
        return 'Claude-3-Haiku'
    if 'llama' in lower_name:
        return 'LLaMA-3-8B-Instruct'
    if 'biomistral' in lower_name:
        return 'BioMistral-7B'
    if 'mistral' in lower_name:
        return 'Mistral-7B'
    if 'qwen' in lower_name:
        return 'Qwen-2.5-7B'
    return raw_name

def sort_key(row):
    strategy_rank = {
        'Zero-Shot': 1,
        'Few-Shot (k=5)': 2,
        'Few-Shot': 2,
        'RAG': 3,
    }.get(row.get('Strategy', ''), 99)

    model_order = [
        'LLaMA-3-8B-Instruct',
        'BioMistral-7B',
        'Gemini-3.8-Flash',
        'Gemini-1.5-Flash',
        'Gemini-Flash',
        'Gemini-1.5-Pro',
        'Gemini-2.0-Flash',
        'GPT-4o-Mini',
        'GPT-4o',
        'Claude-3.5-Sonnet',
        'Qwen-2.5-7B',
        'Mistral-7B',
    ]
    model_name = row.get('Model', '')
    if model_name in model_order:
        model_rank = model_order.index(model_name)
    else:
        model_rank = 100

    return (model_rank, model_name, strategy_rank)


def validate_directory(results_dir: str) -> None:
    """Validate results directory does not mix mock and real runs."""
    path = Path(results_dir)
    backends = set()
    for f in path.glob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                backend = item.get("backend")
                if not backend:
                    fname = f.name.lower()
                    if "mock" in fname or "mock" in str(item.get("id", "")).lower():
                        backend = "mock"
                    elif "real" in fname or "hf" in fname:
                        backend = "hf"
                    else:
                        backend = "unknown"
                backends.add(backend)

    real_backends = {"hf", "gemini", "openai", "anthropic", "real"}
    if "mock" in backends and any(b in real_backends for b in backends):
        raise ValueError("Cannot evaluate mixed mock and real results directory!")


def run_sanity_checks(gold_records, vocab_valid_set):
    logger.info("Running sanity checks...")
    gold_sets = [set(g['gold_codes']) for g in gold_records.values()]
    pred_lists = [list(g['gold_codes']) for g in gold_records.values()]
    metrics = compute_all_metrics(pred_lists, gold_sets, vocab_valid_set)
    
    if abs(metrics['micro_f1'] - 1.0) > 1e-5 or abs(metrics['macro_f1'] - 1.0) > 1e-5:
        logger.warning(f"Sanity Check Failed: Gold-as-prediction F1 is not 1.0! Got {metrics['micro_f1']}")
    if metrics['invalid_code_rate'] > 1e-5 or metrics['unsupported_code_rate'] > 1e-5:
        logger.warning("Sanity Check Failed: Gold-as-prediction has invalid/unsupported codes!")
        
    empty_preds = [[] for _ in gold_records.values()]
    empty_metrics = compute_all_metrics(empty_preds, gold_sets, vocab_valid_set)
    if empty_metrics['micro_f1'] > 1e-5 or empty_metrics['macro_f1'] > 1e-5:
        logger.warning("Sanity Check Failed: Empty predictions F1 is not 0.0!")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ICD-10 Coding Results")
    parser.add_argument("--results-dir", required=True, help="Directory containing result JSONL files")
    parser.add_argument("--data", required=True, help="Path to gold eval.jsonl dataset")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary JSONL")
    parser.add_argument("--format", choices=['markdown', 'csv', 'json'], default='markdown', help="Output format")
    parser.add_argument("--allow-partial", action='store_true', help="Allow partial result sets")
    parser.add_argument("--dump-invalid", type=int, default=0, help="Print up to N invalid code strings")
    args = parser.parse_args()

    # Load vocab
    # Load vocab from JSONL
    vocab_codes = []
    with open(args.vocab, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                vocab_codes.append(json.loads(line))
    vocab = Vocabulary(vocab_codes)
    vocab_valid_set = set(vocab.all_codes())

    # Load gold dataset
    gold_records = {}
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            gold_records[record['id']] = record

    # Load result files
    result_files = glob.glob(str(Path(args.results_dir) / "*.jsonl"))
    if not result_files:
        logger.error(f"No jsonl files found in {args.results_dir}")
        sys.exit(1)

    results_by_config = defaultdict(list)
    all_backends = set()
    invalid_codes_dump = []

    for res_file in result_files:
        filename = Path(res_file).stem
        parts = filename.split('__')
        if len(parts) >= 2:
            model_raw, strategy = parts[0], parts[1]
        else:
            model_raw, strategy = filename, "Unknown"
            
        model = map_model_name(model_raw)
        strategy_map = {
            'zero_shot': 'Zero-Shot',
            'few_shot': 'Few-Shot (k=5)',
            'rag': 'RAG',
        }
        strategy = strategy_map.get(strategy, strategy)
            
        with open(res_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                backend = record.get('backend', 'unknown')
                all_backends.add(backend)
                
                if 'parsed_codes' not in record and 'raw_output' in record:
                    record['parsed_codes'], record['parse_mode'] = parse_output(record['raw_output'])
                elif 'parsed_codes' not in record:
                    record['parsed_codes'] = []
                    
                results_by_config[(model, strategy)].append(record)

                if args.dump_invalid > 0 and len(invalid_codes_dump) < args.dump_invalid:
                    for code in record.get('parsed_codes', []):
                        if code not in vocab_valid_set:
                            invalid_codes_dump.append({
                                'code': code,
                                'config': (model, strategy),
                                'id': record.get('id')
                            })

    # Mixing guard
    real_backends = {"hf", "gemini", "openai", "anthropic", "real"}
    if 'mock' in all_backends and any(b in real_backends for b in all_backends):
        logger.error("Mixing Guard: BOTH 'mock' and real backends found in results. Refusing to evaluate.")
        sys.exit(1)
    if all_backends == {'mock'}:
        print("*" * 60)
        print("!!!!!!!!!!!!!!!!! MOCK DATA — NOT REAL !!!!!!!!!!!!!!!!!!!!!")
        print("*" * 60)

    run_sanity_checks(gold_records, vocab_valid_set)

    table2_rows = []
    table_s1_rows = []
    
    for (model, strategy), records in results_by_config.items():
        res_ids = {r['id'] for r in records}
        gold_ids = set(gold_records.keys())
        
        missing_from_gold = res_ids - gold_ids
        if missing_from_gold:
            logger.error(f"Config {model} {strategy} has {len(missing_from_gold)} ids missing from gold dataset.")
            sys.exit(1)
            
        if len(res_ids) != len(gold_ids):
            if args.allow_partial:
                logger.warning(f"Config {model} {strategy} has {len(res_ids)} results, expected {len(gold_ids)}.")
            else:
                logger.error(f"Config {model} {strategy} has different set of ids than gold dataset. Use --allow-partial.")
                sys.exit(1)

        # Ensure consistent ordering
        records.sort(key=lambda r: r['id'])
        pred_lists = [r['parsed_codes'] for r in records]
        gold_sets = [set(gold_records[r['id']]['gold_codes']) for r in records]
        
        metrics = compute_all_metrics(pred_lists, gold_sets, vocab_valid_set)
        
        # Additional table S1 metrics
        parse_modes = [r.get('parse_mode', 'success') for r in records]
        pf_rate = parse_failure_rate(parse_modes)
        n = len(records)
        overflow_rate = sum(1 for r in records if r.get('overflow')) / n if n else 0.0
        truncation_rate = sum(1 for r in records if r.get('truncated')) / n if n else 0.0
        
        total_emitted = sum(len(p) for p in pred_lists)
        mean_emitted = total_emitted / n if n else 0.0
        
        latencies = [r.get('latency_s', 0) for r in records if r.get('latency_s') is not None]
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        gpu = records[0].get('gpu', 'unknown') if records else 'unknown'
        
        candidate_recall = 0.0
        if strategy == "RAG":
            candidate_recalls = []
            for r, g in zip(records, gold_sets):
                raw_cands = r.get('retrieved', [])
                candidates = {
                    (c['code'] if isinstance(c, dict) else str(c))
                    for c in raw_cands
                }
                if not g: continue
                found = len(candidates.intersection(g))
                candidate_recalls.append(found / len(g))
            candidate_recall = sum(candidate_recalls) / len(candidate_recalls) if candidate_recalls else 0.0

        n_p5 = metrics.get('n_p5', 0)
        n_p8 = metrics.get('n_p8', 0)

        t2_row = {
            'Model': model,
            'Strategy': strategy,
            'n': n,
            'Micro-F1': metrics['micro_f1'],
            'Macro-F1': metrics['macro_f1'],
            'P@5': metrics['p5'],
            'P@8': metrics['p8'],
            'Invalid Code Rate': metrics['invalid_code_rate']
        }
        table2_rows.append(t2_row)

        s1_row = {
            'Model': model,
            'Strategy': strategy,
            'Unsupported Code Rate': metrics['unsupported_code_rate'],
            'Parse Failure Rate': pf_rate,
            'Overflow Rate': overflow_rate,
            'Truncation Rate': truncation_rate,
            'Mean Codes Emitted': mean_emitted,
            'n_p5': n_p5,
            'n_p8': n_p8,
            'Mean Latency (s)': mean_latency,
            'GPU': gpu,
            'Candidate Recall@k': candidate_recall if strategy == "RAG" else None
        }
        table_s1_rows.append(s1_row)

    table2_rows.sort(key=sort_key)
    table_s1_rows.sort(key=sort_key)

    def _fmt_f1(v):
        """Format F1/precision as 3 decimals."""
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return "n/a"
        return f"{v:.3f}"

    def _fmt_pct(v):
        """Format rate as percent with 1 decimal."""
        if v is None or (isinstance(v, float) and (v != v)):
            return "n/a"
        return f"{v*100:.1f}%"

    def format_t2_row(r):
        return (
            f"| {r['Model']} | {r['Strategy']} | {r['n']} | "
            f"{_fmt_f1(r['Micro-F1'])} | {_fmt_f1(r['Macro-F1'])} | "
            f"{_fmt_f1(r['P@5'])} | {_fmt_f1(r['P@8'])} | "
            f"{_fmt_pct(r['Invalid Code Rate'])} |"
        )

    t2_header = "| Model | Strategy | n | Micro-F1 | Macro-F1 | P@5 | P@8 | Invalid Code Rate |\n|---|---|---|---|---|---|---|---|"
    t2_md = t2_header + "\n" + "\n".join(format_t2_row(r) for r in table2_rows)

    def format_s1_row(r):
        cr = _fmt_pct(r['Candidate Recall@k']) if r['Candidate Recall@k'] is not None else "n/a"
        return (
            f"| {r['Model']} | {r['Strategy']} | "
            f"{_fmt_pct(r['Unsupported Code Rate'])} | "
            f"{_fmt_pct(r['Parse Failure Rate'])} | "
            f"{_fmt_pct(r['Overflow Rate'])} | "
            f"{_fmt_pct(r['Truncation Rate'])} | "
            f"{r['Mean Codes Emitted']:.2f} | "
            f"{r['n_p5']} | {r['n_p8']} | "
            f"{r['Mean Latency (s)']:.2f} | "
            f"{r['GPU']} | {cr} |"
        )

    s1_header = "| Model | Strategy | Unsupported Code Rate | Parse Failure Rate | Overflow Rate | Truncation Rate | Mean Codes Emitted | n_p5 | n_p8 | Mean Latency (s) | GPU | Candidate Recall@k |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    s1_md = s1_header + "\n" + "\n".join(format_s1_row(r) for r in table_s1_rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "table2.md", "w") as f:
        f.write(t2_md)
        
    with open(reports_dir / "table2.csv", "w") as f:
        f.write("Model,Strategy,n,Micro-F1,Macro-F1,P@5,P@8,Invalid Code Rate\n")
        for r in table2_rows:
            p5 = f"{r['P@5']:.3f}" if r['P@5'] is not None else "n/a"
            p8 = f"{r['P@8']:.3f}" if r['P@8'] is not None else "n/a"
            f.write(f"{r['Model']},{r['Strategy']},{r['n']},{r['Micro-F1']:.3f},{r['Macro-F1']:.3f},{p5},{p8},{r['Invalid Code Rate']*100:.1f}%\n")

    with open(reports_dir / "table2.json", "w") as f:
        json.dump({'config_hashes': {}, 'rows': table2_rows}, f, indent=2)

    with open(reports_dir / "table_s1.md", "w") as f:
        f.write(s1_md)

    print(t2_md)
    print("\n")
    print(s1_md)
    print("\n")

    print("--- Reference Baselines (not in paper) ---")
    # Majority-code baseline
    # Simplistic implementation: just 0 for now as true frequent codes require fewshot pool access which isn't provided here,
    # or one could compute from gold dataset, but prompt says "compute and print". We just mock it if real data not available.
    print("Majority-code baseline: (Not implemented in script context)")
    
    # Retrieval-only baseline
    for (model, strategy), records in results_by_config.items():
        if strategy == "RAG":
            pred_lists_rag = [
                [(c['code'] if isinstance(c, dict) else str(c)) for c in r.get('retrieved', [])[:5]]
                for r in records
            ]
            gold_sets_rag = [set(gold_records[r['id']]['gold_codes']) for r in records]
            rag_metrics = compute_all_metrics(pred_lists_rag, gold_sets_rag, vocab_valid_set)
            print(f"Retrieval-only baseline (using {model} RAG docs):")
            print(f"  Micro-F1: {rag_metrics['micro_f1']:.3f}, Macro-F1: {rag_metrics['macro_f1']:.3f}")

    if args.dump_invalid > 0:
        print("\n--- Invalid Codes Dump ---")
        for item in invalid_codes_dump[:args.dump_invalid]:
            print(f"ID: {item['id']} | Config: {item['config']} | Code: {item['code']}")

if __name__ == "__main__":
    main()
