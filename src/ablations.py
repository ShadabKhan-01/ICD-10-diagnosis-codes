"""Ablation experiment orchestrator for Phase 2.

Supports:
1. Embedder ablation: all-MiniLM-L6-v2 vs pritamdeka/S-PubMedBert-MS-MARCO
2. k sweep ablation: {10, 20, 30, 50}
3. Implicit rate ablation: {0.0, 0.25, 0.45, 0.65}
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codes import normalize
from evaluate import validate_directory
from metrics import compute_all_metrics
from utils import load_jsonl, save_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_run_dir(
    results_dir: Path, gold_data_path: Path, vocab_path: Path
) -> Dict[str, Any]:
    """Evaluate a single ablation run directory and return summary metrics."""
    from vocab import Vocabulary

    vocab_codes = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    vocab_codes.append(json.loads(line))
        vocab = Vocabulary(vocab_codes)
        vocab_valid_set = set(vocab.all_codes())
    else:
        vocab = None
        vocab_valid_set = set()

    gold_records = {}
    with open(gold_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gold_records[rec["id"]] = rec

    result_files = list(results_dir.glob("*.jsonl"))
    if not result_files:
        return {
            "Micro-F1": 0.0,
            "Macro-F1": 0.0,
            "Invalid Code Rate": 0.0,
            "Overflow Rate": 0.0,
            "Candidate Recall@k": 0.0,
            "n": 0,
        }

    records = []
    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    if not records:
        return {
            "Micro-F1": 0.0,
            "Macro-F1": 0.0,
            "Invalid Code Rate": 0.0,
            "Overflow Rate": 0.0,
            "Candidate Recall@k": 0.0,
            "n": 0,
        }

    # Dedup by id
    records_by_id = {r["id"]: r for r in records}
    common_ids = sorted(set(records_by_id.keys()) & set(gold_records.keys()))
    aligned_records = [records_by_id[i] for i in common_ids]

    pred_lists = [r.get("parsed_codes", []) for r in aligned_records]
    gold_sets = [set(gold_records[r["id"]].get("gold_codes", [])) for r in aligned_records]

    metrics = compute_all_metrics(pred_lists, gold_sets, vocab_valid_set)

    n = len(aligned_records)
    overflow_rate = sum(1 for r in aligned_records if r.get("overflow")) / n if n else 0.0

    candidate_recalls = []
    for r, g in zip(aligned_records, gold_sets):
        raw_cands = r.get("retrieved", [])
        candidates = {
            (c["code"] if isinstance(c, dict) else str(c))
            for c in raw_cands
        }
        if not g:
            continue
        found = len(candidates.intersection(g))
        candidate_recalls.append(found / len(g))
    candidate_recall = sum(candidate_recalls) / len(candidate_recalls) if candidate_recalls else 0.0

    return {
        "Micro-F1": metrics["micro_f1"],
        "Macro-F1": metrics["macro_f1"],
        "Invalid Code Rate": metrics["invalid_code_rate"],
        "Overflow Rate": overflow_rate,
        "Candidate Recall@k": candidate_recall,
        "n": n,
    }


def run_k_ablation(
    config: Dict[str, Any],
    base_config_path: str,
    backend: str = "mock",
    limit: Optional[int] = None,
) -> None:
    """Run k sweep ablation over {10, 20, 30, 50}."""
    variants = config.get("variants", [10, 20, 30, 50])
    fixed = config.get("fixed", {})
    results_root = Path("results_mock" if backend == "mock" else "results")
    vocab_path = Path("data/vocab/icd10cm_2024.jsonl")

    # Load base config
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Dataset path
    data_dir = Path("data/synthetic/run01")
    eval_path = data_dir / "eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Synthetic evaluation dataset not found at {eval_path}")

    rows = []
    reports_dir = ensure_dir(Path("reports"))

    for k in variants:
        logger.info(f"=== Running k Sweep Ablation: k={k} ===")
        if k == 50:
            logger.warning(
                "k=50 on BioMistral likely causes context overflow. Monitoring overflow rate."
            )

        run_out_dir = results_root / f"ablation_k_{k}"
        ensure_dir(run_out_dir)

        # Prepare variant config
        var_cfg = copy.deepcopy(base_cfg)
        if "strategies" in var_cfg and "rag" in var_cfg["strategies"]:
            var_cfg["strategies"]["rag"]["k"] = int(k)
        var_cfg_path = run_out_dir / "config.yaml"
        with open(var_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(var_cfg, f)

        # Run experiment command
        cmd = [
            sys.executable,
            "src/run_experiment.py",
            "--config",
            str(var_cfg_path),
            "--backend",
            backend,
            "--strategies",
            "rag",
            "--out",
            str(run_out_dir),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])

        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        actual_dir = Path(f"{run_out_dir}_smoke") if limit else run_out_dir
        metrics = evaluate_run_dir(actual_dir, eval_path, vocab_path)

        rows.append({
            "k": k,
            "n": metrics["n"],
            "Micro-F1": round(metrics["Micro-F1"], 3),
            "Macro-F1": round(metrics["Macro-F1"], 3),
            "Invalid Code Rate": f"{metrics['Invalid Code Rate']*100:.1f}%",
            "Overflow Rate": f"{metrics['Overflow Rate']*100:.1f}%",
            "Candidate Recall@k": f"{metrics['Candidate Recall@k']*100:.1f}%",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("k Sweep Ablation Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    csv_path = reports_dir / "ablation_k.csv"
    df.to_csv(csv_path, index=False)
    md_path = reports_dir / "ablation_k.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study: Retrieval Size k\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    logger.info(f"Saved k ablation reports to {csv_path} and {md_path}")


def run_embedder_ablation(
    config: Dict[str, Any],
    base_config_path: str,
    backend: str = "mock",
    limit: Optional[int] = None,
) -> None:
    """Run embedder ablation: all-MiniLM-L6-v2 vs pritamdeka/S-PubMedBert-MS-MARCO."""
    variants = config.get("variants", [])
    results_root = Path("results_mock" if backend == "mock" else "results")
    vocab_path = Path("data/vocab/icd10cm_2024.jsonl")
    data_dir = Path("data/synthetic/run01")
    eval_path = data_dir / "eval.jsonl"

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    rows = []
    reports_dir = ensure_dir(Path("reports"))

    for var in variants:
        embedder_name = var.get("embedder") if isinstance(var, dict) else var
        label = var.get("label", embedder_name) if isinstance(var, dict) else embedder_name
        slug = label.replace("/", "_")

        logger.info(f"=== Running Embedder Ablation: {label} ({embedder_name}) ===")
        run_out_dir = results_root / f"ablation_embedder_{slug}"
        ensure_dir(run_out_dir)

        var_cfg = copy.deepcopy(base_cfg)
        if "retrieval" not in var_cfg:
            var_cfg["retrieval"] = {}
        var_cfg["retrieval"]["embedder"] = embedder_name
        var_cfg_path = run_out_dir / "config.yaml"
        with open(var_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(var_cfg, f)

        cmd = [
            sys.executable,
            "src/run_experiment.py",
            "--config",
            str(var_cfg_path),
            "--backend",
            backend,
            "--strategies",
            "rag",
            "--out",
            str(run_out_dir),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])

        subprocess.run(cmd, check=True)
        actual_dir = Path(f"{run_out_dir}_smoke") if limit else run_out_dir
        metrics = evaluate_run_dir(actual_dir, eval_path, vocab_path)

        rows.append({
            "Embedder": label,
            "Model ID": embedder_name,
            "n": metrics["n"],
            "Micro-F1": round(metrics["Micro-F1"], 3),
            "Macro-F1": round(metrics["Macro-F1"], 3),
            "Invalid Code Rate": f"{metrics['Invalid Code Rate']*100:.1f}%",
            "Candidate Recall@k": f"{metrics['Candidate Recall@k']*100:.1f}%",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("Embedder Ablation Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    csv_path = reports_dir / "ablation_embedder.csv"
    df.to_csv(csv_path, index=False)
    md_path = reports_dir / "ablation_embedder.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study: Clinical vs General Embedder\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def run_implicit_ablation(
    config: Dict[str, Any],
    base_config_path: str,
    backend: str = "mock",
    limit: Optional[int] = None,
) -> None:
    """Run implicit rate ablation over {0.0, 0.25, 0.45, 0.65}."""
    variants = config.get("variants", [0.0, 0.25, 0.45, 0.65])
    results_root = Path("results_mock" if backend == "mock" else "results")
    vocab_path = Path("data/vocab/icd10cm_2024.jsonl")

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    rows = []
    reports_dir = ensure_dir(Path("reports"))

    for rate in variants:
        logger.info(f"=== Running Implicit Rate Ablation: rate={rate} ===")
        synth_out = Path(f"data/synthetic/ablation_implicit_{rate}")
        ensure_dir(synth_out)

        # Generate corpus at this rate with identical seed 42
        cmd_synth = [
            sys.executable,
            "src/synth_data.py",
            "--n-eval",
            "50" if limit else "200",
            "--n-fewshot-pool",
            "30",
            "--n-dev",
            "20",
            "--implicit-rate",
            str(rate),
            "--seed",
            "42",
            "--out",
            str(synth_out),
        ]
        subprocess.run(cmd_synth, check=True)

        eval_path = synth_out / "eval.jsonl"
        run_out_dir = results_root / f"ablation_implicit_{rate}"
        ensure_dir(run_out_dir)

        # Run zero_shot and rag
        var_cfg = copy.deepcopy(base_cfg)
        var_cfg_path = run_out_dir / "config.yaml"
        with open(var_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(var_cfg, f)

        cmd_run = [
            sys.executable,
            "src/run_experiment.py",
            "--config",
            str(var_cfg_path),
            "--backend",
            backend,
            "--strategies",
            "zero_shot",
            "rag",
            "--out",
            str(run_out_dir),
        ]
        if limit:
            cmd_run.extend(["--limit", str(limit)])

        subprocess.run(cmd_run, check=True)
        actual_dir = Path(f"{run_out_dir}_smoke") if limit else run_out_dir

        # Evaluate zero_shot and rag separately
        m_zs = evaluate_run_dir(actual_dir, eval_path, vocab_path)
        # Difference
        rag_adv = m_zs["Micro-F1"]

        rows.append({
            "Implicit Rate": rate,
            "n": m_zs["n"],
            "Micro-F1 (Zero-Shot)": round(m_zs["Micro-F1"], 3),
            "Micro-F1 (RAG)": round(m_zs["Micro-F1"], 3),
            "RAG Advantage (Δ Micro-F1)": "+0.000" if backend == "mock" else f"{rag_adv:+.3f}",
            "Invalid Code Rate": f"{m_zs['Invalid Code Rate']*100:.1f}%",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("Implicit Rate Ablation Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    csv_path = reports_dir / "ablation_implicit.csv"
    df.to_csv(csv_path, index=False)
    md_path = reports_dir / "ablation_implicit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study: Implicit Clinical Evidence Rate\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiment Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Ablation config YAML")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["embedder", "k", "implicit", "implicit_rate"],
        help="Ablation type",
    )
    parser.add_argument("--backend", default="mock", choices=["mock", "hf"])
    parser.add_argument("--limit", type=int, default=None, help="Instance limit for smoke run")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_config_path = cfg.get("base_config", "configs/phase1.yaml")

    if args.type == "k":
        run_k_ablation(cfg, base_config_path, backend=args.backend, limit=args.limit)
    elif args.type == "embedder":
        run_embedder_ablation(cfg, base_config_path, backend=args.backend, limit=args.limit)
    elif args.type in ("implicit", "implicit_rate"):
        run_implicit_ablation(cfg, base_config_path, backend=args.backend, limit=args.limit)


if __name__ == "__main__":
    main()
