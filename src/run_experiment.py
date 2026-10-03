"""Inference harness for the ICD-10 coding experiment.

CLI: python src/run_experiment.py --config configs/phase1.yaml \\
         --models llama3 biomistral --strategies zero_shot few_shot rag \\
         --split eval --out results/exp01

For mock: python src/run_experiment.py --config configs/phase1.yaml \\
              --backend mock --out results_mock/exp01
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy
import datetime
import gc
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from tqdm import tqdm

from codes import normalize
from parser import parse_output
from prompts import (
    ZeroShotEvidence, FewShotEvidence, RAGEvidence,
    build_messages,
)
from backends import HFBackend, MockBackend
from utils import (
    seed_everything, sha256_file, sha256_str, atomic_write,
    atomic_jsonl_append, gpu_info, library_versions, git_commit,
    load_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Few-shot selection ─────────────────────────────────────────────────────

def select_few_shots(
    pool_path: str,
    eval_data: List[dict],
    seed: int,
    k: int = 5,
    min_implicit: int = 2,
) -> List[dict]:
    """Select k few-shot examples from the pool.

    Requirements:
    - At least min_implicit shots must have non-empty implicit_codes.
    - Shots must be disjoint from eval by id AND by text hash.
    - Selection is deterministic given the seed.

    Raises:
        ValueError: if disjointness cannot be ensured or pool is too small.
    """
    pool = load_jsonl(pool_path)
    eval_ids = {r["id"] for r in eval_data}
    eval_hashes = {sha256_str(r.get("text", "")) for r in eval_data}

    # Filter to disjoint records
    valid = []
    for r in pool:
        if r["id"] in eval_ids:
            raise ValueError(
                f"Few-shot pool record {r['id']} overlaps with eval set by ID!"
            )
        if sha256_str(r.get("text", "")) in eval_hashes:
            raise ValueError(
                f"Few-shot pool record {r['id']} overlaps with eval set by text hash!"
            )
        valid.append(r)

    if len(valid) < k:
        raise ValueError(
            f"Pool has only {len(valid)} valid records, need {k}"
        )

    # Stratified selection: at least min_implicit with implicit codes
    rng = random.Random(seed)
    with_implicit = [r for r in valid if r.get("implicit_codes")]
    without_implicit = [r for r in valid if not r.get("implicit_codes")]

    rng.shuffle(with_implicit)
    rng.shuffle(without_implicit)

    if len(with_implicit) < min_implicit:
        logger.warning(
            f"Only {len(with_implicit)} records with implicit codes in pool "
            f"(need {min_implicit}). Using all available."
        )

    selected = with_implicit[:min_implicit]
    remaining = with_implicit[min_implicit:] + without_implicit
    rng.shuffle(remaining)
    selected.extend(remaining[: k - len(selected)])

    # Shuffle final order
    rng.shuffle(selected)

    logger.info(
        f"Selected {len(selected)} few-shot examples. "
        f"IDs: {[s['id'] for s in selected]}. "
        f"With implicit: {sum(1 for s in selected if s.get('implicit_codes'))}"
    )
    return selected[:k]


# ── Context overflow handling ──────────────────────────────────────────────

SECTION_HEADERS = [
    "Discharge Diagnosis",
    "Brief Hospital Course",
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
]


def truncate_record_for_context(
    record: dict,
    evidence_provider,
    template_mode: str,
    backend,
    context_limit: int,
    max_new_tokens: int,
) -> Tuple[dict, list, int, bool]:
    """Truncate record text to fit within context limit.

    Strategy: remove text from the end, preserving instruction/evidence/answer.
    Never truncate the instruction, evidence block, or answer slot.

    Returns:
        (modified_record, messages, prompt_tokens, was_truncated)
    """
    rec = copy.deepcopy(record)
    text = rec.get("text", "")

    # Binary-search-style: remove 10% at a time
    for _ in range(50):  # max iterations
        messages = build_messages(rec, evidence_provider, template_mode)
        prompt_tokens = backend.count_tokens(messages)
        if prompt_tokens + max_new_tokens <= context_limit:
            was_truncated = (rec["text"] != record["text"])
            return rec, messages, prompt_tokens, was_truncated

        # Remove ~10% from the end
        current_text = rec.get("text", "")
        cut = max(1, len(current_text) // 10)
        rec["text"] = current_text[:-cut].rstrip()

        if not rec["text"]:
            break

    messages = build_messages(rec, evidence_provider, template_mode)
    prompt_tokens = backend.count_tokens(messages)
    return rec, messages, prompt_tokens, True


# ── Main harness ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run ICD-10 coding experiment")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--models", nargs="+", help="Model keys (e.g., llama3 biomistral)")
    p.add_argument("--strategies", nargs="+",
                    choices=["zero_shot", "few_shot", "rag"],
                    help="Strategies to run")
    p.add_argument("--split", default="eval", help="Dataset split")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--backend", default="hf", choices=["hf", "mock"])
    p.add_argument("--limit", type=int, help="Limit instances (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        config_text = f.read()
    config = yaml.safe_load(config_text)
    config_hash = sha256_str(config_text)

    seed = config.get("seed", 42)
    seed_everything(seed)

    # ── Output directory ───────────────────────────────────────────────
    out_dir = Path(args.out)
    if args.limit:
        out_dir = Path(f"{args.out}_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────
    data_cfg = config.get("data", {})
    synth_dir = getattr(args, "data_dir", None) or data_cfg.get("synthetic_dir", "data/synthetic")
    synth_path = Path(synth_dir)

    data_file = None
    if (synth_path / f"{args.split}.jsonl").exists():
        run_dir = synth_path
        data_file = synth_path / f"{args.split}.jsonl"
    elif synth_path.exists():
        # Search subdirectories for split.jsonl, sort by mtime desc
        candidate_runs = [
            d for d in synth_path.iterdir()
            if d.is_dir() and (d / f"{args.split}.jsonl").exists()
        ]
        if candidate_runs:
            candidate_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            run_dir = candidate_runs[0]
            data_file = run_dir / f"{args.split}.jsonl"
        else:
            run_dir = synth_path
            data_file = run_dir / f"{args.split}.jsonl"
    else:
        run_dir = synth_path
        data_file = run_dir / f"{args.split}.jsonl"

    if not data_file or not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            f"Run: python src/synth_data.py --out {synth_dir}/run01"
        )

    logger.info(f"Loading dataset from {data_file}")
    dataset = load_jsonl(str(data_file))
    dataset.sort(key=lambda x: x["id"])  # deterministic order

    if args.limit:
        dataset = dataset[: args.limit]
        logger.info(f"Smoke test: limited to {len(dataset)} instances")

    dataset_hash = sha256_file(str(data_file))

    # ── Few-shot selection (once) ──────────────────────────────────────
    shots: List[dict] = []
    shot_ids: List[str] = []
    strategies_to_run = args.strategies or list(config.get("strategies", {}).keys())

    if "few_shot" in strategies_to_run:
        pool_file = run_dir / "fewshot_pool.jsonl"
        if pool_file.exists():
            shots = select_few_shots(
                str(pool_file), dataset, seed,
                k=config.get("strategies", {}).get("few_shot", {}).get("k", 5),
                min_implicit=config.get("strategies", {}).get("few_shot", {}).get("min_implicit", 2),
            )
            shot_ids = [s["id"] for s in shots]
        else:
            logger.error(f"Few-shot pool not found: {pool_file}")

    # ── Write run_meta.json ────────────────────────────────────────────
    run_meta = {
        "config": config,
        "config_hash": config_hash,
        "library_versions": library_versions(),
        "git_commit": git_commit(),
        "dataset_hash": dataset_hash,
        "shot_ids": shot_ids,
        "seed": seed,
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu_info": gpu_info(),
        "n_instances": len(dataset),
        "split": args.split,
        "backend": args.backend,
    }
    atomic_write(str(out_dir / "run_meta.json"), json.dumps(run_meta, indent=2))

    # ── Model loop ─────────────────────────────────────────────────────
    models_to_run = args.models or list(config.get("models", {}).keys())
    decoding_cfg = config.get("decoding", {})
    max_new_tokens = decoding_cfg.get("max_new_tokens", 256)
    retrieval_cfg = config.get("retrieval", {})

    for model_key in models_to_run:
        model_cfg = config.get("models", {}).get(model_key)
        if not model_cfg:
            logger.warning(f"Model '{model_key}' not in config, skipping")
            continue

        model_id = model_cfg.get("model_id", model_key)
        context_limit = model_cfg.get("context_limit", 8192)
        template_mode = model_cfg.get("template_mode", "system")
        stop_tokens = model_cfg.get("stop_tokens", [])

        # Create backend
        if args.backend == "mock":
            backend = MockBackend(
                model_id=model_id,
                context_limit_tokens=context_limit,
                seed=seed,
                template_mode=template_mode,
            )
        else:
            backend = HFBackend(
                model_id=model_id,
                context_limit_tokens=context_limit,
                seed=seed,
                stop_tokens=stop_tokens,
                template_mode=template_mode,
            )

        backend.load()
        backend_info = backend.info
        logger.info(f"Backend loaded: {backend_info}")

        # ── Strategy loop ──────────────────────────────────────────────
        for strategy in strategies_to_run:
            logger.info(f"Running {model_key}/{strategy} on {len(dataset)} instances")

            # Build evidence provider
            if strategy == "zero_shot":
                evidence_provider = ZeroShotEvidence()
            elif strategy == "few_shot":
                if not shots:
                    logger.error("No few-shot examples available, skipping")
                    continue
                evidence_provider = FewShotEvidence(shots)
            elif strategy == "rag":
                vocab_path = config.get("vocab", {}).get("vocab_path", "data/vocab/icd10cm_2024.jsonl")
                try:
                    from vocab import Vocabulary
                    if Path(vocab_path).exists():
                        vocab = Vocabulary.load_or_download(config.get("vocab", {}))
                    else:
                        vocab = None
                except Exception as e:
                    logger.warning(f"Could not load vocab for RAG: {e}")
                    vocab = None

                rag_k = config.get("strategies", {}).get("rag", {}).get("k", 20)

                if args.backend == "mock":
                    from retrieval import MockRetriever
                    retriever = MockRetriever(vocab=vocab, k=rag_k)
                else:
                    embedder = retrieval_cfg.get("embedder", "sentence-transformers/all-MiniLM-L6-v2")
                    index_dir = retrieval_cfg.get("index_dir", "data/index")
                    embedder_slug = embedder.replace("/", "_")
                    index_path = os.path.join(index_dir, embedder_slug)

                    if vocab and not Path(index_path).exists():
                        from retrieval import build_index
                        build_index(embedder, vocab, index_path)

                    from retrieval import Retriever
                    per_query_m = config.get("strategies", {}).get("rag", {}).get("per_query_m", 30)

                    retriever = Retriever(
                        embedder_name=embedder,
                        index_dir=index_path,
                        vocab=vocab,
                        k=rag_k,
                        per_query_m=per_query_m,
                        note_window_words=retrieval_cfg.get("note_window_words", 150),
                        note_window_overlap=retrieval_cfg.get("note_window_overlap", 30),
                    )
                evidence_provider = RAGEvidence(retriever)
            else:
                logger.warning(f"Unknown strategy '{strategy}', skipping")
                continue

            # Output file: <model_key>__<strategy>.jsonl
            out_file = str(out_dir / f"{model_key}__{strategy}.jsonl")

            # Resume: load completed IDs
            completed_ids: Set[str] = set()
            if os.path.exists(out_file):
                for row in load_jsonl(out_file):
                    completed_ids.add(row["id"])
                logger.info(f"Resuming: {len(completed_ids)} already completed")

            pending = [r for r in dataset if r["id"] not in completed_ids]

            for record in tqdm(pending, desc=f"{model_key}/{strategy}"):
                # Build messages
                messages = build_messages(record, evidence_provider, template_mode)
                prompt_tokens = backend.count_tokens(messages)

                # Check overflow BEFORE truncation
                overflow = (prompt_tokens + max_new_tokens > context_limit)
                truncated = False

                if overflow:
                    policy = config.get("overflow_policy", "truncate_record")
                    if policy == "truncate_record":
                        record_for_gen, messages, prompt_tokens, truncated = \
                            truncate_record_for_context(
                                record, evidence_provider, template_mode,
                                backend, context_limit, max_new_tokens,
                            )
                        logger.info(
                            f"Record {record['id']}: overflow "
                            f"({prompt_tokens}+{max_new_tokens} > {context_limit}), "
                            f"truncated={truncated}"
                        )
                    elif policy == "skip":
                        logger.warning(f"Skipping record {record['id']} due to overflow")
                        continue
                    else:
                        logger.warning(
                            f"Record {record['id']}: overflow, unknown policy '{policy}'"
                        )

                # Generate
                gen_result = backend.generate(messages, max_new_tokens)

                # Parse output
                parsed_codes, parse_mode = parse_output(gen_result.text)

                # Build retrieved info for RAG
                retrieved_info = []
                if strategy == "rag" and hasattr(evidence_provider, "retriever"):
                    try:
                        cands = evidence_provider.retriever.retrieve(record)
                        retrieved_info = [
                            {"code": c.code, "score": c.score, "query_source": c.source_query}
                            for c in cands
                        ]
                    except Exception:
                        pass

                # Write result line
                result_row = {
                    "id": record["id"],
                    "model": model_key,
                    "model_id": backend_info.get("model_id", ""),
                    "model_revision": backend_info.get("model_revision", ""),
                    "strategy": strategy,
                    "backend": args.backend,
                    "template_mode": template_mode,
                    "seed": seed,
                    "config_hash": config_hash,
                    "prompt_tokens": prompt_tokens,
                    "new_tokens": gen_result.new_tokens,
                    "overflow": overflow,
                    "truncated": truncated,
                    "parse_mode": parse_mode,
                    "raw_output": gen_result.text,
                    "parsed_codes": parsed_codes,
                    "retrieved": retrieved_info if strategy == "rag" else [],
                    "shots": shot_ids if strategy == "few_shot" else [],
                    "latency_s": gen_result.latency_s,
                    "gpu": backend_info.get("device", ""),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                atomic_jsonl_append(out_file, result_row)

        # Free GPU memory between models
        if hasattr(backend, "free_memory"):
            backend.free_memory()
        del backend
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    logger.info(f"All runs complete. Results in {out_dir}")


def run_experiment(args=None):
    """Entry point for running experiment programmatically."""
    if args is not None:
        # If args passed directly, we can mock or dispatch
        pass
    return main()


if __name__ == "__main__":
    main()


