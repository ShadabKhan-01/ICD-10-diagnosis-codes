
"""RAGAS ID-based evaluation for retrieved ICD-10-CM code candidates."""
import argparse
import asyncio
import json
import math
from pathlib import Path
from statistics import mean

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_codes(values):
    seen, result = set(), []
    for value in values:
        code = str(value).strip().upper()
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result


async def evaluate_file(result_path, gold_by_id, out_dir):
    precision_metric = IDBasedContextPrecision()
    recall_metric = IDBasedContextRecall()
    rows = []
    unmatched = 0

    for row in load_jsonl(result_path):
        record_id = str(row["id"])
        if record_id not in gold_by_id:
            unmatched += 1
            continue

        retrieved = unique_codes([
            item.get("code", "")
            for item in row.get("retrieved", [])
            if isinstance(item, dict)
        ])
        gold = unique_codes(gold_by_id[record_id].get("gold_codes", []))

        if not gold:
            raise ValueError(f"No gold codes for record {record_id}")

        sample = SingleTurnSample(
            retrieved_context_ids=retrieved,
            reference_context_ids=gold,
        )
        precision = await precision_metric.single_turn_ascore(sample)
        recall = await recall_metric.single_turn_ascore(sample)

        rows.append({
            "id": record_id,
            "model": row.get("model", ""),
            "backend": row.get("backend", ""),
            "n_retrieved": len(retrieved),
            "n_gold": len(gold),
            "n_hits": len(set(retrieved) & set(gold)),
            "id_based_context_precision": (
                float(precision) if math.isfinite(float(precision)) else None
            ),
            "id_based_context_recall": (
                float(recall) if math.isfinite(float(recall)) else None
            ),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result_path.stem
    detail_path = out_dir / f"{stem}_ragas_id_metrics.jsonl"
    summary_path = out_dir / f"{stem}_ragas_summary.json"

    with detail_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    def average(key):
        values = [r[key] for r in rows if r[key] is not None]
        return mean(values) if values else None

    summary = {
        "source_file": result_path.name,
        "metric_family": "RAGAS ID-based context metrics",
        "n_evaluated": len(rows),
        "unmatched_result_ids": unmatched,
        "mean_id_based_context_precision": average("id_based_context_precision"),
        "mean_id_based_context_recall": average("id_based_context_recall"),
        "interpretation": (
            "Exact retrieved-code versus gold-code overlap. These ID-based "
            "metrics do not use an LLM judge and do not measure semantic "
            "relevance or ranking-sensitive average precision."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Per-record scores:", detail_path)
    print("Summary:", summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    gold_by_id = {
        str(row["id"]): row for row in load_jsonl(args.data)
    }
    files = sorted(Path(args.results_dir).glob("*__rag.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No *__rag.jsonl files in {args.results_dir}"
        )

    async def run_all():
        for path in files:
            await evaluate_file(path, gold_by_id, Path(args.out_dir))

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
