# ASSUMPTIONS.md
## Decisions for: "Zero-Shot, Few-Shot, or Retrieval-Augmented? Prompting Strategies for ICD-10 Coding with Open-Weight LLMs"

> Every decision the paper leaves open is recorded here. Updated as the pipeline is built.
> Items marked **paper-specified** come directly from the manuscript; items marked **our decision** are choices we made.

| # | Item | Default | Source | Rationale |
|---|------|---------|--------|-----------|
| 1 | Vocabulary release | FY2024 ICD-10-CM (CDC/NCHS) | **our decision** | Most recent complete release at time of implementation |
| 2 | Valid code definition | A code is *valid* iff it exists anywhere in the pinned CDC file (header or billable) | **paper wording + our decision** | Paper says "does not exist in ICD-10-CM"; we also log non-billable rate separately |
| 3 | Embedding text format | `"{code}: {long_desc}"` | **our decision** | Provides both the code and its full description to the embedder |
| 4 | Embedder | `sentence-transformers/all-MiniLM-L6-v2` | **paper-specified** | Default in §2 of the paper |
| 5 | RAG queries | Note text (windowed, ~150 words, 30-word overlap) + each medication separately | **paper + windowing clarification** | MiniLM truncates at ~256 word-pieces; windowing prevents silent information loss |
| 6 | RAG k (retrieval candidates) | 20 | **our decision** | Ablation sweeps {10, 20, 30, 50} |
| 7 | RAG merge strategy | Max score per code across all queries | **paper-specified** | §4.3 of the brief |
| 8 | RAG per-query top-m | m = 30 (≥ k) | **our decision** | Ensures sufficient candidates before merge |
| 9 | Few-shot shots | Fixed 5, chosen once by seed, ≥2 with implicit codes | **our decision** | Stratified selection from fewshot_pool |
| 10 | Decoding | Greedy (`do_sample=False`), no temperature/top_p | **paper-specified** | §2 and §4.5 |
| 11 | `max_new_tokens` | 256 | **our decision** | Sufficient for JSON list of 3–15 codes; identical across all arms |
| 12 | Batch size | 1 | **our decision** | Padding changes numerics and breaks determinism |
| 13 | Output format | JSON `{"codes": [...]}` | **our decision** | Clear, parseable; fallback to regex extraction |
| 14 | Eval size | n = 200 (adjustable via CLI) | **our decision** | Sample table size in the paper |
| 15 | Macro-F1 variant | Over codes present in the gold labels of the evaluated instances | **paper §6.3** | "Gold-set variant" as specified |
| 16 | P@k definition | First k emitted codes; only instances with ≥k emitted codes contribute | **paper-specified** | §5 |
| 17 | Overflow policy | `truncate_record` — truncate record text from the end; keep instance; log `overflow` and `truncated` flags | **our decision** | Never truncate instruction, evidence block, or answer slot |
| 18 | Section extraction order (truncation) | Discharge Diagnosis → Brief Hospital Course → Chief Complaint → HPI → PMH | **our decision** | Clinically decisive sections prioritized |
| 19 | MIMIC few-shot pool | Synthetic pool (labelled as such) unless configured otherwise | **our decision** | Avoids data contamination between MIMIC eval and pool |
| 20 | BioMistral system message handling | Prepend system text to user message (Mistral template rejects system role) | **paper-specified** | §4.4 |
| 21 | LLaMA-3 stop tokens | `[eos_token_id, <\|eot_id\|>]` | **paper-specified** | §4.5 |
| 22 | Quantisation | NF4 4-bit via bitsandbytes (`bnb_4bit_compute_dtype=float16`, `bnb_4bit_use_double_quant=True`) | **paper-specified** | §2 |
| 23 | Context limits | LLaMA-3-8B-Instruct: 8192 tokens; BioMistral-7B: 2048 tokens | **paper-specified** | §1 |
| 24 | Synthetic record length | Target 120–220 words (`--max-words` CLI param) | **paper-specified** | Ensures BioMistral few-shot does not systematically overflow |
| 25 | Condition library size | ≥60 conditions across 9+ clinical domains | **paper-specified** | §4.2 |
| 26 | Conditions per record | 3–6, sampled with Zipf-like weighting | **paper-specified** | §4.2 |
| 27 | Distractors per record | 1–3 (negations, family history, distractor meds) | **paper-specified** | §4.2 |
| 28 | Implicit rate | 0.45 default; ablation {0.0, 0.25, 0.45, 0.65} | **paper-specified** | §4.2 |
| 29 | Bootstrap resamples | B = 10,000, seeded, paired | **paper-specified** | §8.1 |
| 30 | Parse failure handling | Log per instance; count `regex` + `empty` as parse failures; report rate | **our decision** | §4.6 |
| 31 | Invalid/non-billable codes | Invalid codes are NOT dropped from predictions (they count in Invalid Code Rate); non-billable rate reported separately as a diagnostic | **paper-specified + our decision** | §4.1, §4.6 |
| 32 | Deterministic algorithms | `torch.use_deterministic_algorithms` where feasible; seeds set for `torch`, `numpy`, `random` | **our decision** | Up to GPU numerics |
| 33 | Record text in results | No raw note text for MIMIC runs; synthetic result files may contain `raw_output` | **paper-specified** | §0, rule 7 |

---

## Changelog

- **Initial creation**: All defaults seeded from the coding agent brief §12.
- **Milestone 1 Implementation**: Verified with official CDC FY2024 release (`icd10cm_order_2024.txt`, SHA-256: `d116081f3da8784e78173008449d23e383ecb8143f03c2e0a2d57223aeaa1a06`). Code count: 97,296 total; 74,044 billable.
- **Milestone 2 Implementation**: Built and validated condition library (`conditions.yaml`) with 77 clinical conditions across 9 domains. Corrected non-canonical `H40.90X0` to official CDC valid code `H40.9`. Generated 280-record synthetic dataset (30 dev, 50 fewshot pool, 200 eval) with 43.38% implicit rate (within ±3 pts of 45% target) and verified disjointness.
- **Milestone 3-4 Implementation**: Implemented `prompts.py` (prompt invariance tested), `parser.py` (JSON and regex fallback), `backends.py` (HF and Mock), `retrieval.py` (`Retriever`, `MockRetriever`, and diagnostic candidate recall), and `run_experiment.py` (crash-safe, atomic resume).
- **Phase 2 Implementation**: Implemented `significance.py` (vectorized paired bootstrap B=10,000) and `ablations.py` (k-sweep, embedder, and implicit rate sweeps).
- **Verification**: Complete test suite of 34 unit and integration tests passing at 100%.


