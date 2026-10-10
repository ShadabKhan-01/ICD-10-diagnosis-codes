# Change log: four-strategy ICD-10-CM coding pipeline

## Round 3 (this release)

### Hierarchy (official ICD-10-CM 2024 tabular XML)
- `src/kg.py build --zip ... --xml ... --vocab ... --out ...` extracts the tabular XML from
  the official CDC ZIP, parses it, validates it, and writes the JSONL hierarchy.
- `Hierarchy.add()` now raises on a code that appears under two different parents or with a
  conflicting description. Before, the first occurrence was silently kept.
- `validate_hierarchy()` checks: every parent exists as a code; child lists agree with
  parent links; there are no cycles; every code is well-formed; and, when `--vocab` is given,
  the code set equals the verified vocabulary exactly. Any failure stops the build and writes
  no output.
- Zip member matching: the tabular XML is any single `.xml` whose file name contains
  "tabular". Zero or several matches stop the build with a message.

### KG review (candidate expansion, ranking, dedup, limit)
- Ranking: semantic seeds keep their RAG order first. Expansions follow, in the order
  seed → its parent → its children (XML order, at most `max_children`). Expansions take the
  score of the seed they came from; they are not reranked. Documented as a fixed choice.
- Deduplication: a code is added at most once, so a parent that is also another seed is not
  repeated.
- Candidate limit: `run_experiment.py` raises if `strategies.kg.k` differs from
  `strategies.rag.k`, so both arms offer the same number of candidates (default 20).
- Leakage: retrievers get only `retrieval_view()` (id, text, medications). Covered by tests.

### Integration
- `zero_shot`, `few_shot`, `rag` and `kg` all run through `run_experiment.py`
  (`--strategies`), write the same row schema, and are reported by `evaluate.py`
  (Table II: Precision, Recall, Micro-F1, Macro-F1, P@5, P@8, invalid rate; Table S1:
  unsupported rate, parse failures, truncation, candidate recall for RAG and KG).

### Tests
- New tests (fixture data): conflicting-parent, orphan-parent, cycle, malformed-code, and
  vocabulary-mismatch rejection; KG/RAG limit mismatch. One new test runs only on the
  official file (`test_real_official_tabular_xml_if_present`).
- The "Test results" section below lists exactly what ran.

### README
- Rewritten for Google Colab: upload or download the official ZIP, extract and build the
  hierarchy, verify it, run the 25-example pilot, and produce the comparison table.

## Round 2

### Added
- `src/kg.py`: KG strategy. Official tabular XML parsing, `KGRetriever` (hybrid), CLI.
- `tests/test_strategies.py`: hierarchy, KG expansion, leakage, prompt invariance,
  precision and recall, end-to-end mock run with evaluation.
- `tests/fixtures/tabular_sample.xml`: a TEST FILE in the official nesting. Not CDC data.
  Used only for parser and expansion tests.

### Changed
- `src/retrieval.py`: `retrieval_view()`.
- `src/prompts.py`: `RAGEvidence` passes `retrieval_view(record)`. Output unchanged.
- `src/run_experiment.py`: `kg` strategy; shared semantic retriever (one index load per
  run); `load_kg_hierarchy()` fails loudly if the file is missing.
- `src/metrics.py`: `micro_precision_recall()`; `micro_precision` and `micro_recall` keys
  added. Existing definitions unchanged.
- `src/evaluate.py`: `kg` → `KG`; Precision and Recall columns.
- `configs/phase1.yaml`: `strategies.kg` (k=20, seed_k=10, max_children=5) and `kg.hierarchy_path`.

### Fixed
- Duplicate BOS for LLaMA-3 8B (`tokenize_special_tokens: false`). Verified on the cached
  Llama-3.1-8B tokenizer (same template family): generate path `[128000, 128000]`, count
  path `[128000]`. Not verified on the 8B checkpoint itself. BioMistral is not changed.

## Earlier rounds (LLaMA debugging)
- `src/utils.py`: `props.total_mem` → `props.total_memory`.
- `src/parser.py`: a code cut off at the token limit is dropped when generation was truncated.
- `src/backends.py`: `generation_truncated` flag; float16 compute forced on CUDA, with a
  check that the dtype in use matches the request; `allow_cpu` opt-in for plumbing tests.
- `src/run_experiment.py`: `generation_truncated` and `compute_dtype` per row; `--allow-cpu`.

## Test results (executed in this environment, CPU only)
- `python -m pytest -q -rs tests/`: **92 passed, 2 skipped.**
- The 2 skips:
  - `test_cuda_load_uses_float16_compute`: needs a CUDA GPU (runs on Colab T4).
  - `test_real_official_tabular_xml_if_present`: needs the OFFICIAL tabular XML in
    `data/raw/`. See "Not validated" below.
- Tests that use **fixture** data (not official): the hierarchy parser and expansion tests,
  validation tests, the end-to-end mock run (uses `tests/fixtures/tabular_sample.xml`).
- Tests that use **official** data: none have run yet. The real-XML test is written to
  validate the official file against the verified vocabulary, but it has not executed.

## Not validated
- The official hierarchy. The CDC tabular ZIP (22.7 MB) could not be downloaded in full in
  this environment; transfers stalled, and the most recent partial file was not a valid ZIP.
  Nothing was substituted. The build and its validation must be run in Colab with the
  official file.
- Any model run: LLaMA-3-8B, BioMistral, or LLaMA-3.2-3B, with any strategy. No GPU here.
- The 25-example pilot and the full experiment.
