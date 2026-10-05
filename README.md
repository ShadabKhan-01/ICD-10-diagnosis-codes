# Zero-Shot, Few-Shot, or Retrieval-Augmented? Prompting Strategies for ICD-10 Coding with Open-Weight LLMs

A complete, reproducible research pipeline for evaluating prompting strategies for automated ICD-10-CM coding using open-weight LLMs.

## Hardware Requirements

- **GPU**: One NVIDIA GPU with ≥12 GB VRAM (e.g., T4 16 GB, free-tier Colab/Kaggle)
- **CPU-only**: Mock backend available for testing (`--backend mock`)
- **OS**: Linux or Windows (WSL2 recommended for GPU). `bitsandbytes` NF4 does **NOT** work on Apple-silicon Macs.
- **Storage**: ~20 GB for two 4-bit models + embeddings

## Quick Start

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Process vocabulary
python src/vocab.py --config configs/phase1.yaml

# 3. Generate synthetic data
python src/synth_data.py --n-eval 200 --n-fewshot-pool 50 --n-dev 30 \
    --implicit-rate 0.45 --seed 42 --max-words 200 --out data/synthetic/run01

# 4. Run tests (CPU, no GPU needed)
python -m pytest tests/ -v

# 5. Smoke test with mock backend (CPU)
python src/run_experiment.py --config configs/phase1.yaml \
    --backend mock --strategies zero_shot few_shot rag \
    --limit 20 --out results_mock/exp01

# 6. Evaluate smoke results
python src/evaluate.py --results-dir results_mock/exp01_smoke \
    --data data/synthetic/run01/eval.jsonl \
    --vocab data/vocab/icd10cm_2024.jsonl --allow-partial --dump-invalid 5

# 7. Full Phase 1 run (requires GPU)
export HF_TOKEN=<your_token>  # Accept LLaMA-3 licence first
python src/run_experiment.py --config configs/phase1.yaml \
    --models llama3 biomistral --strategies zero_shot few_shot rag \
    --split eval --out results/exp01

# 8. Evaluate
python src/evaluate.py --results-dir results/exp01 \
    --data data/synthetic/run01/eval.jsonl \
    --vocab data/vocab/icd10cm_2024.jsonl --dump-invalid 5

# 9. Phase 2: Significance testing
python src/significance.py --results-dir results/exp01 \
    --data data/synthetic/run01/eval.jsonl --seed 42 --B 10000

# 10. Phase 2: Ablations
python src/ablations.py --config configs/ablation_k.yaml --type k
```

## What Each Paper Number Maps To

| Paper Section | Source | Command |
|---|---|---|
| Table II (main results) | `evaluate.py` main table | `python src/evaluate.py --results-dir results/exp01 ...` |
| §3.4 (dataset stats) | `stats.json` in synthetic dir | `python src/synth_data.py ...` |
| §4.3 (invalid examples) | `evaluate.py --dump-invalid` | `python src/evaluate.py ... --dump-invalid 5` |
| §4.4 (supplementary) | Table S1 from `evaluate.py` | Same as Table II |
| §4.5 (MIMIC validation) | MIMIC run | `python src/data_prep.py ...` then same pipeline |
| §4.6 (ablation tables) | `ablations.py` | `python src/ablations.py ...` |
| §4.7 (significance) | `significance.py` | `python src/significance.py ...` |

## Models & Supported Backends

The research pipeline supports evaluating **any open-source** or **commercial API** model:

| Model Category | Examples | Backend | Requirements / Notes |
|---|---|---|---|
| **Open-Source (HuggingFace)** | LLaMA-3-8B-Instruct (`llama3`), BioMistral-7B (`biomistral`), Qwen-2.5-7B (`qwen2.5-7b`) | `hf` | CUDA GPU (4-bit NF4 quantisation, ~5–6 GB VRAM) |
| **Google Gemini API** | `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash` | `gemini` | `export GEMINI_API_KEY="..."` (Runs on CPU/GPU, fast) |
| **OpenAI API** | `gpt-4o-mini`, `gpt-4o` | `openai` | `export OPENAI_API_KEY="..."` (Runs on CPU/GPU) |
| **Anthropic Claude** | `claude-3-5-sonnet`, `claude-3-haiku` | `anthropic` | `export ANTHROPIC_API_KEY="..."` |
| **Local OpenAI-Compatible** | Ollama / vLLM local endpoints | `openai` | Set `OPENAI_BASE_URL="http://localhost:11434/v1"` |
| **Mock (Testing)** | Deterministic CPU mock | `mock` | CPU-only, no downloads or keys required |

### Running Custom Models

You can run any model by passing its key or model ID directly to `--models`:

```bash
# 1. Test Google Gemini (runs directly on CPU or GPU without heavy downloads)
export GEMINI_API_KEY="your-api-key"
python src/run_experiment.py --config configs/phase1.yaml \
    --models gemini-1.5-flash --strategies zero_shot few_shot rag --out results/gemini_exp

# 2. Test OpenAI GPT-4o-mini
export OPENAI_API_KEY="your-api-key"
python src/run_experiment.py --config configs/phase1.yaml \
    --models gpt-4o-mini --strategies zero_shot few_shot rag --out results/openai_exp

# 3. Test any Hugging Face model
python src/run_experiment.py --config configs/phase1.yaml \
    --models Qwen/Qwen2.5-7B-Instruct --strategies zero_shot few_shot rag --out results/qwen_exp
```

## Project Structure

```
├── README.md              # This file
├── ASSUMPTIONS.md         # Every decision the paper leaves open
├── requirements.txt       # Python dependencies
├── Makefile               # Make targets: setup, vocab, data, smoke, phase1, evaluate, test, phase2
├── configs/               # YAML configuration files
│   ├── phase1.yaml        # Main experiment config
│   ├── ablation_*.yaml    # Ablation configs
│   └── mimic.yaml         # MIMIC-IV config
├── data/                  # Data directory (gitignored except fixtures)
│   ├── raw/               # Downloaded CDC files
│   ├── vocab/             # Processed ICD-10-CM vocabulary
│   ├── index/             # Embedding indices
│   └── synthetic/         # Generated synthetic records
├── src/                   # Source code
│   ├── codes.py           # ICD-10 code normalization/validation
│   ├── vocab.py           # CDC vocabulary download/parse
│   ├── synth_data.py      # Synthetic data generator
│   ├── templates/         # Condition library (YAML)
│   ├── retrieval.py       # Embedding index & retrieval
│   ├── prompts.py         # Prompt construction (3 strategies)
│   ├── parser.py          # LLM output parser
│   ├── backends.py        # LLM backend abstraction (HF, Mock)
│   ├── run_experiment.py  # Inference harness
│   ├── metrics.py         # Pure metric functions
│   ├── evaluate.py        # Table generation
│   ├── significance.py    # Paired bootstrap testing (Phase 2)
│   ├── ablations.py       # Ablation orchestrator (Phase 2)
│   ├── data_prep.py       # MIMIC-IV loader (Phase 2)
│   └── utils.py           # Shared utilities
├── tests/                 # pytest test suite
├── results/               # Real experiment results (gitignored)
├── results_mock/          # Mock results (gitignored)
└── reports/               # Auto-generated tables
```

## Expected Runtime

| Stage | Hardware | Time |
|---|---|---|
| Vocab processing | CPU | <1 min |
| Synthetic generation | CPU | <1 min |
| Index building | CPU | ~5 min |
| Smoke test (mock, 20 inst) | CPU | <1 min |
| Full Phase 1 (200×6 configs) | T4 GPU | ~2–6 hours |
| Significance (B=10,000) | CPU | ~2 min |
| Ablations | T4 GPU | ~4–8 hours |

## Reproducibility

- All random choices flow from `seed` in config (default: 42)
- Same inputs + code → same dataset, shots, retrieval, outputs
- GPU numerical non-determinism is noted but minimized via `torch.use_deterministic_algorithms`
- Every result row carries its seed, config hash, and timestamp

## Resumability

All long runs are crash-safe:
- Results are appended one line at a time with `flush+fsync`
- On restart, completed instance IDs are skipped
- At most one instance is lost on crash

## Privacy (MIMIC-IV)

- MIMIC data never leaves the local machine
- Never committed to version control
- Result files contain only IDs, codes, and token counts (no raw note text)
- Output directory must not be inside a git-tracked path

