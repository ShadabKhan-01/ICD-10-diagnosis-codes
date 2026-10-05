# Simplified ICD-10 Clinical Coding Evaluation

A clean, modular, and beginner-friendly version of the ICD-10 LLM coding benchmark. It allows you to test any LLM (Gemini, OpenAI, Mock, local endpoints) across 3 key prompt strategies (Zero-Shot, Few-Shot, RAG) with a **single, simple command**.

---

## 📁 File Structure

Everything is neatly organized into single-purpose files:

| File | Purpose |
|---|---|
| [`prompts.py`](prompts.py) | Prompt templates for Zero-Shot, Few-Shot, and RAG + robust code parsing. |
| [`inputs.py`](inputs.py) | Data loader for clinical notes (`eval.jsonl`), demonstrations, and ICD-10 vocabulary. |
| [`models.py`](models.py) | Unified LLM caller supporting Google Gemini, OpenAI, Claude, and offline Mock. |
| [`config_zero_shot.py`](config_zero_shot.py) | **Configuration 1:** Direct Zero-Shot diagnosis coding. |
| [`config_few_shot.py`](config_few_shot.py) | **Configuration 2:** In-Context Few-Shot learning ($k=5$). |
| [`config_rag.py`](config_rag.py) | **Configuration 3:** Retrieval-Augmented Generation with top-$k$ candidate codes. |
| [`results.py`](results.py) | Metrics calculation (Micro-F1, Macro-F1, Invalid Code Rate) & Table II formatting. |
| [`run.py`](run.py) | **The Master Runner:** Single CLI entry point to run any model and strategy. |

---

## 🚀 How to Run

### 1. Set your API Key (Choose either method)

**Option A (Using `.env` file - Recommended):**
Create or edit `.env` in the project root:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Option B (In PowerShell directly):**
```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

### 2. Run Experiments

#### Run All 3 Strategies on Gemini (e.g. 5 patient records):
```powershell
python simplified_version/run.py --model gemini-flash-lite --limit 5
```

#### Run Only One Strategy (e.g. Zero-Shot or Few-Shot):
```powershell
python simplified_version/run.py --model gemini-flash-lite --strategy zero_shot --limit 5
python simplified_version/run.py --model gemini-flash-lite --strategy few_shot --limit 5
python simplified_version/run.py --model gemini-flash-lite --strategy rag --limit 5
```

#### Compare Different Models One by One:
```powershell
# 1. Test Gemini Flash Lite
python simplified_version/run.py --model gemini-flash-lite --limit 5

# 2. Test OpenAI GPT-4o-mini
python simplified_version/run.py --model gpt-4o-mini --limit 5

# 3. Test Offline Mock (Instant, zero cost)
python simplified_version/run.py --model mock --limit 10
```

#### Run on the Full Evaluation Set (all 200 notes):
```powershell
python simplified_version/run.py --model gemini-flash-lite
```

---

## 📊 Output

Running `run.py` automatically:
1. Prints a live progress bar for every strategy.
2. Prints the benchmark comparison table (**Table II**) directly in your terminal:

```
==========================================================================================
                      EVALUATION RESULTS (TABLE II)
==========================================================================================
Model                    | Strategy           | n    | Micro-F1 | Macro-F1 | P@5    | P@8    | Invalid Rate
------------------------------------------------------------------------------------------
gemini-flash-lite        | Zero-Shot          | 5    | 0.690    | 0.533    | n/a    | n/a    | 0.0%        
gemini-flash-lite        | Few-Shot (k=5)     | 5    | 0.727    | 0.727    | n/a    | n/a    | 0.0%        
gemini-flash-lite        | RAG (k=10)         | 5    | 0.556    | 0.455    | n/a    | n/a    | 0.0%        
==========================================================================================
```

3. Automatically saves all outputs to `simplified_version/results/`:
   - `table2_summary.csv`
   - `table2_summary.md`
   - `<model>__<strategy>.jsonl` (contains raw predictions, latency, and full traces for every patient).
