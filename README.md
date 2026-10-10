# ICD-10-CM coding with open-weight LLMs: four prompting strategies

This project compares four ways of giving an LLM evidence for ICD-10-CM coding, under the
same prompt, parser, data and evaluation:

| Strategy | Evidence the model sees | Where it comes from |
|---|---|---|
| `zero_shot` | nothing extra | prompt only |
| `few_shot` | 5 worked examples | fixed, seeded pool, disjoint from the evaluation records |
| `rag` | top-20 candidate codes | MiniLM search over official code descriptions |
| `kg` | top-20 candidate codes | **hybrid**: the same MiniLM seeds, expanded with the official parent and child codes |

`kg` is a hybrid, not a full knowledge graph. Semantic search picks up to 10 seed codes.
Each seed adds its direct parent and up to 5 direct children, taken from the official
ICD-10-CM 2024 tabular hierarchy. The pool is capped at 20, the same as RAG. No relation is
invented.

The strategies differ only in the evidence block. The system text, instruction, record,
answer format, decoding settings, seed, dataset, parser and metrics are the same
(`tests/test_strategies.py` checks this).

---

## Part 1: Google Colab setup (Tesla T4)

You'll run everything in a Colab notebook. Each step below is one cell. Put `!` in front of
shell commands, or use `%%bash` at the top of a cell.

### 1.1 Choose the GPU
Runtime → Change runtime type → **T4 GPU** → Save.

Check it:
```bash
!nvidia-smi
```

### 1.2 Mount Google Drive (keeps the model cache and results between sessions)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.3 Upload the project ZIP
Either upload `icd10_four_strategies.zip` to `MyDrive/icd10/` in Drive (the easiest), or use
the Files panel on the left of Colab.

Unzip it into a fresh folder:
```bash
!rm -rf /content/icd10 && unzip -q "/content/drive/MyDrive/icd10/icd10_four_strategies.zip" -d /content/icd10
```

### 1.4 Install the pinned packages
Do **not** install `torch`: Colab already provides the CUDA build.
```bash
!pip install -q "transformers==4.56.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "sentence-transformers==5.1.0" jsonlines tabulate pytest
```

### 1.5 Keep the model cache on Drive
```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_home"
```
Run this before any model download.

### 1.6 Hugging Face access (only for LLaMA-3-8B)
LLaMA-3-8B is gated. Accept the Meta licence on its Hugging Face page, then create a token
at https://huggingface.co/settings/tokens and set it in Colab's **Secrets** panel (key icon)
under the name `HF_TOKEN`. Then run:
```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```
BioMistral and LLaMA-3.2-3B do not need a token.

---

## Part 2: Build the official ICD-10-CM hierarchy (required for `kg`)

The hierarchy comes from the **official CDC tabular file**. Nothing in this project is a
substitute for it.

### 2.1 Get the official CDC ZIP
The file is `icd10cm-Table and Index-2024.zip` (about 22.7 MB), from the CDC FTP folder:
`https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024/`

**Option A (recommended, downloads inside Colab):**
```bash
!mkdir -p /content/icd10/data/raw && curl -L --fail --retry 3 -o "/content/icd10/data/raw/icd10cm-Table-and-Index-2024.zip" "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024/icd10cm-Table%20and%20Index-2024.zip"
```

**Option B (from your computer):** download the file in your browser from the folder above,
then upload it into `/content/icd10/data/raw/` with the Colab Files panel, and rename it
`icd10cm-Table-and-Index-2024.zip`.

Check the file is a complete ZIP:
```bash
!python -c "import zipfile; z=zipfile.ZipFile('/content/icd10/data/raw/icd10cm-Table-and-Index-2024.zip'); print([i.filename for i in z.infolist()])"
```
If this prints a list of file names, the ZIP is complete. If it prints `BadZipFile`, the
download was incomplete: run Option A again.

### 2.2 Extract the XML and build the hierarchy
The vocabulary must already exist (`data/vocab/icd10cm_2024.jsonl`, included in the ZIP).
```bash
%cd /content/icd10
!python src/kg.py build --zip data/raw/icd10cm-Table-and-Index-2024.zip --xml data/raw/icd10cm_tabular_2024.xml --vocab data/vocab/icd10cm_2024.jsonl --out data/kg/icd10cm_2024_hierarchy.jsonl
```

What the build does:
1. Extracts the one tabular XML file from the ZIP (its name must contain "tabular").
2. Reads the parent and child relations from the XML.
3. Checks the structure: no missing parents, no cycles, every code well-formed, and the
   descriptions are present.
4. Cross-checks the code set against the verified vocabulary. **Any mismatch stops the
   build**, so a wrong file cannot be used by mistake.
5. Writes `data/kg/icd10cm_2024_hierarchy.jsonl`.

A successful build logs a `Validated hierarchy:` line with the counts. A failure stops with
a message naming the problem, and no output file is written.

### 2.3 Verify the hierarchy
```bash
!python -B -m pytest -q -rs tests/test_strategies.py -k real_official
```
This test runs only on the official file. It should pass, not skip. A skip means the XML was
not extracted to `data/raw/`.

Then spot-check a few codes:
```bash
!python -B -c "
import sys; sys.path.insert(0, 'src')
from kg import Hierarchy
h = Hierarchy.from_jsonl('data/kg/icd10cm_2024_hierarchy.jsonl')
for c in ['E11', 'E11.9', 'J18.9', 'I10']:
    print(c, '| parent:', h.parent_of(c), '| children:', h.children_of(c)[:3], '|', h.desc.get(c, '')[:50])
"
```
`E11.9` should have parent `E11`. `E11` should be a top-level code (parent `None`).

---

## Part 3: The 25-example pilot (all four strategies)

Run this first. It takes a few minutes per strategy on a T4.

Choose one model. Pick one of these keys:
- `llama3` = LLaMA-3-8B-Instruct (needs `HF_TOKEN`)
- `biomistral` = BioMistral-7B
- `llama3.2-3b` = LLaMA-3.2-3B (public, smaller)

```bash
%cd /content/icd10
!python src/run_experiment.py --config configs/phase1.yaml --models llama3 --strategies zero_shot few_shot rag kg --split eval --data data/synthetic/run01/eval.jsonl --limit 25 --out /content/drive/MyDrive/icd10/results/pilot_llama3
```

Output goes to `/content/drive/MyDrive/icd10/results/pilot_llama3_smoke/` (the `_smoke`
suffix is added by `--limit`).

If Colab disconnects, run the same command again. Finished examples are skipped.

**Check each strategy finished:**
```bash
!ls -la /content/drive/MyDrive/icd10/results/pilot_llama3_smoke/
```
You should see four `.jsonl` files: `llama3__zero_shot`, `llama3__few_shot`, `llama3__rag`
and `llama3__kg`, each with 25 lines.

---

## Part 4: The comparison table

```bash
%cd /content/icd10
!python src/evaluate.py --results-dir /content/drive/MyDrive/icd10/results/pilot_llama3_smoke --data data/synthetic/run01/eval.jsonl --vocab data/vocab/icd10cm_2024.jsonl --allow-partial --dump-invalid 5
```

`--allow-partial` is needed because 25 of the 200 eval examples were run. The table shows
Precision, Recall, Micro-F1, Macro-F1, P@5, P@8 and the invalid-code rate, and a second table
shows the unsupported-code rate, parse failures, truncation, and candidate recall for RAG and
KG. The files are saved to `reports/` in the project folder.

For the full evaluation set, run Part 3 without `--limit`, and run the same evaluate command
without `--allow-partial`:
```bash
%cd /content/icd10
!python src/run_experiment.py --config configs/phase1.yaml --models llama3 biomistral --strategies zero_shot few_shot rag kg --split eval --data data/synthetic/run01/eval.jsonl --out /content/drive/MyDrive/icd10/results/exp01
!python src/evaluate.py --results-dir /content/drive/MyDrive/icd10/results/exp01 --data data/synthetic/run01/eval.jsonl --vocab data/vocab/icd10cm_2024.jsonl --dump-invalid 5
```

---

## Tests

```bash
%cd /content/icd10
!python -m pytest -q -rs tests/
```

`-rs` prints the reason for every skipped test. The skips are expected: the CUDA float16
test needs the GPU (it runs on Colab), and the official-hierarchy test runs only after Part 2.

---

## Leakage rules

- `retrieval.retrieval_view()` is the only view of a record that any retriever receives:
  `id`, `text`, `medications`. Gold fields are removed there, and RAG and KG both use it.
- KG expansion reads only the hierarchy file. Its seeds come from the same view.
- The few-shot pool is checked for disjointness from the evaluation records by ID and text hash.

## Models and settings

| Key | Model | Notes |
|---|---|---|
| `llama3` | `meta-llama/Meta-Llama-3-8B-Instruct` | gated; needs `HF_TOKEN` |
| `biomistral` | `BioMistral/BioMistral-7B` | prepended system text (no system role) |
| `llama3.2-3b` | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | public, smaller |

All HF models: 4-bit NF4, float16 compute on CUDA (checked and written to every row),
`max_new_tokens: 256`, greedy decoding, batch size 1, context limit as in `configs/phase1.yaml`.

## Known limits

- KG expansion takes at most 5 children per seed, in XML order, with no relevance ranking.
  This is a fixed, documented choice.
- The duplicate-BOS fix is applied to the LLaMA-3 models (see `CHANGELOG.md`). BioMistral's
  chat template was not available to check, so it keeps the default setting.
- The model runs have not been executed in this environment. Results come from the Colab runs.

## RAGAS ID-Based Retrieval Evaluation

Install the optional evaluation dependencies:

```bash
pip install -r requirements-ragas.txt
```

Evaluate saved RAG outputs without rerunning model inference:

```bash
python src/evaluate_ragas.py \
  --results-dir results/pilot_llama3_smoke \
  --data data/synthetic/run01/eval.jsonl \
  --out-dir results/pilot_llama3_ragas
```

This evaluation uses RAGAS ID-based Context Precision and Context
Recall to compare retrieved ICD-10-CM code IDs against gold codes.
These metrics measure exact code overlap, not semantic relevance
or LLM-judged faithfulness.
