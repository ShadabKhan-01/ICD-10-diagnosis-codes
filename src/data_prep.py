import argparse
import logging
import os
import re
import pandas as pd
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def normalize_icd(code: str) -> str:
    """Convert undotted MIMIC codes to canonical dotted form."""
    code = str(code).strip()
    if not code:
        return code
    if '.' in code:
        return code
    if code.startswith('E'):
        if len(code) > 4:
            return f"{code[:4]}.{code[4:]}"
        return code
    else:
        if len(code) > 3:
            return f"{code[:3]}.{code[3:]}"
        return code

def extract_clinical_sections(text: str) -> str:
    """
    Extracts relevant clinical sections from the discharge summary text using case-insensitive regex.
    Common sections: History of Present Illness, Past Medical History, Brief Hospital Course.
    """
    if not isinstance(text, str):
        return ""

    # Case-insensitive headers
    headers_to_extract = [
        r"history of present illness:?",
        r"past medical history:?",
        r"brief hospital course:?",
        r"discharge diagnoses:?",
        r"discharge condition:?"
    ]

    extracted_text = []

    # Split text by newlines and iterate
    lines = text.split('\n')
    capturing = False
    current_section = []

    # A simple state machine to capture sections based on headers
    for line in lines:
        stripped = line.strip()
        is_header = any(re.match(h, stripped, re.IGNORECASE) for h in headers_to_extract)

        # Stop capturing if we hit a new all-caps/colon header that isn't in our list
        # We'll use a simplistic regex for generic section headers: ALL CAPS followed by colon
        is_generic_header = re.match(r"^[A-Z\s]+:$", stripped)

        if is_header:
            if current_section:
                extracted_text.append('\n'.join(current_section))
                current_section = []
            capturing = True
            current_section.append(stripped)
        elif capturing and is_generic_header and not is_header:
            capturing = False
            extracted_text.append('\n'.join(current_section))
            current_section = []
        elif capturing:
            current_section.append(line)

    if current_section:
        extracted_text.append('\n'.join(current_section))

    return "\n\n".join(extracted_text)

def is_git_tracked(path: Path) -> bool:
    """Check if the given path or any of its parents is a git repository."""
    current = path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return True
        current = current.parent
    return False

def process_mimic_data(config_path: str) -> None:
    """
    Process MIMIC-IV data based on configuration.
    Reads MIMIC data, filters for ICD-10, normalizes codes, samples randomly,
    extracts clinical sections, and enforces privacy guards.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mimic_cfg = config.get('mimic', {})
    discharge_path = mimic_cfg.get('discharge_path', '')
    diagnoses_path = mimic_cfg.get('diagnoses_path', '')
    n_sample = mimic_cfg.get('n_sample', 200)
    seed = mimic_cfg.get('seed', 42)
    output_dir = mimic_cfg.get('output_dir', 'data/mimic')
    vocab_path = mimic_cfg.get('vocab_path', 'data/vocab.txt')

    if not discharge_path or not diagnoses_path or not os.path.exists(discharge_path) or not os.path.exists(diagnoses_path):
        logging.warning("MIMIC-IV paths not available. Skipping MIMIC data prep cleanly. Never fabricating data.")
        return

    out_path = Path(output_dir)

    if is_git_tracked(out_path):
        logging.error("Output directory appears git-tracked. Refusing to write raw/processed MIMIC data for privacy reasons.")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading data from {discharge_path} and {diagnoses_path}")

    try:
        if discharge_path.endswith('.parquet'):
            df_notes = pd.read_parquet(discharge_path)
        else:
            df_notes = pd.read_csv(discharge_path)

        if diagnoses_path.endswith('.parquet'):
            df_diag = pd.read_parquet(diagnoses_path)
        else:
            df_diag = pd.read_csv(diagnoses_path)
    except Exception as e:
        logging.error(f"Failed to read data files: {e}")
        return

    # Filter icd_version == 10
    if 'icd_version' in df_diag.columns:
        initial_diag_count = len(df_diag)
        df_diag = df_diag[df_diag['icd_version'] == 10]
        logging.info(f"Filtered ICD-10 only: {initial_diag_count} -> {len(df_diag)} diagnoses records")

    df_diag['icd_code'] = df_diag['icd_code'].apply(normalize_icd)

    # Load pinned vocabulary if available
    pinned_vocab = set()
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            pinned_vocab = set(line.strip() for line in f if line.strip())

        gold_codes = set(df_diag['icd_code'].unique())
        not_in_vocab = gold_codes - pinned_vocab
        if gold_codes:
            frac_missing = len(not_in_vocab) / len(gold_codes)
            logging.info(f"Fraction of gold codes not in pinned vocabulary: {frac_missing:.2%} ({len(not_in_vocab)} / {len(gold_codes)})")

    # Random sample
    unique_hadm = df_diag['hadm_id'].unique()
    logging.info(f"Unique hospital admissions with ICD-10: {len(unique_hadm)}")

    sampled_hadm = pd.Series(unique_hadm).sample(n=min(n_sample, len(unique_hadm)), random_state=seed)

    df_diag_sampled = df_diag[df_diag['hadm_id'].isin(sampled_hadm)]
    df_notes_sampled = df_notes[df_notes['hadm_id'].isin(sampled_hadm)].copy()

    logging.info(f"Sampled {len(sampled_hadm)} admissions, resulting in {len(df_diag_sampled)} diagnosis records and {len(df_notes_sampled)} notes.")

    # Section extraction logic
    if 'text' in df_notes_sampled.columns:
        df_notes_sampled['extracted_text'] = df_notes_sampled['text'].apply(extract_clinical_sections)
        # Drop raw text to ensure no raw text in results
        df_notes_sampled.drop(columns=['text'], inplace=True)
        logging.info("Extracted clinical sections and dropped raw text for privacy.")

        # Warn about discharge diagnosis
        if any(df_notes_sampled['extracted_text'].str.contains('discharge diagnosis', flags=re.IGNORECASE, na=False)):
            logging.warning("Discharge Diagnosis text sits close to the label and may cause data leakage.")

    # Save processed data
    output_notes_path = out_path / "processed_notes.csv"
    output_diag_path = out_path / "processed_diagnoses.csv"

    df_notes_sampled.to_csv(output_notes_path, index=False)
    df_diag_sampled.to_csv(output_diag_path, index=False)

    logging.info(f"Data preparation completed. Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIMIC-IV Data Preparation")
    parser.add_argument("--config", type=str, required=True, help="Path to MIMIC config")
    args = parser.parse_args()
    process_mimic_data(args.config)
