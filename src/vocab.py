import os
import json
import logging
import hashlib
import urllib.request
import argparse
from pathlib import Path

from codes import normalize, is_well_formed

logger = logging.getLogger(__name__)

class Vocabulary:
    """ICD-10-CM Vocabulary lookup and validation."""
    
    def __init__(self, codes: list[dict]):
        self._codes = {}
        self._billable_codes = set()
        
        for item in codes:
            code = item["code"]
            self._codes[code] = item
            if item["is_billable"]:
                self._billable_codes.add(code)
                
    @classmethod
    def from_file(cls, path: str) -> 'Vocabulary':
        """Parse the fixed-width CDC ICD-10-CM order file."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Order file not found at {path}")
            
        logger.info(f"Parsing order file from {path}")
        
        parsed_codes = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                
                # Check column layout on first line
                if i == 0:
                    if len(line) < 78:
                        raise ValueError(f"Order file line 1 is too short: {len(line)} chars")
                    if line[5] != ' ' or line[13] != ' ' or line[15] != ' ' or line[76] != ' ':
                        raise ValueError("Order file format check failed: unexpected non-blank columns")
                        
                raw_code = line[6:13].strip()
                is_billable_flag = line[14]
                short_desc = line[16:76].strip()
                long_desc = line[77:].strip()
                
                if is_billable_flag not in ('0', '1'):
                    logger.warning(f"Line {i+1}: unexpected billable flag '{is_billable_flag}', defaulting to 0")
                    is_billable = False
                else:
                    is_billable = (is_billable_flag == '1')
                    
                norm_code = normalize(raw_code)
                parsed_codes.append({
                    "code": norm_code,
                    "is_billable": is_billable,
                    "short_desc": short_desc,
                    "long_desc": long_desc
                })
                
        logger.info(f"Successfully parsed {len(parsed_codes)} codes.")
        return cls(parsed_codes)

    @classmethod
    def load_or_download(cls, config: dict) -> 'Vocabulary':
        """Load from processed JSONL if exists, else download and parse."""
        processed_path = config.get("processed_jsonl", "data/vocab/icd10cm_2024.jsonl")
        meta_path = config.get("meta_json", "data/vocab/meta.json")
        raw_path = config.get("raw_txt", "data/raw/icd10cm_order_2024.txt")
        url = config.get("url", "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024/icd10cm_order_2024.txt")
        
        processed_file = Path(processed_path)
        if processed_file.exists():
            logger.info(f"Loading processed vocab from {processed_path}")
            codes = []
            with open(processed_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        codes.append(json.loads(line))
            return cls(codes)
            
        # Needs processing. Check if raw file exists, else download.
        raw_file = Path(raw_path)
        if not raw_file.exists():
            logger.info(f"Raw file {raw_path} not found. Attempting to download from {url}")
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, raw_file)
                logger.info(f"Successfully downloaded to {raw_path}")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise RuntimeError(f"Could not download vocabulary. Please place the CDC order file manually at {raw_path}") from e
                
        # Calculate SHA256 of raw file
        sha256_hash = hashlib.sha256()
        with open(raw_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
        
        # Parse
        vocab = cls.from_file(raw_path)
        
        # Save processed
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        vocab.save_jsonl(processed_path)
        vocab.save_meta(meta_path, file_hash)
        
        return vocab

    def is_valid(self, code: str) -> bool:
        """True if code exists (header or billable) after normalize()"""
        norm_code = normalize(code)
        return norm_code in self._codes

    def is_billable(self, code: str) -> bool:
        """True if code is valid AND billable"""
        norm_code = normalize(code)
        return norm_code in self._billable_codes

    def description(self, code: str) -> str | None:
        """Return long_desc or None"""
        norm_code = normalize(code)
        item = self._codes.get(norm_code)
        if item:
            return item["long_desc"]
        return None

    def all_codes(self) -> list[str]:
        """All codes in dotted canonical form"""
        return list(self._codes.keys())

    def billable_codes(self) -> list[str]:
        """Only billable codes"""
        return list(self._billable_codes)

    def save_jsonl(self, path: str):
        """Save processed vocab as JSONL"""
        logger.info(f"Saving vocabulary JSONL to {path}")
        with open(path, "w", encoding="utf-8") as f:
            for item in self._codes.values():
                f.write(json.dumps(item) + "\n")

    def save_meta(self, path: str, source_file_sha256: str):
        """Save meta.json with vocab stats"""
        logger.info(f"Saving vocabulary metadata to {path}")
        meta = {
            "name": "icd10cm_2024",
            "source_file_sha256": source_file_sha256,
            "code_count": len(self._codes),
            "billable_count": len(self._billable_codes),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Download and process ICD-10-CM vocabulary.")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file.")
    args = parser.parse_args()
    
    # Simple config parsing mock for CLI use
    import yaml # Assuming pyyaml is available or standard
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        config = {}
        
    vocab_config = config.get("vocab", {})
    Vocabulary.load_or_download(vocab_config)
