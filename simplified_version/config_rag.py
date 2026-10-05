"""Configuration 3: Retrieval-Augmented Generation (RAG) Strategy."""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, format_rag_prompt, extract_codes
from inputs import load_vocabulary
from models import call_llm

logger = logging.getLogger(__name__)

# Common clinical administrative stopwords to prevent non-diagnostic terms from dominating
ADMIN_STOPWORDS = {
    'and', 'the', 'for', 'with', 'without', 'not', 'other', 'specified', 'unspecified',
    'due', 'patient', 'note', 'history', 'encounter', 'observation',
    'evaluation', 'examination', 'general', 'suspected', 'reported', 'chief',
    'complaint', 'presents', 'present', 'illness', 'presenting', 'multiple',
    'chronic', 'conditions', 'recent', 'labs', 'showed', 'past', 'medical',
    'hospital', 'course', 'admitted', 'stabilized', 'monitored', 'closely',
    'active', 'issues', 'discharge', 'medications', 'bid', 'daily', 'oral',
    'tablet', 'mg', 'mcg', 'prn', 'classified', 'elsewhere', 'routine'
}


class FastLexicalRetriever:
    """High-speed, zero-dependency clinical keyword/IDF retriever over ICD-10 vocabulary."""

    def __init__(self, vocab: Dict[str, str]):
        self.vocab = vocab
        self.code_terms = {}
        self.term_index = defaultdict(list)
        
        N = len(vocab)
        df = defaultdict(int)
        
        for code, desc in vocab.items():
            terms = set(self._tokenize(desc))
            self.code_terms[code] = terms
            for t in terms:
                df[t] += 1
                self.term_index[t].append(code)

        self.idf = {t: math.log(1.0 + (N / count)) for t, count in df.items()}

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'[a-z0-9]{3,}', text.lower())
        return [w for w in words if w not in ADMIN_STOPWORDS]

    def retrieve(self, record: Dict[str, Any], top_k: int = 10) -> List[Dict[str, str]]:
        """Retrieve top-k candidate ICD-10 codes for a patient record."""
        text = record.get("text", "") + " " + " ".join(record.get("medications", []))
        tokens = self._tokenize(text)
        if not tokens:
            return []

        tf = Counter(tokens)
        scores = defaultdict(float)

        for t, count in tf.items():
            if t in self.term_index:
                weight = self.idf[t] * (1.0 + math.log(count))
                for c in self.term_index[t]:
                    scores[c] += weight

        for c in scores:
            desc_len = len(self.code_terms.get(c, []))
            scores[c] = scores[c] / (math.sqrt(desc_len) + 0.1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"code": c, "desc": self.vocab[c]} for c, _ in ranked]


# Global cached retriever instance
_RETRIEVER_INSTANCE: Optional[FastLexicalRetriever] = None


def get_retriever() -> FastLexicalRetriever:
    """Get or initialize cached retriever."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        vocab = load_vocabulary()
        _RETRIEVER_INSTANCE = FastLexicalRetriever(vocab)
    return _RETRIEVER_INSTANCE


def run_rag(
    records: List[Dict[str, Any]],
    model_name: str,
    top_k: int = 10,
    retriever: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Execute RAG configuration across given patient records."""
    if retriever is None:
        retriever = get_retriever()

    results = []
    print(f"\n[Running RAG] Model: {model_name} | Top-{top_k} Candidates | Instances: {len(records)}")

    for rec in tqdm(records, desc=f"RAG ({model_name})"):
        candidates = retriever.retrieve(rec, top_k=top_k)
        prompt = format_rag_prompt(rec, candidates)
        raw_output = call_llm(model_name, SYSTEM_PROMPT, prompt)
        parsed = extract_codes(raw_output)

        results.append({
            "id": rec["id"],
            "model": model_name,
            "strategy": f"RAG (top-{top_k})",
            "gold_codes": rec.get("gold_codes", []),
            "predicted_codes": parsed,
            "raw_output": raw_output,
            "retrieved_candidates": [c["code"] for c in candidates],
        })

    return results
