"""Retrieval system for RAG strategy.

Embeds all ICD-10-CM code descriptions, retrieves candidates per record
using windowed note text + individual medication queries.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single retrieved ICD-10-CM code candidate."""
    code: str
    desc: str
    score: float
    source_query: str


def build_index(
    embedder_name: str,
    vocab,  # Vocabulary instance
    index_dir: str,
    embedding_format: str = "{code}: {long_desc}",
) -> dict:
    """Embed all ICD-10-CM code descriptions and save the index.

    Format: "{code}: {long_desc}" for each code (configurable).
    L2-normalize all embeddings so cosine = dot product.
    Save: embeddings.npy, codes.json, meta.json.
    Skip rebuild if meta.json matches (same embedder + vocab hash).

    Returns:
        meta dict with embedder info and stats.
    """
    os.makedirs(index_dir, exist_ok=True)

    from utils import sha256_str

    # Compute vocab hash for cache invalidation
    all_codes = sorted(vocab.all_codes())
    vocab_hash = sha256_str(json.dumps(all_codes))

    meta_path = os.path.join(index_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            existing_meta = json.load(f)
        if (existing_meta.get("embedder_name") == embedder_name
                and existing_meta.get("vocab_hash") == vocab_hash):
            logger.info("Index already up-to-date, skipping rebuild.")
            return existing_meta

    logger.info(f"Building index for {len(all_codes)} codes with {embedder_name}...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedder_name)

    # Build texts to embed
    texts = []
    valid_codes = []
    for code in all_codes:
        desc = vocab.description(code) or ""
        text = embedding_format.replace("{code}", code).replace("{long_desc}", desc)
        texts.append(text)
        valid_codes.append(code)

    logger.info(f"Encoding {len(texts)} code descriptions...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize
    )
    embeddings = embeddings.astype(np.float32)

    # Save
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(index_dir, "codes.json"), "w") as f:
        json.dump(valid_codes, f)

    # Get embedder revision if possible
    embedder_revision = "unknown"
    try:
        from huggingface_hub import model_info
        info = model_info(embedder_name)
        embedder_revision = info.sha or "unknown"
    except Exception:
        pass

    meta = {
        "embedder_name": embedder_name,
        "embedder_revision": embedder_revision,
        "vocab_hash": vocab_hash,
        "dimension": int(embeddings.shape[1]),
        "num_codes": len(valid_codes),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Index saved to {index_dir} ({len(valid_codes)} codes, dim={embeddings.shape[1]})")
    del model  # free memory
    return meta


class Retriever:
    """Retrieve ICD-10-CM candidate codes for a clinical record.

    Queries per record:
    - Note text split into overlapping windows (~150 words, 30 overlap)
    - Each medication as a separate query

    Merge: max score per code across all queries.
    Return: global top-k candidates, deduplicated.
    """

    def __init__(
        self,
        embedder_name: str,
        index_dir: str,
        vocab=None,  # Vocabulary for descriptions
        k: int = 20,
        per_query_m: int = 30,
        note_window_words: int = 150,
        note_window_overlap: int = 30,
    ):
        self.k = k
        self.per_query_m = per_query_m
        self.note_window_words = note_window_words
        self.note_window_overlap = note_window_overlap
        self.vocab = vocab

        logger.info(f"Loading retriever from {index_dir}")

        # Load index
        with open(os.path.join(index_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)
        with open(os.path.join(index_dir, "codes.json"), "r") as f:
            self.codes = json.load(f)
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))

        # Build code→description lookup
        self._code_desc: Dict[str, str] = {}
        if vocab is not None:
            for code in self.codes:
                desc = vocab.description(code)
                if desc:
                    self._code_desc[code] = desc

        # Load embedder
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedder_name)

        logger.info(
            f"Retriever ready: {len(self.codes)} codes, k={k}, "
            f"m={per_query_m}, window={note_window_words}w/{note_window_overlap}o"
        )

    def _window_text(self, text: str) -> List[str]:
        """Split text into overlapping windows of ~window_words."""
        words = text.split()
        if not words:
            return []

        windows = []
        step = max(1, self.note_window_words - self.note_window_overlap)
        for i in range(0, len(words), step):
            window = words[i : i + self.note_window_words]
            windows.append(" ".join(window))
            if i + self.note_window_words >= len(words):
                break
        return windows

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries and L2-normalize."""
        if not queries:
            return np.empty((0, self.meta["dimension"]), dtype=np.float32)
        emb = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    def _search(self, query_embeddings: np.ndarray, m: int) -> List[List[Tuple[int, float]]]:
        """For each query, find top-m code indices by cosine similarity."""
        if len(query_embeddings) == 0:
            return []

        # Cosine sim = dot product for L2-normalized vectors
        sims = query_embeddings @ self.embeddings.T  # (num_queries, num_codes)

        results = []
        for i in range(len(query_embeddings)):
            # Partial sort for top-m
            if m < len(self.codes):
                top_idx = np.argpartition(sims[i], -m)[-m:]
                top_idx = top_idx[np.argsort(sims[i][top_idx])[::-1]]
            else:
                top_idx = np.argsort(sims[i])[::-1][:m]
            results.append([(int(idx), float(sims[i][idx])) for idx in top_idx])
        return results

    def retrieve(self, record: dict) -> List[Candidate]:
        """Retrieve top-k candidates for a record.

        Returns:
            List of Candidate objects sorted by score (descending).
        """
        text = record.get("text", "")
        meds = record.get("medications", [])

        # Build queries: windowed note + individual medications
        windows = self._window_text(text)
        queries = windows + meds
        query_sources = (
            [f"note_window_{i}" for i in range(len(windows))]
            + [f"med_{m}" for m in meds]
        )

        if not queries:
            logger.warning(f"No queries for record {record.get('id', '?')}")
            return []

        # Encode and search
        query_emb = self._encode_queries(queries)
        search_results = self._search(query_emb, self.per_query_m)

        # Merge by max score per code
        best: Dict[str, Tuple[float, str]] = {}  # code → (score, source)
        for source, hits in zip(query_sources, search_results):
            for idx, score in hits:
                code = self.codes[idx]
                if code not in best or score > best[code][0]:
                    best[code] = (score, source)

        # Sort by score and take top-k
        sorted_codes = sorted(best.items(), key=lambda x: x[1][0], reverse=True)
        top_k = sorted_codes[: self.k]

        candidates = []
        for code, (score, source) in top_k:
            desc = self._code_desc.get(code, "")
            candidates.append(Candidate(code=code, desc=desc, score=score, source_query=source))

        logger.debug(
            f"Record {record.get('id', '?')}: "
            f"{len(queries)} queries → {len(best)} unique codes → "
            f"{len(candidates)} candidates returned"
        )
        return candidates


def candidate_recall_at_k(
    records: List[dict],
    retriever: "Retriever",
    gold_key: str = "gold_codes",
) -> Tuple[float, float]:
    """Diagnostic: fraction of gold codes found in retrieved candidates.

    Labeled as 'not in paper' — explains RAG results without needing an LLM.

    Returns:
        (mean_recall, mean_candidates_per_record)
    """
    recalls = []
    n_candidates = []
    for record in records:
        gold = set(record.get(gold_key, []))
        if not gold:
            continue
        candidates = retriever.retrieve(record)
        candidate_codes = {c.code for c in candidates}
        recall = len(gold & candidate_codes) / len(gold)
        recalls.append(recall)
        n_candidates.append(len(candidates))

    if not recalls:
        return 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(n_candidates))


class MockRetriever:
    """Deterministic mock retriever for testing without sentence-transformers or GPU."""

    def __init__(self, vocab=None, k: int = 20):
        self.k = k
        self.vocab = vocab
        self.default_codes = [
            ("E11.9", "Type 2 diabetes mellitus without complications"),
            ("I10", "Essential (primary) hypertension"),
            ("E78.5", "Hyperlipidemia, unspecified"),
            ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
            ("I25.10", "Atherosclerotic heart disease of native coronary artery"),
            ("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
            ("E03.9", "Hypothyroidism, unspecified"),
            ("I48.91", "Unspecified atrial fibrillation"),
            ("M10.9", "Gout, unspecified"),
            ("N39.0", "Urinary tract infection, site not specified"),
            ("F41.9", "Anxiety disorder, unspecified"),
            ("G47.33", "Obstructive sleep apnea (adult) (pediatric)"),
            ("E66.9", "Obesity, unspecified"),
            ("D64.9", "Anemia, unspecified"),
            ("E55.9", "Vitamin D deficiency, unspecified"),
            ("I50.9", "Heart failure, unspecified"),
            ("J45.909", "Unspecified asthma, uncomplicated"),
            ("G40.909", "Epilepsy, unspecified, not intractable"),
            ("J18.9", "Pneumonia, unspecified organism"),
            ("I63.9", "Cerebral infarction, unspecified"),
        ]

    def retrieve(self, record: dict) -> List[Candidate]:
        candidates = []
        for i, (code, default_desc) in enumerate(self.default_codes[: self.k]):
            desc = self.vocab.description(code) if self.vocab else default_desc
            desc = desc or default_desc
            score = round(0.95 - (i * 0.03), 3)
            candidates.append(
                Candidate(code=code, desc=desc, score=score, source_query=f"mock_query_{i}")
            )
        return candidates


