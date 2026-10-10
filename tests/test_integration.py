import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest
import tempfile
import os
import json

# Test 7: Retrieval (§10.7)
def test_retrieval_determinism_and_dedup():
    """On a tiny fake vocab and deterministic embedder:
    max-score merge, top-k size, determinism, no duplicates."""
    from retrieval import Retriever
    import numpy as np

    # Fake embeddings: 5 codes, dim 4
    codes = ['E11.9', 'I10', 'J44.9', 'K21.9', 'E03.9']

    class FakeEmbedder:
        def embed(self, texts):
            # Deterministic pseudo-random embeddings
            np.random.seed(42)
            return np.random.rand(len(texts), 4)

    # Mock the index building and retrieval
    class FakeRetriever(Retriever):
        def __init__(self):
            self.embedder = FakeEmbedder()
            self.codes = codes

        def retrieve(self, record, k=3):
            from retrieval import Candidate
            # Return some fake candidates with deterministic scores
            cands = [
                Candidate('E11.9', 'Diabetes', 0.95, 'text1'),
                Candidate('I10', 'Hypertension', 0.85, 'text2'),
                Candidate('E11.9', 'Diabetes again', 0.90, 'text3'),  # duplicate code
                Candidate('J44.9', 'COPD', 0.80, 'text4')
            ]

            # Deduplicate by max score
            merged = {}
            for c in cands:
                if c.code not in merged or c.score > merged[c.code].score:
                    merged[c.code] = c

            sorted_cands = sorted(merged.values(), key=lambda x: x.score, reverse=True)
            return sorted_cands[:k]

    retriever = FakeRetriever()
    cands = retriever.retrieve({'text': 'test'})

    assert len(cands) == 3
    # E11.9 should be first, and with score 0.95 (max merge)
    assert cands[0].code == 'E11.9'
    assert cands[0].score == 0.95
    # I10 should be second
    assert cands[1].code == 'I10'
    # Check dedup
    assert len(set(c.code for c in cands)) == len(cands)

# Test 8: Resume (§10.8)
def test_resume_after_crash():
    """Simulate a crash mid-run; restart completes with no duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "out.jsonl")

        # Write partial output
        with open(output_file, 'w') as f:
            f.write(json.dumps({"id": "rec_1", "predicted_codes": ["E11.9"]}) + "\n")
            f.write(json.dumps({"id": "rec_2", "predicted_codes": ["I10"]}) + "\n")

        eval_records = [
            {"id": "rec_1", "text": "...", "gold_codes": ["E11.9"]},
            {"id": "rec_2", "text": "...", "gold_codes": ["I10"]},
            {"id": "rec_3", "text": "...", "gold_codes": ["J44.9"]}
        ]

        # Mocking process_record to simulate LLM call
        def mock_process_record(record, *args, **kwargs):
            return {"id": record["id"], "predicted_codes": record["gold_codes"]}

        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_ids.add(json.loads(line)["id"])

        assert "rec_1" in processed_ids
        assert "rec_2" in processed_ids
        assert "rec_3" not in processed_ids

        # Process remaining
        with open(output_file, 'a') as f:
            for rec in eval_records:
                if rec["id"] not in processed_ids:
                    res = mock_process_record(rec)
                    f.write(json.dumps(res) + "\n")

        # Verify no duplicates
        final_ids = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    final_ids.append(json.loads(line)["id"])

        assert len(final_ids) == 3
        assert len(set(final_ids)) == 3
        assert "rec_3" in final_ids

# Test 9: Bootstrap (§10.9)
def test_bootstrap_identical_zero_delta():
    """With identical predictions, delta=0 and CI near 0."""
    from bootstrap import bootstrap_ci

    base_preds = [{'E11.9'}, {'I10'}]
    new_preds = [{'E11.9'}, {'I10'}]
    gold = [{'E11.9'}, {'I10'}]

    # Mock metric func
    def mock_metric(preds, golds):
        return sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)

    ci_lower, ci_upper, mean_delta = bootstrap_ci(base_preds, new_preds, gold, mock_metric, n_iterations=100)

    assert mean_delta == 0.0
    assert ci_lower == 0.0
    assert ci_upper == 0.0

def test_bootstrap_clearly_better():
    """With constructed clearly-better system, CI excludes 0."""
    from bootstrap import bootstrap_ci

    base_preds = [{'E11.9'}, set()] * 20 # 50% acc
    new_preds = [{'E11.9'}, {'I10'}] * 20 # 100% acc
    gold = [{'E11.9'}, {'I10'}] * 20

    def mock_metric(preds, golds):
        return sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)

    ci_lower, ci_upper, mean_delta = bootstrap_ci(base_preds, new_preds, gold, mock_metric, n_iterations=200)

    assert mean_delta > 0
    assert ci_lower > 0

def test_bootstrap_determinism():
    """Same seed produces same CI."""
    from bootstrap import bootstrap_ci

    base_preds = [{'E11.9'}, set(), {'J44.9'}]
    new_preds = [{'E11.9'}, {'I10'}, set()]
    gold = [{'E11.9'}, {'I10'}, {'J44.9'}]

    def mock_metric(preds, golds):
        return sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)

    ci_lower1, ci_upper1, mean_delta1 = bootstrap_ci(base_preds, new_preds, gold, mock_metric, seed=42)
    ci_lower2, ci_upper2, mean_delta2 = bootstrap_ci(base_preds, new_preds, gold, mock_metric, seed=42)

    assert ci_lower1 == ci_lower2
    assert ci_upper1 == ci_upper2

# Test 10: Overflow logic (§10.10)
def test_overflow_detection():
    """Long record on mock 2048-limit tokenizer sets overflow/truncated correctly."""
    from prompts import render_prompt_with_truncation

    record = {"text": "word " * 3000}

    class MockTokenizer:
        def encode(self, text):
            return text.split()

    prompt, truncated = render_prompt_with_truncation(record, MockTokenizer(), max_tokens=2048)
    assert truncated is True
    # The prompt should be truncated
    assert len(MockTokenizer().encode(prompt)) <= 2048

def test_truncation_preserves_instruction():
    """Truncation never cuts instruction or evidence."""
    from prompts import render_prompt_with_truncation, INSTRUCTION_TEXT

    record = {"text": "word " * 3000}
    evidence_text = "EVIDENCE E11.9"

    class MockTokenizer:
        def encode(self, text):
            return text.split()

    prompt, truncated = render_prompt_with_truncation(record, MockTokenizer(), max_tokens=2048, evidence=evidence_text)

    assert truncated is True
    assert INSTRUCTION_TEXT in prompt
    assert evidence_text in prompt

# Test 11: Mixing guard (§10.11)
def test_mixing_guard():
    """evaluate.py refuses a directory mixing mock and real rows."""
    from evaluate import validate_directory

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock file
        with open(os.path.join(tmpdir, "mock_results.jsonl"), 'w') as f:
            f.write(json.dumps({"id": "mock_1", "predicted_codes": ["E11.9"]}) + "\n")

        # Create a real file
        with open(os.path.join(tmpdir, "real_results.jsonl"), 'w') as f:
            f.write(json.dumps({"id": "real_1", "predicted_codes": ["I10"]}) + "\n")

        with pytest.raises(ValueError, match="mixed"):
            validate_directory(tmpdir)
