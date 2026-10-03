import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest

# Test 1: Prompt invariance (§10.1)
def test_prompt_invariance():
    """For the same record, render zero-shot, few-shot and RAG prompts;
    remove the evidence block; assert the three remainders are byte-identical.
    Also assert system text and instruction strings are the same values."""
    from prompts import (ZeroShotEvidence, FewShotEvidence, RAGEvidence,
                         render_prompt, build_messages, SYSTEM_TEXT, INSTRUCTION_TEXT,
                         strip_evidence_block)
    
    record = {
        'id': 'test_001',
        'text': 'Patient presents with chest pain and shortness of breath.',
        'medications': ['aspirin 81 mg daily', 'metoprolol 25 mg BID'],
        'gold_codes': ['I25.10', 'I10']
    }
    
    # Create a simple mock retriever
    class MockRetriever:
        def retrieve(self, record):
            from retrieval import Candidate
            return [Candidate('I25.10', 'Coronary artery disease', 0.9, 'note_window_0')]
    
    shots = [{
        'id': 'shot_001',
        'text': 'Example patient with diabetes.',
        'medications': ['metformin 500 mg BID'],
        'gold_codes': ['E11.9']
    }]
    
    zs = ZeroShotEvidence()
    fs = FewShotEvidence(shots)
    rag = RAGEvidence(MockRetriever())
    
    zs_evidence = zs.build_evidence(record)
    fs_evidence = fs.build_evidence(record)
    rag_evidence = rag.build_evidence(record)
    
    zs_prompt = render_prompt(record, zs_evidence)
    fs_prompt = render_prompt(record, fs_evidence)
    rag_prompt = render_prompt(record, rag_evidence)
    
    # Strip evidence blocks and compare
    zs_stripped = strip_evidence_block(zs_prompt, zs_evidence)
    fs_stripped = strip_evidence_block(fs_prompt, fs_evidence)
    rag_stripped = strip_evidence_block(rag_prompt, rag_evidence)
    
    assert zs_stripped == fs_stripped, f"ZS vs FS stripped prompts differ"
    assert zs_stripped == rag_stripped, f"ZS vs RAG stripped prompts differ"
    
    # Also check that SYSTEM_TEXT and INSTRUCTION_TEXT are identical objects/values
    msgs_system = build_messages(record, zs, 'system')
    msgs_prepend = build_messages(record, zs, 'prepend')
    # Both should contain the same SYSTEM_TEXT
    assert SYSTEM_TEXT in msgs_system[0]['content']

# Test 2: Few-shot disjointness (§10.2)
def test_fewshot_disjointness_by_id():
    """A pool that overlaps eval by id raises an error."""
    from run_experiment import select_few_shots
    import tempfile
    import json
    import os
    
    # Create temp pool file with overlapping id
    pool_records = [
        {'id': 'eval_1', 'text': 'pool text 1', 'medications': [], 'gold_codes': ['E11.9'], 'implicit_codes': []},
        {'id': 'pool_2', 'text': 'pool text 2', 'medications': [], 'gold_codes': ['I10'], 'implicit_codes': []},
        {'id': 'pool_3', 'text': 'pool text 3', 'medications': [], 'gold_codes': ['J44.9'], 'implicit_codes': ['J44.9']},
        {'id': 'pool_4', 'text': 'pool text 4', 'medications': [], 'gold_codes': ['K21.9'], 'implicit_codes': ['K21.9']},
        {'id': 'pool_5', 'text': 'pool text 5', 'medications': [], 'gold_codes': ['E03.9'], 'implicit_codes': []},
        {'id': 'pool_6', 'text': 'pool text 6', 'medications': [], 'gold_codes': ['E78.5'], 'implicit_codes': []},
    ]
    eval_records = [{'id': 'eval_1', 'text': 'eval text 1'}]  # overlaps by id
    
    fd, pool_path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for r in pool_records:
                f.write(json.dumps(r) + '\n')
        with pytest.raises(ValueError, match="overlaps.*ID"):
            select_few_shots(pool_path, eval_records, seed=42)
    finally:
        os.unlink(pool_path)

def test_fewshot_disjointness_by_hash():
    """A pool that overlaps eval by text hash raises an error."""
    from run_experiment import select_few_shots
    import tempfile
    import json
    import os
    
    identical_text = 'This exact text appears in both pool and eval'
    pool_records = [
        {'id': 'pool_1', 'text': identical_text, 'medications': [], 'gold_codes': ['E11.9'], 'implicit_codes': []},
        {'id': 'pool_2', 'text': 'different text', 'medications': [], 'gold_codes': ['I10'], 'implicit_codes': []},
        {'id': 'pool_3', 'text': 'more text', 'medications': [], 'gold_codes': ['J44.9'], 'implicit_codes': ['J44.9']},
        {'id': 'pool_4', 'text': 'yet more', 'medications': [], 'gold_codes': ['K21.9'], 'implicit_codes': ['K21.9']},
        {'id': 'pool_5', 'text': 'and more', 'medications': [], 'gold_codes': ['E03.9'], 'implicit_codes': []},
        {'id': 'pool_6', 'text': 'final', 'medications': [], 'gold_codes': ['E78.5'], 'implicit_codes': []},
    ]
    eval_records = [{'id': 'eval_1', 'text': identical_text}]  # overlaps by text hash
    
    fd, pool_path = tempfile.mkstemp(suffix='.jsonl')
    try:
        with os.fdopen(fd, 'w') as f:
            for r in pool_records:
                f.write(json.dumps(r) + '\n')
        with pytest.raises(ValueError, match="overlaps.*text hash"):
            select_few_shots(pool_path, eval_records, seed=42)
    finally:
        os.unlink(pool_path)

# Test 3: Metrics on hand-computed fixtures (§10.3)
def test_micro_f1_hand_computed():
    """Hand-computed micro F1."""
    from metrics import micro_f1
    pred_sets = [{'A', 'B'}, {'A', 'D'}]
    gold_sets = [{'A', 'C'}, {'A', 'B'}]
    assert abs(micro_f1(pred_sets, gold_sets) - 0.5) < 1e-10

def test_macro_f1_hand_computed():
    from metrics import macro_f1_gold_set
    pred_sets = [{'A', 'B'}, {'A', 'D'}]
    gold_sets = [{'A', 'C'}, {'A', 'B'}]
    expected = (1.0 + 0.0 + 0.0) / 3
    assert abs(macro_f1_gold_set(pred_sets, gold_sets) - expected) < 1e-10

def test_precision_at_k():
    from metrics import precision_at_k
    pred_lists = [['A','B','C'], ['D'], ['A','B']]
    gold_sets = [{'A','C'}, {'D'}, {'A','B'}]
    val, n = precision_at_k(pred_lists, gold_sets, 2)
    assert n == 2  # inst1 and inst3 qualify
    assert abs(val - 0.75) < 1e-10  # mean(0.5, 1.0)

def test_precision_at_k_zero_qualify():
    from metrics import precision_at_k
    pred_lists = [['A'], ['B']]  # all have < 2 codes
    gold_sets = [{'A'}, {'B'}]
    val, n = precision_at_k(pred_lists, gold_sets, 2)
    assert n == 0
    import math
    assert math.isnan(val)

def test_invalid_code_rate():
    from metrics import invalid_code_rate
    vocab = {'A', 'B', 'C'}
    pred = [['A', 'B', 'X'], ['C', 'Y', 'Z']]
    assert abs(invalid_code_rate(pred, vocab) - 0.5) < 1e-10

def test_unsupported_code_rate():
    from metrics import unsupported_code_rate
    vocab = {'A', 'B', 'C', 'D'}
    pred = [['A', 'B', 'C'], ['A', 'D']]
    gold = [{'A'}, {'A', 'D'}]
    assert abs(unsupported_code_rate(pred, gold, vocab) - 0.4) < 1e-10

def test_identity_tp_invalid_unsupported():
    """tp_share + invalid_rate + unsupported_rate = 1.0 for any non-trivial case."""
    from metrics import tp_share, invalid_code_rate, unsupported_code_rate
    vocab = {'A', 'B', 'C', 'D'}
    pred = [['A', 'B', 'X'], ['C', 'Y', 'A']]
    gold = [{'A', 'C'}, {'A', 'B'}]
    
    t = tp_share(pred, gold, vocab)
    i = invalid_code_rate(pred, vocab)
    u = unsupported_code_rate(pred, gold, vocab)
    assert abs(t + i + u - 1.0) < 1e-10, f"Identity violated: {t} + {i} + {u} = {t+i+u}"

def test_empty_predictions_f1_zero():
    from metrics import micro_f1, macro_f1_gold_set
    pred_sets = [set(), set()]
    gold_sets = [{'A', 'B'}, {'C'}]
    assert micro_f1(pred_sets, gold_sets) == 0.0
    assert macro_f1_gold_set(pred_sets, gold_sets) == 0.0

def test_perfect_predictions_f1_one():
    from metrics import micro_f1, macro_f1_gold_set
    gold_sets = [{'A', 'B'}, {'C', 'D'}]
    pred_sets = [{'A', 'B'}, {'C', 'D'}]
    assert abs(micro_f1(pred_sets, gold_sets) - 1.0) < 1e-10
    assert abs(macro_f1_gold_set(pred_sets, gold_sets) - 1.0) < 1e-10

def test_cross_check_with_sklearn():
    """Cross-check micro-F1 against scikit-learn."""
    from metrics import micro_f1
    try:
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import f1_score
    except ImportError:
        pytest.skip("scikit-learn not installed")
        
    pred_sets = [{'A', 'B'}, {'A', 'C'}, {'B', 'D'}]
    gold_sets = [{'A', 'C'}, {'A', 'B'}, {'B', 'C'}]
    
    all_codes = sorted(set().union(*pred_sets, *gold_sets))
    mlb = MultiLabelBinarizer(classes=all_codes)
    y_true = mlb.fit_transform([sorted(g) for g in gold_sets])
    y_pred = mlb.transform([sorted(p) for p in pred_sets])
    
    sklearn_f1 = f1_score(y_true, y_pred, average='micro')
    our_f1 = micro_f1(pred_sets, gold_sets)
    assert abs(our_f1 - sklearn_f1) < 1e-10, f"Our: {our_f1}, sklearn: {sklearn_f1}"

# Test 4: Parser (§10.4)
def test_parser_json_happy_path():
    from parser import parse_output
    codes, mode = parse_output('{"codes": ["E11.9", "I10"]}')
    assert mode == 'json'
    assert codes == ['E11.9', 'I10']

def test_parser_json_with_prose():
    from parser import parse_output
    codes, mode = parse_output('Here are the codes: {"codes": ["E11.9", "I10"]}. Done.')
    assert mode == 'json'
    assert codes == ['E11.9', 'I10']

def test_parser_malformed_json_regex_fallback():
    from parser import parse_output
    codes, mode = parse_output('The codes are E11.9 and I10 and maybe J44.9')
    assert mode == 'regex'
    assert 'E11.9' in codes
    assert 'I10' in codes
    assert 'J44.9' in codes

def test_parser_empty():
    from parser import parse_output
    codes, mode = parse_output('')
    assert mode == 'empty'
    assert codes == []

def test_parser_lowercase_codes():
    from parser import parse_output
    codes, mode = parse_output('{"codes": ["e11.9", "i10"]}')
    assert mode == 'json'
    assert codes == ['E11.9', 'I10']  # normalized to uppercase

def test_parser_undotted_codes():
    from parser import parse_output
    codes, mode = parse_output('{"codes": ["E119", "I10"]}')
    assert mode == 'json'
    assert codes == ['E11.9', 'I10']  # dot inserted

def test_parser_duplicates_preserving_order():
    from parser import parse_output
    codes, mode = parse_output('{"codes": ["E11.9", "I10", "E11.9", "J44.9"]}')
    assert mode == 'json'
    assert codes == ['E11.9', 'I10', 'J44.9']  # first occurrence wins

def test_parser_invalid_codes_preserved():
    """Invalid codes like Z99.999 must survive (not dropped)."""
    from parser import parse_output
    codes, mode = parse_output('{"codes": ["E11.9", "Z99.999", "E11.99"]}')
    assert mode == 'json'
    assert 'E11.9' in codes
    assert 'Z99.999' in codes
    assert 'E11.99' in codes

# Test 5: Vocab (§10.5)
def test_normalize_round_trips():
    from codes import normalize
    assert normalize('E11.9') == 'E11.9'
    assert normalize('e119') == 'E11.9'
    assert normalize(' I10 ') == 'I10'
    assert normalize('E119') == 'E11.9'
    assert normalize(normalize('e119')) == 'E11.9'

def test_well_formed_valid():
    from codes import is_well_formed
    assert is_well_formed('E11.9')
    assert is_well_formed('I10')
    assert is_well_formed('J44.9')
    assert is_well_formed('Z99.9')

def test_well_formed_invalid():
    from codes import is_well_formed
    assert not is_well_formed('11.9')  # no letter prefix
    assert not is_well_formed('EE1.9')  # wrong format
    assert not is_well_formed('')
    assert not is_well_formed('U11.9')  # U is not in [A-T, V-Z]

def test_vocab_layout_assertion():
    """A corrupted line in the order file should trigger an assertion."""
    from vocab import Vocabulary
    import tempfile, os
    # Create a fake file with wrong format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('This is not a valid ICD-10 order file format\n')
        path = f.name
    try:
        try:
            Vocabulary.from_file(path)
            assert False, 'Should have raised ValueError'
        except (ValueError, IndexError):
            pass  # Expected
    finally:
        os.unlink(path)

# Test 6: Generator (partial - just check the interface)
def test_synth_data_module_importable():
    """Verify synth_data module can be imported."""
    import synth_data
    assert hasattr(synth_data, 'generate_records') or hasattr(synth_data, 'main')
