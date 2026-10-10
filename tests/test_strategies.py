"""Tests for the four coding strategies, the KG hierarchy, leakage guards, metrics, and
an end-to-end mock run with evaluation (CPU only, no model weights).

Fixture: tests/fixtures/tabular_sample.xml is a small TEST FILE with the official nesting,
not the CDC data. The real-file test is skipped until data/raw holds the official XML.
"""

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_XML = ROOT / "tests" / "fixtures" / "tabular_sample.xml"
sys.path.insert(0, str(ROOT / "src"))

from kg import (  # noqa: E402
    Hierarchy, KGRetriever, extract_tabular_xml_from_zip, parse_tabular_xml, validate_hierarchy,
)
from retrieval import Candidate, retrieval_view  # noqa: E402
from prompts import (  # noqa: E402
    FewShotEvidence, RAGEvidence, ZeroShotEvidence, build_messages, render_prompt,
    strip_evidence_block,
)
from metrics import compute_all_metrics, micro_precision_recall  # noqa: E402

GOLD_KEYS = ("gold_codes", "implicit_codes", "explicit_codes", "distractors")


# ── helpers ────────────────────────────────────────────────────────────────

def _record(gold=("A00.1",)):
    return {
        "id": "rec_t1",
        "text": "Patient with cholera-like diarrhoea and high blood pressure. "
                "Treated with oral rehydration.",
        "medications": ["oral rehydration solution"],
        "gold_codes": list(gold),
        "implicit_codes": [],
        "explicit_codes": list(gold),
        "distractors": ["I10"],
        "n_words": 14,
    }


class _SpyRetriever:
    """Records every record passed to retrieve(), and returns fixed ranked candidates."""

    def __init__(self, cands, k=20):
        self._cands = cands
        self.k = k
        self.seen = []

    def retrieve(self, record):
        self.seen.append(copy.deepcopy(record))
        return list(self._cands)[: self.k]


def _cands(*pairs):
    return [Candidate(code=c, desc=d, score=s, source_query=q) for c, d, s, q in pairs]


@pytest.fixture
def hierarchy():
    return parse_tabular_xml(str(FIXTURE_XML))


# ── KG hierarchy loading ───────────────────────────────────────────────────

def test_fixture_hierarchy_parent_child_relations(hierarchy):
    assert hierarchy.parent_of("A00.0") == "A00"       # undotted A000 normalised
    assert hierarchy.parent_of("A00.1") == "A00"
    assert hierarchy.parent_of("A00") is None           # top-level category
    assert hierarchy.children_of("A00") == ["A00.0", "A00.1"]
    assert hierarchy.children_of("I10") == []
    assert "Cholera" == hierarchy.desc["A00"]


def test_hierarchy_has_only_codes_from_the_xml(hierarchy):
    # No relation may exist for a code that is not in the source file.
    assert set(hierarchy.desc) == {"A00", "A00.0", "A00.1", "A02", "A02.1", "I10"}


def test_xml_without_diag_is_rejected(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<ICD10CM.tabular><chapter><name>1</name></chapter></ICD10CM.tabular>",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="No <diag>"):
        parse_tabular_xml(str(bad))


def test_hierarchy_jsonl_round_trip(hierarchy, tmp_path):
    out = tmp_path / "h.jsonl"
    hierarchy.to_jsonl(str(out))
    loaded = Hierarchy.from_jsonl(str(out))
    assert loaded.desc == hierarchy.desc
    assert loaded.parent == hierarchy.parent
    assert loaded.children == hierarchy.children


def test_zip_extraction_then_parse(tmp_path):
    zip_path = tmp_path / "tables.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(FIXTURE_XML, arcname="Tables/icd10cm_tabular_2024.xml")
        z.writestr("Tables/readme.txt", "ignored")
    xml_out = tmp_path / "tabular.xml"
    sha = extract_tabular_xml_from_zip(str(zip_path), str(xml_out))
    assert len(sha) == 64
    assert len(parse_tabular_xml(str(xml_out))) == 6


def test_real_official_tabular_xml_if_present():
    """Runs only on the OFFICIAL CDC file (data/raw). Fixture data is never used here."""
    real = ROOT / "data" / "raw" / "icd10cm_tabular_2024.xml"
    if not real.exists():
        pytest.skip("official tabular XML not extracted to data/raw (OFFICIAL data not present)")
    h = parse_tabular_xml(str(real))
    vocab_path = ROOT / "data" / "vocab" / "icd10cm_2024.jsonl"
    vocab = None
    if vocab_path.exists():
        import json as _json
        vocab = [_json.loads(l)["code"] for l in vocab_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    report = validate_hierarchy(h, vocab)  # strict: must match the verified vocabulary exactly
    assert report["codes"] > 70000
    assert h.parent_of("E11.9") == "E11"


# ── KG expansion: semantics and bounds ─────────────────────────────────────

def test_kg_orders_seeds_first_then_expansions(hierarchy):
    base = _SpyRetriever(_cands(
        ("A00.1", "Cholera Eltor", 0.9, "note_window_0"),
        ("I10", "Essential hypertension", 0.8, "med_x"),
        ("A02.1", "Salmonella sepsis", 0.7, "note_window_1"),
    ))
    kg = KGRetriever(base, hierarchy, k=10, seed_k=2, max_children=1)
    out = kg.retrieve(_record())
    codes = [c.code for c in out]
    assert codes[:2] == ["A00.1", "I10"]                 # semantic seeds keep their order
    assert codes[2] == "A00"                             # parent of seed A00.1
    assert out[2].source_query == "kg_parent_of:A00.1"
    assert "A02.1" not in codes                          # beyond seed_k, not expanded


def test_kg_expands_only_with_official_relations(hierarchy):
    base = _SpyRetriever(_cands(
        ("A00", "Cholera", 0.9, "note_window_0"),
        ("A02", "Other salmonella", 0.8, "note_window_0"),
    ))
    out = KGRetriever(base, hierarchy, k=20, seed_k=2, max_children=5).retrieve(_record())
    seeds = {"A00", "A02"}
    for c in out:
        if c.code in seeds:
            continue
        # every added code is a direct parent or child of one seed, per the hierarchy
        assert any(hierarchy.parent_of(s) == c.code or c.code in hierarchy.children_of(s)
                   for s in seeds), c.code


def test_kg_respects_k_and_has_no_duplicates(hierarchy):
    base = _SpyRetriever(_cands(
        ("A00", "Cholera", 0.9, "q0"), ("A00.0", "x", 0.8, "q1"), ("A00.1", "y", 0.7, "q2"),
        ("A02", "z", 0.6, "q3"),
    ))
    out = KGRetriever(base, hierarchy, k=3, seed_k=4, max_children=5).retrieve(_record())
    codes = [c.code for c in out]
    assert len(codes) == 3
    assert len(set(codes)) == len(codes)


def test_kg_is_deterministic(hierarchy):
    cands = _cands(("A00.1", "a", 0.9, "q0"), ("I10", "b", 0.8, "q1"))
    a = KGRetriever(_SpyRetriever(cands), hierarchy, k=20, seed_k=2, max_children=5).retrieve(_record())
    b = KGRetriever(_SpyRetriever(cands), hierarchy, k=20, seed_k=2, max_children=5).retrieve(_record())
    assert [(c.code, c.source_query) for c in a] == [(c.code, c.source_query) for c in b]


def test_kg_config_validation(hierarchy):
    with pytest.raises(ValueError):
        KGRetriever(_SpyRetriever([]), hierarchy, k=0)


# ── leakage: gold labels never reach retrieval ─────────────────────────────

def test_retrieval_view_drops_every_gold_field():
    view = retrieval_view(_record())
    assert set(view) == {"id", "text", "medications"}
    for key in GOLD_KEYS:
        assert key not in view


def test_kg_retriever_never_sees_gold_fields(hierarchy):
    base = _SpyRetriever(_cands(("A00.1", "a", 0.9, "q0")))
    KGRetriever(base, hierarchy, k=5, seed_k=1).retrieve(_record(gold=("A00.1", "I10")))
    assert base.seen, "base retriever was not called"
    for seen in base.seen:
        for key in GOLD_KEYS:
            assert key not in seen


def test_kg_candidates_do_not_depend_on_gold(hierarchy):
    cands = _cands(("A00.1", "a", 0.9, "q0"), ("I10", "b", 0.8, "q1"))
    with_gold = KGRetriever(_SpyRetriever(cands), hierarchy, k=20, seed_k=2).retrieve(_record(gold=("A00.1",)))
    no_gold = KGRetriever(_SpyRetriever(cands), hierarchy, k=20, seed_k=2).retrieve(_record(gold=("I10", "A02.1")))
    assert [c.code for c in with_gold] == [c.code for c in no_gold]


def test_rag_evidence_passes_only_the_no_gold_view():
    spy = _SpyRetriever(_cands(("I10", "b", 0.8, "q1")))
    RAGEvidence(spy).build_evidence(_record())
    for seen in spy.seen:
        for key in GOLD_KEYS:
            assert key not in seen


# ── prompt invariance across the four strategies ───────────────────────────

def _four_arms(record, hierarchy):
    shots = [
        {"id": "pool_1", "text": "Shot one.", "medications": ["m1"], "gold_codes": ["I10"]},
        {"id": "pool_2", "text": "Shot two.", "medications": [], "gold_codes": ["A00.0"]},
    ]
    cands = _cands(("A00.1", "Cholera Eltor", 0.9, "q0"), ("I10", "Hypertension", 0.8, "q1"))
    kg = KGRetriever(_SpyRetriever(cands), hierarchy, k=20, seed_k=2)
    return {
        "zero_shot": ZeroShotEvidence(),
        "few_shot": FewShotEvidence(shots),
        "rag": RAGEvidence(_SpyRetriever(cands)),
        "kg": RAGEvidence(kg),
    }


def test_prompt_is_identical_across_four_strategies_except_evidence(hierarchy):
    record = _record()
    remainders = {}
    for name, provider in _four_arms(record, hierarchy).items():
        evidence = provider.build_evidence(record)
        user = build_messages(record, provider, "system")[1]["content"]
        remainders[name] = strip_evidence_block(user, evidence)
    assert len(set(remainders.values())) == 1, "only the evidence block may differ"


def test_zero_shot_prompt_has_no_evidence_block(hierarchy):
    record = _record()
    assert ZeroShotEvidence().build_evidence(record) is None
    assert render_prompt(record, None) == build_messages(record, ZeroShotEvidence(), "system")[1]["content"]


def test_system_text_is_identical_across_strategies(hierarchy):
    record = _record()
    systems = {
        name: build_messages(record, p, "system")[0]["content"]
        for name, p in _four_arms(record, hierarchy).items()
    }
    assert len(set(systems.values())) == 1


# ── metrics: precision and recall ──────────────────────────────────────────

def test_micro_precision_recall_hand_computed():
    preds = [{"A", "B", "C"}, {"D"}]
    golds = [{"A", "E"}, {"D"}]
    # TP = A, D = 2; FP = B, C = 2; FN = E = 1
    p, r = micro_precision_recall(preds, golds)
    assert p == pytest.approx(2 / 4)
    assert r == pytest.approx(2 / 3)


def test_micro_precision_recall_zero_denominators():
    assert micro_precision_recall([set()], [set()]) == (0.0, 0.0)


def test_compute_all_metrics_reports_precision_and_recall():
    m = compute_all_metrics([["A", "B"]], [{"A"}], {"A", "B"})
    assert m["micro_precision"] == pytest.approx(0.5)
    assert m["micro_recall"] == pytest.approx(1.0)
    assert "unsupported_code_rate" in m and "invalid_code_rate" in m


# ── end-to-end: four strategies on mock, logged, then evaluated ────────────

def _tmp_config(tmp_path, hierarchy_jsonl):
    cfg = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8"))
    cfg["kg"]["hierarchy_path"] = str(hierarchy_jsonl)
    cfg["vocab"]["vocab_path"] = str(ROOT / "data" / "vocab" / "icd10cm_2024.jsonl")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


_NEEDED = [ROOT / "data" / "synthetic" / "run01" / "eval.jsonl",
           ROOT / "data" / "synthetic" / "run01" / "fewshot_pool.jsonl"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="synthetic dataset not generated (run src/synth_data.py)")
def test_four_strategies_mock_run_logged_and_evaluated(tmp_path, hierarchy):
    hier_jsonl = tmp_path / "hier.jsonl"
    hierarchy.to_jsonl(str(hier_jsonl))
    cfg_path = _tmp_config(tmp_path, hier_jsonl)
    out = tmp_path / "pilot"

    run = subprocess.run(
        [sys.executable, "-B", "src/run_experiment.py", "--config", str(cfg_path),
         "--models", "mock", "--backend", "mock",
         "--strategies", "zero_shot", "few_shot", "rag", "kg",
         "--data", str(_NEEDED[0]), "--limit", "3", "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    smoke = Path(str(out) + "_smoke")
    files = sorted(p.name for p in smoke.glob("*.jsonl"))
    assert files == ["mock__few_shot.jsonl", "mock__kg.jsonl", "mock__rag.jsonl", "mock__zero_shot.jsonl"]

    rows = {}
    for f in files:
        rows[f] = [json.loads(line) for line in (smoke / f).read_text(encoding="utf-8").splitlines() if line]
        assert len(rows[f]) == 3
        for r in rows[f]:
            # same schema for all four strategies
            for key in ("id", "strategy", "prompt_tokens", "new_tokens", "truncated",
                        "generation_truncated", "parse_mode", "raw_output", "parsed_codes",
                        "retrieved", "shots", "compute_dtype", "backend", "template_mode"):
                assert key in r, (f, key)
    kg_row = rows["mock__kg.jsonl"][0]
    assert kg_row["strategy"] == "kg" and kg_row["retrieved"], "KG rows must log candidates"
    assert len({c["code"] for c in kg_row["retrieved"]}) == len(kg_row["retrieved"])
    assert rows["mock__rag.jsonl"][0]["strategy"] == "rag"

    ev = subprocess.run(
        [sys.executable, "-B", "src/evaluate.py", "--results-dir", str(smoke),
         "--data", str(_NEEDED[0]), "--vocab", str(ROOT / "data" / "vocab" / "icd10cm_2024.jsonl"),
         "--allow-partial"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    assert ev.returncode == 0, ev.stderr[-2000:]
    assert "MOCK DATA" in ev.stdout
    assert "| KG |" in ev.stdout
    assert "| Precision | Recall |" in ev.stdout


# ── hierarchy validation (fixture: structural checks; official: vocabulary match) ──

def test_conflicting_parent_in_xml_is_rejected(tmp_path):
    xml = tmp_path / "conflict.xml"
    xml.write_text(
        "<ICD10CM.tabular>"
        "<diag><name>A00</name><desc>x</desc><diag><name>A000</name><desc>a</desc></diag></diag>"
        "<diag><name>A02</name><desc>y</desc><diag><name>A000</name><desc>a</desc></diag></diag>"
        "</ICD10CM.tabular>", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting parents for A00.0"):
        parse_tabular_xml(str(xml))


def test_fixture_validates_structurally(hierarchy):
    report = validate_hierarchy(hierarchy)
    assert report["codes"] == 6
    assert report["top_level"] == 3          # A00, A02, I10
    assert report["edges"] == 3              # A00->A00.0, A00->A00.1, A02->A02.1


def test_fixture_matches_vocab_when_codes_match(hierarchy):
    report = validate_hierarchy(hierarchy, sorted(hierarchy.desc))
    assert report["in_hierarchy_not_in_vocab"] == 0
    assert report["in_vocab_not_in_hierarchy"] == 0


def test_vocab_mismatch_is_rejected(hierarchy):
    with pytest.raises(ValueError, match="does not match the vocabulary"):
        validate_hierarchy(hierarchy, ["A00", "A00.0"])


def test_orphan_parent_is_rejected():
    h = Hierarchy(desc={"A00.1": "x"}, parent={"A00.1": "A00"}, children={"A00.1": []})
    with pytest.raises(ValueError, match="is not a code in the hierarchy"):
        validate_hierarchy(h)


def test_cycle_is_rejected():
    h = Hierarchy(desc={"A00": "a", "B00": "b"},
                  parent={"A00": "B00", "B00": "A00"},
                  children={"A00": ["B00"], "B00": ["A00"]})
    with pytest.raises(ValueError, match="Cycle"):
        validate_hierarchy(h)


def test_malformed_code_is_rejected():
    h = Hierarchy(desc={"??": "x"}, parent={"??": None}, children={"??": []})
    with pytest.raises(ValueError, match="not well-formed"):
        validate_hierarchy(h)


def test_kg_limit_must_equal_rag_limit(tmp_path, hierarchy):
    if not all(p.exists() for p in _NEEDED):
        pytest.skip("synthetic dataset not generated")
    hier_jsonl = tmp_path / "hier.jsonl"
    hierarchy.to_jsonl(str(hier_jsonl))
    cfg = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8"))
    cfg["kg"]["hierarchy_path"] = str(hier_jsonl)
    cfg["strategies"]["kg"]["k"] = 10  # != rag k (20)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-B", "src/run_experiment.py", "--config", str(cfg_path), "--models", "mock",
         "--backend", "mock", "--strategies", "kg", "--data", str(_NEEDED[0]), "--limit", "2",
         "--out", str(tmp_path / "x")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert run.returncode != 0
    assert "must equal" in run.stderr
