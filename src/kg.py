"""Knowledge-graph strategy: official ICD-10-CM hierarchy + semantic seeds (hybrid).

Source of truth: the CDC/NCHS ICD-10-CM 2024 *tabular* XML. Its nested <diag> elements
define parent-child relations (for example category -> subcategory -> code). Nothing else
is used, and no relationship is invented.

The KG strategy is a HYBRID and is documented as one:
  1. Seeds: the SAME MiniLM retrieval index used by the RAG strategy, queried with the
     record's note windows and medications (never gold codes).
  2. Expansion: each of the top `seed_k` seeds contributes its direct parent and up to
     `max_children` direct children from the official hierarchy.
The candidate pool is capped at the same k as RAG, so the only difference between the
RAG and KG evidence is which codes are offered.

Leakage: the record reaches the retriever only through `retrieval_view()` (id, text,
medications). Gold codes are never read by retrieval or expansion.
"""

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codes import normalize  # noqa: E402
from retrieval import retrieval_view  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Hierarchy:
    """Parent/child relations from the official tabular XML."""
    desc: Dict[str, str] = field(default_factory=dict)
    parent: Dict[str, Optional[str]] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, code: str, desc: str, parent: Optional[str]) -> None:
        """Add one code with its parent. A code may appear only once with one parent.

        A second occurrence under a different parent raises, rather than silently
        keeping whichever came first. An identical repeat is accepted.
        """
        if code in self.parent:
            if self.parent[code] != parent:
                raise ValueError(
                    f"Conflicting parents for {code}: {self.parent[code]!r} and {parent!r}"
                )
            if self.desc.get(code) != desc and desc:
                raise ValueError(f"Conflicting descriptions for {code}")
            return
        self.desc[code] = desc
        self.parent[code] = parent
        self.children.setdefault(code, [])
        if parent is not None:
            self.children.setdefault(parent, [])
            self.children[parent].append(code)

    def parent_of(self, code: str) -> Optional[str]:
        return self.parent.get(code)

    def children_of(self, code: str) -> List[str]:
        return list(self.children.get(code, []))

    def __len__(self) -> int:
        return len(self.desc)

    def to_jsonl(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for code in self.desc:
                f.write(json.dumps({
                    "code": code,
                    "desc": self.desc[code],
                    "parent": self.parent.get(code),
                    "children": self.children.get(code, []),
                }, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "Hierarchy":
        h = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                h.desc[row["code"]] = row["desc"]
                h.parent[row["code"]] = row["parent"]
                h.children[row["code"]] = list(row["children"])
        return h


def parse_tabular_xml(source) -> Hierarchy:
    """Parse nested ICD-10-CM diagnoses, preserving parent-child relationships."""
    h = Hierarchy()
    tree = ET.parse(source)
    root = tree.getroot()

    def walk(el, parent_code=None):
        for child in el:
            if child.tag.split("}")[-1] != "diag":
                walk(child, parent_code)
                continue

            raw_name = (child.findtext("name") or "").strip()
            if not raw_name:
                raise ValueError("<diag> without <name> in tabular XML")

            code = normalize(raw_name)
            desc = (child.findtext("desc") or "").strip()
            h.add(code, desc, parent_code)
            walk(child, code)

    walk(root)

    if len(h) == 0:
        raise ValueError("No <diag> elements found in tabular XML")

    return h


def validate_hierarchy(h: Hierarchy, vocab_codes: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Structural checks and an optional vocabulary cross-check.

    Raises ValueError on any structural error (these are never acceptable):
      - a parent that is not itself a code in the hierarchy
      - a child list that disagrees with the parent map
      - a cycle in the parent chain
      - a code that is not well-formed (see codes.is_well_formed)
      - with vocab_codes: a code in the hierarchy that is not in the vocabulary, or the
        reverse. Cross-checking is what makes the hierarchy match the verified code set.

    Returns a report dict with counts, for logging and for the change log.
    """
    from codes import is_well_formed

    bad_format = [c for c in h.desc if not is_well_formed(c)]
    if bad_format:
        raise ValueError(f"{len(bad_format)} codes are not well-formed, e.g. {bad_format[:5]}")

    for code, parent in h.parent.items():
        if parent is not None and parent not in h.desc:
            raise ValueError(f"Parent {parent} of {code} is not a code in the hierarchy")
        if parent is not None and code not in h.children.get(parent, []):
            raise ValueError(f"{code} missing from the child list of {parent}")
    for parent, kids in h.children.items():
        for kid in kids:
            if h.parent.get(kid) != parent:
                raise ValueError(f"Child {kid} listed under {parent} but its parent is {h.parent.get(kid)}")

    for code in h.desc:
        seen = set()
        node = code
        while node is not None:
            if node in seen:
                raise ValueError(f"Cycle in parent chain at {node}")
            seen.add(node)
            node = h.parent.get(node)

    empty_desc = [c for c, d in h.desc.items() if not d]

    report: Dict[str, Any] = {
        "codes": len(h.desc),
        "top_level": sum(1 for p in h.parent.values() if p is None),
        "edges": sum(len(v) for v in h.children.values()),
        "empty_descriptions": len(empty_desc),
    }

    if vocab_codes is not None:
        vocab = set(vocab_codes)
        only_hier = sorted(set(h.desc) - vocab)
        only_vocab = sorted(vocab - set(h.desc))
        report["in_hierarchy_not_in_vocab"] = len(only_hier)
        report["in_vocab_not_in_hierarchy"] = len(only_vocab)
        report["examples_hierarchy_only"] = only_hier[:5]
        report["examples_vocab_only"] = only_vocab[:5]
        if only_hier:
            raise ValueError(
                f"{len(only_hier)} Hierarchy does not match the vocabulary; hierarchy codes are missing: "
                f"{only_hier[:5]}"
            )
    return report


def extract_tabular_xml_from_zip(zip_path: str, out_path: str) -> str:
    """Extract the tabular XML member from the official CDC zip. Returns its SHA-256."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if m.lower().endswith(".xml") and "tabular" in m.lower().rsplit("/", 1)[-1]]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one tabular XML (name containing 'tabular') in "
                             f"{zip_path}, found {members}")
        data = z.read(members[0])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class KGRetriever:
    """Hybrid KG retriever: semantic seeds from the RAG index, expanded via the hierarchy.

    Output candidates follow the RAG Candidate shape (code, desc, score, source_query), so
    the evidence formatting is identical to RAG. `source_query` records how each code came
    in: the seed's own query, or "kg_parent_of:<seed>" / "kg_child_of:<seed>".
    """

    def __init__(self, base_retriever: Any, hierarchy: Hierarchy, k: int = 20,
                 seed_k: int = 10, max_children: int = 5):
        if seed_k < 1 or max_children < 0 or k < 1:
            raise ValueError("k and seed_k must be >= 1, max_children >= 0")
        self.base = base_retriever
        self.hierarchy = hierarchy
        self.k = k
        self.seed_k = seed_k
        self.max_children = max_children

    def retrieve(self, record: Dict[str, Any]):
        from retrieval import Candidate  # local import keeps module importable without torch

        seeds = self.base.retrieve(retrieval_view(record))[: self.seed_k]
        out: List[Candidate] = []
        seen = set()

        def add(code: str, desc: str, score: float, source: str) -> None:
            if code in seen or len(out) >= self.k:
                return
            seen.add(code)
            out.append(Candidate(code=code, desc=desc, score=score, source_query=source))

        for s in seeds:
            add(s.code, s.desc or self.hierarchy.desc.get(s.code, ""), s.score, s.source_query)
        for s in seeds:
            if len(out) >= self.k:
                break
            parent = self.hierarchy.parent_of(s.code)
            if parent is not None:
                add(parent, self.hierarchy.desc.get(parent, ""), s.score, f"kg_parent_of:{s.code}")
            for child in self.hierarchy.children_of(s.code)[: self.max_children]:
                add(child, self.hierarchy.desc.get(child, ""), s.score, f"kg_child_of:{s.code}")
        return out


def _vocab_codes_from_jsonl(path: str) -> List[str]:
    codes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                codes.append(json.loads(line)["code"])
    return codes


def _cli_build(args) -> None:
    logging.basicConfig(level=logging.INFO)
    xml_path = Path(args.xml)
    if args.zip:
        sha = extract_tabular_xml_from_zip(args.zip, str(xml_path))
        logger.info(f"Extracted {xml_path} from {args.zip} (sha256 {sha})")
    h = parse_tabular_xml(str(xml_path))
    vocab_codes = _vocab_codes_from_jsonl(args.vocab) if args.vocab else None
    report = validate_hierarchy(h, vocab_codes)  # raises on any structural error
    h.to_jsonl(args.out)
    logger.info(f"Validated hierarchy: {report}")
    logger.info(f"Wrote {len(h)} codes to {args.out}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build the ICD-10-CM hierarchy for the KG strategy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="parse the official tabular XML into a hierarchy JSONL")
    b.add_argument("--xml", required=True, help="path to icd10cm-tabular-2024.xml (extracted here if --zip)")
    b.add_argument("--zip", help="official CDC zip containing the tabular XML")
    b.add_argument("--out", required=True, help="output hierarchy JSONL")
    b.add_argument("--vocab", help="verified vocabulary JSONL; the hierarchy must match it exactly")
    b.set_defaults(func=_cli_build)
    args = ap.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
