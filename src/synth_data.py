import argparse
import json
import logging
import random
import hashlib
import yaml
from pathlib import Path

from codes import normalize, is_well_formed
from vocab import Vocabulary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_conditions(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_overlap(text, phrases):
    text_lower = text.lower()
    for p in phrases:
        if p.lower() in text_lower:
            return True
    return False

def generate_record(record_id, conditions, all_codes_list, vocab, implicit_rate, max_words):
    # Sample 3-6 conditions
    n_cond = random.randint(3, 6)
    weights = [c.get('weight', 1.0) for c in conditions]
    
    # Random choices with weights
    selected_conditions = []
    # Weighted sampling without replacement
    remaining_indices = list(range(len(conditions)))
    remaining_weights = list(weights)
    for _ in range(n_cond):
        if sum(remaining_weights) == 0:
            idx = random.choice(remaining_indices)
        else:
            idx = random.choices(remaining_indices, weights=remaining_weights, k=1)[0]
        selected_conditions.append(conditions[idx])
        pos = remaining_indices.index(idx)
        remaining_indices.pop(pos)
        remaining_weights.pop(pos)

    gold_codes = []
    implicit_codes = []
    explicit_codes = []
    medications = []
    labs = []
    explicit_phrases = []
    implicit_objs = []
    distractors = []
    
    for c in selected_conditions:
        code = c['code']
        gold_codes.append(code)
        is_implicit = random.random() < implicit_rate
        if is_implicit and c.get('implicit_signals'):
            implicit_codes.append(code)
            if 'medications' in c['implicit_signals']:
                meds = c['implicit_signals']['medications']
                if meds: medications.append(random.choice(meds))
            if 'labs' in c['implicit_signals']:
                lbs = c['implicit_signals']['labs']
                if lbs: labs.append(random.choice(lbs))
            implicit_objs.append(c)
        else:
            explicit_codes.append(code)
            explicit_phrases.append(random.choice(c['explicit_phrases']))

    # Add 1-3 distractors
    n_distractors = random.randint(1, 3)
    distractor_candidates = [c for c in conditions if c['code'] not in gold_codes]
    distractor_conds = random.sample(distractor_candidates, min(n_distractors, len(distractor_candidates)))
    
    distractor_text_parts = []
    for c in distractor_conds:
        dtype = random.choice(['negation', 'family', 'med'])
        distractors.append(c['code'])
        if dtype == 'negation' and c.get('negation_phrases'):
            distractor_text_parts.append(random.choice(c['negation_phrases']))
        elif dtype == 'family' and c.get('family_history_phrases'):
            distractor_text_parts.append(random.choice(c['family_history_phrases']))
        elif dtype == 'med' and c.get('distractor_meds'):
            medications.append(random.choice(c['distractor_meds']))
        else:
            if c.get('negation_phrases'):
                distractor_text_parts.append(random.choice(c['negation_phrases']))
    
    # Rendering
    sections = []
    sections.append(f"CHIEF COMPLAINT:\nPatient presents for evaluation.")
    
    hpi = "HISTORY OF PRESENT ILLNESS:\nPatient is a 65-year-old presenting with multiple chronic conditions."
    if labs:
        hpi += " Recent labs showed " + ", ".join(labs) + "."
    if distractor_text_parts:
        hpi += " " + " ".join(distractor_text_parts).capitalize() + "."
    sections.append(hpi)
    
    pmh = "PAST MEDICAL HISTORY:\n"
    if explicit_phrases:
        pmh += ", ".join(explicit_phrases).capitalize() + "."
    sections.append(pmh)
    
    hospital_course = "HOSPITAL COURSE:\nPatient was admitted and stabilized. Monitored closely for all active issues."
    sections.append(hospital_course)
    
    meds_section = "DISCHARGE MEDICATIONS:\n" + "\n".join(f"- {m}" for m in medications) if medications else "DISCHARGE MEDICATIONS:\nNone"
    sections.append(meds_section)
    
    dx_section = "DISCHARGE DIAGNOSIS:\n"
    if explicit_phrases:
        dx_section += "\n".join(f"- {p}" for p in explicit_phrases)
    else:
        dx_section += "Pending."
    sections.append(dx_section)
    
    text = "\n\n".join(sections)
    
    # Length constraint
    words = text.split()
    if len(words) > max_words:
        # truncate
        hpi_idx = text.find("HISTORY OF PRESENT ILLNESS:")
        pmh_idx = text.find("PAST MEDICAL HISTORY:")
        if pmh_idx > hpi_idx and hpi_idx != -1:
            pass # Keep it simple, just truncate words but maintain structure if possible, wait, just truncate and append "..."
        
        words = words[:max_words]
        text = " ".join(words) + " ..."
    
    n_words = len(text.split())
    
    # Hard Checks
    # Implicit codes never appear by name in text
    for c in implicit_objs:
        if check_overlap(text, c['explicit_phrases']):
            raise ValueError(f"Implicit code {c['code']} appeared explicitly in text!")
            
    # No gold code is distractor
    overlap = set(gold_codes).intersection(set(distractors))
    if overlap:
        raise ValueError(f"Gold codes and distractors overlap: {overlap}")
        
    for gc in gold_codes:
        if not vocab.is_valid(gc):
            raise ValueError(f"Gold code {gc} is invalid!")
            
    return {
        "id": f"rec_{record_id:05d}",
        "text": text,
        "medications": medications,
        "gold_codes": gold_codes,
        "implicit_codes": implicit_codes,
        "explicit_codes": explicit_codes,
        "distractors": distractors,
        "n_words": n_words
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-fewshot-pool", type=int, default=50)
    parser.add_argument("--n-dev", type=int, default=30)
    parser.add_argument("--implicit-rate", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading vocabulary...")
    vocab = Vocabulary.load_or_download({})
    
    logger.info("Loading conditions...")
    conditions = load_conditions("src/templates/conditions.yaml")
    
    for c in conditions:
        if not vocab.is_valid(c['code']):
            logger.error(f"Template code {c['code']} is invalid in vocabulary!")
            return
            
    total_records = args.n_eval + args.n_fewshot_pool + args.n_dev
    records = []
    
    text_hashes = set()
    all_codes = [c['code'] for c in conditions]
    
    logger.info(f"Generating {total_records} records...")
    attempts = 0
    while len(records) < total_records and attempts < total_records * 10:
        attempts += 1
        try:
            rec = generate_record(len(records) + 1, conditions, all_codes, vocab, args.implicit_rate, args.max_words)
            rec['seed'] = args.seed
            
            # check disjoint
            thash = hashlib.md5(rec['text'].encode()).hexdigest()
            if thash in text_hashes:
                continue
            text_hashes.add(thash)
            records.append(rec)
        except Exception as e:
            logger.debug(f"Skipped record due to error: {e}")
            
    if len(records) < total_records:
        logger.error(f"Could only generate {len(records)} unique records.")
        return
        
    dev_recs = records[:args.n_dev]
    fewshot_recs = records[args.n_dev:args.n_dev+args.n_fewshot_pool]
    eval_recs = records[args.n_dev+args.n_fewshot_pool:]
    
    def write_jsonl(recs, filename):
        for rec in recs:
            rec["split"] = filename.replace(".jsonl", "")
        with open(out_dir / filename, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
                
    write_jsonl(dev_recs, "dev.jsonl")
    write_jsonl(fewshot_recs, "fewshot_pool.jsonl")
    write_jsonl(eval_recs, "eval.jsonl")
    
    # Stats
    all_gold = []
    implicit_count = 0
    total_words = 0
    freqs = {}
    for r in records:
        all_gold.extend(r['gold_codes'])
        implicit_count += len(r['implicit_codes'])
        total_words += r['n_words']
        for gc in r['gold_codes']:
            freqs[gc] = freqs.get(gc, 0) + 1
            
    stats = {
        "total_records": len(records),
        "unique_gold_codes": len(set(all_gold)),
        "mean_gold_codes_per_record": len(all_gold) / len(records) if records else 0,
        "mean_record_length": total_words / len(records) if records else 0,
        "percent_implicit": (implicit_count / len(all_gold) * 100) if all_gold else 0,
        "code_frequencies": freqs
    }
    
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()
