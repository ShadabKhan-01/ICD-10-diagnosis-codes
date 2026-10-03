import json
import os

conditions = [
    {"code": "E11.9", "explicit_phrases": ["type 2 diabetes mellitus", "T2DM", "diabetes mellitus type 2"], "implicit_signals": {"medications": ["metformin 500 mg BID", "glipizide 5 mg daily"], "labs": ["HbA1c 8.4%"]}, "negation_phrases": ["no history of diabetes", "denies diabetes"], "family_history_phrases": ["mother had type 2 diabetes"], "distractor_meds": ["insulin glargine 10 units daily"], "weight": 1.0, "domain": "Cardiometabolic"},
    {"code": "I10", "explicit_phrases": ["essential hypertension", "hypertension", "HTN"], "implicit_signals": {"medications": ["lisinopril 20 mg daily", "amlodipine 5 mg daily"]}, "negation_phrases": ["normotensive", "no history of hypertension"], "family_history_phrases": ["father had hypertension"], "distractor_meds": ["hydralazine 25 mg PRN"], "weight": 1.0, "domain": "Cardiometabolic"},
    {"code": "E78.5", "explicit_phrases": ["hyperlipidemia", "high cholesterol"], "implicit_signals": {"medications": ["atorvastatin 40 mg daily", "rosuvastatin 20 mg daily"], "labs": ["LDL 160 mg/dL"]}, "negation_phrases": ["normal lipid panel"], "family_history_phrases": ["family history of hyperlipidemia"], "distractor_meds": ["fenofibrate 145 mg daily"], "weight": 1.0, "domain": "Cardiometabolic"},
    {"code": "J44.9", "explicit_phrases": ["COPD", "chronic obstructive pulmonary disease"], "implicit_signals": {"medications": ["tiotropium bromide 18 mcg daily", "fluticasone/salmeterol 250/50 mcg BID"]}, "negation_phrases": ["no COPD", "clear lungs"], "family_history_phrases": ["father had COPD"], "distractor_meds": [], "weight": 1.0, "domain": "Respiratory"},
    {"code": "I48.91", "explicit_phrases": ["atrial fibrillation", "a-fib"], "implicit_signals": {"medications": ["apixaban 5 mg BID", "rivaroxaban 20 mg daily"]}, "negation_phrases": ["normal sinus rhythm", "no afib"], "family_history_phrases": [], "distractor_meds": ["dabigatran 150 mg BID"], "weight": 0.8, "domain": "Cardiometabolic"},
    {"code": "E03.9", "explicit_phrases": ["hypothyroidism", "underactive thyroid"], "implicit_signals": {"medications": ["levothyroxine 100 mcg daily"], "labs": ["TSH 10.2 mIU/L"]}, "negation_phrases": ["euthyroid", "no thyroid disease"], "family_history_phrases": ["mother has hypothyroidism"], "distractor_meds": ["liothyronine 25 mcg daily"], "weight": 0.7, "domain": "Endocrine"},
    {"code": "K21.9", "explicit_phrases": ["GERD", "gastroesophageal reflux disease", "acid reflux"], "implicit_signals": {"medications": ["omeprazole 20 mg daily", "pantoprazole 40 mg daily"]}, "negation_phrases": ["no heartburn", "denies GERD"], "family_history_phrases": [], "distractor_meds": ["famotidine 20 mg daily"], "weight": 0.9, "domain": "GI"},
    {"code": "J45.909", "explicit_phrases": ["asthma", "uncomplicated asthma"], "implicit_signals": {"medications": ["albuterol inhaler PRN", "budesonide 180 mcg BID"]}, "negation_phrases": ["no asthma"], "family_history_phrases": ["sibling with asthma"], "distractor_meds": [], "weight": 0.8, "domain": "Respiratory"},
    {"code": "G47.33", "explicit_phrases": ["obstructive sleep apnea", "OSA"], "implicit_signals": {"medications": ["CPAP at night"]}, "negation_phrases": ["no sleep apnea"], "family_history_phrases": [], "distractor_meds": [], "weight": 0.6, "domain": "Neurological"},
    {"code": "I50.9", "explicit_phrases": ["heart failure", "CHF", "congestive heart failure"], "implicit_signals": {"medications": ["furosemide 40 mg daily", "spironolactone 25 mg daily", "entresto 49/51 mg BID"]}, "negation_phrases": ["no heart failure"], "family_history_phrases": [], "distractor_meds": ["bumetanide 1 mg daily"], "weight": 0.7, "domain": "Cardiometabolic"},
    {"code": "E66.9", "explicit_phrases": ["obesity", "obese"], "implicit_signals": {"labs": ["BMI 34"]}, "negation_phrases": ["normal weight"], "family_history_phrases": [], "distractor_meds": ["orlistat 120 mg TID"], "weight": 1.0, "domain": "Cardiometabolic"},
    {"code": "D64.9", "explicit_phrases": ["anemia", "unspecified anemia"], "implicit_signals": {"medications": ["ferrous sulfate 325 mg daily"], "labs": ["Hgb 9.2 g/dL"]}, "negation_phrases": ["no anemia"], "family_history_phrases": [], "distractor_meds": [], "weight": 0.6, "domain": "Hematological"},
    {"code": "E55.9", "explicit_phrases": ["vitamin D deficiency"], "implicit_signals": {"medications": ["cholecalciferol 50,000 IU weekly"], "labs": ["Vitamin D 12 ng/mL"]}, "negation_phrases": ["normal vitamin D"], "family_history_phrases": [], "distractor_meds": ["calcium carbonate 500 mg daily"], "weight": 0.8, "domain": "Endocrine"},
    {"code": "M10.9", "explicit_phrases": ["gout", "gouty arthritis"], "implicit_signals": {"medications": ["allopurinol 300 mg daily", "colchicine 0.6 mg PRN"], "labs": ["Uric acid 8.5 mg/dL"]}, "negation_phrases": ["no history of gout"], "family_history_phrases": [], "distractor_meds": ["probenecid 500 mg BID"], "weight": 0.5, "domain": "Musculoskeletal"},
    {"code": "G40.909", "explicit_phrases": ["epilepsy", "seizure disorder"], "implicit_signals": {"medications": ["levetiracetam 500 mg BID", "lamotrigine 100 mg BID"]}, "negation_phrases": ["no seizures"], "family_history_phrases": [], "distractor_meds": ["phenytoin 100 mg TID"], "weight": 0.4, "domain": "Neurological"},
    {"code": "I25.10", "explicit_phrases": ["coronary artery disease", "CAD", "ASHD"], "implicit_signals": {"medications": ["clopidogrel 75 mg daily", "isosorbide mononitrate 30 mg daily"]}, "negation_phrases": ["no CAD"], "family_history_phrases": ["father had early CAD"], "distractor_meds": ["nitroglycerin 0.4 mg SL PRN"], "weight": 0.9, "domain": "Cardiometabolic"},
    {"code": "F41.9", "explicit_phrases": ["anxiety", "anxiety disorder"], "implicit_signals": {"medications": ["escitalopram 10 mg daily", "buspirone 15 mg BID"]}, "negation_phrases": ["no anxiety"], "family_history_phrases": ["mother with anxiety"], "distractor_meds": ["alprazolam 0.5 mg PRN"], "weight": 0.8, "domain": "Psychiatric"},
    {"code": "N39.0", "explicit_phrases": ["urinary tract infection", "UTI"], "implicit_signals": {"medications": ["nitrofurantoin 100 mg BID for 5 days", "trimethoprim-sulfamethoxazole DS BID for 3 days"], "labs": ["Urinalysis positive for nitrites and leukocytes"]}, "negation_phrases": ["no UTI"], "family_history_phrases": [], "distractor_meds": ["phenazopyridine 200 mg TID PRN"], "weight": 0.4, "domain": "Renal"},
    {"code": "J18.9", "explicit_phrases": ["pneumonia", "community acquired pneumonia"], "implicit_signals": {"medications": ["azithromycin 500 mg day 1 then 250 mg daily", "amoxicillin/clavulanate 875/125 mg BID"]}, "negation_phrases": ["no pneumonia"], "family_history_phrases": [], "distractor_meds": ["ceftriaxone 1g IV daily"], "weight": 0.3, "domain": "Respiratory"},
    {"code": "I63.9", "explicit_phrases": ["stroke", "CVA", "cerebral infarction"], "implicit_signals": {"medications": ["aspirin 81 mg daily", "dipyridamole/aspirin 200/25 mg BID"]}, "negation_phrases": ["no stroke"], "family_history_phrases": ["grandmother had stroke"], "distractor_meds": ["cilostazol 100 mg BID"], "weight": 0.5, "domain": "Neurological"},
]

def add_condition(code, phrases, meds=None, labs=None, domain="Cardiometabolic", weight=0.5):
    implicit = {}
    if meds: implicit["medications"] = meds
    if labs: implicit["labs"] = labs
    return {
        "code": code,
        "explicit_phrases": phrases,
        "implicit_signals": implicit,
        "negation_phrases": [f"no history of {phrases[0]}"],
        "family_history_phrases": [f"family history of {phrases[0]}"],
        "distractor_meds": [],
        "weight": weight,
        "domain": domain
    }

conditions.append(add_condition("E10.9", ["type 1 diabetes mellitus", "T1DM"], ["insulin lispro 5 units AC"], ["HbA1c 7.8%"], "Cardiometabolic", 0.3))
conditions.append(add_condition("I11.9", ["hypertensive heart disease"], ["losartan 50 mg daily"], None, "Cardiometabolic", 0.4))
conditions.append(add_condition("I27.20", ["pulmonary hypertension"], ["sildenafil 20 mg TID"], None, "Cardiometabolic", 0.1))
conditions.append(add_condition("E78.00", ["pure hypercholesterolemia"], ["simvastatin 20 mg daily"], None, "Cardiometabolic", 0.5))
conditions.append(add_condition("J43.9", ["emphysema"], ["umeclidinium 62.5 mcg daily"], None, "Respiratory", 0.3))
conditions.append(add_condition("I26.99", ["pulmonary embolism", "PE"], ["warfarin 5 mg daily", "enoxaparin 1 mg/kg BID"], None, "Respiratory", 0.3))
conditions.append(add_condition("J30.9", ["allergic rhinitis", "allergies"], ["fluticasone nasal spray 2 sprays daily", "cetirizine 10 mg daily"], None, "Respiratory", 0.8))
conditions.append(add_condition("G20", ["Parkinson's disease"], ["carbidopa/levodopa 25/100 mg TID"], None, "Neurological", 0.2))
conditions.append(add_condition("G43.909", ["migraine", "migraine without aura"], ["sumatriptan 50 mg PRN", "rizatriptan 10 mg PRN"], None, "Neurological", 0.6))
conditions.append(add_condition("G30.9", ["Alzheimer's disease"], ["donepezil 10 mg daily", "memantine 10 mg BID"], None, "Neurological", 0.2))
conditions.append(add_condition("G35", ["multiple sclerosis", "MS"], ["glatiramer acetate 20 mg daily"], None, "Neurological", 0.1))
conditions.append(add_condition("K50.90", ["Crohn's disease"], ["adalimumab 40 mg every other week", "mesalamine 800 mg TID"], None, "GI", 0.2))
conditions.append(add_condition("K74.60", ["liver cirrhosis", "cirrhosis"], ["lactulose 30 mL TID", "nadolol 20 mg daily"], None, "GI", 0.2))
conditions.append(add_condition("K85.90", ["acute pancreatitis"], ["pancrelipase 36,000 units with meals"], None, "GI", 0.1))
conditions.append(add_condition("K25.9", ["gastric ulcer"], ["sucralfate 1 gram QID"], None, "GI", 0.2))
conditions.append(add_condition("N18.3", ["chronic kidney disease stage 3", "CKD 3"], None, ["eGFR 45 mL/min"], "Renal", 0.5))
conditions.append(add_condition("N20.0", ["nephrolithiasis", "kidney stones"], ["tamsulosin 0.4 mg daily"], None, "Renal", 0.3))
conditions.append(add_condition("N18.4", ["chronic kidney disease stage 4", "CKD 4"], None, ["eGFR 25 mL/min"], "Renal", 0.2))
conditions.append(add_condition("N18.9", ["chronic kidney disease", "CKD"], ["sevelamer 800 mg TID"], None, "Renal", 0.4))
conditions.append(add_condition("E05.90", ["hyperthyroidism"], ["methimazole 5 mg daily"], ["TSH < 0.01 mIU/L"], "Endocrine", 0.2))
conditions.append(add_condition("E21.3", ["hyperparathyroidism"], ["cinacalcet 30 mg daily"], ["Calcium 10.8 mg/dL"], "Endocrine", 0.1))
conditions.append(add_condition("E27.40", ["adrenal insufficiency"], ["hydrocortisone 20 mg in morning and 10 mg in evening"], None, "Endocrine", 0.1))
conditions.append(add_condition("F32.9", ["major depressive disorder", "depression"], ["sertraline 50 mg daily", "fluoxetine 20 mg daily"], None, "Psychiatric", 0.8))
conditions.append(add_condition("F31.9", ["bipolar disorder"], ["lithium 300 mg TID", "divalproex 500 mg BID"], None, "Psychiatric", 0.3))
conditions.append(add_condition("F43.10", ["PTSD", "post-traumatic stress disorder"], ["prazosin 1 mg at bedtime"], None, "Psychiatric", 0.2))
conditions.append(add_condition("F20.9", ["schizophrenia"], ["risperidone 2 mg daily", "olanzapine 10 mg daily"], None, "Psychiatric", 0.2))
conditions.append(add_condition("F90.9", ["ADHD", "attention deficit hyperactivity disorder"], ["methylphenidate 10 mg BID", "lisdexamfetamine 30 mg daily"], None, "Psychiatric", 0.4))
conditions.append(add_condition("M19.90", ["osteoarthritis", "OA"], ["meloxicam 15 mg daily", "diclofenac 1% gel PRN"], None, "Musculoskeletal", 0.7))
conditions.append(add_condition("M81.0", ["osteoporosis"], ["alendronate 70 mg weekly", "denosumab 60 mg every 6 months"], ["T-score -2.8"], "Musculoskeletal", 0.5))
conditions.append(add_condition("M06.9", ["rheumatoid arthritis", "RA"], ["methotrexate 15 mg weekly", "hydroxychloroquine 200 mg BID"], None, "Musculoskeletal", 0.3))
conditions.append(add_condition("M54.50", ["low back pain"], ["cyclobenzaprine 5 mg PRN"], None, "Musculoskeletal", 0.8))
conditions.append(add_condition("A41.9", ["sepsis", "septicemia"], ["vancomycin 1g IV Q12H"], ["Blood cultures positive"], "Infectious", 0.2))
conditions.append(add_condition("L03.90", ["cellulitis"], ["cephalexin 500 mg QID"], None, "Infectious", 0.4))
conditions.append(add_condition("B20", ["HIV", "human immunodeficiency virus"], ["bictegravir/emtricitabine/tenofovir alafenamide daily"], ["CD4 count 350"], "Infectious", 0.2))
conditions.append(add_condition("A04.72", ["C. diff", "Clostridioides difficile infection"], ["oral vancomycin 125 mg QID"], None, "Infectious", 0.1))
conditions.append(add_condition("B18.2", ["chronic hepatitis C", "HCV"], ["sofosbuvir/velpatasvir 400/100 mg daily"], ["HCV RNA positive"], "Infectious", 0.2))
conditions.extend([
    add_condition("D50.9", ["iron deficiency anemia"], ["ferrous gluconate 324 mg daily"], ["Ferritin 8 ng/mL"], "Hematological", 0.5),
    add_condition("L20.9", ["atopic dermatitis", "eczema"], ["triamcinolone 0.1% cream BID", "hydrocortisone 2.5% cream PRN"], None, "Dermatological", 0.5),
    add_condition("L40.0", ["psoriasis", "plaque psoriasis"], ["calcipotriene 0.005% ointment BID", "secukinumab 150 mg Q4W"], None, "Dermatological", 0.2),
    add_condition("M79.10", ["myalgia", "muscle pain"], ["methocarbamol 750 mg TID"], None, "Musculoskeletal", 0.6),
    add_condition("M79.7", ["fibromyalgia"], ["duloxetine 60 mg daily", "pregabalin 75 mg BID"], None, "Musculoskeletal", 0.4),
    add_condition("N40.1", ["BPH", "benign prostatic hyperplasia"], ["finasteride 5 mg daily", "dutasteride 0.5 mg daily"], None, "Genitourinary", 0.5),
    add_condition("Z87.891", ["history of tobacco use", "former smoker"], ["nicotine patch 21 mg daily"], None, "Other", 0.6),
    add_condition("K59.00", ["constipation"], ["polyethylene glycol 17g daily", "docusate sodium 100 mg BID"], None, "GI", 0.7),
    add_condition("H40.90X0", ["glaucoma"], ["latanoprost 0.005% eye drops at bedtime", "timolol 0.5% eye drops BID"], None, "Ophthalmological", 0.4),
    add_condition("H25.9", ["cataracts"], None, None, "Ophthalmological", 0.5),
    add_condition("F10.20", ["alcohol use disorder", "alcoholism"], ["naltrexone 50 mg daily", "acamprosate 333 mg TID"], None, "Psychiatric", 0.3),
    add_condition("F11.20", ["opioid use disorder"], ["buprenorphine/naloxone 8/2 mg daily", "methadone 40 mg daily"], None, "Psychiatric", 0.2),
    add_condition("F17.210", ["tobacco use disorder", "smoker"], ["varenicline 1 mg BID"], None, "Psychiatric", 0.6),
    add_condition("E79.0", ["hyperuricemia"], ["febuxostat 40 mg daily"], ["Uric acid 9.2 mg/dL"], "Metabolic", 0.3),
    add_condition("D51.0", ["vitamin B12 deficiency anemia", "pernicious anemia"], ["cyanocobalamin 1000 mcg IM monthly"], ["Vitamin B12 150 pg/mL"], "Hematological", 0.3),
    add_condition("G62.9", ["polyneuropathy", "neuropathy"], ["gabapentin 300 mg TID"], None, "Neurological", 0.5),
    add_condition("I83.90", ["varicose veins"], ["compression stockings"], None, "Cardiovascular", 0.4),
    add_condition("R03.0", ["elevated blood pressure"], None, ["BP 135/85"], "Cardiovascular", 0.4),
    add_condition("R73.03", ["prediabetes"], None, ["HbA1c 6.0%"], "Endocrine", 0.6),
    add_condition("I95.9", ["hypotension"], ["midodrine 5 mg TID", "fludrocortisone 0.1 mg daily"], ["BP 85/55"], "Cardiovascular", 0.1),
    add_condition("K58.0", ["IBS with diarrhea", "irritable bowel syndrome"], ["loperamide 2 mg PRN", "dicyclomine 20 mg QID"], None, "GI", 0.3),
])

def dump_yaml(data, out):
    for item in data:
        out.write(f"- code: \"{item['code']}\"\n")
        out.write(f"  explicit_phrases:\n")
        for p in item['explicit_phrases']: out.write(f"    - \"{p}\"\n")
        out.write(f"  implicit_signals:\n")
        if 'medications' in item['implicit_signals']:
            out.write(f"    medications:\n")
            for m in item['implicit_signals']['medications']: out.write(f"      - \"{m}\"\n")
        if 'labs' in item['implicit_signals']:
            out.write(f"    labs:\n")
            for l in item['implicit_signals']['labs']: out.write(f"      - \"{l}\"\n")
        out.write(f"  negation_phrases:\n")
        for p in item['negation_phrases']: out.write(f"    - \"{p}\"\n")
        out.write(f"  family_history_phrases:\n")
        for p in item['family_history_phrases']: out.write(f"    - \"{p}\"\n")
        out.write(f"  distractor_meds:\n")
        for p in item['distractor_meds']: out.write(f"    - \"{p}\"\n")
        out.write(f"  weight: {item['weight']}\n")

os.makedirs("src/templates", exist_ok=True)
with open("src/templates/conditions.yaml", "w") as f:
    dump_yaml(conditions, f)

print(f"Created {len(conditions)} conditions.")
