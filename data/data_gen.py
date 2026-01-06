#!/usr/bin/env python3
"""
Generate linked diagnoses_synth.csv and procedures_synth.csv from encounters_synth.csv

Rules enforced:
- For each encounter (RECORD_ID):
  - Diagnoses: 1–6 rows (at least 1)
  - Procedures: 0–5 rows
  - SEQUENCE_NUMBER unique and increasing within each RECORD_ID group
  - PROCEDURE_DATE within [ADMISSION_DATE, DISCHARGE_DATE]
  - If RV_VENT_ON_ADMIT_DATE == 1 -> at least one ventilation-type procedure exists and is dated on ADMISSION_DATE
  - POA distribution ~ 70–80% Y, ~10–15% N, remainder E/1
  - Some RISK_VARIABLE_DESCRIPTION / VARIABLE_DESCRIPTION blank

Outputs:
- diagnoses_synth.csv
- procedures_synth.csv
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAX_DX_PER_ENC = 6
MAX_PR_PER_ENC = 5

# If True, occasionally start diagnosis sequence numbers at 10–35 to mimic "can exceed"
ALLOW_HIGH_SEQ_START = True
HIGH_SEQ_START_PROB = 0.10

# -----------------------------
# Libraries (synthetic but realistic)
# -----------------------------
DIAG_LIBRARY: List[Tuple[str, str, str, str]] = [
    ("I2510", "Atherosclerotic heart disease of native coronary artery without angina pectoris", "Coronary artery disease", "Lipid Disorders"),
    ("E669", "Obesity, unspecified", "Obesity", "Obesity"),
    ("Z3A40", "40 weeks gestation of pregnancy", "Full-term pregnancy", ""),
    ("E785", "Hyperlipidemia, unspecified", "High cholesterol", "Lipid Disorders"),
    ("N189", "Chronic kidney disease, unspecified", "Chronic kidney disease", "any fluid electrolyte"),
    ("J45909", "Unspecified asthma, uncomplicated", "Asthma", ""),
    ("G3183", "Dementia with Lewy bodies", "Dementia with Lewy bodies", "Dementia"),
    ("E559", "Vitamin D deficiency, unspecified", "Vitamin D deficiency", ""),
    ("Z7982", "Long term (current) use of aspirin", "Long-term aspirin use", "Anticoagulation Status"),
    ("A419", "Sepsis, unspecified organism", "Sepsis", "any fluid electrolyte"),
    ("R6520", "Severe sepsis without septic shock", "Severe sepsis", "any fluid electrolyte"),
    ("J9600", "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia", "Acute respiratory failure", "Respiratory Failure"),
    ("I509", "Heart failure, unspecified", "Heart failure", "CHF"),
    ("E875", "Hypokalemia", "Low potassium", "any fluid electrolyte"),
    ("D650", "Acute posthemorrhagic anemia", "Blood loss anemia", "any Blood Loss Anemia"),
    ("D696", "Thrombocytopenia, unspecified", "Low platelet count", "RV Thrombocytopenia"),
    ("E435", "Unspecified severe protein-calorie malnutrition", "Malnutrition", "NPOA Malnutrition"),
    ("D65", "Disseminated intravascular coagulation [defibrination syndrome]", "Disseminated intravascular coagulation", "NPOA DIC"),
    ("C7800", "Secondary malignant neoplasm of unspecified lung", "Metastatic cancer to lung", "Metastatic Cancer"),
    ("K7200", "Acute and subacute hepatic failure without coma", "Acute liver failure", "Liver Failure"),
    ("O800", "Encounter for full-term uncomplicated delivery", "Uncomplicated delivery", ""),
    ("N170", "Acute kidney failure with tubular necrosis", "Acute kidney injury", "Acute Renal Failure"),
]

# Basic weights to keep certain codes common
COMMON_DX = {"I2510", "E669", "Z3A40", "E785", "N189", "J45909"}

PROC_LIBRARY: List[Tuple[str, str, str, Dict[str, bool]]] = [
    ("5A1945Z", "Respiratory ventilation, less than 24 consecutive hours", "Vent on Admission Day", {"vent": True}),
    ("5A1955Z", "Respiratory ventilation, 24–96 consecutive hours", "Vent within 48h", {"vent": True}),
    ("0W9G3ZZ", "Drainage of thoracic cavity, percutaneous approach", "Respiratory Support", {"vent": False}),
    ("B2111ZZ", "Measurement of cardiac output, monitoring, via central approach", "Hemodynamic Monitoring", {"vent": False}),
    ("B211YZZ", "Measurement of cardiac pressure, monitoring, via peripheral approach", "Hemodynamic Monitoring", {"vent": False}),
    ("30233N1", "Transfusion of nonautologous red blood cells into peripheral vein, percutaneous approach", "Control Bleeding on Admission Day", {"vent": False}),
    ("30243N1", "Transfusion of nonautologous red blood cells into central vein, percutaneous approach", "", {"vent": False}),
    ("0DBP8ZX", "Diagnostic excision of large intestine, via natural or artificial opening endoscopic", "", {"vent": False}),
    ("0SR902Z", "Replacement of right hip joint with cemented synthetic substitute, open approach", "Joint Replacement", {"vent": False}),
    ("0W3P0ZZ", "Control bleeding in upper gastrointestinal tract, open approach", "Control Bleeding on Admission Day", {"vent": False}),
    ("5A1D70Z", "Performance of urinary filtration, intermittent", "Hemodialysis Pre-Transplant", {"vent": False, "dialysis": True}),
    ("3E03329", "Introduction of other anti-infective into peripheral vein, percutaneous approach", "IV Antibiotic Therapy", {"vent": False}),
]

VENT_CODES = {"5A1945Z", "5A1955Z"}


# -----------------------------
# Helpers
# -----------------------------
def maybe_blank(s: str, p_blank: float) -> str:
    return "" if random.random() < p_blank else s

def parse_iso(d: str) -> date:
    return date.fromisoformat(d)

def rand_date_between(a: date, b: date) -> date:
    # inclusive
    if b < a:
        return a
    delta = (b - a).days
    return a + timedelta(days=random.randint(0, delta))

def weighted_choice(items, weights):
    return random.choices(items, weights=weights, k=1)[0]

def poa_draw() -> str:
    # ~70–80% Y, ~10–15% N, remainder E/1
    vals = ["Y", "N", "E", "1"]
    wts  = [0.76, 0.12, 0.09, 0.03]
    return weighted_choice(vals, wts)

def choose_dx_count() -> int:
    # 1–6 (at least 1)
    choices = [1, 2, 3, 4, 5, 6]
    weights = [0.20, 0.23, 0.20, 0.17, 0.12, 0.08]
    return weighted_choice(choices, weights)

def choose_pr_count() -> int:
    # 0–5
    choices = [0, 1, 2, 3, 4, 5]
    weights = [0.25, 0.24, 0.20, 0.16, 0.10, 0.05]
    return weighted_choice(choices, weights)

def pick_dx_list() -> List[Tuple[str, str, str, str]]:
    k = choose_dx_count()
    weights = []
    for code, *_ in DIAG_LIBRARY:
        w = 1.0
        if code in COMMON_DX:
            w *= 2.5
        weights.append(w)

    chosen = random.choices(DIAG_LIBRARY, weights=weights, k=min(k, len(DIAG_LIBRARY)))

    # ensure unique codes within encounter
    out = []
    seen = set()
    for item in chosen:
        if item[0] not in seen:
            out.append(item)
            seen.add(item[0])
    while len(out) < k:
        item = random.choices(DIAG_LIBRARY, weights=weights, k=1)[0]
        if item[0] not in seen:
            out.append(item)
            seen.add(item[0])
    return out[:k]

def pick_proc_list(need_vent: bool, dialysis_hint: bool) -> List[Tuple[str, str, str, Dict[str, bool]]]:
    k = choose_pr_count()
    if k == 0:
        return []

    weights = []
    for code, _, _, tags in PROC_LIBRARY:
        w = 1.0
        if code in {"5A1945Z", "0W9G3ZZ", "B2111ZZ", "B211YZZ", "30233N1", "30243N1"}:
            w *= 1.8
        if need_vent and tags.get("vent", False):
            w *= 4.0
        if dialysis_hint and tags.get("dialysis", False):
            w *= 3.0
        weights.append(w)

    chosen = random.choices(PROC_LIBRARY, weights=weights, k=min(k, len(PROC_LIBRARY)))

    out = []
    seen = set()
    for item in chosen:
        if item[0] not in seen:
            out.append(item)
            seen.add(item[0])
    while len(out) < k:
        item = random.choices(PROC_LIBRARY, weights=weights, k=1)[0]
        if item[0] not in seen:
            out.append(item)
            seen.add(item[0])

    # enforce at least one vent procedure if required
    if need_vent and not any(tags.get("vent", False) for _, _, _, tags in out):
        out[0] = random.choice([p for p in PROC_LIBRARY if p[3].get("vent", False)])

    return out[:k]


# -----------------------------
# Main generation
# -----------------------------
def generate(encounters_csv: str | Path,
             out_dx_csv: str | Path = "diagnoses_synth.csv",
             out_pr_csv: str | Path = "procedures_synth.csv") -> None:

    enc = pd.read_csv(encounters_csv, dtype=str)

    # minimal required columns in encounters_synth
    required = {"RECORD_ID", "ADMISSION_DATE", "DISCHARGE_DATE", "RV_VENT_ON_ADMIT_DATE"}
    missing = required - set(enc.columns)
    if missing:
        raise ValueError(f"encounters_synth.csv is missing required columns: {sorted(missing)}")

    dx_rows: List[List[str]] = []
    pr_rows: List[List[str]] = []

    for _, row in enc.iterrows():
        rid = str(row["RECORD_ID"])
        admit = parse_iso(row["ADMISSION_DATE"])
        disch = parse_iso(row["DISCHARGE_DATE"])
        need_vent = str(row.get("RV_VENT_ON_ADMIT_DATE", "0")) == "1"

        # simple dialysis hint from renal flags if present, else False
        dialysis_hint = str(row.get("RV_ACUTE_RENAL_FAILURE", "0")) == "1" or str(row.get("RV_MALNUTRITION", "0")) == "1"

        # ---- Diagnoses (1–6)
        dx_list = pick_dx_list()

        if ALLOW_HIGH_SEQ_START and random.random() < HIGH_SEQ_START_PROB:
            start_seq = random.randint(10, 35)
            seqs = list(range(start_seq, start_seq + len(dx_list)))
        else:
            seqs = list(range(1, len(dx_list) + 1))

        for seq, (code, formal, friendly, risk) in zip(seqs, dx_list):
            dx_rows.append([
                rid,
                code,
                str(seq),
                poa_draw(),
                formal,
                friendly,
                maybe_blank(risk, p_blank=0.28),
            ])

        # ---- Procedures (0–5)
        proc_list = pick_proc_list(need_vent=need_vent, dialysis_hint=dialysis_hint)

        # If RV_VENT_ON_ADMIT_DATE=1, ensure a vent code exists and is on admission date
        # Build provisional procedure rows
        temp_pr: List[List[str]] = []
        for j, (pcode, pname, vdesc, tags) in enumerate(proc_list, start=1):
            if need_vent and tags.get("vent", False) and random.random() < 0.75:
                pdate = admit
            else:
                pdate = rand_date_between(admit, disch)

            temp_pr.append([
                rid,
                pcode,
                str(j),
                pdate.isoformat(),
                pname,
                maybe_blank(vdesc, p_blank=0.22),
            ])

        if need_vent:
            has_vent = any(r[1] in VENT_CODES for r in temp_pr)
            if not has_vent:
                # inject a ventilation procedure as sequence 1
                temp_pr.insert(0, [
                    rid,
                    "5A1945Z",
                    "1",
                    admit.isoformat(),
                    "Respiratory ventilation, less than 24 consecutive hours",
                    "Vent on Admission Day",
                ])
                # resequence
                for idx in range(len(temp_pr)):
                    temp_pr[idx][2] = str(idx + 1)
            else:
                # force one vent procedure to be on admit date
                for r in temp_pr:
                    if r[1] in VENT_CODES:
                        r[3] = admit.isoformat()
                        break

        pr_rows.extend(temp_pr)

    # Create DataFrames with exact schemas
    dx_cols = [
        "RECORD_ID",
        "DIAGNOSIS_CODE",
        "SEQUENCE_NUMBER",
        "POA",
        "ICD10_DESCRIPTION",
        "ICD10_FRIENDLY_DESCRIPTION",
        "RISK_VARIABLE_DESCRIPTION",
    ]
    pr_cols = [
        "RECORD_ID",
        "PROCEDURE_CODE",
        "SEQUENCE_NUMBER",
        "PROCEDURE_DATE",
        "PROCEDURE_NAME",
        "VARIABLE_DESCRIPTION",
    ]

    dx_df = pd.DataFrame(dx_rows, columns=dx_cols)
    pr_df = pd.DataFrame(pr_rows, columns=pr_cols)

    # -----------------------------
    # Self-checks (validation checklist)
    # -----------------------------
    enc_ids = set(enc["RECORD_ID"].astype(str))

    # All diagnoses/procedures record IDs exist in encounters
    assert set(dx_df["RECORD_ID"]).issubset(enc_ids)
    assert set(pr_df["RECORD_ID"]).issubset(enc_ids)

    # Each encounter has >= 1 diagnosis
    dx_counts = dx_df.groupby("RECORD_ID").size().reindex(enc["RECORD_ID"], fill_value=0)
    assert (dx_counts >= 1).all()

    # Sequence uniqueness per encounter
    assert not dx_df.groupby("RECORD_ID")["SEQUENCE_NUMBER"].apply(lambda s: s.duplicated().any()).any()
    if len(pr_df) > 0:
        assert not pr_df.groupby("RECORD_ID")["SEQUENCE_NUMBER"].apply(lambda s: s.duplicated().any()).any()

    # Procedure dates within encounter window
    if len(pr_df) > 0:
        m = pr_df.merge(enc[["RECORD_ID", "ADMISSION_DATE", "DISCHARGE_DATE", "RV_VENT_ON_ADMIT_DATE"]], on="RECORD_ID", how="left")
        proc_dt = pd.to_datetime(m["PROCEDURE_DATE"])
        adm_dt = pd.to_datetime(m["ADMISSION_DATE"])
        dis_dt = pd.to_datetime(m["DISCHARGE_DATE"])
        assert (proc_dt >= adm_dt).all() and (proc_dt <= dis_dt).all()

        # Vent flag implies ventilation procedure
        must_vent_ids = set(enc.loc[enc["RV_VENT_ON_ADMIT_DATE"].astype(str) == "1", "RECORD_ID"].astype(str))
        vent_ids = set(pr_df.loc[pr_df["PROCEDURE_CODE"].isin(list(VENT_CODES)), "RECORD_ID"].astype(str))
        assert must_vent_ids.issubset(vent_ids)

        # And ventilation date on admission date for those flagged encounters
        vent_m = pr_df[pr_df["PROCEDURE_CODE"].isin(list(VENT_CODES))].merge(
            enc[["RECORD_ID", "ADMISSION_DATE"]], on="RECORD_ID", how="left"
        )
        # for each flagged encounter, at least one vent row matches admit date
        for rid in must_vent_ids:
            rows = vent_m[vent_m["RECORD_ID"].astype(str) == rid]
            assert (rows["PROCEDURE_DATE"].astype(str) == rows["ADMISSION_DATE"].astype(str)).any()

    # Write outputs
    dx_df.to_csv(out_dx_csv, index=False)
    pr_df.to_csv(out_pr_csv, index=False)

    print(f"Wrote: {out_dx_csv} ({len(dx_df)} rows)")
    print(f"Wrote: {out_pr_csv} ({len(pr_df)} rows)")
    print(f"Encounters: {len(enc)} | avg dx/enc: {dx_counts.mean():.2f} | avg pr/enc: {pr_df.groupby('RECORD_ID').size().reindex(enc['RECORD_ID'], fill_value=0).mean() if len(pr_df)>0 else 0:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encounters", default="encounters_synth.csv", help="Path to encounters_synth.csv")
    ap.add_argument("--out_dx", default="diagnoses_synth.csv", help="Output diagnoses CSV")
    ap.add_argument("--out_pr", default="procedures_synth.csv", help="Output procedures CSV")
    args = ap.parse_args()

    generate(args.encounters, args.out_dx, args.out_pr)
