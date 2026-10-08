"""Exact I-SPY2 Arm labels and explicit availability-aware clinical helpers.

The released spreadsheet supplies an assigned arm, not dose, interval treatment,
or an allocation timestamp. Callers must state the meaning and known-at stage.
"""
from __future__ import annotations

import math
import re
from pathlib import Path


OFFICIAL_ARMS = (
    "Paclitaxel",
    "Paclitaxel + ABT 888 + Carboplatin",
    "Paclitaxel + AMG 386",
    "Paclitaxel + AMG 386 + Trastuzumab",
    "Paclitaxel + Ganetespib",
    "Paclitaxel + Ganitumab",
    "Paclitaxel + MK-2206",
    "Paclitaxel + MK-2206 + Trastuzumab",
    "Paclitaxel + Neratinib",
    "Paclitaxel + Pembrolizumab",
    "Paclitaxel + Pertuzumab + Trastuzumab",
    "Paclitaxel + Trastuzumab",
    "T-DM1 + Pertuzumab",
)
ARM_TO_ID = {name: index + 1 for index, name in enumerate(OFFICIAL_ARMS)}
ARM_SEMANTICS = ("prospective_verified", "assigned_arm_scenario")
CLINICAL_FEATURES = (
    "age_at_screening", "hr_positive", "her2_positive", "mammaprint_binary",
)


def stage_index(value):
    if isinstance(value, str) and value in {"T0", "T1", "T2", "T3"}:
        return int(value[1])
    if type(value) is not int or value not in range(4):
        raise ValueError("Known-at stage must be an explicit canonical integer 0..3")
    return value


def normalize_patient_key(value):
    """Match numeric official IDs to an existing ISPY2-prefixed key privately."""
    text = str(value).strip()
    if text.startswith("ISPY2-"):
        text = text[6:]
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not text or text.lower() in {"nan", "none"}:
        raise ValueError("Patient key is missing")
    return text


def arm_id(label):
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return 0
    if not isinstance(label, str) or label.strip() not in ARM_TO_ID:
        raise ValueError("Arm must exactly match one of the 13 official labels")
    return ARM_TO_ID[label.strip()]


def validate_arm_record(record):
    if not isinstance(record, dict) or set(record) != {"label", "known_at_stage", "semantics"}:
        raise ValueError("Arm requires label, explicit known_at_stage and semantics")
    if record["semantics"] not in ARM_SEMANTICS:
        raise ValueError("Declare prospective_verified or assigned_arm_scenario Arm semantics")
    identifier = arm_id(record["label"])
    when = record["known_at_stage"]
    if identifier == 0:
        if when is not None:
            raise ValueError("Unknown Arm must have null known-at stage")
    else:
        stage_index(when)
    return identifier


def visible_arm(record, landmark):
    identifier = validate_arm_record(record)
    visible = identifier > 0 and record["known_at_stage"] <= stage_index(landmark)
    return (identifier if visible else 0, visible,
            record["known_at_stage"] if visible else -1)


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Clinical fields must be numeric or missing") from exc
    return number if math.isfinite(number) else None


def load_official_clinical(path):
    """Read locally; return private keyed records without printing patient data."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Clinical preparation requires the optional pandas/openpyxl data dependencies") from exc
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    required = {"Patient_ID", "Arm", "HR", "HER2", "MP", "pCR", "Age_at_Screening"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Official clinical table is missing columns: {sorted(required - set(frame.columns))}")
    rows = {}
    for _, row in frame.iterrows():
        key = normalize_patient_key(row["Patient_ID"])
        if key in rows:
            raise ValueError("Clinical table has duplicate normalized patient keys")
        label = None if pd.isna(row["Arm"]) else str(row["Arm"]).strip()
        arm_id(label)
        age = _number(row["Age_at_Screening"])
        if age is not None and not 0 < age < 120:
            raise ValueError("Age at screening must be within (0,120) or missing")
        values = [age] + [_number(row[name]) for name in ("HR", "HER2", "MP")]
        if any(v is not None and v not in (0, 1) for v in values[1:]):
            raise ValueError("HR, HER2 and MP must use official binary values or missing")
        pcr = _number(row["pCR"])
        if pcr is not None and pcr not in (0, 1):
            raise ValueError("Final pCR must be binary or missing")
        rows[key] = {"arm": label, "clinical_values": values, "pcr": pcr}
    return rows


def make_arm_record(label, known_at_stage, semantics):
    record = {"label": label, "known_at_stage": known_at_stage if label is not None else None,
              "semantics": semantics}
    validate_arm_record(record)
    return record
