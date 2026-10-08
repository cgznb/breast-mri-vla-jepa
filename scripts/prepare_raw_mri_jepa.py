#!/usr/bin/env python3
"""Join an explicit private MRI index/split with the official I-SPY2 table.

This reads already prepared source-grid images; it does not derive a crop from
future lesions, infer scan times, or manufacture missing interval treatments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mri_vla_jepa.arm import CLINICAL_FEATURES, ARM_SEMANTICS, load_official_clinical, normalize_patient_key, make_arm_record, stage_index
from mri_vla_jepa.constants import PHASES
from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.data import SCHEMA, RawMRIStore


def private_output(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path
    if not relative.parts or relative.parts[0] not in {"data", "runs", "artifacts", "experiments", "checkpoints", "results", "weights"}:
        raise ValueError("Patient manifests must be outside the repository or in a gitignored data/artifacts/runs directory")
    return path


def source_metadata(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare(index_path, split_path, clinical_path, output, *, arm_known_at_stage,
            arm_semantics, clinical_known_at_stage, image_shape=(32, 128, 128), allow_synthetic=False):
    stage_index(arm_known_at_stage)
    stage_index(clinical_known_at_stage)
    index_path = Path(index_path).resolve()
    index = read_json(index_path)
    if index.get("schema") != "responsewm_raw_mri_index_v1":
        raise ValueError("MRI index requires responsewm_raw_mri_index_v1")
    splits = read_json(split_path)
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Explicit patient train/val/test lists are required; empty test is allowed for development")
    membership = {}
    for split, keys in splits.items():
        for key in keys:
            key = normalize_patient_key(key)
            if key in membership:
                raise ValueError("Patient split overlap")
            membership[key] = split
    rows = load_official_clinical(clinical_path)
    patients, seen = [], set()
    for original in index.get("patients", []):
        key = normalize_patient_key(original["patient_key"])
        if key in seen:
            raise ValueError("Duplicate patient in MRI index")
        seen.add(key)
        if key not in membership or key not in rows:
            raise ValueError("MRI index patient is missing from the split or official clinical table")
        row = rows[key]
        # Only runtime contract fields enter newly generated private manifests.
        patient = {"patient_key": original["patient_key"],
                   "geometry": {name: original.get("geometry", {}).get(name) for name in
                                ("source_only", "source_stage", "roi_source", "orientation", "resampling", "grid_id")},
                   "visits": [None if visit is None else
                              {name: visit[name] for name in ("stage", "available_at", "grid_id", "image", "phases")
                               if name in visit} for visit in original["visits"]]}
        patient["split"] = membership[key]
        values = row["clinical_values"]
        patient["clinical"] = {"values": values,
                               "known_at": [clinical_known_at_stage if v is not None else None for v in values]}
        patient["arm"] = make_arm_record(row["arm"], arm_known_at_stage, arm_semantics)
        patient["target"] = {"pcr": row["pcr"], "label_source": "official_ispy2_clinical:pCR"}
        for visit in patient["visits"]:
            if visit is None:
                continue
            if "image" in visit:
                p = Path(visit["image"])
                visit["image"] = str(p.resolve() if p.is_absolute() else (index_path.parent / p).resolve())
            if "phases" in visit:
                visit["phases"] = [str(Path(p).resolve() if Path(p).is_absolute() else (index_path.parent / p).resolve())
                                   for p in visit["phases"]]
        patients.append(patient)
    if seen != set(membership):
        raise ValueError("Split lists must exactly cover the MRI index cohort")
    manifest = {"schema": SCHEMA, "time_basis": "stage_index", "canonical_stages": [0, 1, 2, 3],
                "phase_order": PHASES, "clinical_features": list(CLINICAL_FEATURES), "synthetic": index.get("synthetic", False),
                "image_preprocessing": {"mode": "raw_t0_zscore"},
                "patients": patients, "provenance": {
                    "mri_index": source_metadata(index_path), "split": source_metadata(split_path),
                    "clinical_table": source_metadata(clinical_path), "arm_semantics": arm_semantics,
                    "arm_known_at_stage": arm_known_at_stage, "clinical_known_at_stage": clinical_known_at_stage,
                    "geometry": "Source-grid declarations checked for consistency; original ROI/date truth requires source audit",
                    "treatment": "Official patient-level Arm only; no dose, cycles or interval changes inferred"}}
    output = private_output(output)
    # Validate in the private destination; do not leave a malformed manifest.
    temporary = output.with_name(output.name + ".validation.json")
    write_json(temporary, manifest)
    temporary.chmod(0o600)
    try:
        store = RawMRIStore(temporary, image_shape=image_shape, allow_synthetic=allow_synthetic)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(output, manifest)
    output.chmod(0o600)
    return {"patients": len(patients), "by_split": {s: len(v) for s, v in store.by_split.items()},
            "arm_categories": 13, "clinical_features": 4, "arm_semantics": arm_semantics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-index", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm-known-at-stage", required=True, type=int, choices=range(4))
    parser.add_argument("--arm-semantics", required=True, choices=ARM_SEMANTICS)
    parser.add_argument("--clinical-known-at-stage", required=True, type=int, choices=range(4))
    parser.add_argument("--image-shape", nargs=3, type=int, default=(32, 128, 128), metavar=("D", "H", "W"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    result = prepare(args.mri_index, args.split, args.clinical, args.output,
                     arm_known_at_stage=args.arm_known_at_stage, arm_semantics=args.arm_semantics,
                     clinical_known_at_stage=args.clinical_known_at_stage, image_shape=tuple(args.image_shape),
                     allow_synthetic=args.allow_synthetic)
    import json
    print(json.dumps(result, indent=2))  # Cohort summaries only, never patient rows.


if __name__ == "__main__":
    main()
