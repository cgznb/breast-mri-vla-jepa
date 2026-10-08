#!/usr/bin/env python3
"""Audit existing three-phase ROI32 caches and build a private raw-MRI manifest.

The arrays already use a training-set intensity scale. This script never
rewrites them, reads VQ latents, re-crops images, or derives treatment schedules.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from mri_vla_jepa.arm import CLINICAL_FEATURES, load_official_clinical, make_arm_record, normalize_patient_key
from mri_vla_jepa.constants import PHASES
from mri_vla_jepa.data import ROI32_SHAPE, SCHEMA, RawMRIStore, canonical_image_normalization
from mri_vla_jepa.io import read_json, write_json


SPLITS = ("train", "val", "test")


def _resolve(base, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Inventory asset paths must be nonempty strings")
    path = Path(name)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _stage(visit_id, patient_id):
    if not isinstance(visit_id, str):
        raise ValueError("Inventory visit_id must end in _T0.._T3 or :T0..:T3")
    match = re.fullmatch(r"(.+)[_:]T([0-3])", visit_id)
    if match is None or normalize_patient_key(match[1]) != normalize_patient_key(patient_id):
        raise ValueError("Inventory visit_id must identify its patient and canonical stage")
    return int(match[2])


def _private_path(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path
    if not relative.parts or relative.parts[0] not in {
        "data", "runs", "artifacts", "experiments", "checkpoints", "results", "weights",
    }:
        raise ValueError("Private patient manifests must be outside the repository or in an ignored output directory")
    return path


def _private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".roi32_", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _audit_archive(path):
    if path.suffix != ".npz" or not path.is_file():
        raise ValueError("Each registered ROI32 asset must be an existing .npz file")
    with np.load(path, allow_pickle=False) as archive:
        if not {"image", "support", "roi"}.issubset(archive.files):
            raise ValueError("ROI32 preflight requires image, support and roi arrays")
        image, support, roi = archive["image"], archive["support"], archive["roi"]
    if image.shape != ROI32_SHAPE or image.dtype != np.float32:
        raise ValueError("ROI32 image must be float32 with exact shape [3,32,128,128]")
    if not np.isfinite(image).all():
        raise ValueError("ROI32 image must contain only finite values")
    if support.shape != ROI32_SHAPE or support.dtype != np.bool_:
        raise ValueError("ROI32 support must be bool with exact shape [3,32,128,128]")
    if roi.shape != (1, *ROI32_SHAPE[1:]) or roi.dtype != np.bool_:
        raise ValueError("ROI32 roi must be bool with exact shape [1,32,128,128]")
    if np.any(image[~support] != 0):
        raise ValueError("ROI32 acquisition padding outside support must preserve zero background")
    return {"zero_image_values": int(np.count_nonzero(image == 0)),
            "outside_support_values": int(np.count_nonzero(~support)),
            "roi_voxels": int(roi.sum()), "empty_roi_arrays": int(not roi.any()),
            "image_min": float(image.min()), "image_max": float(image.max())}


def _source_geometry(value):
    if not isinstance(value, dict) or value.get("shape_zyx") != list(ROI32_SHAPE[1:]):
        raise ValueError("Pair source geometry must declare the fixed ROI32 shape")
    spacing = value.get("spacing_zyx_mm", [])
    affine = np.asarray(value.get("crop_affine_ras", []), dtype=np.float64)
    if (not isinstance(spacing, list) or len(spacing) != 3
            or any(type(v) not in {int, float} or not math.isfinite(v) or v <= 0 for v in spacing)
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or not np.allclose(affine[:3, :3], np.diag([spacing[2], spacing[1], spacing[0]]), atol=1e-6)):
        raise ValueError("Pair source geometry requires finite RAS crop affine and positive spacing")
    return {key: copy.deepcopy(value[key]) for key in ("shape_zyx", "spacing_zyx_mm", "crop_affine_ras")}


def prepare(inventory_path, clinical_path, output, report):
    inventory_path, clinical_path = Path(inventory_path).resolve(), Path(clinical_path).resolve()
    inventory = read_json(inventory_path)
    if (inventory.get("schema") != "trjepa_three_phase_source_grid_v1"
            or inventory.get("phase_order") != PHASES
            or inventory.get("spatial_policy") != "source_defined_grid"):
        raise ValueError("Expected registered three-phase source-grid inventory and fixed phase order")
    policy = inventory.get("roi_policy", "")
    if (not isinstance(policy, str) or "T0" not in policy
            or "source" not in policy.lower() or "fixed" not in policy.lower()):
        raise ValueError("Inventory must declare a source-available T0 ROI policy")
    registration_policy = inventory.get("registration_policy")
    if not isinstance(registration_policy, str) or not registration_policy.strip():
        raise ValueError("Inventory must declare the source-grid registration policy")
    normalization = canonical_image_normalization(inventory.get("image_normalization"))
    visits = inventory.get("visits")
    if not isinstance(visits, list) or not visits:
        raise ValueError("Inventory must contain existing visits")
    rows = load_official_clinical(clinical_path)
    patients, assets, file_identities, aliases = {}, {}, set(), {}
    audit = Counter()
    image_min, image_max = math.inf, -math.inf
    for record in visits:
        patient_id = record.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("Inventory patient_id must be a nonempty string")
        key = normalize_patient_key(patient_id)
        if aliases.setdefault(key, patient_id) != patient_id:
            raise ValueError("Inventory contains duplicate patient aliases")
        split, stage = record.get("split"), _stage(record.get("visit_id"), patient_id)
        if split not in SPLITS:
            raise ValueError("Inventory split must be train, val or test")
        if key not in rows:
            raise ValueError("An inventory patient is absent from the official clinical table")
        if key not in patients:
            row = rows[key]
            grid_id = f"registered_roi32:{patient_id}"
            patients[key] = {"patient_key": patient_id, "split": split,
                             "clinical": {"values": copy.deepcopy(row["clinical_values"]),
                                          "known_at": [0 if v is not None else None for v in row["clinical_values"]]},
                             "arm": make_arm_record(row["arm"], 0, "assigned_arm_scenario"),
                             "geometry": {"source_stage": 0, "source_only": True, "grid_id": grid_id,
                                          "orientation": "RAS", "roi_source": "T0", "resampling": "source_grid"},
                             "visits": [None] * 4,
                             "target": {"pcr": row["pcr"], "label_source": "official_ispy2_clinical:pCR"}}
        patient = patients[key]
        if patient["split"] != split:
            raise ValueError("A patient overlaps inventory splits")
        if patient["visits"][stage] is not None:
            raise ValueError("Inventory contains a duplicate patient stage")
        path = _resolve(inventory_path.parent, record.get("file"))
        if str(path) in assets:
            raise ValueError("Inventory contains a duplicate MRI asset")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in file_identities:
            raise ValueError("Different inventory paths alias the same MRI asset")
        file_identities.add(identity)
        assets[str(path)] = (key, stage)
        patient["visits"][stage] = {"stage": stage, "available_at": stage,
                                     "grid_id": patient["geometry"]["grid_id"], "image": str(path)}
        checked = _audit_archive(path)
        image_min, image_max = min(image_min, checked.pop("image_min")), max(image_max, checked.pop("image_max"))
        audit.update(checked)
        audit["archives_checked"] += 1
    if any(patient["visits"][0] is None for patient in patients.values()):
        raise ValueError("Every inventory patient requires an actual T0; missing future slots remain null")

    pair_checks, source_grids, pair_ids = Counter(), {}, set()
    for pair in inventory.get("pairs", []):
        key = normalize_patient_key(pair.get("patient_id"))
        if key not in patients or pair.get("split") != patients[key]["split"]:
            raise ValueError("Pair patient or split conflicts with inventory visits")
        if "pair_id" in pair:
            if pair["pair_id"] in pair_ids:
                raise ValueError("Inventory contains a duplicate pair_id")
            pair_ids.add(pair["pair_id"])
        bound = []
        for field in ("source_file", "target_file"):
            path = str(_resolve(inventory_path.parent, pair.get(field)))
            if path not in assets or assets[path][0] != key:
                raise ValueError("Pair assets must belong to its patient's canonical visits")
            bound.append(assets[path][1])
        source_stage, target_stage = bound
        if source_stage >= target_stage or pair.get("transition") != f"T{source_stage}->T{target_stage}":
            raise ValueError("Pair transition conflicts with canonical visit stages")
        row = rows[key]
        labels = pair.get("labels", {})
        if "pcr" in labels:
            if labels["pcr"] != row["pcr"]:
                raise ValueError("Inventory pair pCR conflicts with the official clinical table")
            pair_checks["pcr_records_checked"] += 1
        treatment = pair.get("conditions", {}).get("treatment", {})
        if "regimen" in treatment:
            if treatment["regimen"] != row["arm"]:
                raise ValueError("Inventory pair Arm conflicts with the official clinical table")
            pair_checks["arm_records_checked"] += 1
        if "source_geometry" in pair:
            geometry = _source_geometry(pair["source_geometry"])
            if key in source_grids and source_grids[key] != geometry:
                raise ValueError("A patient's visits must preserve the same T0 crop geometry")
            source_grids[key] = geometry
        pair_checks["pairs_checked"] += 1
    if set(source_grids) != set(patients):
        raise ValueError("Every patient requires recorded RAS T0 crop geometry in the inventory pairs")
    for key, geometry in source_grids.items():
        patients[key]["geometry"].update(geometry)
    ordered = [patients[key] for key in sorted(patients)]
    partitions = {split: [p["patient_key"] for p in ordered if p["split"] == split] for split in SPLITS}
    manifest = {"schema": SCHEMA, "time_basis": "stage_index", "canonical_stages": [0, 1, 2, 3],
                "phase_order": PHASES, "clinical_features": list(CLINICAL_FEATURES), "synthetic": False,
                "image_preprocessing": {"mode": "preprocessed_roi32"}, "image_normalization": normalization,
                "patients": ordered, "provenance": {
                    "inventory": _identity(inventory_path), "clinical_table": _identity(clinical_path),
                    "roi_policy": policy, "registration_policy": registration_policy,
                    "arm_semantics": "assigned_arm_scenario", "arm_known_at_stage": 0,
                    "clinical_known_at_stage": 0,
                    "availability_assumption": "Assigned Arm and four baseline clinical fields assumed available at T0",
                    "treatment": "Official patient-level assigned Arm; no doses, cycles or interval schedules inferred",
                    "data_policy": "Existing image arrays passed through unchanged; support used only for preflight"}}
    output = _private_path(output)
    split_output = output.with_name(output.stem + ".splits.json")
    report = Path(report).resolve()
    if (len({output, split_output, report}) != 3
            or any(path in {inventory_path, clinical_path} or str(path) in assets for path in (output, split_output, report))):
        raise ValueError("Output, split and report paths must be distinct from each other and all inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".roi32_validation_", suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, allow_nan=False)
    try:
        store = RawMRIStore(temporary, image_shape=ROI32_SHAPE[1:])
    finally:
        temporary.unlink(missing_ok=True)
    summary = {"schema": "registered_roi32_preflight_v1", "status": "passed", "patients": len(ordered),
               "by_split": {split: len(partitions[split]) for split in SPLITS},
               "visits_by_stage": {f"T{stage}": sum(p["visits"][stage] is not None for p in ordered) for stage in range(4)},
               "visit_patterns": dict(sorted(Counter("/".join(f"T{s}" for s, v in enumerate(p["visits"]) if v is not None)
                                                     for p in ordered).items())),
               "clinical_missing": {feature: sum(p["clinical"]["values"][i] is None for p in ordered)
                                    for i, feature in enumerate(CLINICAL_FEATURES)},
               "pcr_counts": {"negative": sum(p["target"]["pcr"] == 0 for p in ordered),
                              "positive": sum(p["target"]["pcr"] == 1 for p in ordered),
                              "missing": sum(p["target"]["pcr"] is None for p in ordered)},
               "array_audit": {**dict(audit), "image_min": image_min, "image_max": image_max},
               "clinical_cross_checks": dict(pair_checks), "patients_with_crop_geometry": len(source_grids),
               "image_preprocessing": copy.deepcopy(store.image_preprocessing), "image_normalization": normalization,
               "arm_semantics": "assigned_arm_scenario", "arm_known_at_stage": 0,
               "development_validation": "Validation split is for model development; no independent test result",
               "normalization_audit_limit": "The original foreground mask is absent; declared training scale checked, not recomputed",
               "support_meaning": "Acquisition coverage, not original intensity-normalization foreground or a loss mask",
               "latent_policy": "VQ latent arrays not loaded"}
    _private_json(output, manifest)
    _private_json(split_output, partitions)
    write_json(report, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="Existing registered three-phase ROI32 inventory JSON")
    parser.add_argument("--clinical", type=Path, required=True,
                        help="Official clinical CSV or XLSX containing Arm and pCR labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inventory, args.clinical, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
