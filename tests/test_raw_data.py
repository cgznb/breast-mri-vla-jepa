import copy

import numpy as np
import pytest
import torch

from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic


@pytest.fixture
def raw(tmp_path):
    path = make_raw_synthetic(tmp_path / "fake", n_train=4, n_val=2, image_shape=(4, 8, 8))
    return path, RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True)


def test_fixed_slots_missing_targets_and_queries(raw):
    path, _ = raw
    value = read_json(path)
    value["patients"][0]["visits"][1] = None
    value["patients"][0]["visits"][2] = None
    value["patients"][0]["target"]["pcr"] = None
    write_json(path, value)
    store = RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True)
    inp, sup = store.batch([(0, 0)])
    assert inp.images.shape == (1, 4, 3, 4, 8, 8)
    assert inp.observed_mask.tolist() == [[True, False, False, False]]
    assert inp.query_mask.tolist() == [[False, True, True, True]]
    assert sup.future_mask.tolist() == [[False, False, False, True]]
    assert not sup.label_mask.any()
    assert store.allowed_landmarks(0) == [0, 3]
    later, _ = store.batch([(0, 2)], supervised=False)
    assert later.observed_mask.tolist() == [[True, False, False, False]]
    assert later.query_mask.tolist() == [[False, False, False, True]]


def test_no_future_file_or_label_reads(raw):
    _, store = raw
    original, _ = store.batch([(0, 0)], supervised=False)
    for visit in store.patients[0]["visits"][1:]:
        store.resolve(visit["image"]).unlink()
    class PoisonTarget(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Inference read a pCR label")
    store.patients[0]["target"] = PoisonTarget()
    after, supervision = store.batch([(0, 0)], supervised=False)
    assert supervision is None
    for field in vars(original):
        assert torch.equal(getattr(original, field), getattr(after, field))


def test_future_intensity_and_missingness_cannot_change_source(raw):
    _, store = raw
    initial, _ = store.batch([(0, 0)])
    future = store.resolve(store.patients[0]["visits"][3]["image"])
    np.save(future, np.load(future) * 1000 + 500)
    poisoned, _ = store.batch([(0, 0)])
    assert torch.equal(initial.images, poisoned.images)
    store.patients[0]["visits"][3] = None
    missing, _ = store.batch([(0, 0)])
    assert torch.equal(initial.query_mask, missing.query_mask)
    assert torch.equal(initial.clinical, missing.clinical)


def test_clinical_training_statistics_and_visibility(raw):
    path, store = raw
    state = store.normalization_state()
    assert state["clinical_mean"][0] == pytest.approx(36.5)
    assert state["clinical_mean"][1:] == [0, 0, 0]
    value = read_json(path)
    value["patients"][4]["clinical"]["values"][0] = 100
    value["patients"][0]["clinical"]["values"][2] = None
    value["patients"][0]["clinical"]["known_at"][2] = None
    value["patients"][0]["clinical"]["known_at"][1] = 2
    write_json(path, value)
    fresh = RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True)
    assert fresh.normalization_state()["clinical_mean"][0] == state["clinical_mean"][0]
    early, _ = fresh.batch([(0, 0)], supervised=False)
    later, _ = fresh.batch([(0, 2)], supervised=False)
    assert early.clinical_mask.tolist() == [[True, False, False, True]]
    assert not early.clinical[0, 1:3].any()
    assert later.clinical_mask.tolist() == [[True, True, False, True]]


def test_arm_visibility(raw):
    _, store = raw
    store.patients[0]["arm"]["known_at_stage"] = 1
    early, _ = store.batch([(0, 0)], supervised=False)
    late, _ = store.batch([(0, 1)], supervised=False)
    assert early.arm_id.item() == 0 and not early.arm_mask.item()
    assert early.arm_known_at.item() == -1
    assert late.arm_id.item() == 1 and late.arm_mask.item()


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["geometry"].update(roi_source="T3"), "source-only"),
    (lambda p: p["geometry"].update(source_only=False), "source-only"),
    (lambda p: p["visits"][1].update(grid_id="future_crop"), "source grid"),
    (lambda p: p["visits"][1].update(available_at=0), "chronology"),
])
def test_source_geometry_and_stage_protections(raw, mutation, match):
    path, _ = raw
    value = read_json(path)
    mutation(value["patients"][0])
    write_json(path, value)
    with pytest.raises(ValueError, match=match):
        RawMRIStore(path, allow_synthetic=True)


def test_patient_and_asset_overlap(raw):
    path, _ = raw
    value = read_json(path)
    patient = copy.deepcopy(value["patients"][0]); patient["split"] = "test"
    value["patients"].append(patient)
    write_json(path, value)
    with pytest.raises(ValueError, match="overlap"):
        RawMRIStore(path, allow_synthetic=True)
    value["patients"].pop()
    value["patients"][1]["visits"][0]["image"] = value["patients"][0]["visits"][0]["image"]
    write_json(path, value)
    with pytest.raises(ValueError, match="asset overlaps"):
        RawMRIStore(path, allow_synthetic=True)


def test_normalized_patient_alias_overlap_without_checksums(raw):
    path, _ = raw
    value = read_json(path)
    value["patients"][0]["patient_key"] = "ISPY2-100"
    write_json(path, value)
    store = RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True)
    assert store.patients[0]["patient_key"] == "ISPY2-100"
    assert "train_patient_hashes" not in store.normalization_state()
    assert store.normalization_state()["clinical_counts"] == [4, 4, 4, 4]
    value["patients"][4]["patient_key"] = "100"
    write_json(path, value)
    with pytest.raises(ValueError, match="patient overlap"):
        RawMRIStore(path, allow_synthetic=True)


def test_normalization_restore_inference_only(raw):
    path, store = raw
    state = store.normalization_state()
    value = read_json(path)
    for p in value["patients"]:
        p["split"] = "test"
    write_json(path, value)
    with pytest.raises(ValueError, match="inference-only"):
        RawMRIStore(path, allow_synthetic=True)
    restored = RawMRIStore(path, allow_synthetic=True, normalization_state=state, image_shape=(4, 8, 8))
    inp, _ = restored.batch([(0, 0)], supervised=False)
    assert inp.clinical[0, 0].item() == pytest.approx((35 - 36.5) / np.std([35, 36, 37, 38]))
    with pytest.raises(ValueError, match="training clinical"):
        RawMRIStore(path, allow_synthetic=True, normalization_state={**state, "fit_split": "test"})


def test_signature_detects_manifest_and_scan_changes(raw):
    path, store = raw
    before = store.signature()
    image = store.resolve(store.patients[0]["visits"][0]["image"])
    np.save(image, np.load(image) + 1)
    assert before != store.signature()


def test_raw_preparation_cli_with_private_fake_cohort(raw, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    path, store = raw
    patients = copy.deepcopy(store.patients)
    for patient in patients:
        for visit in patient["visits"]:
            visit["image"] = str(store.resolve(visit["image"]))
        for key in ("clinical", "arm", "target", "split"):
            patient.pop(key)
    index = tmp_path / "index.json"
    split = tmp_path / "split.json"
    clinical = tmp_path / "clinical.csv"
    output = tmp_path / "private_raw_manifest.json"
    write_json(index, {"schema": "responsewm_raw_mri_index_v1", "synthetic": True, "patients": patients})
    write_json(split, {"train": [p["patient_key"] for p in store.patients[:4]],
                       "val": [p["patient_key"] for p in store.patients[4:]], "test": []})
    clinical.write_text("Patient_ID,Arm,HR,HER2,MP,pCR,Age_at_Screening\n" + "".join(
        f"{p['patient_key']},Paclitaxel,{i % 2},0,1,{i % 2},{35+i}\n" for i, p in enumerate(patients)))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_raw_mri_jepa.py"),
               "--mri-index", str(index), "--split", str(split), "--clinical", str(clinical),
               "--output", str(output), "--arm-known-at-stage", "1", "--arm-semantics", "assigned_arm_scenario",
               "--clinical-known-at-stage", "0", "--image-shape", "4", "8", "8", "--allow-synthetic"]
    # Run the script against this checkout, independent of any installed wheel.
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    run = subprocess.run(command, capture_output=True, text=True, check=True, env=environment)
    assert "synthetic_0" not in run.stdout
    prepared = RawMRIStore(output, image_shape=(4, 8, 8), allow_synthetic=True)
    inp, _ = prepared.batch([(0, 0)], supervised=False)
    assert inp.arm_id.item() == 0
    assert prepared.manifest["synthetic"] is True


def test_nifti_orientation_and_source_grid_resampling(raw, tmp_path):
    nib = pytest.importorskip("nibabel")
    path, _ = raw
    value = read_json(path)
    base = np.arange(4 * 8 * 8, dtype=np.float32).reshape(8, 8, 4)
    affine = np.diag([-2., 2., 3., 1.]); affine[0, 3] = 14
    for stage in range(4):
        phases = []
        for phase in range(3):
            file = tmp_path / f"nifti_T{stage}_p{phase}.nii.gz"
            nib.save(nib.Nifti1Image(base + phase + stage, affine), file)
            phases.append(str(file))
        visit = value["patients"][0]["visits"][stage]
        visit.pop("image"); visit["phases"] = phases
    write_json(path, value)
    store = RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True)
    inp, sup = store.batch([(0, 0)])
    expected = base[::-1].transpose(2, 1, 0)
    expected = (expected - expected.mean()) / expected.std()
    assert np.allclose(inp.images[0, 0, 0].numpy(), expected, atol=1e-5)
    assert sup.future_mask.sum() == 3
    # A different target affine is mapped back to the baseline reference grid.
    target_path = value["patients"][0]["visits"][3]["phases"][0]
    shifted = affine.copy(); shifted[1, 3] += 2
    nib.save(nib.Nifti1Image(base + 3, shifted), target_path)
    second, second_sup = store.batch([(0, 0)])
    assert torch.equal(inp.images, second.images)
    assert not torch.equal(sup.future[0, 3, 0], second_sup.future[0, 3, 0])
