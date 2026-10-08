"""Registered ROI arrays retain their coordinates and source-only boundary."""
import copy
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from mri_vla_jepa.constants import PHASES
from mri_vla_jepa.data import ROI32_SHAPE, RawMRIStore, make_raw_synthetic
from mri_vla_jepa.data_pipeline import file_identity, prepare_image_cache
from mri_vla_jepa.io import read_json, write_json


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_registered_roi32.py"
_SPEC = importlib.util.spec_from_file_location("prepare_registered_roi32", _SCRIPT)
_PREPARER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREPARER)


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cache(tmp_path):
    inventory, clinical = tmp_path / "inventory.json", tmp_path / "clinical.csv"
    visits, pairs = [], []
    views = tmp_path / "views"
    views.mkdir()
    for key, split, stages in (("ISPY2-101", "train", (0, 2, 3)), ("ISPY2-102", "train", (0, 1)),
                               ("ISPY2-103", "val", (0, 3))):
        for stage in stages:
            image = np.zeros(ROI32_SHAPE, dtype=np.float32)
            support = np.ones(ROI32_SHAPE, dtype=bool)
            support[:, :2] = False
            for phase in range(3):
                image[phase, 12:20, 40:60, 40:60] = .25 * (phase + 1) + stage
            roi = np.zeros((1, *ROI32_SHAPE[1:]), dtype=bool)
            roi[:, 14:18, 44:50, 44:50] = True
            file = f"views/{key}_T{stage}.npz"
            # An object latent would fail if either the runtime or preflight read it.
            np.savez(tmp_path / file, image=image, support=support, roi=roi, latent=np.array(["unused"], dtype=object))
            visits.append({"patient_id": key, "visit_id": f"{key}:T{stage}", "split": split, "file": file})
        for source, target in zip(stages, stages[1:]):
            pairs.append({"patient_id": key, "split": split, "pair_id": f"{key}:T{source}->T{target}",
                          "source_file": f"views/{key}_T{source}.npz", "target_file": f"views/{key}_T{target}.npz",
                          "transition": f"T{source}->T{target}", "labels": {"pcr": int(key[-1]) % 2},
                          "conditions": {"treatment": {"regimen": "Paclitaxel"}},
                          "source_geometry": {"shape_zyx": [32, 128, 128], "spacing_zyx_mm": [2., .7, .7],
                                              "crop_affine_ras": [[.7, 0, 0, 1], [0, .7, 0, 2],
                                                                  [0, 0, 2, 3], [0, 0, 0, 1]]}})
    value = {"schema": "trjepa_three_phase_source_grid_v1", "phase_order": PHASES,
             "spatial_policy": "source_defined_grid", "roi_policy": "Fixed source-available T0 predicted ROI",
             "registration_policy": "Rigid fallback retained; pair spatial validity audited upstream",
             "image_normalization": {"mean": 201., "std": 252., "fit_split": "train",
                                     "scope": "unique_train_crop_nonzero_foreground", "background_policy": "preserve_zero",
                                     "already_normalized": True, "normalization_policy": "Foreground only; zero background retained",
                                     "source_sha256": "legacy_metadata_must_not_be_copied"},
             "visits": visits, "pairs": pairs, "provenance": {"sha256": "not_copied"}}
    write_json(inventory, value)
    clinical.write_text("Patient_ID,Arm,HR,HER2,MP,pCR,Age_at_Screening\n"
                        "101,Paclitaxel,1,0,1,1,40\n102,Paclitaxel,0,0,1,0,\n103,Paclitaxel,1,1,0,1,70\n")
    return {"inventory": inventory, "clinical": clinical, "manifest": tmp_path / "private_manifest.json",
            "report": tmp_path / "report.json"}


def _prepare(cache):
    return _PREPARER.prepare(cache["inventory"], cache["clinical"], cache["manifest"], cache["report"])


def test_preparation_passthrough_missing_slots_and_clinical_masks(cache, monkeypatch):
    report = _prepare(cache)
    assert report["by_split"] == {"train": 2, "val": 1, "test": 0}
    assert report["array_audit"]["archives_checked"] == 7
    assert report["clinical_missing"]["age_at_screening"] == 1
    store = RawMRIStore(cache["manifest"])
    monkeypatch.setattr("mri_vla_jepa.data.F.interpolate", lambda *args, **kwargs: pytest.fail("ROI32 was resized"))
    inp, sup = store.batch([(0, 2), (1, 0)])
    assert inp.images.shape == (2, 4, *ROI32_SHAPE)
    assert inp.observed_mask.tolist() == [[True, False, True, False], [True, False, False, False]]
    assert sup.future_mask.tolist() == [[False, False, False, True], [False, True, False, False]]
    for stage in (0, 2):
        with np.load(store.resolve(store.patients[0]["visits"][stage]["image"]), allow_pickle=False) as archive:
            assert np.array_equal(inp.images[0, stage].numpy(), archive["image"])
    with np.load(store.resolve(store.patients[0]["visits"][3]["image"]), allow_pickle=False) as archive:
        assert np.array_equal(sup.future[0, 3].numpy(), archive["image"])
    assert inp.images[:, :, :, :2].count_nonzero() == 0
    assert store.allowed_landmarks(0) == [0, 2, 3]
    assert not inp.clinical_mask[1, 0] and inp.clinical[1, 0] == 0
    assert inp.arm_mask.all() and inp.arm_known_at.tolist() == [0, 0]
    assert store.normalization_state()["clinical_mean"][0] == 40
    assert cache["manifest"].stat().st_mode & 0o777 == 0o600
    assert cache["manifest"].with_name("private_manifest.splits.json").stat().st_mode & 0o777 == 0o600


def test_roi32_inference_never_reads_future_archives_or_label(cache):
    _prepare(cache)
    store = RawMRIStore(cache["manifest"])
    before, _ = store.batch([(0, 0)], supervised=False)
    for visit in store.patients[0]["visits"][1:]:
        if visit is not None:
            store.resolve(visit["image"]).unlink()
    class PoisonTarget(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Inference read a label")
    store.patients[0]["target"] = PoisonTarget()
    after, supervision = store.batch([(0, 0)], supervised=False)
    assert supervision is None
    for key, value in vars(before).items():
        assert torch.equal(value, getattr(after, key))


@pytest.mark.parametrize("bad,match", [
    ("dtype", "float32"), ("shape", "exact shape"), ("nonfinite", "finite"),
])
def test_runtime_rejects_incompatible_npz(cache, bad, match):
    _prepare(cache)
    store = RawMRIStore(cache["manifest"])
    path = store.resolve(store.patients[0]["visits"][0]["image"])
    image = np.zeros(ROI32_SHAPE, dtype=np.float32)
    if bad == "dtype":
        image = image.astype(np.float64)
    elif bad == "shape":
        image = image[:, :-1]
    else:
        image.flat[0] = np.nan
    np.savez(path, image=image)
    with pytest.raises(ValueError, match=match):
        store.batch([(0, 0)], supervised=False)


def test_preprocessing_and_checkpoint_coordinates_must_match(cache):
    _prepare(cache)
    store = RawMRIStore(cache["manifest"])
    state = store.normalization_state()
    assert state["schema"] == "raw_mri_normalization_v2"
    with pytest.raises(ValueError, match="resizing is forbidden"):
        RawMRIStore(cache["manifest"], image_shape=(8, 16, 16))
    with pytest.raises(ValueError, match="image_preprocessing"):
        RawMRIStore(cache["manifest"], image_preprocessing={"mode": "raw_t0_zscore"})
    altered = copy.deepcopy(state)
    altered["image_normalization"]["mean"] += 1
    with pytest.raises(ValueError, match="image_normalization"):
        RawMRIStore(cache["manifest"], normalization_state=altered)
    value = read_json(cache["manifest"])
    value.pop("image_preprocessing")
    write_json(cache["manifest"], value)
    with pytest.raises(ValueError, match="preprocessed_roi32"):
        RawMRIStore(cache["manifest"])


def test_v1_normalization_is_read_only_compatible_for_legacy_raw_inputs(tmp_path):
    path = make_raw_synthetic(tmp_path / "raw", n_train=2, n_val=1, image_shape=(4, 8, 8))
    original = RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True)
    modern = original.normalization_state()
    legacy = {key: copy.deepcopy(modern[key]) for key in (
        "fit_split", "clinical_features", "clinical_mean", "clinical_std", "clinical_counts",
    )}
    legacy.update(schema="raw_mri_normalization_v1", train_patient_hashes=["legacy_private_identity"], intensity="legacy")
    restored = RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True, normalization_state=legacy)
    assert restored.normalization_state() == modern
    before, _ = original.batch([(0, 0)], supervised=False)
    after, _ = restored.batch([(0, 0)], supervised=False)
    assert torch.equal(before.images, after.images)


@pytest.mark.parametrize("change,match", [
    (lambda x: x["visits"].append(copy.deepcopy(x["visits"][0])), "duplicate patient stage"),
    (lambda x: x["visits"][1].update(split="val"), "overlaps inventory splits"),
    (lambda x: x["visits"][1].update(visit_id="ISPY2-101:T1", file=x["visits"][0]["file"]), "duplicate MRI asset"),
    (lambda x: x["visits"][1].update(patient_id="101", visit_id="101:T2"), "duplicate patient aliases"),
    (lambda x: x["pairs"][0]["labels"].update(pcr=0), "pCR conflicts"),
    (lambda x: x["pairs"][0]["conditions"]["treatment"].update(regimen="Paclitaxel + Neratinib"), "Arm conflicts"),
    (lambda x: x["pairs"][0].update(transition="T0->T1"), "transition conflicts"),
    (lambda x: x["pairs"][0].update(source_file=x["visits"][3]["file"]), "belong to its patient"),
    (lambda x: x["image_normalization"].update(fit_split="val"), "training scale"),
    (lambda x: x["image_normalization"].update(background_policy="zscore_background"), "zero background"),
    (lambda x: x.update(phase_order=list(reversed(PHASES))), "phase order"),
    (lambda x: x["pairs"][1]["source_geometry"]["crop_affine_ras"][0].__setitem__(3, 99), "same T0 crop geometry"),
])
def test_preparer_rejects_structural_and_clinical_conflicts(cache, change, match):
    value = read_json(cache["inventory"])
    change(value)
    write_json(cache["inventory"], value)
    with pytest.raises(ValueError, match=match):
        _prepare(cache)
    assert not cache["manifest"].exists() and not cache["report"].exists()


@pytest.mark.parametrize("bad,match", [("padding", "zero background"), ("support", "support must be bool"),
                                      ("roi", "roi must be bool")])
def test_preparer_audits_support_roi_and_zero_padding(cache, bad, match):
    value = read_json(cache["inventory"])
    path = cache["inventory"].parent / value["visits"][0]["file"]
    with np.load(path, allow_pickle=False) as archive:
        image, support, roi = archive["image"], archive["support"], archive["roi"]
    if bad == "padding":
        image.flat[0] = 1
    elif bad == "support":
        support = support.astype(np.uint8)
    else:
        roi = roi[:, :-1]
    np.savez(path, image=image, support=support, roi=roi)
    with pytest.raises(ValueError, match=match):
        _prepare(cache)


def test_new_outputs_do_not_copy_checksums_and_signature_binds_runtime_and_disk(cache):
    _prepare(cache)
    store = RawMRIStore(cache["manifest"])
    outputs = [read_json(cache["manifest"]), read_json(cache["report"]), store.normalization_state(), store.signature()]
    text = json.dumps(outputs)
    assert not re.search(r'"[^" ]*(?:sha256|hash|checksum|digest)[^" ]*"\s*:', text, re.IGNORECASE)
    assert "legacy_metadata_must_not_be_copied" not in text and "not_copied" not in text
    signature = store.signature()
    value = read_json(cache["manifest"])
    value["patients"][0]["clinical"]["values"][0] = 41
    write_json(cache["manifest"], value)
    changed = store.signature()
    assert changed["manifest"] == signature["manifest"]
    assert changed["manifest_on_disk"] != signature["manifest_on_disk"]
    store.patients[0]["clinical"]["values"][0] = 42
    assert store.signature()["manifest"] != signature["manifest"]


def test_prepare_cli_reports_only_aggregates(cache):
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(_SCRIPT), "--inventory", str(cache["inventory"]),
                             "--clinical", str(cache["clinical"]), "--output", str(cache["manifest"]),
                             "--report", str(cache["report"])], check=True, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(_SCRIPT.parents[1] / "src")})
    assert json.loads(result.stdout)["patients"] == 3
    assert "ISPY2-101" not in result.stdout
    assert "ISPY2-101" not in cache["report"].read_text()


def _image_cache(cache):
    _prepare(cache)
    original = RawMRIStore(cache["manifest"])
    directory = cache["manifest"].parent / "derived_images"
    report = prepare_image_cache(original, directory, workers=2)
    return original, RawMRIStore(cache["manifest"], image_cache="derived_images"), directory, report


def test_derived_image_cache_preserves_every_tensor_and_zero_background(cache, monkeypatch):
    original, cached, directory, report = _image_cache(cache)
    assert report["images"] == 7 and report["all_images_value_equal_to_npz"]
    assert report["zero_background_preserved"]
    tasks = [(0, 0), (0, 2), (1, 1), (2, 3)]
    expected = original.batch(tasks)
    real_load = np.load

    def no_npz(path, *args, **kwargs):
        assert Path(path).suffix == ".npy", "Cache path decoded the original NPZ"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", no_npz)
    actual = cached.batch(tasks)
    for reference, observed in zip(expected, actual):
        for name, tensor in vars(reference).items():
            assert torch.equal(tensor, getattr(observed, name))
    assert actual[0].images[:, :, :, :2].count_nonzero() == 0
    assert actual[0].observed_mask[1].tolist() == [True, False, True, False]
    assert not actual[0].clinical_mask[2, 0]
    assert directory.stat().st_mode & 0o777 == 0o700
    assert all(Path(entry["cached"]["path"]).stat().st_mode & 0o777 == 0o600
               for entry in cached.image_cache.index["images"])
    assert not re.search(r'"[^" ]*(?:sha256|hash|checksum|digest)[^" ]*"\s*:',
                         json.dumps(cached.signature()), re.IGNORECASE)
    assert prepare_image_cache(original, directory, workers=2)["reused"]


@pytest.mark.parametrize("kind", ["source", "cached"])
def test_derived_cache_rejects_changed_file_identity_even_after_mapping(cache, kind):
    import os
    _, cached, _, _ = _image_cache(cache)
    cached.batch([(0, 0)], supervised=False)
    source = str(cached.resolve(cached.patients[0]["visits"][0]["image"]))
    entry = cached.image_cache.entries[source][kind]
    path = Path(entry["path"])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000))
    with pytest.raises(ValueError, match="stale"):
        cached.batch([(0, 0)], supervised=False)
    with pytest.raises(ValueError, match="stale"):
        cached.signature()


@pytest.mark.parametrize("field", ["phase_order", "image_normalization", "shape", "population"])
def test_derived_cache_rejects_wrong_declaration_or_population(cache, field):
    original, _, directory, _ = _image_cache(cache)
    index = read_json(directory / "index.json")
    if field == "phase_order":
        index[field].reverse()
    elif field == "image_normalization":
        index[field]["mean"] += 1
    elif field == "shape":
        index[field][1] -= 1
    else:
        index["images"].pop()
    write_json(directory / "index.json", index)
    with pytest.raises(ValueError, match="declaration|population"):
        RawMRIStore(cache["manifest"], image_cache=directory)
    with pytest.raises(ValueError, match="declaration|population"):
        prepare_image_cache(original, directory)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite"])
def test_derived_cache_validates_mapped_arrays(cache, bad):
    _, cached, directory, _ = _image_cache(cache)
    index = read_json(directory / "index.json")
    source = str(cached.resolve(cached.patients[0]["visits"][0]["image"]))
    entry = next(e for e in index["images"] if e["source"]["path"] == source)
    image = np.zeros(ROI32_SHAPE, dtype=np.float32)
    if bad == "shape":
        image = image[:, :-1]
    elif bad == "dtype":
        image = image.astype(np.float64)
    else:
        image.flat[0] = np.nan
    np.save(entry["cached"]["path"], image, allow_pickle=False)
    entry["cached"] = file_identity(entry["cached"]["path"])
    write_json(directory / "index.json", index)
    reloaded = RawMRIStore(cache["manifest"], image_cache=directory)
    with pytest.raises(ValueError, match="finite float32"):
        reloaded.batch([(0, 0)], supervised=False)


def test_derived_cache_index_change_rejects_signature(cache):
    _, cached, directory, _ = _image_cache(cache)
    index = read_json(directory / "index.json")
    index["images"].reverse()
    write_json(directory / "index.json", index)
    with pytest.raises(ValueError, match="index changed"):
        cached.signature()


def test_cached_t0_inference_does_not_read_future_or_target(cache, monkeypatch):
    _, cached, _, _ = _image_cache(cache)
    expected, _ = cached.batch([(0, 0)], supervised=False)
    for visit in cached.patients[0]["visits"][1:]:
        if visit is not None:
            cached.resolve(visit["image"]).unlink()
    original_read = cached.image_cache.read
    baseline = cached.resolve(cached.patients[0]["visits"][0]["image"])

    def baseline_only(source):
        assert source == baseline
        return original_read(source)

    class PoisonTarget(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Inference accessed a label")

    monkeypatch.setattr(cached.image_cache, "read", baseline_only)
    cached.patients[0]["target"] = PoisonTarget()
    actual, supervision = cached.batch([(0, 0)], supervised=False)
    assert supervision is None
    for name, tensor in vars(expected).items():
        assert torch.equal(tensor, getattr(actual, name))


def test_raw_preprocessing_cannot_use_roi32_cache(tmp_path):
    path = make_raw_synthetic(tmp_path / "raw", n_train=2, n_val=1, image_shape=(4, 8, 8))
    with pytest.raises(ValueError, match="requires preprocessed_roi32"):
        RawMRIStore(path, image_shape=(4, 8, 8), allow_synthetic=True, image_cache="unused")
