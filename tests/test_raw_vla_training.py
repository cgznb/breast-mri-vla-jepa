from dataclasses import replace
import json
from pathlib import Path
import re
import numpy as np
import pytest
import torch

from mri_vla_jepa.io import load_checkpoint, stable_hash, write_json
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.training import (
    _patient_equal_score, _scores, evaluate, from_dict, load_config,
    load_trained, predict, train,
)

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path, profile="t0", two_stage=False):
    cfg = load_config(ROOT / f"configs/raw_vla_jepa_smoke_{profile}.yaml")
    cfg.joint_steps = 3
    cfg.validation_every = 1
    if two_stage:
        cfg.schedule = "two_stage"
        cfg.representation_steps = 2
        cfg.freeze_encoder_after_representation = True
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=2, image_shape=cfg.image_shape)
    return cfg, manifest, RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("profile,two_stage", [("t0", False), ("dynamic", False), ("dynamic", True)])
def test_profiles_resume_exact(tmp_path, profile, two_stage):
    cfg, manifest, store = setup(tmp_path, profile, two_stage)
    full = train(store, cfg, tmp_path / "full")
    partial = train(store, cfg, tmp_path / "resumed", stop_after=2)
    assert not partial["completed"]
    resumed = train(store, cfg, tmp_path / "resumed", resume=True)
    assert full["completed"] and resumed["completed"]
    a, b = [load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "resumed")]
    assert a["history"] == b["history"]
    assert_same(a["model"], b["model"])
    assert_same(a["optimizer"], b["optimizer"])
    assert_same(a["sampler_state"], b["sampler_state"])
    assert_same(a["rng"], b["rng"])
    assert a["schema"] == "responsewm_raw_vla_jepa_checkpoint_v2"
    assert a["patient_partitions"] == {
        "train": ["synthetic_0", "synthetic_1", "synthetic_2", "synthetic_3"],
        "val": ["synthetic_4", "synthetic_5"], "test": [],
    }
    if two_stage:
        # Joint freeze coordinates: EMA is synchronized once at the boundary.
        encoder = {k.removeprefix("encoder."): v for k, v in b["model"].items() if k.startswith("encoder.")}
        teacher = {k.removeprefix("teacher_encoder."): v for k, v in b["model"].items() if k.startswith("teacher_encoder.")}
        assert_same(encoder, teacher)
    report = evaluate(tmp_path / "resumed/best.pt", manifest, tmp_path / "evaluation.json",
                      split="val", allow_synthetic=True, bootstrap=0)
    expected = "t0_direct_nll" if profile == "t0" else "patient_equal_prefix_nll"
    assert a["selection_protocol"] == report["selection_protocol"] == expected
    assert report["selection_score"] == report["patient_equal_metrics"]["nll"]
    assert report["per_landmark"]["T0"]["n"] == 2
    assert report["per_landmark"]["T3"]["n"] == (0 if profile == "t0" else 2)
    assert report["patient_equal_metrics"]["prefixes"] == (2 if profile == "t0" else 8)


def test_patient_equal_prefix_nll_and_empty_stage():
    rows = [{"patient_key": "A", "landmark": stage, "pcr": 1,
             "pcr_probability": .1 if stage == 0 else .9, "arm_semantics": "assigned_arm_scenario"}
            for stage in range(4)]
    rows += [{"patient_key": "B", "landmark": 0, "pcr": 0,
              "pcr_probability": .1, "arm_semantics": "assigned_arm_scenario"}]
    rows += [{"patient_key": "C", "landmark": 1, "pcr": None,
              "pcr_probability": .5, "arm_semantics": "assigned_arm_scenario"}]
    a_mean = (-np.log(.1) - 3 * np.log(.9)) / 4
    expected = (a_mean - np.log(.9)) / 2
    result = _patient_equal_score(rows, bootstrap=10, seed=4)
    assert result["nll"] == pytest.approx(expected)
    assert result["patients"] == 2 and result["prefixes"] == 5
    assert result["missing_label_patients"] == 1
    pooled = (-np.log(.1) - 4 * np.log(.9)) / 5
    assert result["nll"] != pytest.approx(pooled)
    # Averaging probabilities before BCE would give another, incorrect criterion.
    assert result["nll"] != pytest.approx((-np.log(.7) - np.log(.9)) / 2)
    t1 = _scores([rows[1]], 1)
    assert t1["selection_metric"] == "landmark_T1_direct_nll"
    assert t1["per_landmark"]["T3"]["n"] == 0
    assert t1["per_landmark"]["T3"]["nll"] is None


def test_resume_does_not_add_off_schedule_validation(tmp_path):
    cfg, _, store = setup(tmp_path, "dynamic")
    cfg.validation_every = 3
    train(store, cfg, tmp_path / "full")
    partial = train(store, cfg, tmp_path / "resumed", stop_after=1)
    assert partial["best_checkpoint"] is None
    assert "validation" not in load_checkpoint(tmp_path / "resumed/last.pt")["history"][0]
    train(store, cfg, tmp_path / "resumed", resume=True)
    full = load_checkpoint(tmp_path / "full/last.pt")
    resumed = load_checkpoint(tmp_path / "resumed/last.pt")
    assert full["history"] == resumed["history"]
    assert full["best_nll"] == resumed["best_nll"]
    for key in ("model", "optimizer", "rng", "sampler_state"):
        assert_same(full[key], resumed[key])


def test_dynamic_missing_visits_stay_canonical(tmp_path):
    cfg, manifest, _ = setup(tmp_path, "dynamic")
    data = json.loads(manifest.read_text())
    data["patients"][4]["visits"][1] = None
    data["patients"][4]["visits"][2] = None
    write_json(manifest, data)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    train(store, cfg, tmp_path / "run")
    report = evaluate(tmp_path / "run/best.pt", manifest, tmp_path / "evaluation.json",
                      split="val", allow_synthetic=True, bootstrap=0)
    assert report["patient_equal_metrics"]["patients"] == 2
    assert report["patient_equal_metrics"]["prefixes"] == 6
    assert [report["per_landmark"][f"T{s}"]["n"] for s in range(4)] == [2, 1, 1, 2]
    assert [r["landmark"] for r in report["rows"] if r["patient_key"] == "synthetic_4"] == [0, 3]


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
def test_source_only_pcr_and_explicit_forecast(tmp_path, profile, monkeypatch):
    cfg, manifest, store = setup(tmp_path, profile)
    train(store, cfg, tmp_path / "run")
    checkpoint = tmp_path / "run/best.pt"
    from mri_vla_jepa.model import RawVLAJEPA, TimeCausalWorldPredictor
    for patient in store.patients:
        for visit in patient["visits"][1:]:
            store.resolve(visit["image"]).unlink()
    with monkeypatch.context() as isolated:
        isolated.setattr(TimeCausalWorldPredictor, "forward", lambda *a, **k: pytest.fail("pCR ran WM"))
        isolated.setattr(RawVLAJEPA, "generate", lambda *a, **k: pytest.fail("pCR ran ODE"))
        isolated.setattr(RawVLAJEPA, "forecast", lambda *a, **k: pytest.fail("pCR forecast states"))
        result = predict(checkpoint, manifest, "synthetic_0", tmp_path / "direct.json", allow_synthetic=True)
        assert 0 <= result["pcr_probability"] <= 1
        assert result["observed_stages"] == [0]
        evaluate(checkpoint, manifest, tmp_path / "evaluation.json", landmark=0,
                 split="val", allow_synthetic=True, bootstrap=0)
    result = predict(checkpoint, manifest, "synthetic_0", tmp_path / "forecast.json",
                     allow_synthetic=True, forecast_states=True)
    arrays = np.load(result["forecast_states"], allow_pickle=False)
    assert arrays["state_prediction"].shape == (1, 4, 4, 16)
    assert np.isfinite(arrays["state_prediction"]).all()
    assert not arrays["state_prediction"][:, 0].any()
    assert arrays["query_mask"].tolist() == [[False, True, True, True]]
    with pytest.raises(ValueError, match="enable_flow"):
        predict(checkpoint, manifest, "synthetic_0", tmp_path / "image.json", allow_synthetic=True, generate=True)
    if profile == "t0":
        with pytest.raises(ValueError, match="T0 checkpoint"):
            predict(checkpoint, manifest, "synthetic_0", tmp_path / "late.json", landmark=1, allow_synthetic=True)
        with pytest.raises(ValueError, match="T0 checkpoint"):
            evaluate(checkpoint, manifest, tmp_path / "late_eval.json", landmark="all_observed",
                     split="val", allow_synthetic=True, bootstrap=0)


def test_resume_identity_and_strict_configs(tmp_path, monkeypatch):
    cfg, manifest, store = setup(tmp_path)
    train(store, cfg, tmp_path / "run", stop_after=1)
    with pytest.raises(ValueError, match="identity"):
        train(store, replace(cfg, lr=cfg.lr * 2), tmp_path / "run", resume=True)
    from mri_vla_jepa import training as raw_vla_training
    with monkeypatch.context() as isolated:
        isolated.setattr(raw_vla_training, "_source_snapshot", lambda: {"changed": "source changed"})
        with pytest.raises(ValueError, match="implementation identity"):
            train(store, cfg, tmp_path / "run", resume=True)
    with monkeypatch.context() as isolated:
        normalization = store.normalization_state()
        normalization["changed"] = True
        isolated.setattr(store, "normalization_state", lambda: normalization)
        with pytest.raises(ValueError, match="normalization"):
            train(store, cfg, tmp_path / "run", resume=True)
    with monkeypatch.context() as isolated:
        isolated.setitem(store.by_split, "train", store.by_split["train"][:-1])
        with pytest.raises(ValueError, match="population/selection identity"):
            train(store, cfg, tmp_path / "run", resume=True)
    with monkeypatch.context() as isolated:
        state = load_checkpoint(tmp_path / "run/last.pt")
        state["selection_protocol"] = "changed"
        isolated.setattr(raw_vla_training, "load_checkpoint", lambda path: state)
        with pytest.raises(ValueError, match="population/selection identity"):
            train(store, cfg, tmp_path / "run", resume=True)
    data = json.loads(manifest.read_text())
    data["provenance"] = {"notes": "changed source declaration"}
    write_json(manifest, data)
    with pytest.raises(ValueError, match="data identity"):
        train(store, cfg, tmp_path / "run", resume=True)
    del data["provenance"]
    write_json(manifest, data)
    image = store.resolve(store.patients[0]["visits"][0]["image"])
    with image.open("ab") as handle:
        handle.write(b"identity change")
    with pytest.raises(ValueError, match="identity"):
        train(store, cfg, tmp_path / "run", resume=True)
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"unknown": True})
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"model": {"unknown": True}})
    with pytest.raises(ValueError, match="schema"):
        from_dict({"schema": "responsewm_raw_jepa_train_v1"})
    with pytest.raises(ValueError, match="enable_flow"):
        replace(cfg, flow_weight=.1).validate()


def assert_no_checksums(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                assert not any(word in key.lower() for word in ("sha256", "checksum", "digest", "hash"))
            assert_no_checksums(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_no_checksums(item)
    elif isinstance(value, str):
        assert re.search(r"\b[0-9a-f]{64}\b", value) is None


def test_new_checkpoint_snapshots_without_checksums(tmp_path):
    cfg, _, store = setup(tmp_path)
    train(store, cfg, tmp_path / "run", stop_after=1)
    state = load_checkpoint(tmp_path / "run/last.pt")
    assert state["config"] == cfg.to_dict()
    for name, source in state["source_snapshot"].items():
        assert source == (ROOT / "src/mri_vla_jepa" / name).read_text(encoding="utf-8")
    assert_no_checksums(state)
    for path in (tmp_path / "run").glob("*.json"):
        assert_no_checksums(json.loads(path.read_text()))


@pytest.mark.parametrize("from_split", ["train", "val"])
def test_legacy_v1_read_only_and_population_guards(tmp_path, monkeypatch, from_split):
    cfg, manifest, store = setup(tmp_path)
    train(store, cfg, tmp_path / "run", stop_after=1)
    legacy = load_checkpoint(tmp_path / "run/last.pt")
    legacy["schema"] = "responsewm_raw_vla_jepa_checkpoint_v1"
    legacy["config_digest"] = cfg.digest
    legacy["patient_partitions"] = {
        split: [stable_hash(key) for key in keys]
        for split, keys in legacy["patient_partitions"].items()
    }
    legacy["normalization"] = {
        key: value for key, value in legacy["normalization"].items()
        if key not in {"image_preprocessing", "image_normalization"}
    }
    legacy["normalization"]["schema"] = "raw_mri_normalization_v1"
    legacy["normalization"]["train_patient_hashes"] = legacy["patient_partitions"]["train"]
    # Legacy checksum coordinates exist only in this in-memory reader fixture.
    from mri_vla_jepa import training as raw_vla_training
    monkeypatch.setattr(raw_vla_training, "load_checkpoint", lambda path: legacy)
    _, loaded_cfg, loaded = load_trained("legacy_v1.pt")
    assert loaded_cfg.to_dict() == cfg.to_dict()
    assert loaded["schema"] == legacy["schema"]
    report = evaluate("legacy_v1.pt", manifest, tmp_path / "legacy_evaluation.json",
                      split="val", allow_synthetic=True, bootstrap=0)
    assert report["per_landmark"]["T0"]["n"] == 2
    assert_no_checksums(report)
    result = predict("legacy_v1.pt", manifest, "synthetic_0", tmp_path / "legacy_predict.json",
                     allow_synthetic=True)
    assert 0 <= result["pcr_probability"] <= 1
    assert_no_checksums(result)
    with pytest.raises(ValueError, match="v1 checkpoints are read-only"):
        train(store, cfg, tmp_path / "run", resume=True)
    data = json.loads(manifest.read_text())
    index = store.by_split[from_split][0]
    data["patients"][index]["split"] = "test"
    changed = tmp_path / "data/legacy_overlap.json"
    write_json(changed, data)
    with pytest.raises(ValueError, match="overlap"):
        evaluate("legacy_v1.pt", changed, tmp_path / "overlap.json", allow_synthetic=True, bootstrap=0)
    legacy["config_digest"] = "invalid legacy coordinate"
    with pytest.raises(ValueError, match="digest mismatch"):
        load_trained("legacy_v1.pt")


@pytest.mark.parametrize("from_split", ["train", "val"])
def test_evaluation_population_guard_canonical_alias(tmp_path, from_split):
    cfg, manifest, _ = setup(tmp_path)
    data = json.loads(manifest.read_text())
    index = 0 if from_split == "train" else 4
    data["patients"][index]["patient_key"] = "ISPY2-100"
    write_json(manifest, data)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    train(store, cfg, tmp_path / "run")
    data["patients"][index]["patient_key"] = "100.0"
    data["patients"][index]["split"] = "test"
    changed = tmp_path / "data/changed.json"
    write_json(changed, data)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(tmp_path / "run/best.pt", changed, tmp_path / "evaluation.json",
                 allow_synthetic=True, bootstrap=0)


def test_unlabelled_training_is_explicit_error(tmp_path):
    cfg, _, store = setup(tmp_path)
    for i in store.by_split["train"]:
        store.patients[i]["target"]["pcr"] = None
    with pytest.raises(ValueError, match="labelled training"):
        train(store, cfg, tmp_path / "run")


def test_packaged_default_smoke_from_external_directory(tmp_path):
    # Simulate the installed console wrapper while the checkout configs are absent.
    import os
    import subprocess
    import sys
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src')
    output = tmp_path / 'default_smoke'
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; from mri_vla_jepa.cli import entrypoint; sys.exit(entrypoint())',
         'smoke-vla-jepa', '--output', str(output)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'smoke_report.json').read_text())
    assert report['engineering_only'] is True
    assert report['clinical_validation'] is False
    assert report['training']['landmarks'] == 't0'
    assert report['training']['completed'] is True
    state = load_checkpoint(output / 'run/last.pt')
    assert set(state['source_snapshot']) == {
        'model.py', 'model_config.py', 'encoder.py', 'flow.py', 'training.py',
        'train_config.py', 'cli.py', 'data.py', 'contracts.py', 'constants.py',
        'arm.py', 'io.py', 'metrics.py', 'data_pipeline.py', 'diagnostics.py',
    }
