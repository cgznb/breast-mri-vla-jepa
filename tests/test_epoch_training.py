from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mri_vla_jepa import training
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import load_checkpoint, stable_hash, write_json
from mri_vla_jepa.train_config import from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]
EPOCH_FIELDS = ("training_unit", "max_epochs", "early_stopping_patience", "early_stopping_min_delta")
SCHEDULER_FIELDS = ("lr_scheduler", "plateau_factor", "plateau_patience", "plateau_min_lr", "plateau_threshold")
PIPELINE_FIELDS = ("image_cache", "prefetch_batches", "loader_workers", "pin_memory", "non_blocking_transfer")


class TinyModel(torch.nn.Module):
    """Cheap stochastic optimizer steps exercise sampler and RNG restoration."""

    def __init__(self, config):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.teacher_encoder = torch.nn.Linear(1, 1).requires_grad_(False)

    def forward(self, inp):
        mean = inp.images.mean(dim=(1, 2, 3, 4, 5)).unsqueeze(1)
        return {"pcr_logit": self.encoder(mean).flatten()}

    def compute_loss(self, inp, sup, **kwargs):
        logits = self(inp)["pcr_logit"]
        loss = (logits - sup.label + .1 * torch.rand_like(logits)).square().mean()
        return loss, {"total": loss.detach(), "task": loss.detach()}

    def update_teacher(self):
        with torch.no_grad():
            for teacher, online in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
                teacher.mul_(.9).add_(online, alpha=.1)


def make_setup(tmp_path, profile="t0", n_train=5):
    cfg = load_config(ROOT / f"configs/raw_vla_jepa_smoke_{profile}.yaml")
    cfg.training_unit = "epochs"
    cfg.max_epochs = 3
    manifest = make_raw_synthetic(tmp_path / "data", n_train=n_train, n_val=2,
                                  image_shape=cfg.image_shape)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    return cfg, manifest, store


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
def test_patient_epoch_coverage_and_partial_batch(tmp_path, monkeypatch, profile):
    cfg, manifest, store = make_setup(tmp_path, profile)
    if profile == "dynamic":
        data = json.loads(manifest.read_text())
        data["patients"][0]["visits"][1:3] = [None, None]
        write_json(manifest, data)
        store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    batches = []
    original_batch = store.batch

    def record(tasks, supervised=True):
        if supervised:
            batches.append(list(tasks))
        return original_batch(tasks, supervised=supervised)

    monkeypatch.setattr(store, "batch", record)
    report = training.train(store, cfg, tmp_path / "run")
    assert report["completed"] and report["stop_reason"] == "max_epochs"
    assert report["completed_epochs"] == 3
    assert report["completed_steps"] == report["total_steps"] == 9
    assert report["batches_per_epoch"] == 3
    for epoch in range(3):
        epoch_batches = batches[epoch * 3:(epoch + 1) * 3]
        assert [len(batch) for batch in epoch_batches] == [2, 2, 1]
        assert sorted(i for batch in epoch_batches for i, stage in batch) == store.by_split["train"]
        for batch in epoch_batches:
            for index, stage in batch:
                assert stage in ([0] if profile == "t0" else store.allowed_landmarks(index))
    state = load_checkpoint(tmp_path / "run/last.pt")
    assert sum("validation" in item for item in state["history"]) == 3
    assert len(state["epoch_history"]) == 3
    for summary in state["epoch_history"]:
        assert summary["patients"] == 5 and summary["batches"] == 3
        step_rows = [row for row in state["history"] if row["epoch"] == summary["epoch"]]
        assert summary["training"]["total"] == pytest.approx(
            sum(row["total"] * row["batch_size"] for row in step_rows) / 5)
    progress = json.loads((tmp_path / "run/progress.json").read_text())
    assert progress["status"] == "completed" and progress["completed_epochs"] == 3
    assert progress["last_epoch"]["validation"]["selection_metric"] == report["selection_protocol"]
    assert json.loads((tmp_path / "run/epoch_history.json").read_text()) == state["epoch_history"]


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
def test_mid_epoch_resume_is_exact(tmp_path, monkeypatch, profile):
    cfg, _, store = make_setup(tmp_path, profile)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "full")
    partial = training.train(store, cfg, tmp_path / "resumed", stop_after=4)
    assert not partial["completed"] and partial["stop_reason"] == "stop_after"
    assert partial["completed_epochs"] == 1
    state = load_checkpoint(tmp_path / "resumed/last.pt")
    assert state["epoch_state"]["position"] == 2
    assert state["epoch_state"]["batches_completed"] == 1
    assert len(state["epoch_state"]["indices"]) == 5
    assert len(state["epoch_history"]) == 1
    assert "validation" not in state["history"][-1]
    with pytest.raises(ValueError, match="config/data identity"):
        training.train(store, replace(cfg, early_stopping_patience=49), tmp_path / "resumed", resume=True)
    resumed = training.train(store, cfg, tmp_path / "resumed", resume=True)
    assert resumed["completed"] and resumed["completed_epochs"] == 3
    left = load_checkpoint(tmp_path / "full/last.pt")
    right = load_checkpoint(tmp_path / "resumed/last.pt")
    for key in ("model", "optimizer", "rng", "sampler_state", "history", "epoch_history", "epoch_state", "best_nll"):
        assert_same(left[key], right[key])


def mock_scores(monkeypatch, values):
    scores = iter(values)

    def score(rows, profile):
        nll = next(scores)
        return {"selection_score": nll, "selection_metric": "t0_direct_nll", "metrics": {"nll": nll}}

    monkeypatch.setattr(training, "_scores", score)


def test_patience_fifty_stops_after_fifty_bad_epochs(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg.max_epochs = 200
    cfg.early_stopping_patience = 50
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    mock_scores(monkeypatch, [.5] * 51)
    report = training.train(store, cfg, tmp_path / "run")
    assert report["completed"] and report["stop_reason"] == "early_stopping"
    assert report["completed_epochs"] == report["completed_steps"] == 51
    assert report["bad_epochs"] == report["early_stopping_patience"] == 50
    best = load_checkpoint(tmp_path / "run/best.pt")
    assert best["epoch_state"]["completed_epochs"] == 1
    last = load_checkpoint(tmp_path / "run/last.pt")
    assert last["epoch_state"]["stopped_early"]
    assert len(last["epoch_history"]) == 51
    # A completed early-stopped run cannot continue and consume more validations.
    again = training.train(store, cfg, tmp_path / "run", resume=True)
    assert again == report


def test_min_delta_patience_reset_and_resume(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg.max_epochs = 20
    cfg.early_stopping_patience = 2
    cfg.early_stopping_min_delta = .01
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    mock_scores(monkeypatch, [1., .995, .98, .98, .98])
    partial = training.train(store, cfg, tmp_path / "run", stop_after=2)
    assert partial["bad_epochs"] == 1 and partial["best_validation_nll"] == 1.
    report = training.train(store, cfg, tmp_path / "run", resume=True)
    assert report["stop_reason"] == "early_stopping" and report["completed_epochs"] == 5
    best = load_checkpoint(tmp_path / "run/best.pt")
    assert best["epoch_state"]["completed_epochs"] == 3
    assert best["epoch_state"]["bad_epochs"] == 0
    assert report["best_validation_nll"] == .98


@pytest.mark.parametrize("changes,message", [
    ({"training_unit": "batches"}, "training_unit"),
    ({"max_epochs": 0}, "max_epochs"),
    ({"max_epochs": True}, "max_epochs"),
    ({"early_stopping_patience": 0}, "early_stopping_patience"),
    ({"early_stopping_patience": False}, "early_stopping_patience"),
    ({"early_stopping_min_delta": -1.}, "early_stopping_min_delta"),
    ({"early_stopping_min_delta": float("nan")}, "early_stopping_min_delta"),
    ({"training_unit": "epochs", "accumulation": 4}, "accumulation=1"),
    ({"training_unit": "epochs", "schedule": "two_stage", "representation_steps": 2}, "one_stage"),
])
def test_epoch_config_validation(changes, message):
    cfg = load_config(ROOT / "configs/raw_vla_jepa_smoke_t0.yaml")
    with pytest.raises(ValueError, match=message):
        replace(cfg, **changes).validate()
    assert from_dict({}).training_unit == "steps"
    assert from_dict({}).early_stopping_patience == 50


@pytest.mark.parametrize("schema", [training.CHECKPOINT_SCHEMA, training.LEGACY_CHECKPOINT_SCHEMA])
def test_pre_epoch_checkpoint_read_compatibility(tmp_path, monkeypatch, schema):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg = replace(cfg, training_unit="steps", max_epochs=200, joint_steps=1)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "run")
    state = load_checkpoint(tmp_path / "run/last.pt")
    state["schema"] = schema
    for field in (*EPOCH_FIELDS, *SCHEDULER_FIELDS, *PIPELINE_FIELDS):
        state["config"].pop(field)
    if schema == training.LEGACY_CHECKPOINT_SCHEMA:
        # Validate the original raw configuration before new defaults are added.
        state["config_digest"] = stable_hash(state["config"])
    monkeypatch.setattr(training, "load_checkpoint", lambda path: state)
    _, loaded_cfg, loaded = training.load_trained("pre_epoch.pt")
    assert loaded_cfg.to_dict() == cfg.to_dict()
    assert loaded["schema"] == schema
    assert loaded_cfg.lr_scheduler == "none"


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
def test_plateau_epoch_rates_and_mid_epoch_resume_are_exact(tmp_path, monkeypatch, profile):
    cfg, _, store = make_setup(tmp_path, profile)
    cfg.lr_scheduler = "plateau"
    cfg.plateau_patience = 0
    cfg.plateau_min_lr = cfg.lr / 4
    cfg.max_epochs = 6
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    mock_scores(monkeypatch, [.5] * 6)
    training.train(store, cfg, tmp_path / "full")
    mock_scores(monkeypatch, [.5] * 6)
    partial = training.train(store, cfg, tmp_path / "resumed", stop_after=7)
    assert partial["completed_epochs"] == 2
    saved = load_checkpoint(tmp_path / "resumed/last.pt")
    assert saved["scheduler_state"]["last_epoch"] == 2
    assert saved["scheduler_state"]["_last_lr"] == [cfg.lr / 2]
    assert saved["epoch_state"]["position"] == 2
    training.train(store, cfg, tmp_path / "resumed", resume=True)
    left = load_checkpoint(tmp_path / "full/last.pt")
    right = load_checkpoint(tmp_path / "resumed/last.pt")
    for key in ("model", "optimizer", "scheduler_state", "rng", "sampler_state", "history",
                "epoch_history", "epoch_state", "best_nll"):
        assert_same(left[key], right[key])
    epochs = left["epoch_history"]
    assert [e["lr"] for e in epochs] == [cfg.lr, cfg.lr, cfg.lr / 2, cfg.lr / 4, cfg.lr / 4, cfg.lr / 4]
    assert [e["next_lr"] for e in epochs] == [cfg.lr, cfg.lr / 2, cfg.lr / 4, cfg.lr / 4, cfg.lr / 4, cfg.lr / 4]
    assert [e["lr_reduced"] for e in epochs] == [False, True, True, False, False, False]
    assert left["scheduler_state"]["last_epoch"] == 6
    for epoch in epochs:
        assert {s["lr"] for s in left["history"] if s["epoch"] == epoch["epoch"]} == {epoch["lr"]}


def test_plateau_uses_selection_score_instead_of_stage_zero_nll(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, "dynamic", n_train=2)
    cfg.lr_scheduler = "plateau"
    cfg.plateau_patience = 1
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    values = iter([(.5, .6), (.6, .4), (.7, .2)])

    def scores(rows, profile):
        selection, stage_zero = next(values)
        return {"selection_score": selection, "selection_metric": "patient_equal_prefix_nll",
                "metrics": {"nll": selection}, "per_landmark": {"T0": {"nll": stage_zero}}}

    monkeypatch.setattr(training, "_scores", scores)
    training.train(store, cfg, tmp_path / "run")
    state = load_checkpoint(tmp_path / "run/last.pt")
    assert state["scheduler_state"]["best"] == .5
    assert state["epoch_history"][-1]["next_lr"] == cfg.lr / 2


@pytest.mark.parametrize("changes,message", [
    ({"lr_scheduler": "cosine"}, "lr_scheduler"),
    ({"lr_scheduler": "plateau"}, "epoch training"),
    ({"plateau_factor": 0}, "plateau_factor"),
    ({"plateau_factor": 1}, "plateau_factor"),
    ({"plateau_factor": True}, "plateau_factor"),
    ({"plateau_factor": float("nan")}, "plateau_factor"),
    ({"plateau_patience": -1}, "plateau_patience"),
    ({"plateau_patience": True}, "plateau_patience"),
    ({"plateau_min_lr": 0}, "plateau_min_lr"),
    ({"plateau_threshold": -1}, "plateau_threshold"),
    ({"training_unit": "epochs", "lr_scheduler": "plateau", "plateau_min_lr": 1}, "plateau_min_lr"),
])
def test_plateau_config_validation(changes, message):
    cfg = load_config(ROOT / "configs/raw_vla_jepa_smoke_t0.yaml")
    with pytest.raises(ValueError, match=message):
        replace(cfg, **changes).validate()


def test_plateau_resume_rejects_missing_or_stale_scheduler_state(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg.lr_scheduler = "plateau"
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "run", stop_after=1)
    saved = load_checkpoint(tmp_path / "run/last.pt")
    saved["scheduler_state"] = None
    monkeypatch.setattr(training, "load_checkpoint", lambda path: saved)
    with pytest.raises(ValueError, match="saved plateau scheduler"):
        training.train(store, cfg, tmp_path / "run", resume=True)
    saved["scheduler_state"] = {"_last_lr": [cfg.lr / 2], "last_epoch": 1}
    with pytest.raises(ValueError, match="optimizer/epoch progress"):
        training.train(store, cfg, tmp_path / "run", resume=True)


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
def test_real_network_epoch_validation(tmp_path, profile):
    cfg, _, store = make_setup(tmp_path, profile, n_train=3)
    cfg.max_epochs = 1
    report = training.train(store, cfg, tmp_path / "run")
    assert report["completed"] and report["completed_epochs"] == 1
    assert report["completed_steps"] == 2
    state = load_checkpoint(tmp_path / "run/last.pt")
    assert "validation" not in state["history"][0]
    assert state["history"][1]["batch_size"] == 1
    assert state["history"][1]["validation"]["selection_metric"] == report["selection_protocol"]
    assert state["epoch_history"][0]["training"]["label_count"] == 3
    for name in ("future_count", "pair_count"):
        assert state["epoch_history"][0]["training"][name] == sum(row[name] for row in state["history"])


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
@pytest.mark.parametrize("scheduler", ["none", "plateau"])
def test_prefetch_preserves_stochastic_training_and_mid_epoch_resume(tmp_path, monkeypatch, profile, scheduler):
    cfg, manifest, store = make_setup(tmp_path, profile)
    # Missing T1 must retain its slot and legal dynamic sampling with prefetch.
    data = json.loads(manifest.read_text())
    data["patients"][0]["visits"][1] = None
    write_json(manifest, data)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    cfg.lr_scheduler = scheduler
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "serial")
    parallel = replace(cfg, prefetch_batches=2, loader_workers=2)
    training.train(store, parallel, tmp_path / "prefetched")
    training.train(store, parallel, tmp_path / "resumed", stop_after=4)
    training.train(store, parallel, tmp_path / "resumed", resume=True)
    reference = load_checkpoint(tmp_path / "serial/last.pt")
    for directory in ("prefetched", "resumed"):
        actual = load_checkpoint(tmp_path / directory / "last.pt")
        for key in ("model", "optimizer", "scheduler_state", "rng", "sampler_state", "history",
                    "epoch_history", "epoch_state", "best_nll"):
            assert_same(reference[key], actual[key])


@pytest.mark.parametrize("profile", ["t0", "dynamic"])
def test_prefetch_real_network_matches_serial_training(tmp_path, profile):
    cfg, _, store = make_setup(tmp_path, profile, n_train=3)
    cfg.max_epochs = 1
    training.train(store, cfg, tmp_path / "serial")
    training.train(store, replace(cfg, prefetch_batches=2), tmp_path / "prefetched")
    reference = load_checkpoint(tmp_path / "serial/last.pt")
    actual = load_checkpoint(tmp_path / "prefetched/last.pt")
    for key in ("model", "optimizer", "rng", "sampler_state", "history", "epoch_history"):
        assert_same(reference[key], actual[key])


@pytest.mark.parametrize("changes,message", [
    ({"prefetch_batches": -1}, "prefetch_batches"),
    ({"prefetch_batches": 5}, "prefetch_batches"),
    ({"prefetch_batches": True}, "prefetch_batches"),
    ({"prefetch_batches": 2}, "epoch training"),
    ({"loader_workers": 0}, "loader_workers"),
    ({"loader_workers": 9}, "loader_workers"),
    ({"image_cache": ""}, "image_cache"),
    ({"pin_memory": True}, "CUDA"),
    ({"non_blocking_transfer": True}, "requires pin_memory"),
])
def test_pipeline_config_validation(changes, message):
    cfg = load_config(ROOT / "configs/raw_vla_jepa_smoke_t0.yaml")
    with pytest.raises(ValueError, match=message):
        replace(cfg, **changes).validate()
