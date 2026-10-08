"""Diagnostics must not perturb training and must respect real visit chains."""
from dataclasses import replace
import random
import json

import numpy as np
import pytest
import torch

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.diagnostics import (aggregate_world_errors, evaluate_world, make_probe_tasks,
    preserve_training_state, representation_metrics, run_training_probe, world_error_rows)
from mri_vla_jepa.io import rng_state
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig


@pytest.fixture
def setup(tmp_path):
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    store = RawMRIStore(make_raw_synthetic(tmp_path, n_train=4, n_val=2, image_shape=(4, 8, 8)),
                        (4, 8, 8), allow_synthetic=True)
    cfg = RawVLATrainConfig(model=RawVLAJEPAConfig(dim=16, token_grid=(1, 2, 2), heads=4,
        encoder_depth=1, fusion_depth=1, predictor_depth=1, state_queries=2, pcr_queries=2),
        image_shape=(4, 8, 8), device="cpu", precision="fp32", diagnostics_batch_size=2,
        diagnostics_patients=4, landmarks="t0", allow_synthetic=True)
    yield store, cfg, RawVLAJEPA(cfg.model)
    torch.set_num_threads(before)


def assert_rng_equal(left, right):
    assert left["python"] == right["python"] and left["numpy"] == right["numpy"]
    assert torch.equal(left["torch"], right["torch"])
    assert len(left["cuda"]) == len(right["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def test_probe_is_read_only_and_resume_state_reproducible(setup):
    store, cfg, model = setup
    model.train(); model.encoder.eval()
    parameter = next(model.encoder.parameters())
    parameter.grad = torch.ones_like(parameter)
    before_grad = parameter.grad.clone()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    flags = [module.training for module in model.modules()]
    rng = rng_state()
    tasks = make_probe_tasks(store, cfg.landmarks, 4)
    report, private = run_training_probe(model, store, cfg, 0, tasks)
    assert_rng_equal(rng, rng_state())
    assert flags == [module.training for module in model.modules()]
    assert torch.equal(parameter.grad, before_grad)
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    replay, _ = run_training_probe(model, store, cfg, 5, tasks, private)
    assert replay["eval_pcr_nll"] == report["eval_pcr_nll"]
    assert replay["gradient_batches"] == report["gradient_batches"]
    assert replay["world"] == report["world"]
    assert replay["teacher_drift"]["l1"] == 0
    assert replay["teacher_drift"]["adjacent_copy_change_l1"] == 0
    assert report["teacher_frozen"]
    assert all(p.grad is None for p in model.teacher_encoder.parameters())


def test_exception_restores_all_rng_and_module_modes(setup):
    _, _, model = setup
    model.train(); model.encoder.eval()
    modes = [module.training for module in model.modules()]
    state = rng_state()
    with pytest.raises(RuntimeError):
        with preserve_training_state(model):
            random.random(); np.random.rand(); torch.rand(2)
            raise RuntimeError("probe failed")
    assert_rng_equal(state, rng_state())
    assert modes == [module.training for module in model.modules()]


def test_disabled_jepa_reports_untrained_world_unavailable(setup, monkeypatch):
    store, cfg, model = setup
    cfg = replace(cfg, jepa_weight=0, reconstruction_weight=0, variance_weight=0,
                  variance_definition="off")
    original = store.batch
    def input_only(tasks, **kwargs):
        assert kwargs["future_supervision"] is False
        return original(tasks, **kwargs)
    monkeypatch.setattr(store, "batch", input_only)
    report, _ = run_training_probe(model, store, cfg, 0, make_probe_tasks(store, "t0", 4))
    assert report["world"]["status"] == "unavailable"
    assert evaluate_world(model, store, cfg)["status"] == "unavailable"
    for batch in report["gradient_batches"]:
        assert not batch["enabled_terms"]["jepa"]
        assert batch["weighted_gradient_norms"]["jepa"]["all_trainable"] == 0
        assert batch["task_auxiliary_cosines"]["jepa"]["encoder"] is None


def test_flow_only_auxiliary_marks_world_head_supervised_and_preserves_rng(setup):
    store, cfg, _ = setup
    cfg = replace(cfg, jepa_weight=0, flow_weight=.1,
                  model=replace(cfg.model, enable_flow=True, fm_channels=8))
    model = RawVLAJEPA(cfg.model)
    result = evaluate_world(model, store, cfg, batch_size=2)
    assert result["status"] == "evaluated" and result["world_head_trained"]
    state = rng_state()
    tasks = make_probe_tasks(store, "t0", 4)
    report, _ = run_training_probe(model, store, cfg, 5, tasks)
    assert_rng_equal(state, rng_state())
    assert report["world_head_trained"]
    assert report["world"]["status"] == "evaluated"
    assert all(batch["weighted_gradient_norms"]["flow"]["world_predictor"] > 0
               for batch in report["gradient_batches"])
    replay, _ = run_training_probe(model, store, cfg, 10, tasks)
    assert replay["gradient_batches"] == report["gradient_batches"]


def test_representation_metrics_detect_patient_collapse_despite_spatial_variation():
    states = torch.zeros(4, 4, 4, 2)
    states[:, 0] = torch.tensor([[-3., -3.], [-1., -1.], [1., 1.], [3., 3.]])
    mask = torch.tensor([[True, False, False, False]] * 4)
    metrics = representation_metrics(states, mask, torch.zeros(4, dtype=torch.long))
    assert metrics["legacy_variance_penalty"] == 0
    assert metrics["patient_axis_variance_penalty"] == pytest.approx(.99)
    assert metrics["anchor_patient_pooled_effective_rank"] == 0


def test_world_patient_equal_reduction_not_prefix_weighted():
    keys = ("teacher_forcing_l1", "autonomous_l1", "teacher_forcing_copy_l1", "autonomous_copy_l1")
    rows = [{"patient": 0, "horizon": 1, **dict.fromkeys(keys, 1.)},
            {"patient": 0, "horizon": 1, **dict.fromkeys(keys, 3.)},
            {"patient": 1, "horizon": 1, **dict.fromkeys(keys, 8.)}]
    result = aggregate_world_errors(rows)["metrics"]["horizon_1"]
    assert result["patients"] == 2 and result["prefix_targets"] == 3
    assert result["autonomous_l1"] == 5
    assert result["autonomous_relative_to_copy"] == 1
    assert result["autonomous_skill_vs_copy"] == 0
    json.dumps(result, allow_nan=False)


def test_world_near_zero_copy_has_explicit_null_skill():
    row = {"patient": 0, "horizon": 1, "teacher_forcing_l1": 1., "autonomous_l1": 2.,
           "teacher_forcing_copy_l1": 0., "autonomous_copy_l1": 1e-10}
    result = aggregate_world_errors([row])["metrics"]["horizon_1"]
    assert result["autonomous_relative_to_copy"] is None
    assert result["autonomous_skill_vs_copy"] is None
    assert result["autonomous_near_zero_copy_patients"] == 1
    with pytest.raises(ValueError, match="finite"):
        aggregate_world_errors([{**row, "autonomous_l1": float("nan")}])


@pytest.mark.parametrize("reconstruction_weight", [0., .1])
def test_no_jepa_probe_never_reads_future_files(setup, monkeypatch, reconstruction_weight):
    store, cfg, model = setup
    cfg = replace(cfg, jepa_weight=0, reconstruction_weight=reconstruction_weight)
    original = store._read_scan
    def legal_only(visit, *args):
        if visit["stage"] != 0:
            raise AssertionError("probe tried to read future MRI")
        return original(visit, *args)
    monkeypatch.setattr(store, "_read_scan", legal_only)
    report, private = run_training_probe(model, store, cfg, 5, make_probe_tasks(store, "t0", 4))
    assert not report["world_head_trained"]
    assert private["known_mask"].sum() == 4


def test_missing_visit_excludes_long_chains_but_keeps_later_adjacent_pair(setup):
    store, _, model = setup
    inp, sup = store.batch([(0, 0), (1, 2)])
    future = sup.future.clone(); mask = sup.future_mask.clone()
    future[0, 1] = 0; mask[0, 1] = False
    sup = replace(sup, future=future, future_mask=mask)
    model.eval()
    rows, _, _ = world_error_rows(model, inp, sup, [0, 1])
    continuous = [row for row in rows if row["scope"] == "continuous_chain"]
    adjacent = [row for row in rows if row["scope"] == "teacher_forcing_all_adjacent"]
    assert {(row["patient"], row["horizon"]) for row in continuous} == {(1, 1)}
    assert {(row["patient"], row["source_stage"], row["target_stage"]) for row in adjacent} == {(0, 2, 3), (1, 2, 3)}
    aggregate = aggregate_world_errors(rows)
    assert aggregate["teacher_forcing_all_adjacent"]["metrics"]["T2_to_T3"]["patients"] == 2
    assert aggregate["metrics"]["horizon_1"]["patients"] == 1


def test_future_targets_never_enter_autonomous_predictions(setup):
    store, _, model = setup
    inp, sup = store.batch([(0, 0), (1, 0)])
    model.eval()
    first, _, _ = world_error_rows(model, inp, sup, [0, 1])
    # Keep teacher targets changed only in supervision and intercept free states.
    original = model._rollout(inp)
    changed = replace(sup, future=sup.future + sup.future_mask[:, :, None, None, None, None] * 7)
    world_error_rows(model, inp, changed, [0, 1])
    assert torch.equal(model._rollout(inp), original)
    assert len([row for row in first if row["scope"] == "continuous_chain"]) == 6


def test_fixed_sampling_uses_local_rng_and_rejects_changed_resume_tasks(setup):
    store, cfg, model = setup
    state = rng_state()
    tasks = make_probe_tasks(store, "all_observed", 4, 71)
    assert tasks == make_probe_tasks(store, "all_observed", 4, 71)
    assert_rng_equal(state, rng_state())
    _, private = run_training_probe(model, store, cfg, 0, make_probe_tasks(store, "t0", 4))
    with pytest.raises(ValueError, match="tasks changed"):
        run_training_probe(model, store, cfg, 5, tuple(reversed(private["tasks"])), private)


def test_full_world_eval_counts_all_legal_prefixes_and_no_private_ids(setup):
    store, cfg, model = setup
    cfg = replace(cfg, landmarks="all_observed")
    result = evaluate_world(model, store, cfg, batch_size=2)
    assert result["patients"] == 2 and result["legal_prefixes"] == 8
    assert [result["metrics"][f"horizon_{h}"]["prefix_targets"] for h in (1, 2, 3)] == [6, 4, 2]
    assert result["verified_registration_subgroup"]["status"] == "unavailable"
    assert "patient_key" not in str(result) and "synthetic_" not in str(result)
