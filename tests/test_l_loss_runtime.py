"""Single-factor loss ablations, target isolation, and matched-patient variance."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.data_pipeline import BatchLoader
from mri_vla_jepa.io import load_checkpoint
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig, load_config
from mri_vla_jepa.training import _checkpoint_config_matches, predict, train


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def tiny_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(81)
    yield
    torch.set_num_threads(previous)


def _batch():
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    images = torch.zeros(2, 4, 3, 4, 8, 8)
    images[mask] = torch.randn(3, 3, 4, 8, 8)
    queries = torch.tensor([[False, True, True, True], [False, False, True, True]])
    inp = RawMRIInput(images, mask, torch.tensor([0, 1]), torch.randn(2, 4),
                      torch.ones(2, 4, dtype=torch.bool), torch.tensor([1, 2]),
                      torch.ones(2, dtype=torch.bool), torch.tensor([0, 0]), queries)
    future = torch.full_like(images, float("nan"))
    future[queries] = torch.randn(int(queries.sum()), 3, 4, 8, 8)
    sup = RawMRISupervision(future, queries.clone(), torch.tensor([1., 0.]),
                            torch.ones(2, dtype=torch.bool))
    return inp, sup


def _model():
    return RawVLAJEPA(RawVLAJEPAConfig(dim=24, token_grid=(1, 2, 2), heads=4,
                      encoder_depth=1, fusion_depth=1, predictor_depth=1,
                      state_queries=2, pcr_queries=2, dropout=0)).eval()


@pytest.mark.parametrize("jepa,reconstruction,variance,definition", [
    (0, 0, 0, "off"), (.5, 0, .01, "legacy"),
    (0, .1, .01, "legacy"), (.5, .1, .01, "patient_axis_stage_token_fp32"),
])
def test_supervised_loss_configs(jepa, reconstruction, variance, definition):
    cfg = RawVLATrainConfig(jepa_weight=jepa, reconstruction_weight=reconstruction,
                           variance_weight=variance, variance_definition=definition)
    assert cfg.validate() is cfg


def test_empty_representation_and_inconsistent_variance_rejected():
    cfg = RawVLATrainConfig(jepa_weight=0, reconstruction_weight=0, variance_weight=0)
    cfg.schedule, cfg.representation_steps = "two_stage", 1
    with pytest.raises(ValueError, match="active auxiliary loss"):
        cfg.validate()
    cfg.model.enable_flow, cfg.flow_weight = True, 1
    with pytest.raises(ValueError, match="active auxiliary loss"):
        cfg.validate()
    with pytest.raises(ValueError, match="variance_weight=0"):
        RawVLATrainConfig(variance_definition="off").validate()
    with pytest.raises(ValueError, match="Unsupported variance_definition"):
        RawVLATrainConfig(variance_definition="typo").validate()


def test_legacy_config_missing_new_fields_loads_defaults():
    cfg = RawVLATrainConfig()
    saved = cfg.to_dict()
    for name in ("variance_definition", "diagnostics_every_epochs", "diagnostics_patients",
                 "diagnostics_batch_size", "diagnostics_seed"):
        saved.pop(name)
    assert _checkpoint_config_matches(saved, cfg)
    cfg.variance_definition = "patient_axis_stage_token_fp32"
    assert not _checkpoint_config_matches(saved, cfg)


def test_pcr_only_skips_all_auxiliaries_and_target_values(monkeypatch):
    model, (inp, sup) = _model(), _batch()
    sup = replace(sup, future=torch.full_like(sup.future, float("nan")))

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled auxiliary branch was executed")

    monkeypatch.setattr(model.teacher_encoder, "forward", forbidden)
    monkeypatch.setattr(model.world_predictor, "forward", forbidden)
    monkeypatch.setattr(model.reconstruction, "forward", forbidden)
    monkeypatch.setattr(model, "variance_penalty", forbidden)
    loss, terms = model.compute_loss(inp, sup, jepa_weight=0, flow_weight=0,
                   reconstruction_weight=0, variance_weight=0, variance_definition="off")
    expected = F.binary_cross_entropy_with_logits(model(inp)["pcr_logit"], sup.label)
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    for name in ("jepa", "flow", "reconstruction", "variance", "pair_count"):
        assert terms[name] == 0
    loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.world_predictor.parameters())
    assert all(p.grad is None for p in model.reconstruction.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.teacher_encoder.parameters())


def test_default_full_loss_retains_exact_legacy_formula():
    model, (inp, sup) = _model(), _batch()
    loss, terms = model.compute_loss(inp, sup, jepa_weight=.5, flow_weight=0)
    output, observed = model._backbone(inp)
    spatial = observed.transpose(1, 2).reshape(len(observed), model.cfg.dim, *model.cfg.token_grid)
    reconstructed = F.interpolate(model.reconstruction(spatial), size=inp.images.shape[-3:],
                                   mode="trilinear", align_corners=False)
    expected_task = F.binary_cross_entropy_with_logits(output["pcr_logit"], sup.label)
    expected_jepa = model.teacher_forcing_loss(inp, sup, output["dynamics_features"])
    expected_reconstruction = F.mse_loss(reconstructed, inp.images[inp.observed_mask])
    expected_variance = F.relu(1 - (observed.flatten(0, 1).var(0, unbiased=False) + 1e-4).sqrt()).mean()
    expected = expected_task + .5 * expected_jepa + .1 * expected_reconstruction + .01 * expected_variance
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    torch.testing.assert_close(terms["variance"], expected_variance, atol=0, rtol=0)


def test_patient_variance_does_not_count_spatial_or_stage_diversity():
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    repeated = torch.tensor([[-2.], [2.]])
    observed = torch.stack((repeated, repeated + 100, repeated, repeated + 100)).requires_grad_()
    legacy = RawVLAJEPA.variance_penalty(observed, mask, "legacy")
    corrected = RawVLAJEPA.variance_penalty(observed, mask, "patient_axis_stage_token_fp32")
    assert legacy == 0
    assert corrected.item() == pytest.approx(.99)
    corrected.backward()
    assert torch.isfinite(observed.grad).all()


def test_patient_variance_fp32_valid_stage_mean_and_singleton_skip():
    mask = torch.tensor([[True, True, False, False], [True, False, True, False],
                         [True, False, True, False]])
    states = torch.zeros(3, 4, 2, 2, dtype=torch.bfloat16)
    states[0, 0], states[1, 0], states[2, 0] = 0, .5, 1
    states[0, 1] = 1000
    states[1, 2], states[2, 2] = 0, .25
    observed = states[mask].requires_grad_()
    actual = RawVLAJEPA.variance_penalty(observed, mask, "patient_axis_stage_token_fp32")
    expected = torch.stack([F.relu(1 - (states[mask[:, s], s].float().var(0, unbiased=False)
                                      + 1e-4).sqrt()).mean() for s in (0, 2)]).mean()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    assert observed.grad[1].count_nonzero() == 0
    single = RawVLAJEPA.variance_penalty(observed[:1], mask[:1] & torch.tensor([True, False, False, False]),
                                       "patient_axis_stage_token_fp32")
    assert single.dtype == torch.float32 and single == 0


@pytest.mark.parametrize("prefetch", [0, 2])
def test_labels_without_future_reads_and_prefetch(tmp_path, monkeypatch, prefetch):
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=2)
    store = RawMRIStore(manifest, (8, 16, 16), allow_synthetic=True)
    original_read = store._read_scan

    def observed_only(visit, reference=None):
        assert visit["stage"] == 0, "Future file was read"
        return original_read(visit, reference)

    monkeypatch.setattr(store, "_read_scan", observed_only)
    tasks = [(0, 0), (1, 0)]
    with BatchLoader(store, prefetch_batches=prefetch, future_supervision=False) as loader:
        if prefetch:
            loader.plan([tasks])
        inp, sup = loader.fetch(tasks)
        loader.finish_epoch()
    assert sup.label_mask.all() and sup.label.tolist() == [0, 1]
    assert not sup.future_mask.any() and sup.future.count_nonzero() == 0
    assert inp.observed_mask[:, 0].all()


@pytest.mark.parametrize("reconstruction,variance,definition", [(0, 0, "off"), (.1, .01, "legacy")])
def test_pcr_or_observed_losses_train_without_future_targets(tmp_path, monkeypatch,
                                                            reconstruction, variance, definition):
    cfg = load_config(ROOT / "configs/raw_vla_jepa_smoke_t0.yaml")
    cfg.joint_steps, cfg.validation_every, cfg.batch_size = 2, 1, 2
    cfg.jepa_weight, cfg.reconstruction_weight, cfg.variance_weight = 0, reconstruction, variance
    cfg.variance_definition = definition
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=2, image_shape=cfg.image_shape)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    original_read = store._read_scan

    def observed_only(visit, reference=None):
        assert visit["stage"] == 0, "Future file was read during training"
        return original_read(visit, reference)

    monkeypatch.setattr(store, "_read_scan", observed_only)
    assert train(store, cfg, tmp_path / "run")["completed"]
    state = load_checkpoint(tmp_path / "run/last.pt")
    assert all(row["future_count"] == 0 and row["jepa"] == 0 for row in state["history"])
    with pytest.raises(ValueError, match="trained JEPA world head"):
        predict(tmp_path / "run/best.pt", manifest, "synthetic_0", tmp_path / "forecast.json",
                allow_synthetic=True, forecast_states=True)


def _assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for name in a:
            _assert_same(a[name], b[name])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_same(left, right)
    else:
        assert a == b


def test_flow_trained_world_head_can_forecast_without_jepa(tmp_path, monkeypatch):
    from mri_vla_jepa import training
    model = RawVLAJEPA(replace(_model().cfg, enable_flow=True)).eval()
    cfg = RawVLATrainConfig(model=model.cfg, image_shape=(4, 8, 8), landmarks="t0",
                           jepa_weight=0, flow_weight=1, reconstruction_weight=0,
                           variance_weight=0, variance_definition="off", device="cpu",
                           precision="fp32", allow_synthetic=True).validate()
    manifest = make_raw_synthetic(tmp_path / "data", n_train=2, n_val=2, image_shape=cfg.image_shape)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    monkeypatch.setattr(training, "load_trained", lambda *args: (model, cfg,
                        {"normalization": store.normalization_state()}))
    report = predict("flow_checkpoint.pt", manifest, "synthetic_0", tmp_path / "forecast.json",
                     allow_synthetic=True, forecast_states=True)
    assert report["forecast_shape"] == [1, 4, 4, 24]


def test_diagnostic_epoch_resume_exact_and_training_unchanged(tmp_path):
    cfg = load_config(ROOT / "configs/raw_vla_jepa_smoke_dynamic.yaml")
    cfg.training_unit, cfg.accumulation, cfg.batch_size = "epochs", 1, 2
    cfg.max_epochs, cfg.early_stopping_patience = 3, 5
    cfg.diagnostics_every_epochs, cfg.diagnostics_patients, cfg.diagnostics_batch_size = 1, 4, 2
    cfg.variance_definition = "patient_axis_stage_token_fp32"
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=2, image_shape=cfg.image_shape)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    train(store, cfg, tmp_path / "full")
    train(store, cfg, tmp_path / "resumed", stop_after=3)
    train(store, cfg, tmp_path / "resumed", resume=True)
    full, resumed = [load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "resumed")]
    for name in ("model", "optimizer", "sampler_state", "history", "epoch_state", "epoch_history", "diagnostics_state"):
        _assert_same(full[name], resumed[name])
    # Diagnostic probes and sampled update measurements consume no training randomness.
    cfg.diagnostics_every_epochs = 0
    train(store, cfg, tmp_path / "without_diagnostics")
    reference = load_checkpoint(tmp_path / "without_diagnostics/last.pt")
    _assert_same(full["model"], reference["model"])
    _assert_same(full["optimizer"], reference["optimizer"])
    assert len(full["diagnostics_state"]["reports"]) == 3
    assert all(0 <= row["training"]["gradient_clip_fraction"] <= 1 for row in full["epoch_history"])
    assert all(row["training"]["sampled_optimizer_update_norm"] > 0 for row in full["epoch_history"])
    samples = [row for row in full["history"] if "optimizer_update_norm" in row]
    assert len(samples) == 3
    assert all(row["optimizer_update_scope"] == "first_batch_of_diagnostic_epoch" for row in samples)
