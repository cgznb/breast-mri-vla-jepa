"""Independent leakage and missing-visit checks for raw-MRI VLA-JEPA."""
from dataclasses import replace

import pytest
import torch

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.model import RawVLAJEPA, RawVLAJEPAConfig


@pytest.fixture(autouse=True)
def tiny_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(previous)


def _model(*, enable_flow=False):
    return RawVLAJEPA(RawVLAJEPAConfig(
        dim=24, heads=4, token_grid=(1, 2, 2),
        encoder_depth=1, fusion_depth=1, predictor_depth=1,
        state_queries=2, pcr_queries=2, fm_channels=8,
        enable_flow=enable_flow,
    )).eval()


def _case(landmark=0, available=(True, True, True, True)):
    """Fake scans stay in their canonical slots; hidden supervision is poisoned."""
    stages = torch.arange(4)[None]
    real = torch.tensor([available], dtype=torch.bool)
    observed = real & (stages <= landmark)
    query = stages > landmark
    scans = torch.randn(1, 4, 3, 4, 8, 8)
    images = torch.zeros_like(scans)
    images[observed] = scans[observed]
    inp = RawMRIInput(
        images, observed, torch.tensor([landmark]),
        torch.tensor([[.4, 1., 0., 1.]]), torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([2]), torch.tensor([True]), torch.tensor([0]), query,
    )
    future_mask = query & real
    future = torch.full_like(scans, float("nan"))
    future[future_mask] = scans[future_mask]
    sup = RawMRISupervision(future, future_mask, torch.tensor([1.]),
                            torch.tensor([True]))
    return inp, sup


def _teacher_details(model, inp, sup, dynamics=None):
    if dynamics is None:
        dynamics = model(inp)["dynamics_features"]
    return model.teacher_forcing_loss(inp, sup, dynamics, return_details=True)


def _has_gradient(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0
               for p in module.parameters())


def test_true_s1_is_teacher_input_for_s2_but_never_for_s1_or_pcr():
    model = _model()
    inp, sup = _case()
    direct = model(inp)
    _, before = _teacher_details(model, inp, sup, direct["dynamics_features"])
    future = sup.future.clone()
    # Change spatial content, rather than a scale/offset removed by GroupNorm.
    future[:, 1] = torch.randn_like(future[:, 1])
    _, after = _teacher_details(model, inp, replace(sup, future=future),
                                direct["dynamics_features"])
    assert before["pair_mask"].tolist() == [[True, True, True]]
    torch.testing.assert_close(after["teacher_states"][:, 0],
                               before["teacher_states"][:, 0], atol=0, rtol=0)
    assert not torch.allclose(after["teacher_states"][:, 1],
                              before["teacher_states"][:, 1])
    torch.testing.assert_close(after["prediction"][:, 0],
                               before["prediction"][:, 0], atol=0, rtol=0)
    assert not torch.allclose(after["prediction"][:, 1], before["prediction"][:, 1])
    torch.testing.assert_close(model(inp)["pcr_logit"], direct["pcr_logit"],
                               atol=0, rtol=0)


def test_true_s3_is_target_only_and_cannot_change_world_inputs_or_predictions():
    model = _model()
    inp, sup = _case()
    dynamics = model(inp)["dynamics_features"]
    _, before = _teacher_details(model, inp, sup, dynamics)
    future = sup.future.clone()
    future[:, 3] = torch.randn_like(future[:, 3])
    _, after = _teacher_details(model, inp, replace(sup, future=future), dynamics)
    torch.testing.assert_close(after["source_states"], before["source_states"],
                               atol=0, rtol=0)
    torch.testing.assert_close(after["prediction"], before["prediction"],
                               atol=0, rtol=0)
    torch.testing.assert_close(after["teacher_states"][:, :3],
                               before["teacher_states"][:, :3], atol=0, rtol=0)
    assert not torch.allclose(after["targets"][:, 2], before["targets"][:, 2])


def test_future_images_and_labels_do_not_change_ordinary_forward():
    model = _model()
    inp, sup = _case()
    before = model(inp)
    future = sup.future.clone()
    future[sup.future_mask] = torch.randn_like(future[sup.future_mask])
    changed = replace(sup, future=future, label=1 - sup.label)
    model.compute_loss(inp, changed, flow_weight=0,
                        reconstruction_weight=0, variance_weight=0)
    after = model(inp)
    for name in before:
        torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)


def test_ordinary_pcr_does_not_execute_teacher_world_predictor_or_flow(monkeypatch):
    model = _model(enable_flow=True)
    inp, _ = _case()

    def forbidden(*args, **kwargs):
        raise AssertionError("pCR-only inference executed a training or generation branch")

    monkeypatch.setattr(model.teacher_encoder, "forward", forbidden)
    monkeypatch.setattr(model.world_predictor, "forward", forbidden)
    monkeypatch.setattr(model.flow, "forward", forbidden)
    output = model(inp)
    assert output["pcr_logit"].shape == (1,)
    assert torch.isfinite(output["pcr_logit"]).all()


def test_missing_t1_masks_both_adjacent_pairs_without_compressing_visits():
    model = _model()
    inp, sup = _case(available=(True, False, True, True))
    loss, details = _teacher_details(model, inp, sup)
    assert details["pair_mask"].tolist() == [[False, False, True]]
    assert details["prediction"].shape == (1, 3, 4, 24)
    assert torch.isfinite(loss)
    _, terms = model.compute_loss(inp, sup, flow_weight=0)
    assert terms["pair_count"].item() == 1


def test_t1_landmark_uses_observed_s1_and_supervises_only_remaining_pairs():
    model = _model()
    inp, sup = _case(landmark=1)
    _, details = _teacher_details(model, inp, sup)
    assert details["pair_mask"].tolist() == [[False, True, True]]
    future = sup.future.clone()
    # T1 is observed input; the unselected supervision slot is never read.
    future[:, 1] = torch.randn_like(future[:, 1])
    _, changed = _teacher_details(model, inp, replace(sup, future=future))
    torch.testing.assert_close(changed["teacher_states"], details["teacher_states"],
                               atol=0, rtol=0)


def test_t3_is_a_valid_task_landmark_with_zero_world_loss_and_finite_backward():
    model = _model()
    inp, sup = _case(landmark=3)
    loss, terms = model.compute_loss(inp, sup, flow_weight=0,
                                    reconstruction_weight=0, variance_weight=0)
    assert torch.isfinite(loss) and loss.item() > 0
    assert terms["task"].item() > 0
    assert terms["world_model"].item() == 0
    assert terms["pair_count"].item() == 0
    loss.backward()
    assert _has_gradient(model.encoder) and _has_gradient(model.pcr_head)
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert model.forecast_states(inp).count_nonzero() == 0


def test_masked_nan_targets_and_absent_labels_are_not_encoded_or_reduced():
    model = _model()
    inp, sup = _case(available=(True, False, True, True))
    sup = replace(sup, label=torch.tensor([float("nan")]),
                  label_mask=torch.tensor([False]))
    loss, details = _teacher_details(model, inp, sup)
    assert torch.isfinite(loss)
    assert torch.isfinite(details["teacher_states"]).all()
    total, terms = model.compute_loss(inp, sup, flow_weight=0,
                                     reconstruction_weight=0, variance_weight=0)
    assert torch.isfinite(total) and terms["task"].item() == 0
    total.backward()
    assert _has_gradient(model.world_predictor)
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert all(p.grad is None for p in model.teacher_encoder.parameters())


def test_no_future_targets_retains_a_graph_connected_zero_world_loss():
    model = _model()
    inp, sup = _case(available=(True, False, False, False))
    sup = replace(sup, label=torch.tensor([float("nan")]),
                  label_mask=torch.tensor([False]))
    loss, terms = model.compute_loss(inp, sup, flow_weight=0,
                                    reconstruction_weight=0, variance_weight=0)
    assert loss.requires_grad and loss.item() == 0
    assert terms["world_model"].item() == terms["pair_count"].item() == 0
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_missing_current_scan_allows_classification_but_rejects_autonomous_rollout():
    model = _model()
    inp, _ = _case(landmark=1, available=(True, False, True, True))
    assert torch.isfinite(model(inp)["pcr_logit"]).all()
    with pytest.raises(ValueError, match="current|landmark"):
        model.forecast_states(inp)
