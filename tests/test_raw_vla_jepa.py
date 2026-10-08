"""Optimization and rollout contracts of the isolated VLA-style raw model."""
from dataclasses import replace

import pytest
import torch

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.model import RawVLAJEPAConfig, RawVLAJEPA, TimeCausalWorldPredictor


@pytest.fixture(autouse=True)
def tiny_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(81)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return RawVLAJEPAConfig(dim=24, token_grid=(1, 2, 2), heads=4,
                            encoder_depth=1, fusion_depth=1, predictor_depth=1,
                            state_queries=2, pcr_queries=2, fm_channels=8, ema_decay=.8)


@pytest.fixture
def model(cfg):
    return RawVLAJEPA(cfg)


@pytest.fixture
def cpu_bfloat16_math(monkeypatch):
    # Exercise scatter/gradient dtype handling independently of CPU-specific
    # fused attention and oneDNN BF16 kernels. PyTorch 2.5's fused eval path
    # misses CPU autocast; BF16 backward kernel support also varies by CPU ISA.
    monkeypatch.setattr(torch.backends.mha, "get_fastpath_enabled", lambda: False)
    with torch.backends.mkldnn.flags(enabled=False):
        yield


@pytest.fixture
def batch():
    observed = torch.tensor([[True, False, False, False], [True, True, False, False]])
    images = torch.zeros(2, 4, 3, 4, 8, 8)
    images[observed] = torch.randn(3, 3, 4, 8, 8)
    queries = torch.tensor([[False, True, True, True], [False, False, True, True]])
    inp = RawMRIInput(images, observed, torch.tensor([0, 1]), torch.randn(2, 4),
                      torch.ones(2, 4, dtype=torch.bool), torch.tensor([1, 13]),
                      torch.ones(2, dtype=torch.bool), torch.tensor([0, 0]), queries)
    future = torch.full_like(images, float("nan"))
    future[queries] = torch.randn(int(queries.sum()), 3, 4, 8, 8)
    sup = RawMRISupervision(future, queries.clone(), torch.tensor([1., 0.]),
                            torch.ones(2, dtype=torch.bool))
    return inp, sup


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_config_and_backbone_shapes(model, batch):
    inp, _ = batch
    assert not model.cfg.enable_flow and model.flow is None
    output = model(inp)
    assert set(output) == {"pcr_logit", "dynamics_features", "pcr_features"}
    assert output["pcr_logit"].shape == (2,)
    assert output["dynamics_features"].shape == (2, 4, 2, 24)
    assert output["pcr_features"].shape == (2, 2, 24)
    assert output["dynamics_features"][~inp.query_mask].count_nonzero() == 0


def test_world_block_causality_and_inactive_nan_isolation(cfg):
    world = TimeCausalWorldPredictor(cfg).eval()
    states = torch.randn(2, 3, 4, 24)
    transitions = torch.randn(2, 3, 2, 24)
    mask = torch.tensor([[True, True, True], [False, True, True]])
    original = world(states, transitions, mask)
    changed = states.clone()
    changed[:, 2] += torch.randn_like(changed[:, 2]) * 20
    modified = world(changed, transitions, mask)
    torch.testing.assert_close(original[:, :2], modified[:, :2], atol=0, rtol=0)
    assert not torch.allclose(original[:, 2], modified[:, 2])
    # Same block action/state interaction is unrestricted, not token causal.
    changed = states.clone()
    changed[0, 0, -1] += torch.randn_like(changed[0, 0, -1]) * 20
    assert not torch.allclose(world(changed, transitions, mask)[0, 0, 0], original[0, 0, 0])
    poisoned = states.clone()
    poisoned[~mask] = float("nan")
    poisoned_z = transitions.clone()
    poisoned_z[~mask] = float("nan")
    isolated = world(poisoned, poisoned_z, mask)
    torch.testing.assert_close(original, isolated, atol=0, rtol=0)
    empty = world(poisoned, poisoned_z, torch.zeros_like(mask))
    assert torch.isfinite(empty).all() and empty.count_nonzero() == 0


def test_teacher_forcing_patient_mean_pair_reduction(model, batch):
    inp, sup = batch
    output = model(inp)
    loss, detail = model.teacher_forcing_loss(inp, sup, output["dynamics_features"], True)
    assert detail["prediction"].shape == (2, 3, 4, 24)
    assert torch.equal(detail["pair_mask"], torch.tensor([[True, True, True], [False, True, True]]))
    expected = torch.stack((detail["pair_losses"][0].mean(), detail["pair_losses"][1, 1:].mean())).mean()
    torch.testing.assert_close(loss, expected)
    assert not detail["teacher_states"].requires_grad


def test_joint_world_and_task_gradients_do_not_update_teacher(model, batch):
    inp, sup = batch
    loss, terms = model.compute_loss(inp, sup, flow_weight=0)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
    assert terms["pair_count"] == 5
    torch.testing.assert_close(terms["jepa"], terms["world_model"])
    loss.backward()
    assert has_grad(model.encoder) and has_grad(model.fusion)
    assert has_grad(model.world_predictor) and has_grad(model.pcr_head)
    assert model.dynamics_queries.grad.abs().sum() > 0
    assert model.pcr_queries.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.teacher_encoder.parameters())


def test_pcr_gradient_reaches_dynamics_queries_without_world_head(model, batch):
    inp, sup = batch
    loss, _ = model.compute_loss(inp, sup, jepa_weight=0, flow_weight=0,
                                 reconstruction_weight=0, variance_weight=0)
    loss.backward()
    assert has_grad(model.pcr_head) and has_grad(model.encoder)
    assert model.dynamics_queries.grad.abs().sum() > 0
    assert not has_grad(model.world_predictor)


def test_representation_stage_disables_pcr_loss_and_head_gradients(model, batch):
    inp, sup = batch
    loss, terms = model.compute_loss(inp, sup, representation_only=True)
    assert terms["task"] == terms["flow"] == 0
    loss.backward()
    assert has_grad(model.encoder) and has_grad(model.world_predictor)
    assert not has_grad(model.pcr_head)
    assert model.pcr_queries.grad is None or model.pcr_queries.grad.count_nonzero() == 0


def test_ema_exact_update_and_teacher_always_eval(model):
    before = [p.detach().clone() for p in model.teacher_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(.25)
    model.update_teacher()
    for old, current, teacher in zip(before, model.encoder.parameters(), model.teacher_encoder.parameters(), strict=True):
        torch.testing.assert_close(teacher, old * .8 + current * .2)
    model.train()
    assert model.encoder.training and not model.teacher_encoder.training


def test_free_forecast_is_recursive_and_input_only(model, batch, monkeypatch):
    inp, _ = batch
    model.eval()
    predictions = model.forecast(inp)
    assert predictions["state_prediction"].shape == (2, 4, 4, 24)
    assert predictions["state_prediction"][~inp.query_mask].count_nonzero() == 0
    torch.testing.assert_close(predictions["pcr_logit"], model(inp)["pcr_logit"], atol=2e-6, rtol=2e-6)
    calls = []
    original = model.world_predictor.forward
    def record(source, dynamics, mask):
        calls.append((source.detach().clone(), mask.detach().clone()))
        return original(source, dynamics, mask)
    monkeypatch.setattr(model.world_predictor, "forward", record)
    result = model.forecast_states(inp)
    assert len(calls) == 3
    torch.testing.assert_close(calls[1][0][0, 1], result[0, 1], atol=0, rtol=0)
    torch.testing.assert_close(calls[2][0][0, 2], result[0, 2], atol=0, rtol=0)
    assert not calls[0][1][1].any()  # T1 patient waits until transition 1->2.


def test_sparse_forecast_requests_compute_intermediates_but_return_requested_only(model, batch):
    inp, _ = batch
    model.eval()
    complete = model.forecast_states(inp)
    sparse_mask = torch.zeros_like(inp.query_mask)
    sparse_mask[:, 3] = True
    sparse = model.forecast_states(replace(inp, query_mask=sparse_mask))
    torch.testing.assert_close(sparse[:, 3], complete[:, 3], atol=0, rtol=0)
    assert sparse[:, :3].count_nonzero() == 0


def test_terminal_missing_current_mri_does_not_need_rollout(model, batch, monkeypatch):
    inp, _ = batch
    terminal = replace(inp, landmark=torch.tensor([3, 3]), query_mask=torch.zeros_like(inp.query_mask))
    def forbidden(*args, **kwargs):
        raise AssertionError("Empty forecast must not execute world or teacher")
    monkeypatch.setattr(model.world_predictor, "forward", forbidden)
    monkeypatch.setattr(model.teacher_encoder, "forward", forbidden)
    assert model.forecast_states(terminal).count_nonzero() == 0
    assert torch.isfinite(model(terminal)["pcr_logit"]).all()


def test_missing_all_labels_and_future_is_finite(model, batch):
    inp, sup = batch
    missing = replace(sup, future=torch.full_like(sup.future, float("nan")),
                      future_mask=torch.zeros_like(sup.future_mask),
                      label=torch.full_like(sup.label, float("nan")), label_mask=torch.zeros_like(sup.label_mask))
    loss, terms = model.compute_loss(inp, missing, reconstruction_weight=0, variance_weight=0)
    assert loss.requires_grad and loss == 0
    assert terms["pair_count"] == 0
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_optional_flow_uses_free_forecast_and_backpropagates(cfg, batch, monkeypatch):
    model = RawVLAJEPA(replace(cfg, enable_flow=True))
    inp, sup = batch
    conditions = []
    original_flow = model.flow.forward
    def record(value, tau, source, future, stage):
        conditions.append(future.detach().clone())
        return original_flow(value, tau, source, future, stage)
    monkeypatch.setattr(model.flow, "forward", record)
    model.eval()
    free = model.forecast_states(inp)
    loss, _ = model.compute_loss(inp, sup, jepa_weight=0, task_weight=0,
                                 reconstruction_weight=0, variance_weight=0, flow_weight=1)
    # Inference can use fused no-grad attention; training uses its backward-
    # capable counterpart. Their roundoff need not be bit-identical.
    torch.testing.assert_close(conditions[0], free[sup.future_mask], atol=2e-6, rtol=2e-6)
    loss.backward()
    assert has_grad(model.flow) and has_grad(model.world_predictor)
    assert has_grad(model.encoder) and model.dynamics_queries.grad.abs().sum() > 0


def test_optional_generation_returns_seeded_future_without_changing_pcr(cfg, batch):
    model = RawVLAJEPA(replace(cfg, enable_flow=True)).eval()
    inp, _ = batch
    direct = model(inp)["pcr_logit"].sigmoid()
    first = model.generate(inp, steps=1, seed=4)
    second = model.generate(inp, steps=1, seed=4)
    assert first["future_images"].shape == inp.images.shape
    assert first["future_images"][~inp.query_mask].count_nonzero() == 0
    assert not first["future_images"].requires_grad
    torch.testing.assert_close(first["future_images"], second["future_images"], atol=0, rtol=0)
    torch.testing.assert_close(first["pcr_probability"], direct, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("enable_flow", [False, True])
def test_bfloat16_autocast_world_loss_and_free_forecast_have_safe_scatter_dtypes(
        cfg, batch, enable_flow, cpu_bfloat16_math):
    model = RawVLAJEPA(replace(cfg, enable_flow=enable_flow))
    inp, sup = batch
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, terms = model.compute_loss(inp, sup)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    assert has_grad(model.encoder) and has_grad(model.world_predictor)
    if enable_flow:
        assert has_grad(model.flow)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        states = model.forecast_states(inp)
    assert states.dtype == inp.images.dtype
    assert torch.isfinite(states).all() and not states.requires_grad


def test_direct_pcr_tiny_cohort_optimization(model, batch):
    inp, sup = batch
    optimizer = torch.optim.Adam(model.parameters(), lr=.008)
    kwargs = dict(jepa_weight=0, flow_weight=0, reconstruction_weight=0, variance_weight=0)
    initial = model.compute_loss(inp, sup, **kwargs)[0].item()
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(inp, sup, **kwargs)
        loss.backward()
        optimizer.step()
    assert model.compute_loss(inp, sup, **kwargs)[0].item() < initial * .2
