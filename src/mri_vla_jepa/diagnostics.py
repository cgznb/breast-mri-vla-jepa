"""Fixed, read-only loss and world probes; these never select a checkpoint."""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .io import autocast, restore_rng, rng_state


@contextmanager
def preserve_training_state(model):
    """Restore all RNGs and individual module modes, including frozen submodules."""
    state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    mps = torch.mps.get_rng_state() if next(model.parameters()).device.type == "mps" else None
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(state)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def make_probe_tasks(store, profile, patients=32, seed=20261004):
    if type(patients) is not int or patients < 1:
        raise ValueError("Probe patient count must be positive")
    if profile not in {"t0", "all_observed"}:
        raise ValueError("Probe profile must be t0 or all_observed")
    indices = list(store.by_split["train"])
    chooser = random.Random(seed)
    chooser.shuffle(indices)
    stages = random.Random(seed + 1)
    return tuple((i, 0 if profile == "t0" else stages.choice(store.allowed_landmarks(i)))
                 for i in indices[:patients])


def _groups(names):
    def group(name):
        if name.startswith("encoder."):
            return "encoder"
        if name.startswith(("fusion.", "fusion_norm.")):
            return "fusion"
        for key in ("pcr_head", "world_predictor", "reconstruction", "flow"):
            if name.startswith(key + "."):
                return key
        return "conditioning_and_queries"
    result = {key: [i for i, name in enumerate(names) if group(name) == key]
              for key in ("encoder", "fusion", "conditioning_and_queries", "pcr_head",
                          "world_predictor", "reconstruction", "flow")}
    result["shared_encoder_fusion"] = result["encoder"] + result["fusion"]
    result["all_trainable"] = list(range(len(names)))
    return result


def weighted_gradient_metrics(model, terms, weights):
    """autograd.grad leaves existing parameter .grad values untouched."""
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*named)
    groups = _groups(names)
    gradients = {}
    for key, weight in weights.items():
        value = weight * terms[key]
        gradients[key] = (torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                          if weight and value.requires_grad else (None,) * len(parameters))
    vectors = {key: {group: torch.cat([(grads[i].detach().float() if grads[i] is not None
                    else torch.zeros_like(parameters[i], dtype=torch.float32)).flatten()
                    for i in indices]) if indices else parameters[0].new_zeros(0)
                    for group, indices in groups.items()} for key, grads in gradients.items()}
    norms = {key: {group: float(torch.linalg.vector_norm(vector))
                  for group, vector in values.items()} for key, values in vectors.items()}
    cosines = {}
    for key in weights:
        if key == "task":
            continue
        cosines[key] = {}
        for group in ("encoder", "fusion", "shared_encoder_fusion"):
            denominator = norms["task"][group] * norms[key][group]
            cosines[key][group] = (float(torch.dot(vectors["task"][group], vectors[key][group]))
                                  / denominator if denominator else None)
    finite = all(math.isfinite(value) for values in norms.values() for value in values.values())
    if not finite:
        raise ValueError("Non-finite fixed-probe gradients")
    return {"weighted_gradient_norms": norms, "task_auxiliary_cosines": cosines,
            "enabled_terms": {key: bool(weight) for key, weight in weights.items()},
            "all_gradient_values_finite": finite}


def _effective_rank(features):
    if len(features) < 2:
        return None
    singular = torch.linalg.svdvals(features.double() - features.double().mean(0))
    values = singular.square()
    if float(values.sum()) == 0:
        return 0.0
    probabilities = values[values > 0] / values.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def representation_metrics(states, mask, landmarks):
    states = states.float().cpu()
    mask, landmarks = mask.cpu(), landmarks.cpu()
    selected = states[mask]
    by_stage = {}
    penalties = []
    for stage in range(4):
        values = states[mask[:, stage], stage]
        variance = values.var(0, unbiased=False) if len(values) >= 2 else None
        penalty = float(F.relu(1 - (variance + 1e-4).sqrt()).mean()) if variance is not None else None
        if penalty is not None:
            penalties.append(penalty)
        pooled = values.mean(1)
        by_stage[f"T{stage}"] = {"patients": len(values), "patient_axis_variance_mean":
                                 float(variance.mean()) if variance is not None else None,
                                 "patient_axis_variance_penalty": penalty,
                                 "patient_pooled_effective_rank": _effective_rank(pooled)}
    anchors = states[torch.arange(len(states)), landmarks].mean(1)
    return {"mean_absolute": float(selected.abs().mean()), "rms": float(selected.square().mean().sqrt()),
            "global_std": float(selected.std(unbiased=False)),
            "legacy_variance_penalty": float(F.relu(1 - (selected.flatten(0, 1).var(0, unbiased=False) + 1e-4).sqrt()).mean()),
            "patient_axis_variance_penalty": float(np.mean(penalties)) if penalties else None,
            "anchor_patient_pooled_variance": float(anchors.var(0, unbiased=False).mean()),
            "anchor_patient_pooled_effective_rank": _effective_rank(anchors), "per_stage": by_stage}


def world_error_rows(model, inp, sup, patient_indices):
    """Targets use this model's EMA teacher; rollout only receives legal input."""
    with torch.no_grad():
        output = model(inp)
        _, details = model.teacher_forcing_loss(inp, sup, output["dynamics_features"], True)
        forecast = model._rollout(inp, output["dynamics_features"])
        states, known = details["teacher_states"].float(), details["known_mask"]
        rows = []
        for row, index in enumerate(patient_indices):
            start = int(inp.landmark[row])
            for source in range(3):
                if not bool(details["pair_mask"][row, source]):
                    continue
                real = states[row, source + 1]
                rows.append({"scope": "teacher_forcing_all_adjacent", "patient": index,
                    "source_stage": source, "target_stage": source + 1,
                    "teacher_forcing_l1": float((details["prediction"][row, source].float() - real).abs().mean()),
                    "teacher_forcing_copy_l1": float((states[row, source] - real).abs().mean())})
            for horizon in range(1, 4 - start):
                target = start + horizon
                # Gaps are excluded instead of inventing adjacent physical visits.
                if not bool(known[row, start:target + 1].all()):
                    continue
                real = states[row, target]
                rows.append({"scope": "continuous_chain", "patient": index, "horizon": horizon,
                    "teacher_forcing_l1": float((details["prediction"][row, target - 1].float() - real).abs().mean()),
                    "autonomous_l1": float((forecast[row, target].float() - real).abs().mean()),
                    "teacher_forcing_copy_l1": float((states[row, target - 1] - real).abs().mean()),
                    "autonomous_copy_l1": float((states[row, start] - real).abs().mean())})
    return rows, states.detach().cpu(), known.detach().cpu()


def aggregate_world_errors(rows):
    keys = ("teacher_forcing_l1", "autonomous_l1", "teacher_forcing_copy_l1", "autonomous_copy_l1")
    if any(not math.isfinite(row[key]) or row[key] < 0 for row in rows for key in keys if key in row):
        raise ValueError("World evaluation errors must be finite and nonnegative")
    adjacent = [row for row in rows if row.get("scope") == "teacher_forcing_all_adjacent"]
    rows = [row for row in rows if row.get("scope", "continuous_chain") == "continuous_chain"]
    result = {}
    for horizon in (1, 2, 3, "all"):
        selected = [row for row in rows if horizon == "all" or row["horizon"] == horizon]
        grouped = {}
        for row in selected:
            grouped.setdefault(row["patient"], []).append(row)
        means = {key: float(np.mean([np.mean([row[key] for row in values])
                                    for values in grouped.values()])) if grouped else None for key in keys}
        for name in ("teacher_forcing", "autonomous"):
            error, copy = means[f"{name}_l1"], means[f"{name}_copy_l1"]
            ratio = error / copy if copy is not None and copy > 1e-8 else None
            means[f"{name}_relative_to_copy"] = ratio
            means[f"{name}_skill_vs_copy"] = 1 - ratio if ratio is not None else None
            means[f"{name}_minus_copy"] = error - copy if error is not None else None
            means[f"{name}_near_zero_copy_patients"] = int(sum(
                np.mean([row[f"{name}_copy_l1"] for row in values]) <= 1e-8 for values in grouped.values()))
        result[f"horizon_{horizon}"] = {"patients": len(grouped), "prefix_targets": len(selected), **means}
    return {"status": "evaluated", "world_head_trained": True,
            "reduction": "mean eligible prefix-targets within patient, then mean patients",
            "relative_error_guard": "ratio and skill are null if aggregate copy L1 <= 1e-8 or no eligible patients",
            "target_coordinates": "same checkpoint EMA teacher for model and both copy baselines",
            "chain_policy": "real continuous visits only; target and all intermediate visits required",
            "teacher_forcing_copy": "real immediate predecessor",
            "autonomous_copy": "real landmark state held constant", "metrics": result,
            "teacher_forcing_all_adjacent": _aggregate_adjacent(adjacent)}


def _aggregate_adjacent(rows):
    result = {}
    for source in (0, 1, 2, "all"):
        selected = [row for row in rows if source == "all" or row["source_stage"] == source]
        grouped = {}
        for row in selected:
            grouped.setdefault(row["patient"], []).append(row)
        means = {key: float(np.mean([np.mean([row[key] for row in values])
                                    for values in grouped.values()])) if grouped else None
                 for key in ("teacher_forcing_l1", "teacher_forcing_copy_l1")}
        error, copy = means["teacher_forcing_l1"], means["teacher_forcing_copy_l1"]
        ratio = error / copy if copy is not None and copy > 1e-8 else None
        result["all" if source == "all" else f"T{source}_to_T{source + 1}"] = {
            "patients": len(grouped), "prefix_pairs": len(selected), **means,
            "relative_to_copy": ratio, "skill_vs_copy": 1 - ratio if ratio is not None else None,
            "near_zero_copy_patients": int(sum(np.mean([row["teacher_forcing_copy_l1"]
                for row in values]) <= 1e-8 for values in grouped.values()))}
    return {"scope": "all real adjacent supervised pairs, including pairs after a prior missing visit",
            "reduction": "mean eligible prefix-pairs within patient, then mean patients", "metrics": result}


def _teacher_drift(states, mask, previous):
    if previous is None:
        return {"status": "first_probe", "l1": None, "relative_to_previous_rms": None,
                "adjacent_copy_change_l1": None}
    before = previous["teacher_states"]
    if before.shape != states.shape or not torch.equal(previous["known_mask"], mask):
        raise ValueError("Fixed-probe teacher coverage changed on resume")
    drift = float((states[mask] - before[mask]).abs().mean())
    scale = float(before[mask].square().mean().sqrt())
    pairs = mask[:, :-1] & mask[:, 1:]
    now_copy = (states[:, 1:] - states[:, :-1]).abs().mean((-1, -2))
    old_copy = (before[:, 1:] - before[:, :-1]).abs().mean((-1, -2))
    counts = pairs.sum(1)
    delta = (now_copy - old_copy).abs().masked_fill(~pairs, 0).sum(1) / counts.clamp_min(1)
    return {"status": "evaluated", "previous_epoch": previous["epoch"], "l1": drift,
            "relative_to_previous_rms": drift / scale if scale else None,
            "adjacent_copy_change_l1": float(delta[counts > 0].mean()) if bool((counts > 0).any()) else None}


def run_training_probe(model, store, cfg, epoch, tasks, previous=None):
    """Return a public aggregate and small private CPU teacher state for resume."""
    tasks = tuple((int(index), int(stage)) for index, stage in tasks)
    if not tasks or len({i for i, _ in tasks}) != len(tasks):
        raise ValueError("Training probe needs one fixed legal prefix per distinct patient")
    if any(i not in store.by_split["train"] or stage not in store.allowed_landmarks(i) for i, stage in tasks):
        raise ValueError("Training probe tasks must be legal training prefixes")
    if previous is not None and tuple(map(tuple, previous["tasks"])) != tasks:
        raise ValueError("Training probe tasks changed on resume")
    weights = {key: getattr(cfg, key + "_weight") for key in ("task", "jepa", "reconstruction", "variance", "flow")}
    world_enabled = bool(cfg.jepa_weight or cfg.flow_weight)
    gradients, nll, online, teachers, masks, observed_masks, landmarks, world_rows = [], [], [], [], [], [], [], []
    with preserve_training_state(model):
        if cfg.flow_weight:
            torch.manual_seed(cfg.diagnostics_seed)
            if str(cfg.device).startswith("cuda"):
                torch.cuda.manual_seed_all(cfg.diagnostics_seed)
        for offset in range(0, len(tasks), cfg.diagnostics_batch_size):
            chunk = tasks[offset:offset + cfg.diagnostics_batch_size]
            inp, sup = store.batch(chunk, future_supervision=bool(cfg.jepa_weight or cfg.flow_weight))
            inp, sup = inp.to(cfg.device), sup.to(cfg.device)
            with torch.enable_grad(), autocast(cfg.device, cfg.precision):
                loss, terms = model.compute_loss(inp, sup, variance_definition=cfg.variance_definition,
                    **{key + "_weight": value for key, value in weights.items()})
                gradients.append({"patients": len(chunk), **weighted_gradient_metrics(model, terms, weights)})
            del loss, terms
            with torch.no_grad():
                # NLL uses FP32 evaluation, matching ordinary validation precision.
                logits = model(inp)["pcr_logit"].float()
                nll.extend(F.binary_cross_entropy_with_logits(logits[sup.label_mask],
                           sup.label[sup.label_mask], reduction="none").cpu().tolist())
                values = inp.images.new_zeros(len(chunk), 4, math.prod(model.cfg.token_grid), model.cfg.dim)
                values[inp.observed_mask] = model.encoder(inp.images[inp.observed_mask]).float()
                if world_enabled:
                    rows, states, known = world_error_rows(model, inp, sup, [i for i, _ in chunk])
                    world_rows.extend(rows)
                else:
                    states, known = model._teacher_states(inp, sup)
                    states, known = states.float().cpu(), known.cpu()
                online.append(values.cpu()); teachers.append(states); masks.append(known)
                observed_masks.append(inp.observed_mask.cpu()); landmarks.append(inp.landmark.cpu())
    states, known, anchors = torch.cat(teachers), torch.cat(masks), torch.cat(landmarks)
    report = {"epoch": epoch, "scope": "fixed training probe; not full validation or checkpoint selection",
        "patients": len(tasks), "labelled_patients": len(nll), "prefixes": len(tasks),
        "landmark_counts": {f"T{s}": sum(stage == s for _, stage in tasks) for s in range(4)},
        "eval_pcr_nll": float(np.mean(nll)) if nll else None,
        "gradient_semantics": "separate weighted objectives in eval mode; preclip module L2 norms; no optimizer update",
        "gradient_batches": gradients,
        "observed_representation": representation_metrics(torch.cat(online), torch.cat(observed_masks), anchors),
        "teacher_representation": representation_metrics(states, known, anchors),
        "teacher_drift": _teacher_drift(states, known, previous),
        "world": aggregate_world_errors(world_rows) if world_enabled else
                 {"status": "unavailable", "world_head_trained": False,
                  "reason": "JEPA and flow disabled; world predictor is untrained"},
        "world_head_trained": world_enabled,
        "teacher_frozen": all(not p.requires_grad and p.grad is None for p in model.teacher_encoder.parameters()),
        "rng_and_module_modes_preserved": True}
    private = {"epoch": epoch, "tasks": tasks, "teacher_states": states, "known_mask": known}
    return report, private


def evaluate_world(model, store, cfg, *, split="val", batch_size=16):
    """Full split evaluation, all legal prefixes for dynamic, T0 for fixed models."""
    if split not in {"train", "val", "test"} or not store.by_split[split]:
        raise ValueError("World evaluation requires a nonempty valid split")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("World evaluation batch size must be positive")
    if not (cfg.jepa_weight or cfg.flow_weight):
        return {"status": "unavailable", "world_head_trained": False,
                "reason": "JEPA and flow disabled; world predictor is untrained",
                "split": split, "patients": len(store.by_split[split])}
    tasks = [(i, stage) for i in store.by_split[split]
             for stage in ([0] if cfg.landmarks == "t0" else store.allowed_landmarks(i))]
    rows = []
    with preserve_training_state(model), torch.no_grad():
        for offset in range(0, len(tasks), batch_size):
            chunk = tasks[offset:offset + batch_size]
            inp, sup = store.batch(chunk)
            values, _, _ = world_error_rows(model, inp.to(cfg.device), sup.to(cfg.device), [i for i, _ in chunk])
            rows.extend(values)
    return {**aggregate_world_errors(rows), "split": split, "patients": len(store.by_split[split]),
            "legal_prefixes": len(tasks), "precision": "fp32", "landmarks": cfg.landmarks,
            "independent_test": split == "test", "development_validation": split == "val",
            "verified_registration_subgroup": {"status": "unavailable",
                "reason": "manifest does not retain per-transition spatial verification metadata"},
            "interpretation": "raw latent L1 between checkpoints with different teachers is not a common coordinate metric"}
