"""VLA-style MRI JEPA training with fixed-T0 or patient-equal dynamic pCR selection."""
from __future__ import annotations

from pathlib import Path
import math
import random
import numpy as np
import torch

from .arm import normalize_patient_key
from .io import (autocast, load_checkpoint, restore_rng, rng_state,
                 save_checkpoint, seed_all, stable_hash, write_json)
from .metrics import classification_metrics, patient_bootstrap
from .data import RawMRIStore
from .data_pipeline import BatchLoader
from .model import RawVLAJEPA
from .train_config import RawVLATrainConfig, from_dict, load_config

CHECKPOINT_SCHEMA = "responsewm_raw_vla_jepa_checkpoint_v2"
LEGACY_CHECKPOINT_SCHEMA = "responsewm_raw_vla_jepa_checkpoint_v1"


def _training_rng(device):
    state = rng_state()
    if str(device).startswith("mps"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_training_rng(state, device):
    restore_rng(state)
    if str(device).startswith("mps"):
        torch.mps.set_rng_state(state["mps"])


def _configure(model, cfg, phase):
    frozen = phase == "joint" and cfg.schedule == "two_stage" and cfg.freeze_encoder_after_representation
    model.encoder.requires_grad_(not frozen)
    model.teacher_encoder.requires_grad_(False)
    model.train()
    if frozen:
        model.encoder.eval()
    return frozen


def _score(rows, bootstrap=0, seed=0):
    labelled = [row for row in rows if row["pcr"] is not None]
    if not labelled:
        raise ValueError("Direct pCR evaluation needs at least one labelled patient")
    labels = [r["pcr"] for r in labelled]
    probabilities = [r["pcr_probability"] for r in labelled]
    result = classification_metrics(labels, probabilities)
    result["missing_labels"] = len(rows) - len(labelled)
    if bootstrap:
        result["bootstrap"] = patient_bootstrap(labels, probabilities,
                            [r["patient_key"] for r in labelled], repetitions=bootstrap, seed=seed)
    return result


def _source_snapshot():
    root = Path(__file__).resolve().parent
    return {name: (root / name).read_text(encoding="utf-8") for name in
            ("model.py", "model_config.py", "encoder.py", "flow.py", "training.py",
             "train_config.py", "cli.py", "data.py", "contracts.py", "constants.py",
             "arm.py", "io.py", "metrics.py", "data_pipeline.py", "diagnostics.py")}


def _patient_partitions(store):
    return {split: sorted(normalize_patient_key(store.patients[i]["patient_key"])
                          for i in store.by_split[split]) for split in ("train", "val", "test")}


def _protocol(profile):
    return "direct_pcr_fixed_t0" if profile == "t0" else "direct_pcr_all_observed"


def _selection_protocol(profile):
    return "t0_direct_nll" if profile == "t0" else "patient_equal_prefix_nll"


def _evaluation_profile(cfg, landmark):
    if landmark is None:
        return cfg.landmarks
    if landmark == "all_observed":
        if cfg.landmarks == "t0":
            raise ValueError("T0 checkpoint only permits landmark=0 evaluation")
        return landmark
    if type(landmark) is not int or not 0 <= landmark <= 3:
        raise ValueError("landmark must be T0..T3 or all_observed")
    if cfg.landmarks == "t0" and landmark != 0:
        raise ValueError("T0 checkpoint only permits landmark=0 evaluation")
    return landmark


def _rows(model, store, *, split, profile, batch_size=1, device="cpu"):
    indices = store.by_split[split]
    if not indices:
        raise ValueError(f"No patients in {split} split")
    tasks = []
    for i in indices:
        stages = store.allowed_landmarks(i) if profile == "all_observed" else [0 if profile == "t0" else profile]
        tasks.extend((i, stage) for stage in stages)
    model.eval()
    rows = []
    with torch.no_grad():
        for offset in range(0, len(tasks), batch_size):
            chunk = tasks[offset:offset + batch_size]
            # Future MRI and target labels never enter the prediction input.
            inp, _ = store.batch(chunk, supervised=False)
            output = model(inp.to(device))
            probabilities = output["pcr_logit"].float().sigmoid().cpu().tolist()
            for (i, stage), probability in zip(chunk, probabilities):
                patient = store.patients[i]
                label = patient.get("target", {}).get("pcr")
                if label is not None and (type(label) not in {int, float} or label not in (0, 1)):
                    raise ValueError("Evaluation pCR label must be binary or missing")
                rows.append({"patient_key": patient["patient_key"], "landmark": stage,
                             "pcr_probability": probability, "pcr": label,
                             "arm_semantics": patient["arm"]["semantics"]})
    return rows


def _patient_equal_score(rows, bootstrap=0, seed=0):
    """Mean prefix NLL within each patient, then mean over patients."""
    groups = {}
    all_ids = {normalize_patient_key(r["patient_key"]) for r in rows}
    for row in rows:
        if row["pcr"] is not None:
            groups.setdefault(normalize_patient_key(row["patient_key"]), []).append(row)
    if not groups:
        raise ValueError("Direct pCR evaluation needs at least one labelled patient")
    patient_nll, patient_brier = [], []
    for patient_rows in groups.values():
        labels = np.asarray([r["pcr"] for r in patient_rows], dtype=np.float64)
        probabilities = np.asarray([r["pcr_probability"] for r in patient_rows], dtype=np.float64)
        if (not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 1
                or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError("Invalid or inconsistent patient labels/probabilities")
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        patient_nll.append(float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean()))
        patient_brier.append(float(((probabilities - labels) ** 2).mean()))
    result = {"n": len(groups), "patients": len(groups), "prefixes": sum(map(len, groups.values())),
              "missing_label_patients": len(all_ids) - len(groups),
              "nll": float(np.mean(patient_nll)), "brier": float(np.mean(patient_brier))}
    if bootstrap:
        rng = np.random.default_rng(seed)
        values = {"nll": np.asarray(patient_nll), "brier": np.asarray(patient_brier)}
        sampled = rng.integers(0, len(groups), size=(bootstrap, len(groups)))
        result["bootstrap"] = {
            name: {"lower": float(np.quantile(vector[sampled].mean(1), .025)),
                   "upper": float(np.quantile(vector[sampled].mean(1), .975)),
                   "valid_replicates": bootstrap} for name, vector in values.items()}
    return result


def _scores(rows, profile, bootstrap=0, seed=0):
    per_landmark = {}
    for stage in range(4):
        selected = [r for r in rows if r["landmark"] == stage]
        labelled = sum(r["pcr"] is not None for r in selected)
        per_landmark[f"T{stage}"] = (_score(selected, bootstrap, seed) if labelled else
                                      {"n": 0, "missing_labels": len(selected), "nll": None,
                                       "brier": None, "auroc": None, "auprc": None})
    patient_equal = _patient_equal_score(rows, bootstrap, seed)
    metrics = patient_equal if profile == "all_observed" else _score(rows, bootstrap, seed)
    metric = (_selection_protocol(profile) if isinstance(profile, str) else
              ("t0_direct_nll" if profile == 0 else f"landmark_T{profile}_direct_nll"))
    return {"selection_metric": metric,
            "selection_score": metrics["nll"], "metrics": metrics,
            "per_landmark": per_landmark, "patient_equal_metrics": patient_equal}


def _checkpoint_config_matches(saved, cfg):
    # Older read-only checkpoints omitted newly introduced default fields.
    normalized = dict(saved)
    defaults = RawVLATrainConfig()
    for name in ("training_unit", "max_epochs", "early_stopping_patience", "early_stopping_min_delta",
                 "lr_scheduler", "plateau_factor", "plateau_patience", "plateau_min_lr", "plateau_threshold"):
        normalized.setdefault(name, getattr(defaults, name))
    for name in ("image_cache", "prefetch_batches", "loader_workers", "pin_memory", "non_blocking_transfer"):
        normalized.setdefault(name, getattr(defaults, name))
    for name in ("variance_definition", "diagnostics_every_epochs", "diagnostics_patients",
                 "diagnostics_batch_size", "diagnostics_seed"):
        normalized.setdefault(name, getattr(defaults, name))
    return normalized == cfg.to_dict()


def train(store, cfg, output, *, resume=False, stop_after=None):
    cfg.validate()
    with BatchLoader(store, prefetch_batches=cfg.prefetch_batches, workers=cfg.loader_workers,
                     pin_memory=cfg.pin_memory,
                     future_supervision=bool(cfg.jepa_weight or cfg.flow_weight)) as loader:
        return _train(store, cfg, output, resume=resume, stop_after=stop_after, loader=loader)


def _train(store, cfg, output, *, resume, stop_after, loader):
    cfg.validate()
    if bool(cfg.image_cache) != (store.image_cache is not None):
        raise ValueError("Store image cache differs from the training configuration")
    if cfg.image_cache:
        configured_cache = Path(cfg.image_cache)
        if not configured_cache.is_absolute():
            configured_cache = store.path.parent / configured_cache
        if configured_cache.resolve() != store.image_cache.directory:
            raise ValueError("Store image cache directory differs from the training configuration")
    if store.image_shape != tuple(cfg.image_shape) or store.clinical_dim != cfg.model.clinical_dim:
        raise ValueError("Raw store shape/clinical dimension disagree with the model configuration")
    if store.manifest.get("synthetic", False) and not cfg.allow_synthetic:
        raise ValueError("Synthetic training must be explicitly enabled")
    if not store.by_split["train"] or not store.by_split["val"]:
        raise ValueError("Training and profile-matched validation splits are required")
    if not any(store.patients[i].get("target", {}).get("pcr") is not None for i in store.by_split["train"]):
        raise ValueError("Joint pCR training requires labelled training patients")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive global optimizer-step limit")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    last = out / "last.pt"
    if last.exists() and not resume:
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    if resume and not last.exists():
        raise FileNotFoundError("Resume requires last.pt")
    seed_all(cfg.seed, threads=cfg.threads, deterministic=cfg.deterministic)
    model = RawVLAJEPA(cfg.model).to(cfg.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                    factor=cfg.plateau_factor, patience=cfg.plateau_patience,
                    min_lr=cfg.plateau_min_lr, threshold=cfg.plateau_threshold,
                    threshold_mode="abs") if cfg.lr_scheduler == "plateau" else None)
    sampler = random.Random(cfg.seed + 1)
    data_signature, source_snapshot = store.signature(), _source_snapshot()
    normalization = store.normalization_state()
    partitions = _patient_partitions(store)
    protocol, selection_protocol = _protocol(cfg.landmarks), _selection_protocol(cfg.landmarks)
    completed, best_nll, history = 0, float("inf"), []
    epoch_mode = cfg.training_unit == "epochs"
    batches_per_epoch = math.ceil(len(store.by_split["train"]) / cfg.batch_size)
    total_steps = batches_per_epoch * cfg.max_epochs if epoch_mode else cfg.total_steps
    epoch_history = []
    epoch_state = {"completed_epochs": 0, "indices": [], "position": 0,
                   "batches_completed": 0, "examples_seen": 0, "term_sums": {},
                   "gradient_norm_sum": 0.0, "bad_epochs": 0, "stopped_early": False}
    diagnostics_state = None
    if cfg.diagnostics_every_epochs:
        from .diagnostics import make_probe_tasks
        diagnostics_state = {"tasks": make_probe_tasks(store, cfg.landmarks,
                             patients=cfg.diagnostics_patients, seed=cfg.diagnostics_seed),
                             "previous": None, "reports": []}
    if resume:
        state = load_checkpoint(last)
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Resume requires a v2 VLA checkpoint schema; v1 checkpoints are read-only")
        if not _checkpoint_config_matches(state["config"], cfg) or state.get("data_signature") != data_signature:
            raise ValueError("Resume config/data identity differs from the saved run")
        if state.get("source_snapshot") != source_snapshot:
            raise ValueError("Resume implementation identity differs from the saved run")
        if state.get("normalization") != normalization:
            raise ValueError("Resume normalization coordinates changed")
        if (state.get("patient_partitions") != partitions or state.get("landmarks") != cfg.landmarks
                or state.get("selection_protocol") != selection_protocol):
            raise ValueError("Resume population/selection identity differs from the saved run")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        completed, best_nll, history = state["completed_steps"], state["best_nll"], state["history"]
        if epoch_mode:
            epoch_state, epoch_history = state["epoch_state"], state["epoch_history"]
        if cfg.diagnostics_every_epochs:
            saved_diagnostics = state.get("diagnostics_state")
            if (not isinstance(saved_diagnostics, dict)
                    or tuple(saved_diagnostics.get("tasks", ())) != diagnostics_state["tasks"]):
                raise ValueError("Resume fixed diagnostic probe differs from the saved run")
            diagnostics_state = saved_diagnostics
        saved_scheduler = state.get("scheduler_state")
        if scheduler is not None:
            if saved_scheduler is None:
                raise ValueError("Resume requires the saved plateau scheduler state")
            scheduler.load_state_dict(saved_scheduler)
            if (scheduler.get_last_lr() != [group["lr"] for group in optimizer.param_groups]
                    or scheduler.last_epoch != epoch_state["completed_epochs"]):
                raise ValueError("Resume scheduler state disagrees with optimizer/epoch progress")
        elif saved_scheduler is not None:
            raise ValueError("Resume contains scheduler state but lr_scheduler is none")
        sampler.setstate(state["sampler_state"])
        _restore_training_rng(state["rng"], cfg.device)
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "normalization.json", normalization)

    def snapshot():
        return {"schema": CHECKPOINT_SCHEMA, "config": cfg.to_dict(),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "completed_steps": completed, "phase": cfg.phase(completed), "best_nll": best_nll,
                "history": history, "rng": _training_rng(cfg.device), "sampler_state": sampler.getstate(),
                "source_snapshot": source_snapshot, "patient_partitions": partitions,
                "data_signature": data_signature, "normalization": normalization,
                "protocol": protocol, "landmarks": cfg.landmarks, "selection_protocol": selection_protocol,
                "training_unit": cfg.training_unit,
                "epoch_state": epoch_state if epoch_mode else None, "epoch_history": epoch_history,
                "diagnostics_state": diagnostics_state,
                "synthetic": bool(store.manifest.get("synthetic", False))}

    def progress(status, stop_reason=None):
        write_json(out / "progress.json", {
            "status": status, "training_unit": cfg.training_unit,
            "completed_steps": completed, "total_steps": total_steps,
            "completed_epochs": epoch_state["completed_epochs"] if epoch_mode else None,
            "current_epoch": (min(epoch_state["completed_epochs"] + 1, cfg.max_epochs)
                              if epoch_mode else None),
            "max_epochs": cfg.max_epochs if epoch_mode else None,
            "batch_in_epoch": epoch_state["batches_completed"] if epoch_mode else None,
            "batches_per_epoch": batches_per_epoch if epoch_mode else None,
            "patients_seen_in_epoch": epoch_state["examples_seen"] if epoch_mode else None,
            "bad_epochs": epoch_state["bad_epochs"] if epoch_mode else None,
            "early_stopping_patience": cfg.early_stopping_patience if epoch_mode else None,
            "lr_scheduler": cfg.lr_scheduler, "current_lr": optimizer.param_groups[0]["lr"],
            "best_validation_nll": best_nll if math.isfinite(best_nll) else None,
            "last_epoch": epoch_history[-1] if epoch_history else None,
            "last_step": history[-1] if history else None, "stop_reason": stop_reason,
        })

    limit = min(total_steps, stop_after) if stop_after is not None else total_steps
    progress("running")
    while completed < limit and not epoch_state["stopped_early"]:
        phase = cfg.phase(completed)
        if (phase == "joint" and cfg.schedule == "two_stage" and cfg.freeze_encoder_after_representation
                and completed == cfg.representation_steps):
            # Frozen targets must use the same coordinates as the frozen online encoder.
            model.teacher_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        frozen = _configure(model, cfg, phase)
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        averages = {}
        for _ in range(cfg.accumulation):
            if epoch_mode:
                if not epoch_state["indices"]:
                    epoch_state["indices"] = list(store.by_split["train"])
                    sampler.shuffle(epoch_state["indices"])
                offset = epoch_state["position"]
                indices = epoch_state["indices"][offset:offset + cfg.batch_size]
                if cfg.prefetch_batches and not loader.planned:
                    # Planning uses a clone; only consumed batches advance the checkpointed sampler.
                    preview = random.Random()
                    preview.setstate(sampler.getstate())
                    planned = []
                    for start in range(offset, len(epoch_state["indices"]), cfg.batch_size):
                        chunk = epoch_state["indices"][start:start + cfg.batch_size]
                        planned.append([(i, 0 if cfg.landmarks == "t0" else preview.choice(store.allowed_landmarks(i)))
                                        for i in chunk])
                    loader.plan(planned)
            else:
                indices = [sampler.choice(store.by_split["train"]) for _ in range(cfg.batch_size)]
            tasks = [(i, 0 if cfg.landmarks == "t0" else sampler.choice(store.allowed_landmarks(i))) for i in indices]
            inp, sup = loader.fetch(tasks)
            inp, sup = (inp.to(cfg.device, non_blocking=cfg.non_blocking_transfer),
                        sup.to(cfg.device, non_blocking=cfg.non_blocking_transfer))
            with autocast(cfg.device, cfg.precision):
                loss, terms = model.compute_loss(inp, sup, task_weight=cfg.task_weight,
                    jepa_weight=cfg.jepa_weight, flow_weight=cfg.flow_weight,
                    reconstruction_weight=cfg.reconstruction_weight, variance_weight=cfg.variance_weight,
                    representation_only=phase == "representation", variance_definition=cfg.variance_definition)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite VLA-JEPA loss at step {completed + 1}")
            (loss / cfg.accumulation).backward()
            names = list(terms)
            values = torch.stack([terms[name].detach().float() for name in names]).cpu().tolist()
            for name, value in zip(names, values):
                averages[name] = averages.get(name, 0.) + value / cfg.accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite VLA-JEPA gradients")
        sample_update = (cfg.diagnostics_every_epochs and epoch_mode
                         and (epoch_state["completed_epochs"] + 1) % cfg.diagnostics_every_epochs == 0
                         and epoch_state["batches_completed"] == 0)
        update_parameters = ([p for p in model.parameters() if p.requires_grad and p.grad is not None]
                             if sample_update else [])
        before_update = [p.detach().clone() for p in update_parameters]
        optimizer.step()
        update_norm = (float(torch.stack([(p.detach().float() - before.float()).square().sum()
                       for p, before in zip(update_parameters, before_update, strict=True)]).sum().sqrt())
                       if sample_update else None)
        if not frozen:
            model.update_teacher()
        completed += 1
        entry = {"step": completed, "phase": phase, **averages, "gradient_norm": float(gradient_norm),
                 "lr": lr_used}
        if sample_update:
            entry.update(optimizer_update_norm=update_norm,
                         optimizer_update_scope="first_batch_of_diagnostic_epoch")
        epoch_finished = False
        if epoch_mode:
            epoch_state["position"] += len(indices)
            epoch_state["examples_seen"] += len(indices)
            epoch_state["batches_completed"] += 1
            for name, value in averages.items():
                weight = 1 if name.endswith("_count") else len(indices)
                epoch_state["term_sums"][name] = epoch_state["term_sums"].get(name, 0.0) + value * weight
            epoch_state["gradient_norm_sum"] += float(gradient_norm) * len(indices)
            if cfg.diagnostics_every_epochs:
                epoch_state["clipped_batches"] = (epoch_state.get("clipped_batches", 0)
                                                   + int(gradient_norm > cfg.grad_clip))
                if sample_update:
                    epoch_state["sampled_optimizer_update_norm"] = update_norm
            entry.update(epoch=epoch_state["completed_epochs"] + 1,
                         batch_in_epoch=epoch_state["batches_completed"], batch_size=len(indices))
            epoch_finished = epoch_state["position"] == len(epoch_state["indices"])
        validate = (epoch_finished if epoch_mode else phase == "joint" and
                    (completed % cfg.validation_every == 0 or completed == total_steps))
        improved = False
        if validate:
            rows = _rows(model, store, split="val", profile=cfg.landmarks,
                         batch_size=cfg.validation_batch_size, device=cfg.device)
            scores = _scores(rows, cfg.landmarks)
            entry["validation"] = scores
            if not math.isfinite(scores["selection_score"]):
                raise FloatingPointError("Non-finite profile-matched validation NLL")
            if scheduler is not None:
                scheduler.step(scores["selection_score"])
            min_delta = cfg.early_stopping_min_delta if epoch_mode else 0.0
            if scores["selection_score"] < best_nll - min_delta:
                best_nll, improved = scores["selection_score"], True
        if epoch_finished:
            loader.finish_epoch()
            next_lr = optimizer.param_groups[0]["lr"]
            entry["next_lr"] = next_lr
            epoch_state["completed_epochs"] += 1
            epoch_state["bad_epochs"] = 0 if improved else epoch_state["bad_epochs"] + 1
            epoch_state["stopped_early"] = epoch_state["bad_epochs"] >= cfg.early_stopping_patience
            count = epoch_state["examples_seen"]
            training_terms = {name: value if name.endswith("_count") else value / count
                              for name, value in epoch_state["term_sums"].items()}
            training_terms["gradient_norm"] = epoch_state["gradient_norm_sum"] / count
            if cfg.diagnostics_every_epochs:
                training_terms["gradient_clip_fraction"] = (epoch_state.get("clipped_batches", 0)
                                                             / epoch_state["batches_completed"])
                if "sampled_optimizer_update_norm" in epoch_state:
                    training_terms["sampled_optimizer_update_norm"] = epoch_state["sampled_optimizer_update_norm"]
            epoch_history.append({"epoch": epoch_state["completed_epochs"], "completed_steps": completed,
                                  "patients": count, "batches": epoch_state["batches_completed"],
                                  "lr": lr_used, "next_lr": next_lr,
                                  "lr_reduced": next_lr < lr_used, "lr_scheduler": cfg.lr_scheduler,
                                  "training": training_terms, "validation": scores,
                                  "best_validation_nll": best_nll, "improved": improved,
                                  "bad_epochs": epoch_state["bad_epochs"]})
            if cfg.diagnostics_every_epochs and epoch_state["completed_epochs"] % cfg.diagnostics_every_epochs == 0:
                from .diagnostics import run_training_probe
                probe, previous = run_training_probe(model, store, cfg, epoch_state["completed_epochs"],
                                      diagnostics_state["tasks"], previous=diagnostics_state["previous"])
                diagnostics_state["reports"].append(probe)
                diagnostics_state["previous"] = previous
                write_json(out / "diagnostics.json", diagnostics_state["reports"])
            epoch_state.update(indices=[], position=0, batches_completed=0, examples_seen=0,
                               term_sums={}, gradient_norm_sum=0.0)
            epoch_state.pop("clipped_batches", None)
            epoch_state.pop("sampled_optimizer_update_norm", None)
            write_json(out / "epoch_history.json", epoch_history)
        history.append(entry)
        if improved:
            save_checkpoint(out / "best.pt", snapshot())
            write_json(out / "best_validation.json", {"protocol": protocol, "landmarks": cfg.landmarks,
                       "selection_protocol": selection_protocol, "split": "val", "independent_test": False,
                       **scores, "rows": rows})
        if completed % cfg.checkpoint_every == 0 or completed == limit or epoch_finished:
            save_checkpoint(last, snapshot())
            write_json(out / "history.json", history)
        progress("running")
    finished = completed == total_steps or epoch_state["stopped_early"]
    stop_reason = ("early_stopping" if epoch_state["stopped_early"] else
                   ("max_epochs" if epoch_mode else "max_steps") if finished else "stop_after")
    report = {"completed": finished, "completed_steps": completed,
              "total_steps": total_steps, "schedule": cfg.schedule, "protocol": protocol,
              "lr_scheduler": cfg.lr_scheduler, "final_lr": optimizer.param_groups[0]["lr"],
              "training_unit": cfg.training_unit, "stop_reason": stop_reason,
              "completed_epochs": epoch_state["completed_epochs"] if epoch_mode else None,
              "max_epochs": cfg.max_epochs if epoch_mode else None,
              "batches_per_epoch": batches_per_epoch if epoch_mode else None,
              "early_stopping_patience": cfg.early_stopping_patience if epoch_mode else None,
              "early_stopping_min_delta": cfg.early_stopping_min_delta if epoch_mode else None,
              "bad_epochs": epoch_state["bad_epochs"] if epoch_mode else None,
              "landmarks": cfg.landmarks, "selection_protocol": selection_protocol,
              "selection_split": "val", "independent_test": False,
              "best_validation_nll": best_nll if math.isfinite(best_nll) else None,
              "last_checkpoint": str(last), "best_checkpoint": str(out / "best.pt") if (out / "best.pt").exists() else None,
              "synthetic": bool(store.manifest.get("synthetic", False))}
    write_json(out / "train_report.json", report)
    progress("completed" if finished else "paused", stop_reason)
    return report


def load_trained(checkpoint, device="cpu"):
    state = load_checkpoint(checkpoint)
    if state.get("schema") not in {CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA}:
        raise ValueError("Expected a VLA-JEPA direct-pCR checkpoint")
    cfg = from_dict(state["config"])
    if state["schema"] == LEGACY_CHECKPOINT_SCHEMA and state.get("config_digest") != stable_hash(state["config"]):
        raise ValueError("Checkpoint configuration digest mismatch")
    if state["schema"] == CHECKPOINT_SCHEMA and not _checkpoint_config_matches(state["config"], cfg):
        raise ValueError("Checkpoint configuration snapshot mismatch")
    if (state.get("protocol") != _protocol(cfg.landmarks) or state.get("landmarks") != cfg.landmarks
            or state.get("selection_protocol") != _selection_protocol(cfg.landmarks)):
        raise ValueError("Checkpoint pCR selection protocol mismatch")
    model = RawVLAJEPA(cfg.model).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, cfg, state


def evaluate(checkpoint, manifest, output, *, split="test", device="cpu", allow_synthetic=False,
             landmark=None, bootstrap=1000):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if type(bootstrap) is not int or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    model, cfg, state = load_trained(checkpoint, device)
    profile = _evaluation_profile(cfg, landmark)
    torch.set_num_threads(cfg.threads)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic, state["normalization"])
    evaluation_ids = {normalize_patient_key(store.patients[i]["patient_key"]) for i in store.by_split[split]}
    if state["schema"] == LEGACY_CHECKPOINT_SCHEMA:
        # Legacy hashes remain a read-only compatibility check, never an output field.
        evaluation_ids = {stable_hash(key) for key in evaluation_ids}
    partitions = state["patient_partitions"]
    if split in {"val", "test"} and evaluation_ids.intersection(partitions["train"]):
        raise ValueError("Evaluation patients overlap checkpoint training patients")
    if split == "test" and evaluation_ids.intersection(partitions["val"]):
        raise ValueError("Independent test patients overlap checkpoint selection patients")
    rows = _rows(model, store, split=split, profile=profile,
                 batch_size=cfg.validation_batch_size, device=device)
    report = {"protocol": _protocol(profile) if isinstance(profile, str) else
                         ("direct_pcr_fixed_t0" if profile == 0 else "direct_pcr_observed_landmark"),
              "landmarks": profile, "checkpoint_landmarks": cfg.landmarks,
              "selection_protocol": state["selection_protocol"], "split": split,
              "independent_test": split == "test", "synthetic": bool(store.manifest.get("synthetic", False)),
              "clinical_validation": False if store.manifest.get("synthetic", False) else None,
              **_scores(rows, profile, bootstrap, cfg.seed), "rows": rows}
    write_json(output, report)
    Path(output).chmod(0o600)
    return report


def predict(checkpoint, manifest, patient_key, output, *, landmark=0, device="cpu", allow_synthetic=False,
            forecast_states=False, generate=False, steps=20, seed=0):
    model, cfg, state = load_trained(checkpoint, device)
    if (forecast_states or generate) and cfg.jepa_weight == 0 and cfg.flow_weight == 0:
        raise ValueError("Future prediction requires a checkpoint with a trained JEPA world head or flow loss")
    _evaluation_profile(cfg, landmark)
    if type(landmark) is not int:
        raise ValueError("Prediction landmark must be T0..T3")
    torch.set_num_threads(cfg.threads)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic, state["normalization"])
    normalized = normalize_patient_key(patient_key)
    matches = [i for i, p in enumerate(store.patients) if normalize_patient_key(p["patient_key"]) == normalized]
    if len(matches) != 1:
        raise ValueError("patient_key must identify exactly one patient")
    inp, _ = store.batch([(matches[0], landmark)], supervised=False)
    inp = inp.to(device)
    with torch.no_grad():
        features = model(inp)
        probability = float(features["pcr_logit"].float().sigmoid()[0])
    report = {"patient_key": store.patients[matches[0]]["patient_key"], "landmark": landmark,
              "pcr_probability": probability, "protocol": "direct_pcr", "checkpoint_landmarks": cfg.landmarks,
              "observed_stages": inp.observed_mask[0].nonzero().flatten().cpu().tolist(),
              "requested_stages": inp.query_mask[0].nonzero().flatten().cpu().tolist(),
              "arm_semantics": store.patients[matches[0]]["arm"]["semantics"],
              "arm_visible": bool(inp.arm_mask[0]), "synthetic": bool(store.manifest.get("synthetic", False))}
    if forecast_states:
        with torch.no_grad():
            forecast = model.forecast(inp)
        state_path = Path(output).with_suffix(".states.npz")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, state_prediction=forecast["state_prediction"].cpu().numpy(),
                            query_mask=forecast["query_mask"].cpu().numpy())
        report["forecast_states"] = str(state_path.resolve())
        report["forecast_shape"] = list(forecast["state_prediction"].shape)
    if generate:
        if not cfg.model.enable_flow:
            raise ValueError("MRI generation requires a checkpoint trained with enable_flow=true")
        with torch.no_grad():
            generated = model.generate(inp, steps=steps, seed=seed)
        image_path = Path(output).with_suffix(".mri.npz")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(image_path, future_images=generated["future_images"].cpu().numpy(),
                            query_mask=generated["query_mask"].cpu().numpy(),
                            dynamics_features=features["dynamics_features"].cpu().numpy(),
                            normalization=store.normalization_state()["intensity"])
        report["generated_mri"] = str(image_path.resolve())
        report["generated_shape"] = list(generated["future_images"].shape)
    write_json(output, report)
    Path(output).chmod(0o600)
    return report
