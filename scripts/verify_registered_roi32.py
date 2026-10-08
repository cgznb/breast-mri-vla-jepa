#!/usr/bin/env python3
"""Engineering checks on real ROI32 inputs; short runs are not clinical results."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.training import load_trained, train
from mri_vla_jepa.train_config import load_config


def verify(manifest, output, *, steps=4, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new verification output directory")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("Real BF16 verification requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device must support BF16")
    output.mkdir(parents=True, mode=0o700)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/registered_roi32_t0.yaml")
    torch.set_num_threads(cfg.threads)
    store = RawMRIStore(manifest, cfg.image_shape)
    index = store.by_split["train"][0]
    inp, sup = store.batch([(index, 0)])
    started = time.monotonic()
    cpu_model = RawVLAJEPA(copy.deepcopy(cfg.model))
    loss, terms = cpu_model.compute_loss(
        inp, sup, task_weight=cfg.task_weight, jepa_weight=cfg.jepa_weight,
        flow_weight=cfg.flow_weight, reconstruction_weight=cfg.reconstruction_weight,
        variance_weight=cfg.variance_weight,
    )
    loss.backward()
    if not all(torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("CPU full-resolution loss is not finite")
    gradients = [p.grad for p in cpu_model.encoder.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("CPU encoder gradients are absent or not finite")
    if any(p.grad is not None or p.requires_grad for p in cpu_model.teacher_encoder.parameters()):
        raise AssertionError("EMA teacher received gradients")
    report = {
        "engineering_only": True, "clinical_validation": False,
        "input_shape": list(inp.images.shape),
        "patients_by_split": {s: len(store.by_split[s]) for s in ("train", "val", "test")},
        "image_preprocessing": store.normalization_state()["image_preprocessing"],
        "cpu_full_resolution": {
            "loss": float(loss.detach()), "finite_gradients": True,
            "teacher_without_gradients": True, "seconds": time.monotonic() - started,
        },
        "cuda": {"torch": torch.__version__, "device": device,
                 "gpu": torch.cuda.get_device_name(device), "precision": "bf16"},
        "runs": {},
    }
    del cpu_model, loss, terms, gradients
    for profile in ("t0", "dynamic"):
        run_cfg = load_config(root / f"configs/registered_roi32_{profile}.yaml")
        run_cfg.joint_steps = steps
        run_cfg.validation_every = steps
        run_cfg.checkpoint_every = steps
        run_cfg.device = device
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        run = output / profile
        run.mkdir(mode=0o700)
        training = train(store, run_cfg, run)
        model, restored_cfg, state = load_trained(run / "best.pt", device)
        if model.flow is not None or restored_cfg.flow_weight != 0:
            raise AssertionError("Image generation was enabled")
        if any(p.requires_grad for p in model.teacher_encoder.parameters()):
            raise AssertionError("Restored teacher is trainable")
        with torch.no_grad():
            forecast = model.forecast(inp.to(device))["state_prediction"]
        if forecast.shape != (1, 4, 32, 128) or not torch.isfinite(forecast).all():
            raise AssertionError("Autonomous state forecast is invalid")
        finite_history = all(
            all(torch.isfinite(torch.tensor(entry[key])).item() for key in ("loss", "gradient_norm"))
            for entry in state["history"]
        )
        if not finite_history or state["completed_steps"] != steps:
            raise AssertionError("Short training did not finish with finite gradients")
        final_scores = state["history"][-1]["validation"]
        report["runs"][profile] = {
            "training": training, "checkpoint_schema": state["schema"],
            "finite_losses_and_gradients": True, "teacher_without_gradients": True,
            "flow_enabled": False, "forecast_shape": list(forecast.shape),
            "validation": final_scores,
            "seconds": time.monotonic() - started,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        }
        del model, state, forecast
    write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = verify(args.manifest, args.output, steps=args.steps, device=args.device)
    import json
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
