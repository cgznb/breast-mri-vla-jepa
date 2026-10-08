#!/usr/bin/env python3
"""Verify physical full-resolution batch 16 training on real ROI32 inputs."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config


def check(manifest: Path, output: Path, device: str = "cuda", config: Path | None = None):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The physical batch-16 gate requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The physical batch-16 gate requires CUDA BF16")
    torch.set_num_threads(4)
    torch.manual_seed(20261003)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config or root / "configs/registered_roi32_t0.yaml")
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cudnn.benchmark = False
    if cfg.model.enable_flow or cfg.flow_weight:
        raise ValueError("The ROI32 gate requires flow generation disabled")
    store = RawMRIStore(manifest, cfg.image_shape, image_cache=cfg.image_cache)
    selected = sorted(
        store.by_split["train"],
        key=lambda index: -sum(visit is not None for visit in store.patients[index]["visits"]),
    )[:16]
    if len(selected) != 16:
        raise ValueError("At least 16 training patients are required")
    if any(any(store.patients[index]["visits"][stage] is None for stage in range(4)) for index in selected):
        raise ValueError("The worst-case batch gate requires 16 complete four-visit patients")
    report = {
        "engineering_only": True,
        "clinical_validation": False,
        "physical_batch_size": 16,
        "microbatch_substitution": False,
        "accumulation": 1,
        "selection": "16 training patients with complete T0-T3 visits",
        "image_shape": [16, 4, 3, 32, 128, 128],
        "image_preprocessing": copy.deepcopy(store.image_preprocessing),
        "model": asdict(cfg.model),
        "lr_scheduler": cfg.lr_scheduler,
        "data_pipeline": {key: getattr(cfg, key) for key in (
            "image_cache", "prefetch_batches", "loader_workers", "pin_memory", "non_blocking_transfer")},
        "loss_weights": {
            "task": cfg.task_weight, "jepa": cfg.jepa_weight,
            "flow": cfg.flow_weight, "reconstruction": cfg.reconstruction_weight,
            "variance": cfg.variance_weight,
        },
        "variance_definition": cfg.variance_definition,
        "cuda": {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": device, "gpu": torch.cuda.get_device_name(device),
            "total_memory_mib": torch.cuda.get_device_properties(device).total_memory / 1024 ** 2,
            "precision": "bf16", "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "cases": [],
        "passed": False,
    }
    for landmark in (0, 2, 3):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        model = optimizer = inp = sup = loss = terms = gradients = None
        calls = {"online_encoder_visit_batches": [], "teacher_encoder_visit_batches": []}
        case = {
            "landmark": landmark, "physical_batch_size": 16,
            "expected_observed_visits": 16 * (landmark + 1),
            "passed": False,
        }
        try:
            future_enabled = cfg.jepa_weight > 0 or cfg.flow_weight > 0
            inp, sup = store.batch([(index, landmark) for index in selected], pin_memory=cfg.pin_memory,
                                   future_supervision=future_enabled)
            if list(inp.images.shape) != report["image_shape"]:
                raise AssertionError("The gate must use full B16 ROI32 tensors")
            case["observed_visits"] = int(inp.observed_mask.sum())
            case["future_visits"] = int(sup.future_mask.sum())
            case["observed_per_patient"] = (landmark + 1)
            if case["observed_visits"] != case["expected_observed_visits"]:
                raise AssertionError("Incomplete legal prefix in the worst-case gate")
            if cfg.pin_memory:
                inp, sup = inp.pin_memory(), sup.pin_memory()
                if not all(tensor.is_pinned() for record in (inp, sup) for tensor in vars(record).values()):
                    raise AssertionError("Every transfer source must use pinned memory")
            inp, sup = (inp.to(device, non_blocking=cfg.non_blocking_transfer),
                        sup.to(device, non_blocking=cfg.non_blocking_transfer))
            model = RawVLAJEPA(copy.deepcopy(cfg.model)).to(device).train()
            model.encoder.register_forward_pre_hook(
                lambda _module, args: calls["online_encoder_visit_batches"].append(len(args[0]))
            )
            model.teacher_encoder.register_forward_pre_hook(
                lambda _module, args: calls["teacher_encoder_visit_batches"].append(len(args[0]))
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, terms = model.compute_loss(
                    inp, sup, task_weight=cfg.task_weight, jepa_weight=cfg.jepa_weight,
                    flow_weight=cfg.flow_weight, reconstruction_weight=cfg.reconstruction_weight,
                    variance_weight=cfg.variance_weight,
                    variance_definition=cfg.variance_definition,
                )
            if not all(torch.isfinite(value).all().item() for value in terms.values()):
                raise FloatingPointError("Nonfinite ROI32 loss term")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                raise FloatingPointError("Absent or nonfinite ROI32 gradients")
            if not any(parameter.grad is not None for parameter in model.encoder.parameters()):
                raise AssertionError("The online MRI encoder did not receive gradients")
            if any(parameter.grad is not None or parameter.requires_grad for parameter in model.teacher_encoder.parameters()):
                raise AssertionError("The EMA teacher received gradients")
            if cfg.jepa_weight == 0 and cfg.flow_weight == 0:
                if calls["teacher_encoder_visit_batches"] or case["future_visits"]:
                    raise AssertionError("Disabled world losses must not load or encode future targets")
                if any(parameter.grad is not None for parameter in model.world_predictor.parameters()):
                    raise AssertionError("Disabled world predictor received gradients")
            if cfg.reconstruction_weight == 0 and any(
                    parameter.grad is not None for parameter in model.reconstruction.parameters()):
                raise AssertionError("Disabled reconstruction head received gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError("Nonfinite total gradient norm")
            optimizer.step()
            model.update_teacher()
            if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise FloatingPointError("Nonfinite weights after optimizer and EMA updates")
            torch.cuda.synchronize(device)
            case.update(
                passed=True, losses={key: float(value.detach()) for key, value in terms.items()},
                gradient_norm=float(gradient_norm), finite_gradients=True,
                teacher_without_gradients=True, optimizer_step=True, ema_update=True,
                **calls,
            )
            if landmark == 3:
                case["jepa_note"] = "No future transition at T3; configured JEPA weight remains positive"
            print(f"Physical B16 gate landmark={landmark}: passed", flush=True)
        except torch.cuda.OutOfMemoryError:
            case["error"] = "CUDA out of memory at physical B16; batch size was not reduced"
            print(f"Physical B16 gate landmark={landmark}: CUDA out of memory", flush=True)
        except (AssertionError, FloatingPointError) as exc:
            case["error"] = str(exc)
            print(f"Physical B16 gate landmark={landmark}: failed", flush=True)
        finally:
            case["seconds"] = time.monotonic() - started
            case["peak_allocated_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            case["peak_reserved_memory_mib"] = torch.cuda.max_memory_reserved(device) / 1024 ** 2
            report["cases"].append(case)
            del model, optimizer, inp, sup, loss, terms, gradients
            gc.collect()
            torch.cuda.empty_cache()
            write_json(output, report)
        if not case["passed"]:
            break
    report["passed"] = len(report["cases"]) == 3 and all(case["passed"] for case in report["cases"])
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/registered_roi32_batch16_gate.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    report = check(args.manifest.resolve(), args.output.resolve(), args.device, args.config)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
