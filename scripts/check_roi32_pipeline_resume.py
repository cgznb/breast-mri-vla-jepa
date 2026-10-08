#!/usr/bin/env python3
"""Verify stochastic epoch/resume paths before the independent A1/A2 queue."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import math
from pathlib import Path

import torch

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import load_checkpoint, write_json
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import evaluate, predict, train

ROOT = Path(__file__).resolve().parents[1]


def require_equal(left, right, path="state", *, rtol=0., atol=0.):
    if isinstance(left, torch.Tensor):
        equal = (torch.allclose(left, right, rtol=rtol, atol=atol)
                 if left.is_floating_point() else torch.equal(left, right))
        if not equal:
            raise AssertionError(f"Resume differs at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Resume keys differ at {path}")
        for key in left:
            require_equal(left[key], right[key], f"{path}.{key}", rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Resume lengths differ at {path}")
        for n, (a, b) in enumerate(zip(left, right)):
            require_equal(a, b, f"{path}.{n}", rtol=rtol, atol=atol)
    elif isinstance(left, float):
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"Resume differs at {path}")
    elif left != right:
        raise AssertionError(f"Resume differs at {path}")


def tensor_difference(left, right):
    squared, reference, maximum = 0., 0., 0.

    def visit(a, b):
        nonlocal squared, reference, maximum
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise AssertionError("Resume tensor shape or finite-value check failed")
            if a.ndim == 0:
                require_equal(a, b)
            delta = (a.double() - b.double())
            squared += float(delta.square().sum())
            reference += float(a.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise AssertionError("Resume tensor state keys differ")
            for key in a:
                visit(a[key], b[key])
        else:
            require_equal(a, b)

    visit(left, right)
    return {"max_abs_difference": maximum, "relative_l2_difference": math.sqrt(squared / max(reference, 1e-30))}


def cuda_comparison(reference, actual, control, cfg):
    measurements = {}
    for key in ("model", "optimizer"):
        if key == "optimizer":
            require_equal(reference[key]["param_groups"], actual[key]["param_groups"])
            a, b, c = (state[key]["state"] for state in (reference, actual, control))
        else:
            a, b, c = (state[key] for state in (reference, actual, control))
        resumed, repeated = tensor_difference(a, b), tensor_difference(a, c)
        limit = max(5 * repeated["relative_l2_difference"], 1e-3 if key == "model" else .01)
        if resumed["relative_l2_difference"] > limit:
            raise AssertionError(f"CUDA resume {key} drift exceeds repeat-control tolerance")
        measurements[key] = {"resumed": resumed, "duplicate_uninterrupted": repeated, "relative_l2_limit": limit}
    for term in ("loss", "task", "jepa", "reconstruction", "variance", "gradient_norm"):
        values = [state["history"] for state in (reference, actual, control)]
        resumed = max(abs(a[term] - b[term]) for a, b in zip(values[0], values[1]))
        repeated = max(abs(a[term] - b[term]) for a, b in zip(values[0], values[2]))
        floor = .01 if term == "gradient_norm" else .001
        limit = max(5 * repeated, floor)
        if resumed > limit:
            raise AssertionError(f"CUDA resume {term} drift exceeds repeat-control tolerance")
        measurements[term] = {"resumed_max_abs_difference": resumed,
                              "duplicate_uninterrupted_max_abs_difference": repeated, "absolute_limit": limit}
    for a, b in zip(reference["history"], actual["history"]):
        for key in ("step", "epoch", "batch_in_epoch", "batch_size", "label_count", "future_count", "pair_count", "lr"):
            require_equal(a[key], b[key], key)
    return measurements


def resumed_case(cfg, manifest, output, *, stop, total):
    def store():
        return RawMRIStore(manifest, cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache)

    train(store(), cfg, output / "full", stop_after=total)
    partial = train(store(), cfg, output / "resumed", stop_after=stop)
    if partial["completed"]:
        raise AssertionError("The resume gate must stop before completing training")
    train(store(), cfg, output / "resumed", stop_after=total, resume=True)
    reference = load_checkpoint(output / "full/last.pt")
    actual = load_checkpoint(output / "resumed/last.pt")
    for key in ("scheduler_state", "rng", "sampler_state", "best_nll", "completed_steps"):
        require_equal(reference[key], actual[key], key)
    cuda = cfg.device.startswith("cuda")
    if cuda:
        train(store(), cfg, output / "repeat_control", stop_after=total)
        control = load_checkpoint(output / "repeat_control/last.pt")
        numeric = cuda_comparison(reference, actual, control, cfg)
        del control
    else:
        for key in ("model", "optimizer", "history", "epoch_history", "epoch_state"):
            require_equal(reference[key], actual[key], key)
    for key in ("indices", "position", "batches_completed", "examples_seen", "completed_epochs", "bad_epochs"):
        require_equal(reference["epoch_state"][key], actual["epoch_state"][key], key)
    difference = max(float((value - actual["model"][name]).abs().max())
                     for name, value in reference["model"].items())
    result = {"profile": cfg.landmarks, "dropout": cfg.model.dropout,
              "lr_scheduler": cfg.lr_scheduler, "physical_batch_size": cfg.batch_size,
              "completed_steps": actual["completed_steps"], "stop_after": stop,
              "completed_epochs": actual["epoch_state"]["completed_epochs"],
              "exact_model_optimizer_rng_sampler_history": not cuda,
              "exact_sampler_rng_and_consumed_cursor": True,
              "model_optimizer_history_numerically_close": True,
              "model_max_abs_difference": difference, "passed": True}
    if cuda:
        result["numerical_check"] = numeric
        result["numerical_note"] = "Original CUDA BF16 kernels are not bitwise deterministic; duplicate uninterrupted control measured"
    del reference, actual
    gc.collect()
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def check(manifest, output, report_path):
    if output.exists():
        raise FileExistsError("Choose a fresh verification output directory")
    output.mkdir(parents=True, mode=0o700)
    report = {"engineering_only": True, "clinical_validation": False,
              "synthetic": [], "real_cuda": [], "passed": False}
    for experiment in ("a1", "a2"):
        for profile in ("t0", "dynamic"):
            cfg = load_config(ROOT / f"configs/raw_vla_jepa_smoke_{profile}.yaml")
            cfg = replace(cfg, training_unit="epochs", max_epochs=3, prefetch_batches=2,
                          lr=1e-4, lr_scheduler="plateau" if experiment == "a1" else "none")
            cfg.model = replace(cfg.model, dropout=.1 if experiment == "a1" else .3)
            case_output = output / "synthetic" / f"{experiment}_{profile}"
            synthetic = make_raw_synthetic(case_output / "data", n_train=5, n_val=2,
                                           image_shape=cfg.image_shape, seed=cfg.seed)
            case = resumed_case(cfg, synthetic, case_output, stop=4, total=9)
            evaluated = evaluate(case_output / "resumed/best.pt", synthetic, case_output / "evaluation.json",
                                 split="val", allow_synthetic=True, bootstrap=0)
            predicted = predict(case_output / "resumed/best.pt", synthetic, "synthetic_0",
                                case_output / "prediction.json", allow_synthetic=True, forecast_states=True)
            case["evaluation_prefixes"] = len(evaluated["rows"])
            case["forecast_shape"] = predicted["forecast_shape"]
            report["synthetic"].append(case)
            print(f"Synthetic {experiment}/{profile}: exact epoch resume, evaluation and forecast passed", flush=True)
    report["real_cuda_uses_production_kernel_settings"] = True
    report["production_deterministic_setting_changed"] = False
    for profile in ("t0", "dynamic"):
        cfg = load_config(ROOT / f"configs/registered_roi32_epoch200_a1_{profile}_seed17.yaml")
        case = resumed_case(cfg, manifest, output / "real_cuda" / profile, stop=2, total=4)
        report["real_cuda"].append(case)
        print(f"Real CUDA BF16 B16 {profile}: exact sampler/RNG and numerically consistent resume passed", flush=True)
    report["passed"] = True
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/roi32_pipeline_verification_20261004")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/roi32_pipeline_resume_20261004.json")
    args = parser.parse_args()
    check(args.manifest, args.output, args.report)


if __name__ == "__main__":
    main()
