#!/usr/bin/env python3
"""Verify loss ablations, stochastic resume, and real physical batch 16."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from pathlib import Path

import torch

from check_registered_roi32_batch16 import check as batch_gate
from check_roi32_pipeline_resume import require_equal, resumed_case
from mri_vla_jepa.data import make_raw_synthetic
from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import make_probe_tasks, run_training_probe
from mri_vla_jepa.io import load_checkpoint, rng_state, write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import evaluate, predict


ROOT = Path(__file__).resolve().parents[1]


def check(manifest, output, report_path):
    if output.exists():
        raise FileExistsError("Choose a fresh loss-suite verification directory")
    output.mkdir(parents=True, mode=0o700)
    report = {"engineering_only": True, "clinical_validation": False,
              "synthetic": [], "real_cuda": [], "passed": False}
    for experiment in ("l1", "l2", "l3", "l4"):
        for profile in ("t0", "dynamic"):
            formal = load_config(ROOT / f"configs/registered_roi32_epoch200_{experiment}_{profile}_seed17.yaml")
            small = load_config(ROOT / f"configs/raw_vla_jepa_smoke_{profile}.yaml")
            cfg = replace(
                small, training_unit="epochs", max_epochs=6, early_stopping_patience=50,
                lr=formal.lr, lr_scheduler=formal.lr_scheduler,
                task_weight=formal.task_weight, jepa_weight=formal.jepa_weight,
                reconstruction_weight=formal.reconstruction_weight,
                variance_weight=formal.variance_weight,
                variance_definition=formal.variance_definition,
                prefetch_batches=2, diagnostics_every_epochs=2,
                diagnostics_patients=5, diagnostics_batch_size=2,
                diagnostics_seed=formal.diagnostics_seed,
            )
            cfg.model = replace(cfg.model, dropout=formal.model.dropout)
            cfg.validate()
            case_output = output / "synthetic" / f"{experiment}_{profile}"
            synthetic = make_raw_synthetic(case_output / "data", n_train=5, n_val=2,
                                           image_shape=cfg.image_shape, seed=cfg.seed)
            result = resumed_case(cfg, synthetic, case_output, stop=4, total=18)
            reference = load_checkpoint(case_output / "full/last.pt")
            resumed = load_checkpoint(case_output / "resumed/last.pt")
            diagnostic_keys = [key for key in reference if "diagnostic" in key or "probe" in key]
            if not diagnostic_keys:
                raise AssertionError("Diagnostic state must be saved for reproducible resume")
            for key in diagnostic_keys:
                require_equal(reference[key], resumed[key], key)
            evaluated = evaluate(case_output / "resumed/best.pt", synthetic,
                                 case_output / "evaluation.json", split="val",
                                 allow_synthetic=True, bootstrap=0)
            predicted = predict(case_output / "resumed/best.pt", synthetic, "synthetic_0",
                                case_output / "prediction.json", allow_synthetic=True,
                                forecast_states=cfg.jepa_weight > 0)
            result.update(experiment=experiment, variance_definition=cfg.variance_definition,
                          diagnostic_resume_exact=True, diagnostic_every_epochs=2,
                          evaluation_prefixes=len(evaluated["rows"]),
                          world_head_trained=cfg.jepa_weight > 0)
            if cfg.jepa_weight > 0:
                result["forecast_shape"] = predicted["forecast_shape"]
            report["synthetic"].append(result)
            write_json(report_path, report)
            print(f"Synthetic {experiment}/{profile}: loss, diagnostic resume and evaluation passed", flush=True)
            del reference, resumed
            gc.collect()
    for experiment in ("l1", "l2", "l3", "l4"):
        for profile in ("t0", "dynamic"):
            config = ROOT / f"configs/registered_roi32_epoch200_{experiment}_{profile}_seed17.yaml"
            gate = batch_gate(manifest, output / f"{experiment}_{profile}_batch16.json", config=config)
            report["real_cuda"].append({"experiment": experiment, "profile": profile,
                                        "config": str(config.relative_to(ROOT)), **gate})
            write_json(report_path, report)
            if not gate["passed"]:
                raise AssertionError(f"Physical batch16 failed for {experiment}/{profile}")
    # Exercise both the no-future and patient-axis paths across a CUDA restart.
    for experiment, profile in (("l1", "t0"), ("l4", "dynamic")):
        cfg = load_config(ROOT / f"configs/registered_roi32_epoch200_{experiment}_{profile}_seed17.yaml")
        result = resumed_case(cfg, manifest, output / "real_cuda_resume" / f"{experiment}_{profile}",
                              stop=2, total=4)
        result.update(experiment=experiment, variance_definition=cfg.variance_definition)
        report.setdefault("real_cuda_resume", []).append(result)
        write_json(report_path, report)
        print(f"CUDA BF16 {experiment}/{profile}: consumed cursor/RNG resume passed", flush=True)
    for experiment in ("l1", "l2", "l3", "l4"):
        for profile in ("t0", "dynamic"):
            cfg = load_config(ROOT / f"configs/registered_roi32_epoch200_{experiment}_{profile}_seed17.yaml")
            store = RawMRIStore(manifest, cfg.image_shape, image_cache=cfg.image_cache)
            model = RawVLAJEPA(cfg.model).to(cfg.device).train()
            original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            modes = [module.training for module in model.modules()]
            random_state = rng_state()
            tasks = make_probe_tasks(store, cfg.landmarks, cfg.diagnostics_patients, cfg.diagnostics_seed)
            torch.cuda.reset_peak_memory_stats()
            diagnostic, _ = run_training_probe(model, store, cfg, 5, tasks)
            require_equal(random_state, rng_state(), "diagnostic_rng")
            for name, value in model.state_dict().items():
                require_equal(original[name], value.detach().cpu(), f"diagnostic_model.{name}")
            if modes != [module.training for module in model.modules()]:
                raise AssertionError("Diagnostic changed CUDA model train/eval modes")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError("Diagnostic left parameter gradients behind")
            if not diagnostic["teacher_frozen"]:
                raise AssertionError("Diagnostic gave gradients to the EMA teacher")
            report.setdefault("real_cuda_diagnostics", []).append({
                "experiment": experiment, "profile": profile, "passed": True,
                "initial_model_probe": True, "clinical_validation": False,
                "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "probe": diagnostic,
            })
            write_json(report_path, report)
            print(f"CUDA BF16 {experiment}/{profile}: B16 fixed probe read-only and finite", flush=True)
            del model, store, original, diagnostic
            gc.collect()
            torch.cuda.empty_cache()
    report["passed"] = True
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/roi32_loss_verification_20261004")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/roi32_loss_verification_20261004.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Loss-suite gates require CUDA BF16")
    check(args.manifest.resolve(), args.output.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
