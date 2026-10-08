#!/usr/bin/env python3
"""Measure exact ROI32 cache loading and verify BF16 losses and gradients."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import gc
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.data_pipeline import BatchLoader
from mri_vla_jepa.io import autocast, seed_all, write_json
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config

ROOT = Path(__file__).resolve().parents[1]


def require_equal(left, right):
    for reference, actual in zip(left, right):
        for name, value in vars(reference).items():
            if not torch.equal(value, getattr(actual, name)):
                raise AssertionError(f"Cached batch differs in {name}")


def measured_batch(store, tasks, repeats, *, pin_memory=False):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        batch = store.batch(tasks, pin_memory=pin_memory)
        times.append(time.perf_counter() - started)
        del batch
    return {"seconds": times, "median_seconds": statistics.median(times)}


def gradients(model, cfg, batch, *, pin, reproducible=False):
    model.train()
    model.zero_grad(set_to_none=True)
    inp, sup = batch
    if pin:
        inp, sup = inp.pin_memory(), sup.pin_memory()
        if not all(t.is_pinned() for record in (inp, sup) for t in vars(record).values()):
            raise AssertionError("Transfer sources must all be pinned")
    started = time.perf_counter()
    inp, sup = inp.to(cfg.device, non_blocking=pin), sup.to(cfg.device, non_blocking=pin)
    torch.cuda.synchronize(cfg.device)
    transfer = time.perf_counter() - started
    # Both paths use exactly the same stochastic model state.
    torch.manual_seed(cfg.seed + 100)
    torch.cuda.manual_seed_all(cfg.seed + 100)
    started = time.perf_counter()
    with (sdpa_kernel(SDPBackend.MATH) if reproducible else nullcontext()), autocast(cfg.device, cfg.precision):
        loss, terms = model.compute_loss(
            inp, sup, task_weight=cfg.task_weight, jepa_weight=cfg.jepa_weight,
            flow_weight=cfg.flow_weight, reconstruction_weight=cfg.reconstruction_weight,
            variance_weight=cfg.variance_weight)
    loss.backward()
    torch.cuda.synchronize(cfg.device)
    compute = time.perf_counter() - started
    if any(p.requires_grad or p.grad is not None for p in model.teacher_encoder.parameters()):
        raise AssertionError("EMA teacher must remain frozen and without gradients")
    result = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
    if not result or not all(torch.isfinite(g).all() for g in result.values()):
        raise FloatingPointError("Absent or nonfinite gradients")
    scalars = {name: float(value.detach()) for name, value in terms.items()}
    if not all(torch.isfinite(torch.tensor(value)) for value in scalars.values()):
        raise FloatingPointError("Nonfinite loss terms")
    return scalars, result, transfer, compute


def check(manifest, config, output, repeats):
    cfg = load_config(config)
    if not cfg.image_cache or not cfg.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("This gate requires an enabled image cache and CUDA")
    seed_all(cfg.seed, threads=cfg.threads, deterministic=False)
    original = RawMRIStore(manifest, cfg.image_shape)
    cached = RawMRIStore(manifest, cfg.image_shape, image_cache=cfg.image_cache)
    indices = original.by_split["train"][:16]
    tasks = [(i, 0) for i in indices]
    if len(tasks) != 16:
        raise ValueError("This benchmark requires a physical batch of sixteen patients")
    report = {"engineering_only": True, "physical_batch_size": 16,
              "threads": cfg.threads, "repeats": repeats,
              "visits_in_batch": sum(v is not None for i in indices for v in original.patients[i]["visits"]),
              "context": "Single process, warm page cache, full input and future supervision",
              "image_cache": str(cached.image_cache.directory), "passed": False}
    # Warm both file paths; timing includes assembly and the public CPU validations.
    for store in (original, cached):
        batch = store.batch(tasks)
        del batch
    report["npz_serial"] = measured_batch(original, tasks, repeats)
    report["npy_serial"] = measured_batch(cached, tasks, repeats)
    report["npy_direct_pinned"] = measured_batch(cached, tasks, repeats, pin_memory=True)
    report["serial_preparation_speedup"] = (report["npz_serial"]["median_seconds"] /
                                            report["npy_serial"]["median_seconds"])
    print(f"B16 preparation: NPZ {report['npz_serial']['median_seconds']:.4f}s; "
          f"NPY {report['npy_serial']['median_seconds']:.4f}s; "
          f"speedup {report['serial_preparation_speedup']:.2f}x", flush=True)
    left, right = original.batch(tasks), cached.batch(tasks, pin_memory=True)
    require_equal(left, right)
    report["all_batch_tensors_equal"] = True
    model = RawVLAJEPA(copy.deepcopy(cfg.model)).to(cfg.device)
    control_terms, control_grads, _, _ = gradients(model, cfg, left, pin=False)
    repeated_terms, repeated_grads, _, _ = gradients(model, cfg, left, pin=False)
    report["production_kernel_repeat_control"] = {
        "same_npz_batch_and_seed": True, "loss_terms_equal": control_terms == repeated_terms,
        "gradient_max_abs_difference": max(float((control_grads[name] - repeated_grads[name]).abs().max())
                                           for name in control_grads)}
    del control_grads, repeated_grads
    # Math attention and deterministic cuDNN isolate data equivalence from kernel reductions.
    torch.backends.cudnn.deterministic = True
    first_terms, first_grads, first_transfer, first_compute = gradients(model, cfg, left, pin=False, reproducible=True)
    second_terms, second_grads, second_transfer, second_compute = gradients(model, cfg, right, pin=True, reproducible=True)
    if first_terms != second_terms or first_grads.keys() != second_grads.keys():
        raise AssertionError("Cached BF16 loss terms or gradient membership differs")
    max_difference = max(float((first_grads[name] - second_grads[name]).abs().max()) for name in first_grads)
    if max_difference != 0:
        raise AssertionError(f"Cached BF16 gradients differ: maximum absolute difference {max_difference}")
    report["cuda_comparison"] = {
        "loss_terms_equal": True, "gradients_equal": True, "gradient_max_abs_difference": max_difference,
        "verification_cudnn_deterministic": True, "production_deterministic_setting_changed": False,
        "verification_attention_backend": "math",
        "teacher_without_gradients": True, "losses": second_terms,
        "pageable_blocking_transfer_seconds": first_transfer,
        "pinned_nonblocking_transfer_seconds": second_transfer,
        "forward_backward_seconds": [first_compute, second_compute],
        "note": "Transfers use the compute stream; no separate copy/compute stream overlap is claimed"}
    del model, left, right, first_grads, second_grads
    gc.collect()
    torch.cuda.empty_cache()
    plans = [[(indices[(n + j) % len(indices)], 0) for j in range(16)] for n in range(repeats + 1)]
    waits, pinned_checks = [], []
    with BatchLoader(cached, prefetch_batches=cfg.prefetch_batches, workers=cfg.loader_workers,
                     pin_memory=cfg.pin_memory) as loader:
        started = time.perf_counter()
        loader.plan(plans)
        for n, plan in enumerate(plans):
            before = time.perf_counter()
            inp, sup = loader.fetch(plan)
            waits.append(time.perf_counter() - before)
            pinned_checks.append(all(t.is_pinned() for record in (inp, sup) for t in vars(record).values()))
            inp_gpu, sup_gpu = inp.to(cfg.device, non_blocking=True), sup.to(cfg.device, non_blocking=True)
            torch.cuda.synchronize(cfg.device)
            del inp, sup, inp_gpu, sup_gpu
            # Controlled compute window based on the measured forward/backward workload.
            time.sleep(second_compute)
        loader.finish_epoch()
        elapsed = time.perf_counter() - started
    if not all(pinned_checks):
        raise AssertionError("Prefetched batch did not use pinned memory")
    report["prefetch"] = {"depth": cfg.prefetch_batches, "workers": cfg.loader_workers,
                          "first_batch_wait_seconds": waits[0], "steady_batch_wait_seconds": waits[1:],
                          "steady_median_wait_seconds": statistics.median(waits[1:]),
                          "all_tensors_pinned": True, "elapsed_seconds": elapsed,
                          "compute_window_seconds": second_compute,
                          "note": "Synthetic wait experiment; actual training throughput must be read from live runs"}
    report["passed"] = True
    write_json(output, report)
    print(f"Steady prefetch median wait: {report['prefetch']['steady_median_wait_seconds']:.4f}s; "
          "BF16 losses and gradients exactly equal", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/registered_roi32_epoch200_a1_t0_seed17.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/roi32_pipeline_benchmark_20261004.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least two")
    check(args.manifest, args.config, args.output, args.repeats)


if __name__ == "__main__":
    main()
