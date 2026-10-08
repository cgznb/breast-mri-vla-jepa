"""Audit saved ROI32 histories and probe loss gradients without updating weights."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import autocast, seed_all, write_json
from mri_vla_jepa.train_config import from_dict
from mri_vla_jepa.training import _patient_partitions, _source_snapshot, load_trained

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("t0_seed17", "t0_seed43", "dynamic_seed17", "dynamic_seed43")
TERMS = ("task", "jepa", "reconstruction", "variance")
WEIGHT_KEYS = ("task_weight", "jepa_weight", "reconstruction_weight", "variance_weight")
LABELS = ("pCR BCE", "0.5 x JEPA L1", "0.1 x reconstruction MSE", "0.01 x variance")
COLORS = ("#2468ad", "#dc7044", "#279779", "#8d64a4")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def title(name):
    return name.replace("t0_", "Fixed T0 / ").replace("dynamic_", "Dynamic / ").replace("seed", "seed ")


def audit_histories(run_root, output):
    summary, epoch_rows = {}, []
    for name in RUNS:
        run = run_root / name
        config, steps, epochs = (read_json(run / item) for item in
                                ("config.json", "history.json", "epoch_history.json"))
        report = read_json(run / "train_report.json")
        weights = dict(zip(TERMS, (config[key] for key in WEIGHT_KEYS)))
        expected_batches = math.ceil(764 / config["batch_size"])
        assert report["completed"] and report["stop_reason"] == "early_stopping"
        assert len(steps) == len(epochs) * expected_batches == report["completed_steps"]
        assert all(s["step"] == i + 1 for i, s in enumerate(steps))
        assert [e["epoch"] for e in epochs] == list(range(1, len(epochs) + 1))
        best = min(epochs, key=lambda e: e["validation"]["selection_score"])
        assert best["validation"]["selection_score"] == read_json(run / "best_validation.json")["selection_score"]
        clip = config["grad_clip"]
        residuals = []
        for step in steps:
            values = [step[key] for key in ("loss", *TERMS, "flow", "world_model", "gradient_norm",
                                           "label_count", "future_count", "pair_count")]
            assert all(math.isfinite(v) for v in values)
            assert step["gradient_norm"] >= 0
            assert step["jepa"] == step["world_model"]
            assert step["flow"] == 0 and config["flow_weight"] == 0
            weighted = sum(weights[key] * step[key] for key in TERMS)
            residuals.append(abs(weighted - step["loss"]))
        assert max(residuals) < 2e-5
        gradient = np.asarray([s["gradient_norm"] for s in steps], dtype=np.float64)
        by_epoch = {}
        for step in steps:
            by_epoch.setdefault(step["epoch"], []).append(step)
        rows = []
        for epoch in epochs:
            number, train = epoch["epoch"], epoch["training"]
            batch_rows = by_epoch[number]
            norms = np.asarray([s["gradient_norm"] for s in batch_rows])
            assert epoch["patients"] == sum(s["batch_size"] for s in batch_rows) == 764
            assert sum(s["label_count"] for s in batch_rows) == 764
            assert epoch["batches"] == len(batch_rows) == expected_batches
            assert [s["batch_in_epoch"] for s in batch_rows] == list(range(1, expected_batches + 1))
            assert all(s["batch_size"] == 16 for s in batch_rows[:-1]) and batch_rows[-1]["batch_size"] == 12
            for key in ("loss", *TERMS, "flow", "gradient_norm"):
                replay = sum(s[key] * s["batch_size"] for s in batch_rows) / 764
                assert math.isclose(train[key], replay, abs_tol=1e-10)
            row = {"run": name, "epoch": number,
                   "training_loss": train["loss"], "training_task": train["task"],
                   "validation_selection_nll": epoch["validation"]["selection_score"],
                   "gradient_mean": train["gradient_norm"],
                   "gradient_median": float(np.median(norms)),
                   "gradient_p95": float(np.quantile(norms, .95)),
                   "gradient_max": float(norms.max()),
                   "clipped_fraction": float((norms > clip).mean()),
                   "weighted_world_task_ratio": weights["jepa"] * train["jepa"] / (weights["task"] * train["task"]),
                   "is_best_epoch": number == best["epoch"]}
            for key in TERMS:
                row[f"raw_{key}"] = train[key]
                row[f"weighted_{key}"] = weights[key] * train[key]
            for stage, metrics in epoch["validation"]["per_landmark"].items():
                for key in ("auroc", "auprc", "nll", "brier"):
                    row[f"validation_{stage}_{key}"] = metrics[key]
            rows.append(row)
            epoch_rows.append(row)
        points = {}
        for label, e in (("first", epochs[0]), ("best", best), ("final", epochs[-1])):
            weighted = {key: weights[key] * e["training"][key] for key in TERMS}
            points[label] = {"epoch": e["epoch"], "weighted_terms": weighted,
                             "weighted_term_shares": {key: value / sum(weighted.values()) for key, value in weighted.items()},
                             "training_task": e["training"]["task"],
                             "validation_selection_nll": e["validation"]["selection_score"],
                             "gradient_mean_preclip": e["training"]["gradient_norm"]}
        summary[name] = {"steps": len(steps), "stop_epoch": len(epochs), "best_epoch": best["epoch"],
                         "selection_protocol": best["validation"]["selection_metric"],
                         "config": {key: config[key] for key in (*WEIGHT_KEYS, "batch_size", "lr", "grad_clip", "precision")},
                         "all_logged_values_finite": True,
                         "max_loss_sum_residual": max(residuals), "epoch_aggregation_replayed": True,
                         "gradient_preclip": {"median": float(np.median(gradient)),
                                              "p95": float(np.quantile(gradient, .95)),
                                              "max": float(gradient.max()),
                                              "clipped_steps": int((gradient > clip).sum()),
                                              "clipped_fraction": float((gradient > clip).mean())},
                         "points": points, "epochs": rows}
        print(f"{name}: {len(steps)} finite steps, best epoch {best['epoch']}, "
              f"clipped {(gradient > clip).mean():.2%}", flush=True)
    result = {"scope": "saved formal training histories; no training updates",
              "gradient_semantics": "total L2 norm returned before clip_grad_norm_; threshold 1.0",
              "epoch_semantics": "batch objective means weighted by patient count; not a global voxel or pair mean",
              "validation": "102 development patients, no independent test",
              "runs": summary}
    write_json(output / "history_audit.json", result)
    with (output / "epoch_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    return result


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def history_figures(audit, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    for kind in ("pcr_curves", "weighted_loss_curves", "gradient_curves", "balance_curves"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
        for ax, (name, run) in zip(axes.flat, audit["runs"].items()):
            rows = run["epochs"]
            x = np.asarray([r["epoch"] for r in rows])
            ax.set_title(title(name))
            ax.set_xlabel("Epoch")
            ax.axvline(run["best_epoch"], color="#777777", ls=":", lw=1.2, label="Best NLL epoch")
            ax.grid(alpha=.18)
            if kind == "pcr_curves":
                ax.plot(x, [r["training_task"] for r in rows], color=COLORS[0], label="Train pCR BCE")
                ax.plot(x, [r["validation_selection_nll"] for r in rows], color=COLORS[1], label="Validation selection NLL")
                ax.set_ylabel("BCE / NLL")
                best = run["points"]["best"]
                ax.scatter([best["epoch"]], [best["validation_selection_nll"]], color=COLORS[1], s=28, zorder=4)
            elif kind == "weighted_loss_curves":
                for key, label, color in zip(TERMS, LABELS, COLORS):
                    ax.plot(x, [r[f"weighted_{key}"] for r in rows], label=label, color=color)
                ax.set_ylabel("Loss contribution after coefficient")
            elif kind == "gradient_curves":
                ax.plot(x, [r["gradient_mean"] for r in rows], color=COLORS[0], label="Epoch mean (patient weighted)")
                ax.plot(x, [r["gradient_p95"] for r in rows], color=COLORS[1], lw=1.1, label="Within-epoch step P95")
                ax.axhline(run["config"]["grad_clip"], color=COLORS[2], ls="--", label="Clip threshold = 1.0")
                ax.set_yscale("log")
                ax.set_ylabel("Total gradient L2 norm BEFORE clipping")
            else:
                ax.plot(x, [r["weighted_world_task_ratio"] for r in rows], color=COLORS[1], label="Weighted JEPA / pCR loss")
                ax.axhline(1, color=COLORS[2], ls="--", label="Equal scalar contributions")
                ax.set_ylabel("Scalar loss ratio")
            ax.legend(loc="best", fontsize=8)
        titles = {"pcr_curves": "Full recorded training: pCR fit versus development validation",
                  "weighted_loss_curves": "Weighted training losses (JEPA and world_model are the SAME term)",
                  "gradient_curves": "Historical gradients: finite, frequently clipped, larger late in training",
                  "balance_curves": "Loss magnitudes alone do not determine gradient influence"}
        fig.suptitle(titles[kind], fontsize=13)
        save_figure(fig, output, kind)


def group_name(name):
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith(("fusion.", "fusion_norm.")):
        return "fusion"
    if name.startswith("pcr_head."):
        return "pcr_head"
    if name.startswith("world_predictor."):
        return "world_predictor"
    if name.startswith("reconstruction."):
        return "reconstruction_head"
    return "conditioning_and_queries"


def cosine(a, b):
    denominator = torch.linalg.vector_norm(a.double()) * torch.linalg.vector_norm(b.double())
    return None if denominator.item() == 0 else float(torch.dot(a.double(), b.double()) / denominator)


def probe_checkpoints(run_root, manifest, output, batches, seed, device):
    results = []
    saved_files = [run_root / name / f"{which}.pt" for name in RUNS for which in ("best", "last")]
    identities = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in saved_files]
    normalization = None
    for name in RUNS:
        for which in ("best", "last"):
            model, cfg, state = load_trained(run_root / name / f"{which}.pt", device=device)
            assert state["source_snapshot"] == _source_snapshot()
            assert state["config"] == from_dict(read_json(run_root / name / "config.json")).to_dict()
            store = RawMRIStore(manifest, cfg.image_shape, normalization_state=state["normalization"])
            assert state["patient_partitions"] == _patient_partitions(store)
            assert state["data_signature"] == store.signature()
            if normalization is None:
                normalization = store.normalization_state()
            assert normalization == store.normalization_state()
            parameters = [(key, value) for key, value in model.named_parameters() if value.requires_grad]
            names, params = zip(*parameters)
            groups = {key: [i for i, name_ in enumerate(names) if group_name(name_) == key]
                      for key in ("encoder", "fusion", "pcr_head", "world_predictor",
                                  "reconstruction_head", "conditioning_and_queries")}
            groups["shared_encoder_fusion"] = groups["encoder"] + groups["fusion"]
            groups["all_trainable"] = list(range(len(params)))
            original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            chooser = random.Random(seed)
            indices = list(store.by_split["train"])
            chooser.shuffle(indices)
            stage_chooser = random.Random(seed + 1)
            tasks = [(i, 0 if cfg.landmarks == "t0" else stage_chooser.choice(store.allowed_landmarks(i)))
                     for i in indices[:batches * cfg.batch_size]]
            assert len(tasks) == batches * 16
            model.train()
            assert not model.teacher_encoder.training
            for batch_number in range(batches):
                chunk = tasks[batch_number * 16:(batch_number + 1) * 16]
                seed_all(seed + batch_number, threads=cfg.threads, deterministic=cfg.deterministic)
                inp, sup = store.batch(chunk)
                inp, sup = inp.to(device), sup.to(device)
                model.zero_grad(set_to_none=True)
                with autocast(device, cfg.precision):
                    loss, terms = model.compute_loss(inp, sup, task_weight=cfg.task_weight,
                        jepa_weight=cfg.jepa_weight, flow_weight=cfg.flow_weight,
                        reconstruction_weight=cfg.reconstruction_weight, variance_weight=cfg.variance_weight)
                assert torch.isfinite(loss)
                weights = dict(zip(TERMS, (getattr(cfg, key) for key in WEIGHT_KEYS)))
                total_grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                gradients = {}
                all_finite = True
                for index, key in enumerate(TERMS):
                    gradients[key] = torch.autograd.grad(weights[key] * terms[key], params,
                                                        retain_graph=index < len(TERMS) - 1, allow_unused=True)
                vectors = {}
                for key, grads in {**gradients, "total": total_grads}.items():
                    all_finite &= all(g is None or bool(torch.isfinite(g).all()) for g in grads)
                    vectors[key] = {group: torch.cat([(grads[i].detach().float() if grads[i] is not None
                                                      else torch.zeros_like(params[i], dtype=torch.float32)).reshape(-1)
                                                     for i in selected]) for group, selected in groups.items()}
                assert all_finite
                sum_vector = sum(vectors[key]["all_trainable"] for key in TERMS)
                residual = float(torch.linalg.vector_norm(sum_vector - vectors["total"]["all_trainable"]))
                total_norm = float(torch.linalg.vector_norm(vectors["total"]["all_trainable"]))
                relative_residual = residual / max(total_norm, 1e-12)
                # Separate BF16 backward paths round before their FP32 parameter gradients are added.
                assert residual < max(1e-5, total_norm * (.02 if cfg.precision == "bf16" else 1e-4)), relative_residual
                no_teacher_grads = all(not p.requires_grad and p.grad is None for p in model.teacher_encoder.parameters())
                assert no_teacher_grads
                record = {"run": name, "checkpoint": which,
                          "checkpoint_epoch": state["epoch_state"]["completed_epochs"],
                          "checkpoint_step": state["completed_steps"], "probe_batch": batch_number + 1,
                          "batch_size": len(chunk),
                          "landmark_counts": {f"T{s}": sum(t == s for _, t in chunk) for s in range(4)},
                          "observed_visits": int(inp.observed_mask.sum()),
                          "future_visits": int(sup.future_mask.sum()), "pair_count": float(terms["pair_count"]),
                          "weighted_losses": {key: float(terms[key].detach()) * weights[key] for key in TERMS},
                          "weighted_gradient_norms": {key: {group: float(torch.linalg.vector_norm(value))
                                                            for group, value in vector.items()}
                                                      for key, vector in vectors.items()},
                          "task_auxiliary_cosines": {
                              key: {group: cosine(vectors["task"][group], vectors[key][group])
                                    for group in ("encoder", "fusion", "shared_encoder_fusion")}
                              for key in ("jepa", "reconstruction", "variance")},
                          "all_gradient_values_finite": all_finite, "teacher_frozen_and_without_gradients": no_teacher_grads,
                          "gradient_sum_residual_l2": residual,
                          "gradient_sum_relative_residual": relative_residual,
                          "total_gradient_norm_preclip": total_norm,
                          "hypothetical_clip_multiplier": min(1.0, cfg.grad_clip / (total_norm + 1e-6))}
                results.append(record)
                print(f"Probe {name}/{which} batch {batch_number + 1}: loss={float(loss.detach()):.4f}, "
                      f"total_grad={total_norm:.3f}, shared JEPA/pCR="
                      f"{record['weighted_gradient_norms']['jepa']['shared_encoder_fusion'] / record['weighted_gradient_norms']['task']['shared_encoder_fusion']:.3f}, "
                      f"cos={record['task_auxiliary_cosines']['jepa']['shared_encoder_fusion']:.3f}", flush=True)
                del inp, sup, loss, terms, total_grads, gradients, vectors, sum_vector
                gc.collect()
            assert all(torch.equal(original[key], value.detach().cpu()) for key, value in model.state_dict().items())
            for r in results:
                if r["run"] == name and r["checkpoint"] == which:
                    r["model_state_unchanged"] = True
            del model, state, store, original, params, parameters, total_norm
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    assert identities == [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in saved_files]
    result = {"scope": "read-only gradient probes, not historical per-loss gradient averages",
              "seed": seed, "training_batches_per_checkpoint": batches,
              "sampling": "same shuffled train patients across all runs; fixed T0 or reproducible legal dynamic stages",
              "precision": "CUDA BF16 matching formal training; FP32 gradients",
              "mode": "train with dropout, frozen eval-mode EMA teacher; no optimizer or EMA updates",
              "shared_norm_scope": "encoder plus fusion and fusion_norm; excludes private heads and conditioning/query embeddings",
              "cosine_semantics": "positive aligned, negative conflicting; null denotes a zero gradient in this group",
              "checkpoint_files_unchanged": True, "records": results}
    write_json(output / "gradient_probes.json", result)
    return result


def probe_figure(probes, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
    for ax, name in zip(axes.flat, RUNS):
        for index, (key, label, color) in enumerate(zip(TERMS, LABELS, COLORS)):
            values = [[r["weighted_gradient_norms"][key]["shared_encoder_fusion"] for r in probes["records"]
                       if r["run"] == name and r["checkpoint"] == which] for which in ("best", "last")]
            x = np.arange(2) + (index - 1.5) * .18
            ax.bar(x, [np.mean(v) for v in values], .17, color=color, label=label, alpha=.85)
            for point, vector in zip(x, values):
                ax.scatter(np.repeat(point, len(vector)), vector, s=15, c="#333333", zorder=4)
        ax.set_xticks([0, 1], ["Best checkpoint", "Final checkpoint"])
        ax.set_yscale("log")
        ax.set_title(title(name))
        ax.set_ylabel("Weighted gradient L2: shared encoder + fusion")
        ax.grid(axis="y", alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle(f"Checkpoint probes: {probes['training_batches_per_checkpoint']} B16 training batches "
                 "per checkpoint (not historical averages)", fontsize=12)
    save_figure(fig, output, "checkpoint_gradient_probes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/registered_roi32_epoch200_20261003")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/training_process_20261004")
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--probe-seed", type=int, default=20261004)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.probe_batches < 1 or args.probe_batches * 16 > 764:
        parser.error("probe-batches must select 1..47 non-overlapping B16 training batches")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_histories(args.run_root, args.output)
    history_figures(audit, args.output)
    if not args.history_only:
        probes = probe_checkpoints(args.run_root, args.manifest, args.output,
                                   args.probe_batches, args.probe_seed, args.device)
        probe_figure(probes, args.output)
    print(f"Analysis artifacts: {args.output}", flush=True)


if __name__ == "__main__":
    main()
