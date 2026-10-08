#!/usr/bin/env python3
"""Run the baseline, A1/A2, or loss-ablation ROI32 experiments in a CUDA queue."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import load_checkpoint, read_json, write_json
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import (CHECKPOINT_SCHEMA, _checkpoint_config_matches,
                                  _patient_partitions, _protocol, _selection_protocol, _source_snapshot)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/registered_roi32_epoch200_20261003"
OVERFITTING_OUTPUT = ROOT / "runs/registered_roi32_a1_a2_20261004"
LOSS_OUTPUT = ROOT / "runs/registered_roi32_l1_l4_20261004"
SUITE_OUTPUTS = {"baseline": DEFAULT_OUTPUT, "a1_a2": OVERFITTING_OUTPUT, "losses": LOSS_OUTPUT}
LOSS_SETTINGS = {
    "l1": {"jepa_weight": 0.0, "reconstruction_weight": 0.0,
           "variance_weight": 0.0, "variance_definition": "off"},
    "l2": {"jepa_weight": 0.5, "reconstruction_weight": 0.0,
           "variance_weight": 0.01, "variance_definition": "legacy"},
    "l3": {"jepa_weight": 0.0, "reconstruction_weight": 0.1,
           "variance_weight": 0.01, "variance_definition": "legacy"},
    "l4": {"jepa_weight": 0.5, "reconstruction_weight": 0.1,
           "variance_weight": 0.01, "variance_definition": "patient_axis_stage_token_fp32"},
}
LOSS_DIAGNOSTICS = {"diagnostics_every_epochs": 5, "diagnostics_patients": 32,
                    "diagnostics_batch_size": 16, "diagnostics_seed": 20261004}
WORLD_EVALUATOR = ROOT / "scripts/evaluate_registered_roi32_world.py"
WORLD_MEMORY_MIB = 10240
TRAINING_MEMORY_MIB = {
    "l1": {"t0": 4096, "dynamic": 7680},
    "l2": {"t0": 6656, "dynamic": 9216},
    "l3": {"t0": 4608, "dynamic": 7680},
    "l4": {"t0": 6656, "dynamic": 9216},
}


def training_memory_mib(job):
    """Conservative measured peaks include diagnostics and CUDA context margin."""
    profile = "t0" if job["profile"] == "t0" else "dynamic"
    return TRAINING_MEMORY_MIB.get(job.get("experiment"), TRAINING_MEMORY_MIB["l4"])[profile]


def world_memory_mib(job):
    return 0 if job.get("world_device") == "cpu" else WORLD_MEMORY_MIB


def validate_memory_budget(budget, planned_jobs, suite, hardware=False):
    if budget is None:
        return
    required = max([training_memory_mib(job) for job in planned_jobs]
                   + ([world_memory_mib(job) for job in planned_jobs] if suite == "losses" else []))
    if budget < required:
        raise ValueError(f"GPU memory budget must accommodate every individual worker (at least {required} MiB)")
    if hardware:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        try:
            result = subprocess.run(["nvidia-smi", f"--id={visible}", "--query-gpu=memory.total",
                                     "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                    text=True, timeout=10)
            capacity = int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise RuntimeError("Cannot verify GPU capacity for the memory-aware queue") from exc
        if budget > capacity:
            raise ValueError(f"GPU memory budget exceeds visible device capacity ({capacity} MiB)")


def admission_reason(reservation, worker_count, reserved, max_concurrent, memory_budget):
    if worker_count >= max_concurrent:
        return "concurrency_limit"
    if memory_budget is not None and reserved + reservation > memory_budget:
        return "gpu_memory_budget"
    return None


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def validate_epoch_config(cfg, profile, seed):
    expected = ("epochs", 200, 50, 16, 1, seed, "t0" if profile == "t0" else "all_observed")
    actual = (cfg.training_unit, cfg.max_epochs, cfg.early_stopping_patience,
              cfg.batch_size, cfg.accumulation, cfg.seed, cfg.landmarks)
    if actual != expected or cfg.model.enable_flow or cfg.flow_weight != 0:
        raise ValueError("Formal configuration differs from the authorized epoch experiment")


def jobs(output, suite="baseline"):
    if suite not in SUITE_OUTPUTS:
        raise ValueError("Unknown ROI32 experiment suite")
    if suite == "losses":
        result = []
        for experiment, loss_settings in LOSS_SETTINGS.items():
            for seed in (17, 43):
                for profile in ("t0", "dynamic"):
                    control = "a1" if profile == "t0" else "a2"
                    name = f"{experiment}_{profile}_seed{seed}"
                    control_name = f"{control}_{profile}_seed{seed}"
                    config = ROOT / f"configs/registered_roi32_epoch200_{name}.yaml"
                    reference = copy.deepcopy(load_config(
                        ROOT / f"configs/registered_roi32_epoch200_{control_name}.yaml").to_dict())
                    reference.update(loss_settings)
                    reference.update(LOSS_DIAGNOSTICS)
                    cfg = load_config(config)
                    validate_epoch_config(cfg, profile, seed)
                    if cfg.to_dict() != reference:
                        raise ValueError(f"{experiment.upper()} must differ from its locked A* by only "
                                         "the planned loss setting and shared diagnostics")
                    result.append({"name": name, "profile": profile, "seed": seed,
                                   "experiment": experiment, "config": str(config),
                                   "output": str(output / name), "status": "queued",
                                   "world_device": "cpu" if cfg.jepa_weight == cfg.flow_weight == 0 else "cuda",
                                   "l0_reference": str(OVERFITTING_OUTPUT / control_name)})
        return result
    result = []
    for seed in (17, 43):
        for profile in ("t0", "dynamic"):
            baseline = ROOT / f"configs/registered_roi32_epoch200_{profile}_seed{seed}.yaml"
            for experiment in (("a1", "a2") if suite == "a1_a2" else (None,)):
                prefix = f"{experiment}_" if experiment else ""
                name = f"{prefix}{profile}_seed{seed}"
                config = ROOT / f"configs/registered_roi32_epoch200_{name}.yaml"
                cfg = load_config(config)
                validate_epoch_config(cfg, profile, seed)
                if experiment:
                    reference = copy.deepcopy(load_config(baseline).to_dict())
                    reference.update(image_cache="image_cache_20261004", prefetch_batches=2,
                                     loader_workers=2, pin_memory=True, non_blocking_transfer=True)
                    if experiment == "a1":
                        reference["lr_scheduler"] = "plateau"
                    else:
                        reference["model"]["dropout"] = .3
                    if cfg.to_dict() != reference:
                        raise ValueError(f"{experiment.upper()} must differ from its baseline by only the planned factor")
                result.append({"name": name, "profile": profile, "seed": seed,
                               "experiment": experiment or "a0", "config": str(config),
                               "output": str(output / name), "status": "queued"})
    return result


def validate_completed_run(job, manifest):
    """A completed report must belong to matching best/last checkpoints."""
    run = Path(job["output"])
    for name in ("best.pt", "last.pt", "train_report.json", "best_validation.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(f"Completed run lacks {name}")
    cfg = load_config(job["config"])
    store = RawMRIStore(manifest, cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache)
    signature, sources = store.signature(), _source_snapshot()
    partitions, normalization = _patient_partitions(store), store.normalization_state()
    report = read_json(run / "train_report.json")
    if (not report.get("completed") or report.get("landmarks") != cfg.landmarks
            or report.get("selection_protocol") != _selection_protocol(cfg.landmarks)
            or report.get("stop_reason") not in {"early_stopping", "max_epochs"}):
        raise ValueError("Completed run report differs from the requested experiment")
    for name in ("best.pt", "last.pt"):
        state = load_checkpoint(run / name)
        if (state.get("schema") != CHECKPOINT_SCHEMA or not _checkpoint_config_matches(state["config"], cfg)
                or state.get("source_snapshot") != sources or state.get("data_signature") != signature
                or state.get("normalization") != normalization or state.get("patient_partitions") != partitions
                or state.get("protocol") != _protocol(cfg.landmarks)
                or state.get("selection_protocol") != _selection_protocol(cfg.landmarks)):
            raise ValueError("Completed checkpoint config/data/source identity differs from the requested experiment")
        if cfg.lr_scheduler == "plateau" and state.get("scheduler_state") is None:
            raise ValueError("Completed A1 checkpoint lacks scheduler state")
        if state["best_nll"] != report["best_validation_nll"]:
            raise ValueError("Completed checkpoint/report best NLL differs")
        if name == "last.pt" and (state["completed_steps"] != report["completed_steps"]
                or state["epoch_state"]["completed_epochs"] != report["completed_epochs"]):
            raise ValueError("Completed checkpoint/report progress differs")
    if read_json(run / "best_validation.json")["selection_score"] != report["best_validation_nll"]:
        raise ValueError("Completed best validation score differs from the report")
    return report


def private_json(path, value):
    write_json(path, value)
    path.chmod(0o600)


def process_info(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        proc = Path(f"/proc/{pid}")
        command = proc.joinpath("cmdline").read_bytes().split(b"\0")
        fields = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    if fields[0] in ("Z", "X"):
        return None
    return {"command": [os.fsdecode(part) for part in command if part],
            "start_ticks": fields[19]}


def has_argument(command, flag, value):
    return any(part == flag and index + 1 < len(command) and command[index + 1] == str(value)
               for index, part in enumerate(command))


def alive_controller(output):
    for receipt in (output / "controller.json", output / "launch.json"):
        if receipt.is_file():
            pid = read_json(receipt).get("controller_pid")
            info = process_info(pid)
            if info and str(Path(__file__).resolve()) in info["command"] and has_argument(
                    info["command"], "--output", output):
                return pid
    return None


def matching_training_process(pid, job, manifest, start_ticks=None):
    info = process_info(pid)
    if info is None or (start_ticks is not None and info["start_ticks"] != start_ticks):
        return None
    command = info["command"]
    if (str(ROOT / "train.py") in command and "train-vla-jepa" in command
            and has_argument(command, "--config", job["config"])
            and has_argument(command, "--manifest", manifest)
            and has_argument(command, "--output", job["output"])):
        return info
    return None


def recorded_children(previous):
    result = {job["name"]: job for job in previous.get("jobs", [])
              if type(job.get("child_pid")) is int}
    # v1 kept its only child's PID on the controller, rather than on the job.
    name, pid = previous.get("current_job"), previous.get("child_pid")
    if name and type(pid) is int and name not in result:
        item = next((job for job in previous.get("jobs", []) if job["name"] == name), {})
        result[name] = {**item, "child_pid": pid}
    return result


def matching_world_process(pid, job, manifest, checkpoint, start_ticks=None):
    if checkpoint not in {"best", "last"}:
        return None
    info = process_info(pid)
    if info is None or (start_ticks is not None and info["start_ticks"] != start_ticks):
        return None
    command, run = info["command"], Path(job["output"])
    if (str(WORLD_EVALUATOR) in command
            and has_argument(command, "--checkpoint", run / f"{checkpoint}.pt")
            and has_argument(command, "--manifest", manifest)
            and has_argument(command, "--output", run / f"world_{checkpoint}.json")):
        return info
    return None


def alive_world_children(output, manifest, suite):
    receipt = output / "controller.json"
    if not receipt.is_file():
        return []
    current = {job["name"]: job for job in jobs(output, suite)}
    result = []
    for job in read_json(receipt).get("jobs", []):
        record = job.get("world_evaluation", {})
        if job["name"] in current and matching_world_process(
                record.get("child_pid"), current[job["name"]], manifest,
                record.get("checkpoint"), record.get("process_start_ticks")):
            result.append(record["child_pid"])
    return result


def evaluate_completed_world(job, manifest, persist):
    """Evaluate best/last independently; always rerun on resume, without training."""
    run = Path(job["output"])
    expected_status = "evaluated" if load_config(job["config"]).jepa_weight > 0 else "unavailable"
    old = job.get("world_evaluation", {})
    while matching_world_process(old.get("child_pid"), job, manifest, old.get("checkpoint"),
                                 old.get("process_start_ticks")):
        # A controller replacement must wait for its orphan evaluator before starting another.
        time.sleep(3)
    results = {}
    log_path = run / "world_evaluation.log"
    with log_path.open("a") as log:
        log_path.chmod(0o600)
        for name in ("best", "last"):
            checkpoint, target = run / f"{name}.pt", run / f"world_{name}.json"
            identity = {"path": str(checkpoint.resolve()), "size_bytes": checkpoint.stat().st_size,
                        "mtime_ns": checkpoint.stat().st_mtime_ns}
            command = [sys.executable, str(WORLD_EVALUATOR), "--checkpoint", str(checkpoint),
                       "--manifest", str(manifest), "--output", str(target),
                       "--device", job.get("world_device", "cuda"), "--batch-size", "16", "--split", "val"]
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            info = matching_world_process(child.pid, job, manifest, name)
            job["world_evaluation"] = {"status": "running", "checkpoint": name,
                                      "child_pid": child.pid,
                                      "process_start_ticks": info["start_ticks"] if info else None,
                                      "started_at": now()}
            persist()
            code = child.wait()
            job["world_evaluation"].update(child_pid=None, exit_code=code, ended_at=now())
            persist()
            if code != 0:
                raise RuntimeError(f"{name} world evaluation exited with code {code}")
            result = read_json(target)
            after = {"path": str(checkpoint.resolve()), "size_bytes": checkpoint.stat().st_size,
                     "mtime_ns": checkpoint.stat().st_mtime_ns}
            if (after != identity or result.get("checkpoint_identity") != identity
                    or result.get("checkpoint_file_unchanged") is not True
                    or result.get("runtime_source_matches_checkpoint") is not True
                    or result.get("status") != expected_status):
                raise ValueError(f"{name} world evaluation checkpoint/source identity differs")
            results[name] = {"output": str(target), "status": result["status"],
                             "checkpoint_identity": identity}
    job["world_evaluation"] = {"status": "completed", "ended_at": now(), "results": results}
    persist()
    return results


def alive_training_children(output, manifest, suite="baseline"):
    receipt = output / "controller.json"
    if not receipt.is_file():
        return []
    previous = read_json(receipt)
    current = {job["name"]: job for job in jobs(output, suite)}
    return [item["child_pid"] for name, item in recorded_children(previous).items()
            if name in current and matching_training_process(item["child_pid"], current[name], manifest,
                                                             item.get("process_start_ticks"))]


def status(output):
    receipt = output / "controller.json"
    value = read_json(receipt if receipt.is_file() else output / "launch.json")
    value["controller_alive"] = alive_controller(output) is not None
    children = recorded_children(value)
    for job in value.get("jobs", []):
        progress = Path(job["output"]) / "progress.json"
        if progress.is_file():
            job["progress"] = read_json(progress)
        child = children.get(job["name"], {})
        job["child_alive"] = matching_training_process(child.get("child_pid"), job,
                                                       value.get("manifest"),
                                                       child.get("process_start_ticks")) is not None
        world = job.get("world_evaluation", {})
        job["world_child_alive"] = matching_world_process(world.get("child_pid"), job,
                    value.get("manifest"), world.get("checkpoint"), world.get("process_start_ticks")) is not None
    return value


def work(output, manifest, resume, max_concurrent=2, suite="baseline", gpu_memory_budget_mib=None):
    with (output / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An experiment controller already holds this queue") from exc
        receipt = output / "controller.json"
        previous = read_json(receipt) if receipt.is_file() else {}
        if not resume and alive_training_children(output, manifest, suite):
            raise RuntimeError("Live training children require --resume to adopt them")
        if not resume and alive_world_children(output, manifest, suite):
            raise RuntimeError("Live world evaluators require --resume to wait for them")
        if resume and previous.get("manifest", str(manifest)) != str(manifest):
            raise ValueError("Cannot resume a queue against a different manifest")
        if resume and previous.get("suite", "baseline") != suite:
            raise ValueError("Cannot resume a queue against a different experiment suite")
        planned_jobs = jobs(output, suite)
        validate_memory_budget(gpu_memory_budget_mib, planned_jobs, suite)
        state = {"schema": "registered_roi32_epoch_queue_v2", "controller_pid": os.getpid(),
                 "started_at": now(), "status": "running", "manifest": str(manifest),
                 "suite": suite,
                 "device_policy": f"At most {max_concurrent} concurrent training/evaluation workers"
                                  + (f"; measured reservations <= {gpu_memory_budget_mib} MiB"
                                     if gpu_memory_budget_mib is not None else ""),
                 "max_concurrent": max_concurrent,
                 "gpu_memory_budget_mib": gpu_memory_budget_mib,
                 "max_epochs": 200, "patience": 50, "batch_size": 16,
                 "seeds": [17, 43], "current_jobs": [], "current_job": None, "child_pid": None,
                 "jobs": planned_jobs}
        active, active_world = {}, {}
        old_children = recorded_children(previous) if resume else {}
        old_jobs = {job["name"]: job for job in previous.get("jobs", [])} if resume else {}

        for job in state["jobs"]:
            job.update(training_memory_mib=training_memory_mib(job), world_memory_mib=world_memory_mib(job))

        def workers():
            result = [{"name": name, "kind": "training", "child_pid": item["pid"],
                       "memory_reservation_mib": item["memory_mib"]} for name, item in active.items()]
            result.extend({"name": name, "kind": "world", "child_pid": item["pid"],
                           "memory_reservation_mib": item["memory_mib"], "adopted": True}
                          for name, item in active_world.items())
            result.extend({"name": job["name"], "kind": "world",
                           "child_pid": job.get("world_evaluation", {}).get("child_pid"),
                           "memory_reservation_mib": world_memory_mib(job)}
                          for job in state["jobs"] if job["status"] == "evaluating"
                          and job["name"] not in active_world)
            return result

        def can_admit(job, reservation):
            current = workers()
            reason = admission_reason(reservation, len(current),
                                      sum(item["memory_reservation_mib"] for item in current),
                                      max_concurrent, gpu_memory_budget_mib)
            job["deferred_reason"] = reason
            return reason is None

        def persist():
            state["current_jobs"] = list(active)
            first = next(iter(active), None)
            state["current_job"] = first
            state["child_pid"] = active[first]["pid"] if first else None
            state["current_world_jobs"] = [worker["name"] for worker in workers() if worker["kind"] == "world"]
            state["current_workers"] = workers()
            state["gpu_memory_reserved_mib"] = sum(worker["memory_reservation_mib"] for worker in workers())
            state["updated_at"] = now()
            private_json(receipt, state)

        for job in state["jobs"]:
            run = Path(job["output"])
            report_path = run / "train_report.json"
            old = old_children.get(job["name"], {})
            info = matching_training_process(old.get("child_pid"), job, manifest,
                                             old.get("process_start_ticks"))
            if info is not None:
                if any(item["pid"] == old["child_pid"] for item in active.values()):
                    raise ValueError("One recorded training PID cannot belong to multiple jobs")
                job.update(status="running", child_pid=old["child_pid"],
                           process_start_ticks=info["start_ticks"], adopted=True, adopted_at=now(),
                           started_at=old.get("started_at", now()))
                active[job["name"]] = {"pid": old["child_pid"], "process": None, "log": None,
                                       "memory_mib": training_memory_mib(job)}
                continue
            if resume and report_path.is_file() and read_json(report_path).get("completed"):
                job.update(status="evaluation_pending" if suite == "losses" else "completed",
                           report=validate_completed_run(job, manifest))
                if suite == "losses" and old_jobs.get(job["name"], {}).get("world_evaluation"):
                    record = old_jobs[job["name"]]["world_evaluation"]
                    job["world_evaluation"] = record
                    world_info = matching_world_process(record.get("child_pid"), job, manifest,
                                      record.get("checkpoint"), record.get("process_start_ticks"))
                    if world_info is not None:
                        if any(item["pid"] == record["child_pid"] for item in active_world.values()):
                            raise ValueError("One recorded world PID cannot belong to multiple jobs")
                        job.update(status="evaluating_adopted", world_adopted_at=now())
                        active_world[job["name"]] = {"pid": record["child_pid"],
                            "memory_mib": 0 if has_argument(world_info["command"], "--device", "cpu")
                                          else WORLD_MEMORY_MIB}
        if len(active) + len(active_world) > max_concurrent:
            raise ValueError("Existing live GPU workers exceed --max-concurrent")
        persist()
        try:
            while True:
                for job in state["jobs"]:
                    if job["name"] not in active_world:
                        continue
                    record = job["world_evaluation"]
                    if matching_world_process(record.get("child_pid"), job, manifest,
                            record.get("checkpoint"), record.get("process_start_ticks")):
                        continue
                    del active_world[job["name"]]
                    job.update(status="evaluation_pending")
                    job["world_evaluation"] = {**record, "child_pid": None,
                                                "status": "adopted_finished", "ended_at": now()}
                    persist()
                for job in state["jobs"]:
                    if job["name"] not in active:
                        continue
                    item = active[job["name"]]
                    code = item["process"].poll() if item["process"] is not None else None
                    if item["process"] is not None and code is None:
                        continue
                    if item["process"] is None and matching_training_process(
                            item["pid"], job, manifest, job.get("process_start_ticks")):
                        continue
                    if item["log"] is not None:
                        item["log"].close()
                    del active[job["name"]]
                    report_path = Path(job["output"]) / "train_report.json"
                    report = read_json(report_path) if report_path.is_file() else None
                    success = (code in (None, 0) and report is not None and report.get("completed"))
                    if success:
                        try:
                            report = validate_completed_run(job, manifest)
                        except (OSError, ValueError, KeyError) as exc:
                            success = False
                            job["failure_reason"] = str(exc)
                    success_status = "evaluation_pending" if suite == "losses" else "completed"
                    job.update(status=success_status if success else "failed", exit_code=code,
                               ended_at=now(), child_pid=None)
                    if report is not None:
                        job["report"] = report
                    if not success:
                        job.setdefault("failure_reason", "Training exited without a successful completed report")
                    persist()
                for job in state["jobs"]:
                    if job["status"] != "evaluation_pending":
                        continue
                    if not can_admit(job, world_memory_mib(job)):
                        continue
                    job["status"] = "evaluating"
                    persist()
                    try:
                        evaluate_completed_world(job, manifest, persist)
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        job.update(status="evaluation_failed", evaluation_failure_reason=str(exc))
                    else:
                        job.update(status="completed")
                    persist()
                # A finite suite can fill spare capacity with lighter jobs without starving larger ones.
                candidates = ([job for job in state["jobs"] if job["status"] == "queued"]
                              if gpu_memory_budget_mib is not None else state["jobs"])
                if gpu_memory_budget_mib is not None:
                    candidates.sort(key=training_memory_mib)
                for job in candidates:
                    if job["status"] != "queued":
                        continue
                    if not can_admit(job, training_memory_mib(job)):
                        continue
                    run = Path(job["output"])
                    run.mkdir(mode=0o700, exist_ok=True)
                    command = [sys.executable, str(ROOT / "train.py"), "train-vla-jepa",
                               "--config", job["config"], "--manifest", str(manifest), "--output", str(run)]
                    if resume and (run / "last.pt").is_file():
                        command.append("--resume")
                    log_path = run / "training.log"
                    log = log_path.open("a")
                    log_path.chmod(0o600)
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT)
                    except OSError as exc:
                        log.close()
                        job.update(status="failed", ended_at=now(), failure_reason=str(exc))
                        persist()
                        continue
                    info = matching_training_process(child.pid, job, manifest)
                    job.update(status="running", started_at=now(), child_pid=child.pid, adopted=False,
                               process_start_ticks=info["start_ticks"] if info else None)
                    active[job["name"]] = {"pid": child.pid, "process": child, "log": log,
                                           "memory_mib": training_memory_mib(job)}
                    persist()
                pending = any(job["status"] in {"queued", "evaluation_pending"} for job in state["jobs"])
                persist()
                if not active and not active_world and not pending:
                    break
                if not active and not active_world and pending:
                    state.update(status="failed", failure_reason="No pending worker fits the admission policy")
                    persist()
                    raise RuntimeError(state["failure_reason"])
                time.sleep(3)
        finally:
            # Children survive a controller replacement and are adopted by --resume.
            for item in active.values():
                if item["log"] is not None:
                    item["log"].close()
        failed = any(job["status"] in {"failed", "evaluation_failed"} for job in state["jobs"])
        state.update(status="failed" if failed else "completed", ended_at=now())
        persist()
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITE_OUTPUTS), default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-concurrent", type=int, choices=range(1, 7), default=2,
                        help="Maximum simultaneous training/evaluation workers (default: 2)")
    parser.add_argument("--gpu-memory-budget-mib", type=int,
                        help="Admit workers only within conservative measured GPU reservations")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (args.output or SUITE_OUTPUTS[args.suite]).resolve()
    manifest = args.manifest.resolve()
    if args.status:
        print(json.dumps(status(output), indent=2))
        return 0
    os.umask(0o077)
    if not manifest.is_file():
        raise FileNotFoundError("Prepare the registered ROI32 manifest before launching")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    planned_jobs = jobs(output, args.suite)
    validate_memory_budget(args.gpu_memory_budget_mib, planned_jobs, args.suite, hardware=True)
    if args.worker:
        return work(output, manifest, args.resume, args.max_concurrent, args.suite, args.gpu_memory_budget_mib)
    with (output / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = alive_controller(output)
        if pid is not None:
            raise RuntimeError(f"Experiment controller {pid} is already running")
        children = alive_training_children(output, manifest, args.suite)
        if children and not args.resume:
            raise RuntimeError(f"Training processes {children} are still running; use --resume to adopt")
        if len(children) > args.max_concurrent:
            raise ValueError("Existing live children exceed --max-concurrent")
        evaluators = alive_world_children(output, manifest, args.suite)
        if evaluators and not args.resume:
            raise RuntimeError("Live world evaluators require --resume to wait for them")
        if len(children) + len(evaluators) > args.max_concurrent:
            raise ValueError("Existing GPU workers exceed --max-concurrent")
        if (output / "launch.json").exists() and not args.resume:
            raise FileExistsError("Use --resume for an existing queue or choose a new output directory")
        if args.detach:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--output", str(output), "--manifest", str(manifest),
                       "--suite", args.suite,
                       "--max-concurrent", str(args.max_concurrent)]
            if args.resume:
                command.append("--resume")
            if args.gpu_memory_budget_mib is not None:
                command.extend(["--gpu-memory-budget-mib", str(args.gpu_memory_budget_mib)])
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            environment["PYTHONUNBUFFERED"] = "1"
            with (output / "controller.log").open("a") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt = {"controller_pid": child.pid, "started_at": now(), "output": str(output),
                       "manifest": str(manifest), "max_concurrent": args.max_concurrent,
                       "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
                       "suite": args.suite, "status": "launched", "jobs": planned_jobs}
            private_json(output / "launch.json", receipt)
            print(json.dumps(receipt, indent=2))
            return 0
        return work(output, manifest, args.resume, args.max_concurrent, args.suite, args.gpu_memory_budget_mib)


if __name__ == "__main__":
    sys.exit(main())
