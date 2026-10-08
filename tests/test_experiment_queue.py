import importlib.util
from pathlib import Path
import sys

import pytest

from mri_vla_jepa.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "roi32_queue", ROOT / "scripts/run_registered_roi32_experiments.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def fake_jobs(output, suite="baseline"):
    return [{"name": f"{profile}_seed{seed}", "profile": profile, "seed": seed,
             "config": str(ROOT / f"configs/{profile}_{seed}.yaml"),
             "output": str(output / f"{profile}_seed{seed}"), "status": "queued"}
            for seed in (17, 43) for profile in ("t0", "dynamic")]


def test_process_match_requires_exact_run_and_start_time(monkeypatch, tmp_path):
    job = fake_jobs(tmp_path)[0]
    manifest = tmp_path / "manifest.json"
    command = [sys.executable, str(ROOT / "train.py"), "train-vla-jepa",
               "--config", job["config"], "--manifest", str(manifest), "--output", job["output"]]
    monkeypatch.setattr(queue, "process_info", lambda pid: {"command": command, "start_ticks": "123"})
    assert queue.matching_training_process(101, job, manifest, "123")
    assert queue.matching_training_process(101, job, manifest, "456") is None
    assert queue.matching_training_process(101, {**job, "output": job["output"] + "_other"}, manifest) is None
    command[-1] += "_other"
    assert queue.matching_training_process(101, job, manifest) is None


def test_recorded_children_reads_old_and_parallel_receipts(tmp_path):
    old = {"current_job": "t0_seed17", "child_pid": 101, "jobs": fake_jobs(tmp_path)}
    assert queue.recorded_children(old)["t0_seed17"]["child_pid"] == 101
    new_jobs = fake_jobs(tmp_path)
    new_jobs[0]["child_pid"], new_jobs[1]["child_pid"] = 101, 102
    current = {"current_jobs": [job["name"] for job in new_jobs[:2]], "jobs": new_jobs}
    assert sorted(job["child_pid"] for job in queue.recorded_children(current).values()) == [101, 102]


@pytest.mark.parametrize("failed_name", [None, "dynamic_seed17"])
def test_parallel_adoption_fills_slots_without_restarting_child(tmp_path, monkeypatch, failed_name):
    output = tmp_path / "runs"
    output.mkdir()
    manifest = tmp_path / "manifest.json"
    old_jobs = fake_jobs(output)
    old_jobs[0].update(status="running", started_at="original-start")
    write_json(output / "controller.json", {
        "manifest": str(manifest), "controller_pid": 100, "current_job": "t0_seed17",
        "child_pid": 101, "jobs": old_jobs})
    monkeypatch.setattr(queue, "jobs", fake_jobs)
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest:
                        read_json(Path(job["output"]) / "train_report.json"))
    clock = {"tick": 0}
    starts = []
    snapshots = []

    def match(pid, job, manifest, start_ticks=None):
        if pid is None:
            return None
        if pid == 101 and clock["tick"] >= 2:
            write_json(Path(job["output"]) / "train_report.json", {"completed": True})
            return None
        return {"start_ticks": str(pid)}

    class Child:
        def __init__(self, command, **kwargs):
            run = Path(command[command.index("--output") + 1])
            self.name, self.run = run.name, run
            self.pid = 200 + len(starts)
            self.started = clock["tick"]
            starts.append((self.name, self.started))

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            if self.name == failed_name:
                return 1
            write_json(self.run / "train_report.json", {"completed": True})
            return 0

    def advance(seconds):
        state = read_json(output / "controller.json")
        snapshots.append(state)
        assert len(state["current_jobs"]) <= 2
        assert len({job["child_pid"] for job in state["jobs"] if job["status"] == "running"}) == len(
            state["current_jobs"])
        clock["tick"] += 1

    monkeypatch.setattr(queue, "matching_training_process", match)
    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue.time, "sleep", advance)
    result = queue.work(output, manifest, resume=True, max_concurrent=2)
    assert result == (1 if failed_name else 0)
    assert starts == [("dynamic_seed17", 0), ("t0_seed43", 2), ("dynamic_seed43", 2)]
    assert snapshots[0]["current_jobs"] == ["t0_seed17", "dynamic_seed17"]
    state = read_json(output / "controller.json")
    assert state["status"] == ("failed" if failed_name else "completed")
    assert state["current_jobs"] == [] and state["child_pid"] is None
    assert state["jobs"][0]["adopted"] and state["jobs"][0]["started_at"] == "original-start"
    assert [job["name"] for job in state["jobs"] if job["status"] == "failed"] == (
        [failed_name] if failed_name else [])


def test_detached_resume_allows_recorded_orphan_and_passes_limit(tmp_path, monkeypatch, capsys):
    output = tmp_path / "runs"
    output.mkdir()
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {})
    previous = {"current_job": "t0_seed17", "child_pid": 101, "jobs": fake_jobs(output)}
    write_json(output / "controller.json", previous)
    monkeypatch.setattr(queue, "jobs", fake_jobs)
    monkeypatch.setattr(queue, "alive_controller", lambda output: None)
    monkeypatch.setattr(queue, "alive_training_children", lambda output, manifest, suite: [101])
    launched = []

    class Controller:
        pid = 200

        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setattr(queue.subprocess, "Popen", Controller)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--output", str(output), "--manifest", str(manifest),
                                     "--resume", "--detach", "--max-concurrent", "2"])
    assert queue.main() == 0
    assert "--resume" in launched[0]
    assert launched[0][launched[0].index("--max-concurrent") + 1] == "2"
    assert launched[0][launched[0].index("--suite") + 1] == "baseline"
    assert read_json(output / "controller.json") == previous
    assert read_json(output / "launch.json")["max_concurrent"] == 2
    capsys.readouterr()


def test_live_controller_blocks_a_second_launcher(tmp_path, monkeypatch):
    output = tmp_path / "runs"
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {})
    monkeypatch.setattr(queue, "jobs", fake_jobs)
    monkeypatch.setattr(queue, "alive_controller", lambda output: 100)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--output", str(output), "--manifest", str(manifest),
                                     "--resume", "--detach"])
    with pytest.raises(RuntimeError, match="controller 100 is already running"):
        queue.main()


def test_a1_a2_queue_has_eight_single_factor_configurations(tmp_path):
    jobs = queue.jobs(tmp_path, "a1_a2")
    assert len(jobs) == 8 and len({job["name"] for job in jobs}) == 8
    assert {(job["experiment"], job["profile"], job["seed"]) for job in jobs} == {
        (experiment, profile, seed) for experiment in ("a1", "a2")
        for profile in ("t0", "dynamic") for seed in (17, 43)}
    for job in jobs:
        path = Path(job["config"])
        assert path.read_bytes() == (ROOT / "src/mri_vla_jepa/resources/configs" / path.name).read_bytes()
        cfg = queue.load_config(path)
        assert cfg.lr_scheduler == ("plateau" if job["experiment"] == "a1" else "none")
        assert cfg.model.dropout == (.1 if job["experiment"] == "a1" else .3)


def test_loss_queue_has_sixteen_fresh_configurations_and_locked_controls(tmp_path):
    jobs = queue.jobs(tmp_path, "losses")
    assert len(jobs) == 16 and len({job["name"] for job in jobs}) == 16
    assert [job["name"] for job in jobs[:4]] == [
        "l1_t0_seed17", "l1_dynamic_seed17", "l1_t0_seed43", "l1_dynamic_seed43"]
    assert {(job["experiment"], job["profile"], job["seed"]) for job in jobs} == {
        (experiment, profile, seed) for experiment in ("l1", "l2", "l3", "l4")
        for profile in ("t0", "dynamic") for seed in (17, 43)}
    for job in jobs:
        path = Path(job["config"])
        assert path.read_bytes() == (ROOT / "src/mri_vla_jepa/resources/configs" / path.name).read_bytes()
        cfg = queue.load_config(path)
        expected_loss = queue.LOSS_SETTINGS[job["experiment"]]
        assert cfg.task_weight == 1.0 and cfg.flow_weight == 0.0 and not cfg.model.enable_flow
        assert all(getattr(cfg, key) == value for key, value in expected_loss.items())
        assert all(getattr(cfg, key) == value for key, value in queue.LOSS_DIAGNOSTICS.items())
        assert cfg.lr_scheduler == ("plateau" if job["profile"] == "t0" else "none")
        assert cfg.model.dropout == (.1 if job["profile"] == "t0" else .3)
        assert cfg.image_cache == "image_cache_20261004"
        assert cfg.prefetch_batches == 2 and cfg.loader_workers == 2
        assert cfg.pin_memory and cfg.non_blocking_transfer
        assert cfg.device == "cuda" and cfg.precision == "bf16"
        control = "a1" if job["profile"] == "t0" else "a2"
        assert Path(job["l0_reference"]) == queue.OVERFITTING_OUTPUT / (
            f"{control}_{job['profile']}_seed{job['seed']}")
        assert Path(job["output"]).parent == tmp_path
        assert job["output"] != job["l0_reference"]
    assert queue.LOSS_OUTPUT == ROOT / "runs/registered_roi32_l1_l4_20261004"


@pytest.mark.parametrize("field,value", [("reconstruction_weight", .1), ("lr", 5e-5),
                                         ("diagnostics_every_epochs", 0)])
def test_loss_queue_rejects_unplanned_changes(tmp_path, monkeypatch, field, value):
    original = queue.load_config

    def changed(path):
        cfg = original(path)
        if Path(path).name == "registered_roi32_epoch200_l2_t0_seed17.yaml":
            setattr(cfg, field, value)
        return cfg

    monkeypatch.setattr(queue, "load_config", changed)
    with pytest.raises(ValueError, match=r"locked A\*"):
        queue.jobs(tmp_path, "losses")


def test_unknown_experiment_suite_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown ROI32 experiment suite"):
        queue.jobs(tmp_path, "unknown")


def test_queue_cannot_resume_a_different_suite(tmp_path):
    write_json(tmp_path / "controller.json", {"suite": "baseline"})
    with pytest.raises(ValueError, match="different experiment suite"):
        queue.work(tmp_path, tmp_path / "manifest.json", resume=True, suite="a1_a2")


def test_completed_report_without_checkpoints_is_rejected(tmp_path):
    job = fake_jobs(tmp_path)[0]
    write_json(Path(job["output"]) / "train_report.json", {"completed": True})
    with pytest.raises(FileNotFoundError, match="best.pt"):
        queue.validate_completed_run(job, tmp_path / "manifest.json")


def test_a1_a2_detached_launcher_propagates_suite(tmp_path, monkeypatch, capsys):
    manifest, output = tmp_path / "manifest.json", tmp_path / "runs"
    write_json(manifest, {})
    launched = []

    class Controller:
        pid = 200

        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setattr(queue.subprocess, "Popen", Controller)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--suite", "a1_a2", "--output", str(output),
                                     "--manifest", str(manifest), "--detach", "--max-concurrent", "3"])
    assert queue.main() == 0
    assert launched[0][launched[0].index("--suite") + 1] == "a1_a2"
    assert launched[0][launched[0].index("--max-concurrent") + 1] == "3"
    receipt = read_json(output / "launch.json")
    assert receipt["suite"] == "a1_a2" and len(receipt["jobs"]) == 8
    capsys.readouterr()


def test_losses_detached_launcher_uses_default_output_and_three_slots(tmp_path, monkeypatch, capsys):
    manifest, output = tmp_path / "manifest.json", tmp_path / "losses"
    write_json(manifest, {})
    launched = []

    class Controller:
        pid = 200

        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setitem(queue.SUITE_OUTPUTS, "losses", output)
    monkeypatch.setattr(queue.subprocess, "Popen", Controller)
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--suite", "losses", "--manifest", str(manifest),
                                     "--detach", "--max-concurrent", "3"])
    assert queue.main() == 0
    command = launched[0]
    assert command[command.index("--suite") + 1] == "losses"
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--max-concurrent") + 1] == "3"
    assert "--resume" not in command
    receipt = read_json(output / "launch.json")
    assert receipt["suite"] == "losses" and len(receipt["jobs"]) == 16
    assert all(Path(job["output"]).parent == output for job in receipt["jobs"])
    capsys.readouterr()


@pytest.mark.parametrize("experiment,status", [("l1", "unavailable"), ("l2", "evaluated")])
def test_world_hook_independently_binds_best_and_last(tmp_path, monkeypatch, experiment, status):
    job = next(job for job in queue.jobs(tmp_path, "losses") if job["experiment"] == experiment)
    run = Path(job["output"])
    run.mkdir()
    for name in ("best", "last"):
        (run / f"{name}.pt").write_bytes(name.encode())
    launched, snapshots = [], []

    class Evaluator:
        pid = 500

        def __init__(self, command, **kwargs):
            self.command = command
            launched.append(command)

        def wait(self):
            checkpoint = Path(self.command[self.command.index("--checkpoint") + 1])
            target = Path(self.command[self.command.index("--output") + 1])
            write_json(target, {"status": status, "checkpoint_file_unchanged": True,
                       "runtime_source_matches_checkpoint": True,
                       "checkpoint_identity": {"path": str(checkpoint.resolve()),
                           "size_bytes": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns}})
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Evaluator)
    monkeypatch.setattr(queue, "process_info", lambda pid: {"command": launched[-1], "start_ticks": "1"}
                        if launched else None)
    results = queue.evaluate_completed_world(job, tmp_path / "manifest.json",
        lambda: snapshots.append(dict(job["world_evaluation"])))
    assert list(results) == ["best", "last"]
    assert [Path(command[command.index("--checkpoint") + 1]).name for command in launched] == ["best.pt", "last.pt"]
    assert all("--resume" not in command and "train-vla-jepa" not in command for command in launched)
    assert all(command[command.index("--batch-size") + 1] == "16" for command in launched)
    assert all(command[command.index("--device") + 1] == ("cpu" if experiment == "l1" else "cuda")
               for command in launched)
    assert job["world_evaluation"]["status"] == "completed"
    assert len([record for record in snapshots if record["status"] == "running" and record.get("child_pid")]) == 2
    # Resume-completed always recomputes, rather than trusting an old aggregate.
    queue.evaluate_completed_world(job, tmp_path / "manifest.json", lambda: None)
    assert len(launched) == 4


def test_world_hook_checks_process_arguments_and_start_ticks(tmp_path, monkeypatch):
    job = queue.jobs(tmp_path, "losses")[0]
    run, manifest = Path(job["output"]), tmp_path / "manifest.json"
    command = [sys.executable, str(queue.WORLD_EVALUATOR), "--checkpoint", str(run / "best.pt"),
               "--manifest", str(manifest), "--output", str(run / "world_best.json")]
    monkeypatch.setattr(queue, "process_info", lambda pid: {"command": command, "start_ticks": "10"})
    assert queue.matching_world_process(500, job, manifest, "best", "10")
    assert queue.matching_world_process(500, job, manifest, "last", "10") is None
    assert queue.matching_world_process(500, job, manifest, "best", "11") is None


@pytest.mark.parametrize("failure", [False, True])
def test_loss_queue_world_evaluation_finishes_before_refilling_gpu_slot(tmp_path, monkeypatch, failure):
    output, manifest = tmp_path / "runs", tmp_path / "manifest.json"
    output.mkdir()
    monkeypatch.setattr(queue, "jobs", fake_jobs)
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest:
                        read_json(Path(job["output"]) / "train_report.json"))
    clock, starts, evaluated = {"tick": 0}, [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.pid, self.started = 200 + len(starts), clock["tick"]
            starts.append(self.run.name)

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            write_json(self.run / "train_report.json", {"completed": True})
            return 0

    def evaluate(job, manifest, persist):
        state = read_json(output / "controller.json")
        assert len(state["current_jobs"]) <= 2
        evaluated.append((job["name"], len(starts)))
        if failure and job["name"] == "t0_seed17":
            raise RuntimeError("world evaluator failed")

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue, "matching_training_process", lambda pid, *args:
                        {"start_ticks": str(pid)} if pid is not None else None)
    monkeypatch.setattr(queue, "evaluate_completed_world", evaluate)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: clock.update(tick=clock["tick"] + 1))
    assert queue.work(output, manifest, False, max_concurrent=3, suite="losses") == int(failure)
    state = read_json(output / "controller.json")
    assert len(evaluated) == 4
    assert evaluated[0] == ("t0_seed17", 3)
    assert state["jobs"][0]["report"]["completed"]
    assert state["jobs"][0]["status"] == ("evaluation_failed" if failure else "completed")
    # A failed world evaluator retries on resume without another training launch.
    assert queue.work(output, manifest, True, max_concurrent=3, suite="losses") == int(failure)
    assert len(starts) == 4 and len(evaluated) == 8


@pytest.mark.parametrize("experiment,t0,dynamic", [
    ("l1", 4096, 7680), ("l2", 6656, 9216), ("l3", 4608, 7680),
    ("l4", 6656, 9216), ("a1", 6656, 9216), ("unknown", 6656, 9216)])
def test_memory_reservations_follow_measured_profile_peaks(experiment, t0, dynamic):
    assert queue.training_memory_mib({"experiment": experiment, "profile": "t0"}) == t0
    assert queue.training_memory_mib({"experiment": experiment, "profile": "dynamic"}) == dynamic
    assert queue.world_memory_mib({"world_device": "cpu"}) == 0
    assert queue.world_memory_mib({"world_device": "cuda"}) == 10240


def test_memory_budget_rejects_unrunnable_worker_and_excess_hardware_capacity(tmp_path, monkeypatch):
    planned = queue.jobs(tmp_path, "losses")
    with pytest.raises(ValueError, match="at least 10240 MiB"):
        queue.validate_memory_budget(9216, planned, "losses")
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs:
                        type("Capacity", (), {"stdout": "32607\n"})())
    queue.validate_memory_budget(29696, planned, "losses", hardware=True)
    with pytest.raises(ValueError, match="exceeds visible device capacity"):
        queue.validate_memory_budget(32768, planned, "losses", hardware=True)


def test_memory_aware_queue_prefers_lighter_jobs_and_defers_large_candidate(tmp_path, monkeypatch):
    output, manifest = tmp_path / "runs", tmp_path / "manifest.json"
    output.mkdir()
    planned = fake_jobs(output)
    planned = [planned[1], planned[3], planned[0], planned[2]]
    for job in planned:
        job["experiment"] = "l1"
    monkeypatch.setattr(queue, "jobs", lambda output, suite: planned)
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest:
                        read_json(Path(job["output"]) / "train_report.json"))
    clock, starts, snapshots = {"tick": 0}, [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.pid, self.started = 200 + len(starts), clock["tick"]
            starts.append((self.run.name, self.started))

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            write_json(self.run / "train_report.json", {"completed": True})
            return 0

    def advance(seconds):
        state = read_json(output / "controller.json")
        snapshots.append(state)
        assert state["gpu_memory_reserved_mib"] <= 16384
        assert len(state["current_workers"]) <= 6
        clock["tick"] += 1

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue, "matching_training_process", lambda pid, *args:
                        {"start_ticks": str(pid)} if pid is not None else None)
    monkeypatch.setattr(queue.time, "sleep", advance)
    assert queue.work(output, manifest, False, max_concurrent=6,
                      gpu_memory_budget_mib=16384) == 0
    assert starts == [("t0_seed17", 0), ("t0_seed43", 0),
                      ("dynamic_seed17", 0), ("dynamic_seed43", 2)]
    assert snapshots[0]["jobs"][1]["deferred_reason"] == "gpu_memory_budget"
    assert snapshots[0]["gpu_memory_reserved_mib"] == 15872


def test_light_priority_fills_two_slots_instead_of_one_larger_fitting_job(tmp_path, monkeypatch):
    output, manifest = tmp_path / "runs", tmp_path / "manifest.json"
    output.mkdir()
    names = ["l1_dynamic_seed17", "l1_dynamic_seed43", "l2_dynamic_seed17",
             "l2_t0_seed17", "l2_t0_seed43"]
    available = {job["name"]: job for job in queue.jobs(output, "losses")}
    planned = [available[name] for name in names]
    old = [dict(job) for job in planned]
    for index, job in enumerate(old[:2]):
        job.update(status="running", child_pid=100 + index, process_start_ticks=str(100 + index))
    write_json(output / "controller.json", {"manifest": str(manifest), "jobs": old})
    monkeypatch.setattr(queue, "jobs", lambda output, suite: [dict(job) for job in planned])
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest:
                        read_json(Path(job["output"]) / "train_report.json"))
    clock, starts, snapshots = {"tick": 0}, [], []

    def match(pid, job, manifest, start_ticks=None):
        if pid is None:
            return None
        if pid in {100, 101} and clock["tick"] >= 3:
            write_json(Path(job["output"]) / "train_report.json", {"completed": True})
            return None
        return {"start_ticks": str(pid)}

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.pid, self.started = 200 + len(starts), clock["tick"]
            starts.append((self.run.name, self.started))

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            write_json(self.run / "train_report.json", {"completed": True})
            return 0

    def advance(seconds):
        snapshots.append(read_json(output / "controller.json"))
        clock["tick"] += 1

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue, "matching_training_process", match)
    monkeypatch.setattr(queue.time, "sleep", advance)
    assert queue.work(output, manifest, True, max_concurrent=6, gpu_memory_budget_mib=29696) == 0
    assert starts[:2] == [("l2_t0_seed17", 0), ("l2_t0_seed43", 0)]
    assert starts[2][0] == "l2_dynamic_seed17" and starts[2][1] >= 2
    assert len(snapshots[0]["current_workers"]) == 4
    assert snapshots[0]["gpu_memory_reserved_mib"] == 28672
    assert snapshots[0]["jobs"][2]["deferred_reason"] == "gpu_memory_budget"


def test_adopted_cuda_world_worker_is_counted_before_memory_admission(tmp_path, monkeypatch):
    output, manifest = tmp_path / "runs", tmp_path / "manifest.json"
    output.mkdir()
    planned = fake_jobs(output)
    for job in planned:
        job.update(experiment="l1", world_device="cpu")
    planned[3].update(experiment="l2", world_device="cuda")
    # Put the large queued dynamic job before the lighter queued T0 job.
    planned = [planned[0], planned[1], planned[3], planned[2]]
    old = [dict(job) for job in planned]
    write_json(Path(old[0]["output"]) / "train_report.json", {"completed": True})
    old[0].update(status="evaluating", world_evaluation={
        "status": "running", "child_pid": 501, "checkpoint": "best", "process_start_ticks": "501"})
    old[1].update(status="running", child_pid=102, process_start_ticks="102")
    write_json(output / "controller.json", {"suite": "losses", "manifest": str(manifest), "jobs": old})
    monkeypatch.setattr(queue, "jobs", lambda output, suite: [dict(job) for job in planned])
    monkeypatch.setattr(queue, "validate_completed_run", lambda job, manifest:
                        read_json(Path(job["output"]) / "train_report.json"))
    clock, starts, evaluated, snapshots = {"tick": 0}, [], [], []

    def match_train(pid, job, manifest, start_ticks=None):
        if pid is None:
            return None
        if pid == 102 and clock["tick"] >= 3:
            write_json(Path(job["output"]) / "train_report.json", {"completed": True})
            return None
        return {"start_ticks": str(pid)}

    def match_world(pid, job, manifest, checkpoint, start_ticks=None):
        return {"start_ticks": "501", "command": ["--device", "cuda"]} if (
            pid == 501 and clock["tick"] < 2) else None

    class Child:
        def __init__(self, command, **kwargs):
            self.run = Path(command[command.index("--output") + 1])
            self.pid, self.started = 200 + len(starts), clock["tick"]
            starts.append((self.run.name, self.started))

        def poll(self):
            if clock["tick"] - self.started < 2:
                return None
            write_json(self.run / "train_report.json", {"completed": True})
            return 0

    def evaluate(job, manifest, persist):
        assert job["name"] != "t0_seed17" or clock["tick"] >= 2
        state = read_json(output / "controller.json")
        assert len(state["current_workers"]) <= 4
        assert state["gpu_memory_reserved_mib"] <= 24576
        evaluated.append(job["name"])

    def advance(seconds):
        state = read_json(output / "controller.json")
        snapshots.append(state)
        assert state["gpu_memory_reserved_mib"] <= 24576
        clock["tick"] += 1

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue, "matching_training_process", match_train)
    monkeypatch.setattr(queue, "matching_world_process", match_world)
    monkeypatch.setattr(queue, "evaluate_completed_world", evaluate)
    monkeypatch.setattr(queue.time, "sleep", advance)
    assert queue.work(output, manifest, True, max_concurrent=4, suite="losses",
                      gpu_memory_budget_mib=24576) == 0
    assert starts[0] == ("t0_seed43", 0)
    assert ("dynamic_seed43", 0) not in starts
    assert len(evaluated) == 4
    first = snapshots[0]
    assert first["current_world_jobs"] == ["t0_seed17"]
    assert first["gpu_memory_reserved_mib"] == 22016
    assert first["jobs"][2]["deferred_reason"] == "gpu_memory_budget"
    assert first["jobs"][0]["status"] == "evaluating_adopted"
    assert any(worker.get("adopted") and worker["memory_reservation_mib"] == 10240
               for worker in first["current_workers"])


def test_detached_launcher_propagates_memory_budget_and_six_worker_ceiling(tmp_path, monkeypatch, capsys):
    manifest, output = tmp_path / "manifest.json", tmp_path / "runs"
    write_json(manifest, {})
    launched = []

    class Controller:
        pid = 200

        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setattr(queue.subprocess, "Popen", Controller)
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs:
                        type("Capacity", (), {"stdout": "32607\n"})())
    monkeypatch.setattr(sys, "argv", [str(spec.origin), "--suite", "losses", "--output", str(output),
        "--manifest", str(manifest), "--detach", "--max-concurrent", "6", "--gpu-memory-budget-mib", "29696"])
    assert queue.main() == 0
    command = launched[0]
    assert command[command.index("--max-concurrent") + 1] == "6"
    assert command[command.index("--gpu-memory-budget-mib") + 1] == "29696"
    assert read_json(output / "launch.json")["gpu_memory_budget_mib"] == 29696
    capsys.readouterr()
