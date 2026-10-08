"""Replay saved pCR predictions and compare completed ROI32 experiments on CPU."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mri_vla_jepa.arm import normalize_patient_key
from mri_vla_jepa.metrics import classification_metrics
from mri_vla_jepa.training import _scores


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("auroc", "auprc", "nll", "brier")
EXPERIMENTS = ("L0", "L1", "L2", "L3", "L4")
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#6D6E71")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_metrics(value):
    return {name: value.get(name) for name in ("n", "positives", *METRICS)}


def patient_nll(rows):
    values = {}
    for row in rows:
        key = normalize_patient_key(row["patient_key"])
        p = np.clip(row["pcr_probability"], 1e-12, 1 - 1e-12)
        y = row["pcr"]
        values.setdefault(key, []).append(float(-(y * np.log(p) + (1 - y) * np.log1p(-p))))
    return {key: float(np.mean(loss)) for key, loss in values.items()}


def collect_runs(manifest):
    val = {normalize_patient_key(p["patient_key"]): p for p in manifest["patients"] if p["split"] == "val"}
    stages = {key: {v["available_at"] for v in p["visits"] if v is not None} for key, p in val.items()}
    complete = {key for key, values in stages.items() if values == {0, 1, 2, 3}}
    assert len(val) == 102 and len(complete) == 84
    records, predictions, vectors, errors = [], {}, {}, []
    for experiment in ("A0", "A1", "A2", "L1", "L2", "L3", "L4"):
        for profile in ("t0", "dynamic"):
            for seed in (17, 43):
                if experiment == "A0":
                    run = ROOT / f"runs/registered_roi32_epoch200_20261003/{profile}_seed{seed}"
                elif experiment.startswith("A"):
                    run = ROOT / f"runs/registered_roi32_a1_a2_20261004/{experiment.lower()}_{profile}_seed{seed}"
                else:
                    run = ROOT / f"runs/registered_roi32_l1_l4_20261004/{experiment.lower()}_{profile}_seed{seed}"
                saved, history, report, cfg = (read(run / name) for name in
                    ("best_validation.json", "epoch_history.json", "train_report.json", "config.json"))
                best = min(history, key=lambda row: row["validation"]["selection_score"])
                assert report["completed"] and report["bad_epochs"] == 50
                assert report["completed_epochs"] == len(history) == best["epoch"] + 50
                assert saved["selection_score"] == best["validation"]["selection_score"] == report["best_validation_nll"]
                rows = saved["rows"]
                index = {(normalize_patient_key(row["patient_key"]), row["landmark"]): row for row in rows}
                expected = {(key, stage) for key in val for stage in ([0] if profile == "t0" else stages[key])}
                assert len(index) == len(rows) and set(index) == expected
                assert all(row["pcr"] == val[key]["target"]["pcr"]
                           and row["arm_semantics"] == "assigned_arm_scenario" for (key, _), row in index.items())
                replay = _scores(rows, "t0" if profile == "t0" else "all_observed")
                error = abs(replay["selection_score"] - saved["selection_score"])
                for stage, score in saved["per_landmark"].items():
                    for metric in METRICS:
                        if score[metric] is not None:
                            error = max(error, abs(score[metric] - replay["per_landmark"][stage][metric]))
                assert error <= 1e-12
                errors.append(error)
                name = f"{experiment}_{profile}_seed{seed}"
                predictions[name], vectors[name] = index, patient_nll(rows)
                common_rows = [row for (key, _), row in index.items() if key in complete]
                common = {f"T{stage}": compact_metrics(classification_metrics(
                    [r["pcr"] for r in common_rows if r["landmark"] == stage],
                    [r["pcr_probability"] for r in common_rows if r["landmark"] == stage]))
                    for stage in ([0] if profile == "t0" else range(4))}
                record = {"name": name, "experiment": experiment, "profile": profile, "seed": seed,
                    "run_path": str(run.relative_to(ROOT)), "best_epoch": best["epoch"], "stop_epoch": len(history),
                    "selection_metric": saved["selection_metric"], "best_selection_nll": saved["selection_score"],
                    "last_selection_nll": history[-1]["validation"]["selection_score"],
                    "best_train_bce": best["training"]["task"], "last_train_bce": history[-1]["training"]["task"],
                    "per_landmark": {key: compact_metrics(value) for key, value in saved["per_landmark"].items()},
                    "complete_four_visit_metrics": common,
                    "complete_four_visit_selection_nll": float(np.mean(list(patient_nll(common_rows).values()))),
                    "replay_max_metric_error": error, "history": history,
                    "loss_weights": {key: cfg[key] for key in
                                    ("task_weight", "jepa_weight", "reconstruction_weight", "variance_weight")}}
                records.append(record)
    locked = []
    for experiment in EXPERIMENTS:
        for profile in ("t0", "dynamic"):
            for seed in (17, 43):
                actual = ("A1" if profile == "t0" else "A2") if experiment == "L0" else experiment
                original = next(r for r in records if (r["experiment"], r["profile"], r["seed"]) == (actual, profile, seed))
                item = copy.deepcopy(original)
                item.update(experiment=experiment, name=f"{experiment}_{profile}_seed{seed}", source_experiment=actual)
                locked.append(item)
                predictions[item["name"]], vectors[item["name"]] = predictions[original["name"]], vectors[original["name"]]
    audit = {"unique_runs": len(records), "locked_comparison_runs": len(locked),
        "replayed_stage_metric_rows": sum(1 if r["profile"] == "t0" else 4 for r in records),
        "max_metric_replay_error": max(errors), "validation_patients": len(val),
        "validation_positives": sum(p["target"]["pcr"] for p in val.values()),
        "stage_patients": {f"T{s}": sum(s in v for v in stages.values()) for s in range(4)},
        "complete_four_visit_patients": len(complete),
        "complete_four_visit_positives": sum(val[key]["target"]["pcr"] for key in complete),
        "patient_stage_labels_match_manifest": True, "duplicate_prediction_rows": 0}
    return records, locked, vectors, audit


def summarize(records):
    result = []
    for experiment in dict.fromkeys(r["experiment"] for r in records):
        for profile in ("t0", "dynamic"):
            subset = [r for r in records if r["experiment"] == experiment and r["profile"] == profile]
            row = {"experiment": experiment, "profile": profile, "seeds": [17, 43]}
            for key in ("best_selection_nll", "last_selection_nll", "best_epoch", "stop_epoch", "best_train_bce", "last_train_bce"):
                row[key] = float(np.mean([r[key] for r in subset]))
            row["per_landmark"] = {stage: {metric: float(np.mean([r["per_landmark"][stage][metric] for r in subset]))
                for metric in METRICS} for stage in (["T0"] if profile == "t0" else [f"T{s}" for s in range(4)])}
            row["complete_four_visit_metrics"] = {stage: {metric: float(np.mean([
                r["complete_four_visit_metrics"][stage][metric] for r in subset])) for metric in METRICS}
                for stage in row["per_landmark"]}
            result.append(row)
    return result


def paired_bootstrap(locked, vectors, repetitions):
    keys = sorted(vectors[locked[0]["name"]])
    draws = np.random.default_rng(20261004).integers(0, len(keys), size=(repetitions, len(keys)))
    result = []
    for experiment in EXPERIMENTS[1:]:
        for profile in ("t0", "dynamic"):
            differences = []
            for seed in (17, 43):
                candidate = vectors[f"{experiment}_{profile}_seed{seed}"]
                reference = vectors[f"L0_{profile}_seed{seed}"]
                assert set(candidate) == set(reference) == set(keys)
                differences.append(np.asarray([candidate[key] - reference[key] for key in keys]))
            for label, delta in (("17", differences[0]), ("43", differences[1]),
                                  ("mean_of_two_seeds", np.mean(differences, axis=0))):
                estimates = delta[draws].mean(axis=1)
                lower, upper = np.quantile(estimates, [.025, .975])
                result.append({"experiment": experiment, "profile": profile, "seed_scope": label,
                    "delta_nll_vs_l0": float(delta.mean()), "lower_95": float(lower), "upper_95": float(upper),
                    "patients": len(keys), "repetitions": repetitions})
    return result


def stage_rows(records, common=False):
    rows = []
    for record in records:
        values = record["complete_four_visit_metrics"] if common else record["per_landmark"]
        for stage, score in values.items():
            if score.get("n", 84) == 0:
                continue
            rows.append({"experiment": record["experiment"], "profile": record["profile"], "seed": record["seed"],
                "stage": stage, **score, "best_epoch": record["best_epoch"], "stop_epoch": record["stop_epoch"]})
    return rows


def figures(locked, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, profile in enumerate(("t0", "dynamic")):
        for column, metric in enumerate(("best_selection_nll", "last_selection_nll")):
            ax = axes[row, column]
            for x, (experiment, color) in enumerate(zip(EXPERIMENTS, COLORS)):
                subset = [r for r in locked if r["experiment"] == experiment and r["profile"] == profile]
                values = [r[metric] for r in subset]
                for offset, value, marker in zip((-.09, .09), values, ("o", "D")):
                    ax.scatter([x + offset], [value], c=color, marker=marker, s=35, zorder=3)
                ax.plot([x - .22, x + .22], [np.mean(values)] * 2, color=color, lw=2)
            ax.set_xticks(range(5), EXPERIMENTS)
            ax.set_title(f"{'Fixed T0' if profile == 't0' else 'Dynamic'} / {'best' if column == 0 else 'last'}")
            ax.set_ylabel("Validation selection NLL (lower is better)")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("pCR NLL: circle = seed 17; diamond = seed 43; line = arithmetic mean")
    save_figure(fig, output, "selection_nll")

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
    for row, profile in enumerate(("t0", "dynamic")):
        for column, experiment in enumerate(EXPERIMENTS):
            ax = axes[row, column]
            for seed, color in ((17, COLORS[0]), (43, COLORS[2])):
                record = next(r for r in locked if (r["experiment"], r["profile"], r["seed"]) == (experiment, profile, seed))
                epochs = record["history"]
                x = [e["epoch"] for e in epochs]
                ax.plot(x, [e["training"]["task"] for e in epochs], color=color, ls="--", lw=1,
                        label=f"Seed {seed} train BCE")
                ax.plot(x, [e["validation"]["selection_score"] for e in epochs], color=color, lw=1.3,
                        label=f"Seed {seed} validation NLL")
                ax.scatter([record["best_epoch"]], [record["best_selection_nll"]], color=color, s=18, zorder=3)
            ax.set_title(f"{experiment} / {'T0' if profile == 't0' else 'dynamic'}")
            ax.set_ylim(0, .8 if profile == "t0" else 1.85)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.17)
            if column == 0:
                ax.set_ylabel("Train BCE / validation NLL")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("All locked L0-L4 histories; dots mark the NLL-selected best checkpoints")
    save_figure(fig, output, "pcr_training_curves")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, constrained_layout=True)
    for ax, seed in zip(axes, (17, 43)):
        for experiment, color in zip(EXPERIMENTS, COLORS):
            record = next(r for r in locked if (r["experiment"], r["profile"], r["seed"]) == (experiment, "dynamic", seed))
            ax.plot(range(4), [record["complete_four_visit_metrics"][f"T{s}"]["auroc"] for s in range(4)],
                    color=color, marker="o", label=experiment)
        ax.set_xticks(range(4), [f"T{s}" for s in range(4)])
        ax.set_title(f"Seed {seed}")
        ax.set_ylabel("AUROC on the same 84 patients")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle("Best dynamic checkpoints / complete-four-visit cohort (28 pCR positives)")
    save_figure(fig, output, "common84_stage_auroc")


def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=170)
    plt.close(fig)


def world_figure(world, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, constrained_layout=True)
    for row, profile in enumerate(("t0", "dynamic")):
        for column, checkpoint in enumerate(("best", "last")):
            ax = axes[row, column]
            for x, experiment in enumerate(("L0", "L2", "L4")):
                color = COLORS[EXPERIMENTS.index(experiment)]
                records = [r for r in world["records"] if (r["loss"], r["profile"], r["checkpoint"])
                           == (experiment, profile, checkpoint)]
                records.sort(key=lambda r: r["seed"])
                values = [r["metrics"]["horizon_all"]["autonomous_relative_to_copy"] for r in records]
                assert len(values) == 2
                for offset, value, marker in zip((-.09, .09), values, ("o", "D")):
                    ax.scatter([x + offset], [value], c=color, marker=marker, s=35, zorder=3)
                ax.plot([x - .22, x + .22], [np.mean(values)] * 2, color=color, lw=2)
            ax.axhline(1, color="#555555", ls="--", label="Copy baseline")
            ax.set_xticks(range(3), ("L0", "L2", "L4"))
            ax.set_title(f"{'Fixed T0' if profile == 't0' else 'Dynamic'} / pCR-{checkpoint}")
            ax.set_ylabel("Autonomous all-horizon L1 / landmark-copy L1")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("World relative to copy: circle = seed 17; diamond = seed 43; ratio < 1 beats copy")
    save_figure(fig, output, "world_relative_to_copy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/l_loss_analysis_20261004")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1000 exploratory bootstrap replicates")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, locked, vectors, audit = collect_runs(read(ROOT / "data/registered_roi32/manifest.json"))
    comparisons = paired_bootstrap(locked, vectors, args.bootstrap)
    public = lambda rows: [{key: value for key, value in row.items() if key != "history"} for row in rows]
    result = {"scope": "saved predictions/histories only; no new inference or training", "audit": audit,
        "selection": "Fixed T0 uses T0 NLL; dynamic averages legal prefixes within patient then patients",
        "mean_semantics": "Arithmetic mean of the two seed metrics, not prediction ensembling",
        "bootstrap_scope": "Exploratory paired patient resampling conditional on already selected models; "
                           "seed variability, checkpoint/config selection and multiple comparisons are not accounted for; not independent test CIs",
        "historical_runs": public(records), "locked_l_runs": public(locked),
        "historical_means": summarize(records), "locked_l_means": summarize(locked),
        "paired_nll_bootstrap": comparisons}
    (output / "pcr_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "pcr_per_seed.csv", stage_rows(records))
    write_csv(output / "pcr_locked_per_seed.csv", stage_rows(locked))
    write_csv(output / "pcr_common84_per_seed.csv", stage_rows(locked, common=True))
    write_csv(output / "pcr_paired_nll_bootstrap.csv", comparisons)
    run_rows = [{key: row[key] for key in ("experiment", "profile", "seed", "best_epoch", "stop_epoch",
                "best_selection_nll", "last_selection_nll", "best_train_bce", "last_train_bce")} for row in locked]
    write_csv(output / "pcr_run_summary.csv", run_rows)
    figures(locked, output)
    if (output / "world_analysis.json").is_file():
        world_figure(read(output / "world_analysis.json"), output)
    print(json.dumps({"audit": audit, "means": [{"experiment": r["experiment"], "profile": r["profile"],
        "best_nll": r["best_selection_nll"], "last_nll": r["last_selection_nll"]} for r in result["locked_l_means"]],
        "paired_mean_seed_nll": [r for r in comparisons if r["seed_scope"] == "mean_of_two_seeds"]}, indent=2))


if __name__ == "__main__":
    main()
