"""Aggregate saved training histories and fixed probes without model inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/l_loss_analysis_20261004"
TERMS = ("task", "jepa", "reconstruction", "variance", "flow")


def read(path):
    return json.loads(path.read_text())


def stats(values):
    values = [x for x in values if x is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None}


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return not isinstance(value, (float, int)) or math.isfinite(value)


def epoch_summary(row, weights):
    training = row["training"]
    weighted = {term: training[term] * weights[term] for term in TERMS}
    return {"epoch": row["epoch"], "lr": row["lr"],
            "training_raw_terms": {key: training[key] for key in TERMS},
            "training_weighted_terms": weighted,
            "training_total_loss": training["loss"],
            "weighted_sum_minus_total": sum(weighted.values()) - training["loss"],
            "patient_weighted_batch_preclip_gradient_norm": training["gradient_norm"],
            "gradient_clip_fraction": training.get("gradient_clip_fraction"),
            "sampled_optimizer_update_norm": training.get("sampled_optimizer_update_norm"),
            "validation_selection_nll": row["validation"]["selection_score"],
            "validation_per_landmark": {key: {k: metrics[k] for k in ("n", "auroc", "auprc", "nll", "brier")}
                                        for key, metrics in row["validation"]["per_landmark"].items()}}


def representation_summary(rep):
    return {"rms": rep["rms"], "global_std": rep["global_std"],
            "patient_axis_variance_penalty": rep["patient_axis_variance_penalty"],
            "legacy_variance_penalty": rep["legacy_variance_penalty"],
            "anchor_patient_pooled_rms_std": math.sqrt(rep["anchor_patient_pooled_variance"]),
            "anchor_patient_pooled_effective_rank": rep["anchor_patient_pooled_effective_rank"],
            "per_stage": {stage: {"patients": row["patients"],
                           "patient_axis_rms_std": (math.sqrt(row["patient_axis_variance_mean"])
                                                    if row["patient_axis_variance_mean"] is not None else None),
                           "patient_pooled_effective_rank": row["patient_pooled_effective_rank"]}
                          for stage, row in rep["per_stage"].items()}}


def probe_summary(probe):
    batches = []
    for index, batch in enumerate(probe["gradient_batches"]):
        norms = batch["weighted_gradient_norms"]
        shared_task = norms["task"]["shared_encoder_fusion"]
        encoder_task = norms["task"]["encoder"]
        batches.append({"batch": index, "patients": batch["patients"],
                        "weighted_shared_norm": {term: norms[term]["shared_encoder_fusion"] for term in TERMS},
                        "weighted_all_trainable_norm": {term: norms[term]["all_trainable"] for term in TERMS},
                        "shared_auxiliary_to_task_norm_ratios": {
                            term: norms[term]["shared_encoder_fusion"] / shared_task if shared_task else None
                            for term in TERMS if term != "task"},
                        "encoder_auxiliary_to_task_norm_ratios": {
                            term: norms[term]["encoder"] / encoder_task if encoder_task else None
                            for term in TERMS if term != "task"},
                        "shared_task_auxiliary_cosines": {term: batch["task_auxiliary_cosines"][term]["shared_encoder_fusion"]
                                                         for term in TERMS if term != "task"},
                        "encoder_task_auxiliary_cosines": {term: batch["task_auxiliary_cosines"][term]["encoder"]
                                                          for term in TERMS if term != "task"}})
    return {"epoch": probe["epoch"], "scope": probe["scope"], "patients": probe["patients"],
            "landmark_counts": probe["landmark_counts"], "eval_pcr_nll": probe["eval_pcr_nll"],
            "teacher_drift": probe["teacher_drift"],
            "observed_representation": representation_summary(probe["observed_representation"]),
            "teacher_representation": representation_summary(probe["teacher_representation"]),
            "gradient_batches": batches,
            "per_batch_statistics": {
                "shared_auxiliary_to_task_norm_ratios": {term: stats([b["shared_auxiliary_to_task_norm_ratios"][term] for b in batches])
                                                        for term in TERMS if term != "task"},
                "encoder_auxiliary_to_task_norm_ratios": {term: stats([b["encoder_auxiliary_to_task_norm_ratios"][term] for b in batches])
                                                        for term in TERMS if term != "task"},
                "shared_task_auxiliary_cosines": {term: stats([b["shared_task_auxiliary_cosines"][term] for b in batches])
                                                  for term in TERMS if term != "task"}}}


def run_summary(name, path, condition, profile, seed):
    config = read(path / "config.json")
    epochs = read(path / "epoch_history.json")
    history = read(path / "history.json")
    report = read(path / "train_report.json")
    weights = {term: config[term + "_weight"] for term in TERMS}
    best = min(epochs, key=lambda row: row["validation"]["selection_score"])
    last = epochs[-1]
    probes = read(path / "diagnostics.json") if (path / "diagnostics.json").exists() else []
    selected = {"first": probes[0], "best_nearest": min(probes, key=lambda row: abs(row["epoch"] - best["epoch"])),
                "late": probes[-1]} if probes else {}
    steps_norm = [row["gradient_norm"] for row in history]
    after_best = [row for row in epochs if row["epoch"] > best["epoch"]]
    result = {"name": name, "condition": condition, "profile": profile, "seed": seed,
              "path": str(path.relative_to(ROOT)), "weights": weights,
              "dropout": config["model"]["dropout"], "lr_scheduler": config["lr_scheduler"],
              "variance_definition": config.get("variance_definition", "legacy"),
              "stop_reason": report["stop_reason"], "completed_epochs": report["completed_epochs"],
              "completed_steps": report["completed_steps"],
              "best": epoch_summary(best, weights), "last": epoch_summary(last, weights),
              "best_to_last": {"train_bce_change": last["training"]["task"] - best["training"]["task"],
                               "train_jepa_raw_change": last["training"]["jepa"] - best["training"]["jepa"],
                               "train_jepa_weighted_change": weights["jepa"] * (last["training"]["jepa"] - best["training"]["jepa"]),
                               "validation_selection_nll_change": last["validation"]["selection_score"] - best["validation"]["selection_score"],
                               "validation_per_landmark_changes": {
                                   stage: {key: (last["validation"]["per_landmark"][stage][key] - scores[key]
                                                 if scores[key] is not None and last["validation"]["per_landmark"][stage][key] is not None else None)
                                           for key in ("auroc", "auprc", "nll", "brier")}
                                   for stage, scores in best["validation"]["per_landmark"].items()}},
              "preclip_step_gradient_norm": stats(steps_norm),
              "overall_clipped_step_fraction": sum(x > config["grad_clip"] for x in steps_norm) / len(steps_norm),
              "after_best_clipped_step_fraction": (sum(row["gradient_norm"] > config["grad_clip"] for row in history
                                                      if row.get("epoch", 0) > best["epoch"])
                                                  / sum(row.get("epoch", 0) > best["epoch"] for row in history)),
              "after_best_epoch_gradient_norm": stats([row["training"]["gradient_norm"] for row in after_best]),
              "first_lr_reduction_epoch": next((row["epoch"] for row in epochs if row["lr_reduced"]), None),
              "final_lr": report["final_lr"],
              "periodic_probe_count": len(probes), "selected_probes": {key: probe_summary(probe) for key, probe in selected.items()},
              "all_epoch_history_and_probes_finite": finite(epochs) and finite(history) and finite(probes),
              "teacher_frozen_all_probes": all(probe["teacher_frozen"] for probe in probes) if probes else None,
              "rng_and_modes_preserved_all_probes": all(probe["rng_and_module_modes_preserved"] for probe in probes) if probes else None,
              "weighted_sum_absolute_error_max": max(abs(sum(weights[k] * row["training"][k] for k in TERMS) - row["training"]["loss"]) for row in epochs)}
    if probes:
        updates = [{"epoch": row["epoch"], "lr": row["lr"],
                    "first_batch_optimizer_update_norm": row["training"]["sampled_optimizer_update_norm"]}
                   for row in epochs if "sampled_optimizer_update_norm" in row["training"]]
        result["sampled_optimizer_updates"] = {
            "scope": "first batch of each diagnostic epoch, after clipping and AdamW; not an epoch mean",
            "samples": updates,
            "norm_statistics": stats([row["first_batch_optimizer_update_norm"] for row in updates])}
        result["probe_across_time"] = {
            "epoch_range": [probes[0]["epoch"], probes[-1]["epoch"]],
            "teacher_rms": stats([p["teacher_representation"]["rms"] for p in probes]),
            "teacher_interval_drift_l1": stats([p["teacher_drift"]["l1"] for p in probes]),
            "teacher_interval_relative_drift": stats([p["teacher_drift"]["relative_to_previous_rms"] for p in probes]),
            "observed_anchor_effective_rank": stats([p["observed_representation"]["anchor_patient_pooled_effective_rank"] for p in probes]),
            "gradient_metrics": {term: {
                "shared_ratio": stats([b["weighted_gradient_norms"][term]["shared_encoder_fusion"] / b["weighted_gradient_norms"]["task"]["shared_encoder_fusion"]
                                       for p in probes for b in p["gradient_batches"] if b["weighted_gradient_norms"]["task"]["shared_encoder_fusion"]]),
                "shared_cosine": stats([b["task_auxiliary_cosines"][term]["shared_encoder_fusion"] for p in probes for b in p["gradient_batches"]]),
                "negative_cosine_fraction": (sum(b["task_auxiliary_cosines"][term]["shared_encoder_fusion"] < 0 for p in probes for b in p["gradient_batches"]
                                                if b["task_auxiliary_cosines"][term]["shared_encoder_fusion"] is not None)
                                             / sum(b["task_auxiliary_cosines"][term]["shared_encoder_fusion"] is not None for p in probes for b in p["gradient_batches"]))
                                             if any(b["task_auxiliary_cosines"][term]["shared_encoder_fusion"] is not None for p in probes for b in p["gradient_batches"]) else None}
                                     for term in ("jepa", "reconstruction", "variance")}}
    return result


def fmt(x, digits=4):
    return "NA" if x is None else f"{x:.{digits}f}"


def make_markdown(result):
    lines = ["# L0-L4 训练稳定性、过拟合与加权梯度分析", "",
             "仅重读已保存 JSON；没有训练、推理或权重改动。102 名验证患者用于开发选模，非独立测试。", "",
             "## 统计口径", "",
             "- L0 固定 T0 使用已锁定 A1；动态使用已锁定 A2。L1 pCR-only，L2 关闭重构，L3 关闭 JEPA，L4 采用患者轴 FP32 方差。",
             "- 不比较不同损失定义的 raw total loss。表中展示 pCR BCE 与实际乘权重后的各项。",
             "- epoch 的 gradient_norm 是患者数加权的批次预裁剪 L2 norm；整体 clipping 比例逐更新统计，阈值为 1。",
             "- probe 每 5 epoch 固定 32 名训练患者、两批 B16、eval 模式；nearest 是距 best epoch 最近的 probe，不能标作 best.pt 精确诊断。",
             "- probe 梯度来自每个已乘权重的目标，表中为两批各自比值/余弦的统计，不是 32 患者整体梯度。",
             "- T0 probe 32 个 T0；动态 probe T0/T1/T2/T3=10/10/8/4。动态 observed 各阶段患者数32/21/12/4，T3有效秩样本尤其有限。",
             "- patient-axis RMS std=sqrt(mean variance)，不是逐维 std 的算术均值；effective rank 使用32名患者池化向量的谱熵，不等同于特征矩阵秩。",
             "- L0 没有新 probe 数据，不能据此直接判断方差定义是否优于 L0。", "",
             "## 每个配置和种子的 best 到 last", "",
             "|配置|版本|种子|best/stop epoch|训练 BCE best→last|验证 NLL best→last|T0 AUROC best→last|T0 Brier best→last|预裁剪 norm best→last|整体裁剪比例|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        b, l = row["best"], row["last"]
        lines.append(f"|{row['condition']}|{row['profile']}|{row['seed']}|{b['epoch']}/{l['epoch']}|{fmt(b['training_raw_terms']['task'])}→{fmt(l['training_raw_terms']['task'])}|{fmt(b['validation_selection_nll'])}→{fmt(l['validation_selection_nll'])}|{fmt(b['validation_per_landmark']['T0']['auroc'])}→{fmt(l['validation_per_landmark']['T0']['auroc'])}|{fmt(b['validation_per_landmark']['T0']['brier'])}→{fmt(l['validation_per_landmark']['T0']['brier'])}|{fmt(b['patient_weighted_batch_preclip_gradient_norm'],2)}→{fmt(l['patient_weighted_batch_preclip_gradient_norm'],2)}|{fmt(row['overall_clipped_step_fraction']*100,1)}%|")
    lines += ["", "## 各项实际加权损失 best 到 last", "", "|配置|版本|种子|BCE|0.5×JEPA|0.1×重构|0.01×方差|", "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        b, l = row["best"]["training_weighted_terms"], row["last"]["training_weighted_terms"]
        lines.append(f"|{row['condition']}|{row['profile']}|{row['seed']}|" + "|".join(f"{fmt(b[k])}→{fmt(l[k])}" for k in ("task", "jepa", "reconstruction", "variance")) + "|")
    lines += ["", "## 固定训练 probe：首个、最接近 best 和末期", "", "|配置|版本|种子|probe epochs first/near/late|eval训练 NLL first→near→late|观察患者池化 RMS std first→near→late|观察患者池化有效秩 first→near→late|教师RMS first→near→late|末次5 epoch教师漂移L1|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        if not row["selected_probes"]:
            continue
        f, b, l = [row["selected_probes"][key] for key in ("first", "best_nearest", "late")]
        triples = ["→".join(fmt(p["eval_pcr_nll"]) for p in (f, b, l)),
                   "→".join(fmt(p["observed_representation"]["anchor_patient_pooled_rms_std"]) for p in (f, b, l)),
                   "→".join(fmt(p["observed_representation"]["anchor_patient_pooled_effective_rank"], 2) for p in (f, b, l)),
                   "→".join(fmt(p["teacher_representation"]["rms"]) for p in (f, b, l))]
        lines.append(f"|{row['condition']}|{row['profile']}|{row['seed']}|{f['epoch']}/{b['epoch']}/{l['epoch']}|" + "|".join(triples) + f"|{fmt(l['teacher_drift']['l1'])}|")
    lines += ["", "## 固定 probe 加权共享梯度", "", "两批算术均值仅用于摘要；JSON 保留逐批数值与最小/最大值。共享参数=encoder+fusion。", "", "|配置|版本|种子|probe near/late|JEPA:BCE norm near→late|重构:BCE norm near→late|方差:BCE norm near→late|JEPA/BCE余弦 near→late|重构/BCE余弦 near→late|方差/BCE余弦 near→late|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        if not row["selected_probes"]:
            continue
        b, l = row["selected_probes"]["best_nearest"], row["selected_probes"]["late"]
        values = []
        for metric in ("shared_auxiliary_to_task_norm_ratios", "shared_task_auxiliary_cosines"):
            for term in ("jepa", "reconstruction", "variance"):
                values.append(f"{fmt(b['per_batch_statistics'][metric][term]['mean'],3)}→{fmt(l['per_batch_statistics'][metric][term]['mean'],3)}")
        lines.append(f"|{row['condition']}|{row['profile']}|{row['seed']}|{b['epoch']}/{l['epoch']}|" + "|".join(values) + "|")
    lines += ["", "## 有证据支持的结论", "",
              "1. 训练数值正常，但泛化过程不正常：20/20 配置/种子在 best 到 last 训练 BCE 下降、验证 NLL 上升；T0 AUROC 均下降、Brier 均上升。L1 关闭全部辅助损失仍出现同样模式，因此过拟合不能只归因于 JEPA。所有 history/probe 数值有限，16个新实验所有诊断中 EMA 教师不参与反向梯度；教师仍通过 EMA 更新。",
              "2. 固定 T0 的最佳 NLL 在 0.5301–0.5439，L0-L4 两种子均值在 0.5355–0.5388，现有损失消融没有提供稳定的大幅改善。动态 L1 mean best NLL=0.5490，L0=0.5589；两种子分别改善约0.0173和0.0024。L1 last mean NLL=0.9996，低于L0=1.4200，但仍远高于自身best。",
              "3. 逐更新裁剪率86.8%–97.8%，阈值1使优化长期依赖裁剪。T0末期患者加权批次norm=3.15–6.02，动态=7.36–13.39；动态最大单步norm到92.91。裁剪后更新并未产生非有限值，这不是梯度爆炸已导致训练崩溃的证据；也不能只看最终loss有限便认定优化设置合适。后续调LR/裁剪应各自做单因素实验。",
              "4. scalar损失和共享梯度量级不同。near-best probe中，0.5×JEPA的encoder+fusion梯度仅为BCE的约0.6%–2.0%（两批比值摘要）；0.1×重构约13.5%–61.3%，0.01×方差约0.8%–3.8%。JEPA主要训练其独立world predictor，不能因为JEPA scalar后期上升就认定它压过pCR共享梯度。余弦有正有负，种子、批次和epoch变化显著，不支持对所有实验统一宣称严重梯度冲突。",
              "5. 没有全体患者表示完全塌缩的证据。新实验near-best观察患者池化有效秩约1.74–4.62，末期约3.08–6.60，患者池化RMS std并非零；L4自身也随训练增长。低有效秩提示冗余或任务信号集中，但该32患者小probe不足以判断128维总体分布。教师RMS约1主要受到归一化影响，单独看RMS不能排除塌缩。",
              "6. JEPA损失上升包含移动目标因素：L2/L4 late动态每5 epoch教师drift L1=0.1137–0.1269，教师RMS=1.0105–1.0150；T0末期drift约0.0098–0.0117。其late教师坐标仍在变化，原始JEPA L1跨checkpoint不是固定靶标；该诊断是训练集，不替代独立验证集world/copy评估。是否未来预测过拟合必须同时看同一教师坐标的copy相对误差。",
              "7. L4修正方差定义的最佳pCR提升非常小：T0 meanNLL与L0差约+0.000024，动态约-0.002784；没有解决late过拟合。L0没有新probe，不能声称L4已比L0改善有效秩。", "",
              "## 下一步实验建议", "",
              "- 动态pCR优先以L1为监督基线，先单独比较constant LR与plateau调度，保持dropout0.3、weight_decay0.01、输入和患者不变。这是由late恶化和L1最佳NLL改善提出的新实验，不能把T0已有A1结果当作dynamic L1调度证据。",
              "- 再独立比较L1 weight_decay0.01与0.05，保持已有最佳调度或基线调度统一。补临床+Arm/MRI-only，并增加种子以检查0.01量级的NLL差是否稳定；不因两个种子立即改变数据筛选。",
              "- 如果未来预测是独立目标，先依据完整验证world/copy报告判断是否有skill，再对L2或L4做一个单因素版本，例如固定EMA教师坐标与持续EMA更新对照。JEPA权重0.5不能由scalar量级直接断言太大或太小；0.1/0.5/1.0仅是随后可测试的权重网格。",
              "- 重构暂不加大权重。若保留该任务，先在同一模型与pCR设置下比较当前全分辨率MSE与2×4×4匹配粗尺度目标；小型decoder另列模型实验，避免同时改权重、解码器和数据。",
              "- 每个新实验继续用best.pt报告pCR，保存late用于过拟合诊断；patience50只决定何时停止，不能把last.pt当最终最佳模型。102名开发验证患者已被反复用于选模，最终泛化需独立测试或另定患者级评估协议。"]
    return "\n".join(lines) + "\n"


def main():
    runs = []
    for condition in ("L0", "L1", "L2", "L3", "L4"):
        for profile in ("t0", "dynamic"):
            for seed in (17, 43):
                if condition == "L0":
                    name = f"{'a1' if profile == 't0' else 'a2'}_{profile}_seed{seed}"
                    path = ROOT / "runs/registered_roi32_a1_a2_20261004" / name
                else:
                    name = f"{condition.lower()}_{profile}_seed{seed}"
                    path = ROOT / "runs/registered_roi32_l1_l4_20261004" / name
                runs.append(run_summary(name, path, condition, profile, seed))
    result = {"schema": "roi32_l_training_diagnostics_analysis_v1", "generated_from": "saved JSON only; no model inference",
              "scope": "20 locked L0-L4 runs, both profiles, seeds 17 and 43",
              "development_validation_patients": 102, "independent_test": False,
              "semantics": {"gradient_norm": "full step preclip L2 norm; epoch value weighted by batch patient count",
                            "clip_fraction": "count of updates whose preclip norm > 1 divided by number of updates",
                            "probe_gradients": "separate weighted objectives in eval mode, per B16 batch; arithmetic means are summaries, not whole-probe gradients",
                            "representation_std": "sqrt(mean patient-axis variance) reported as RMS standard deviation",
                            "teacher_drift": "interval to prior fixed probe; five-epoch interval, not accumulated drift from epoch zero",
                            "world_raw_errors": "different EMA coordinates across checkpoints; compare against same-teacher copy to infer forecast skill",
                            "disabled_jepa": "L1/L3 teacher diagnostics use legal observed histories only, not read future targets"},
              "runs": runs}
    result["verification"] = {
        "run_count": len(runs), "new_run_count": sum(r["condition"] != "L0" for r in runs),
        "periodic_probe_count": sum(r["periodic_probe_count"] for r in runs),
        "all_runs_completed_with_early_stopping": all(r["stop_reason"] == "early_stopping" for r in runs),
        "all_best_to_last_train_bce_down": all(r["best_to_last"]["train_bce_change"] < 0 for r in runs),
        "all_best_to_last_validation_nll_up": all(r["best_to_last"]["validation_selection_nll_change"] > 0 for r in runs),
        "all_best_to_last_t0_auroc_down": all(r["best_to_last"]["validation_per_landmark_changes"]["T0"]["auroc"] < 0 for r in runs),
        "all_best_to_last_t0_brier_up": all(r["best_to_last"]["validation_per_landmark_changes"]["T0"]["brier"] > 0 for r in runs),
        "maximum_weighted_sum_vs_saved_total_absolute_error": max(r["weighted_sum_absolute_error_max"] for r in runs),
        "all_saved_numeric_values_finite": all(r["all_epoch_history_and_probes_finite"] for r in runs),
        "all_new_run_teacher_frozen_checks_true": all(r["teacher_frozen_all_probes"] for r in runs if r["condition"] != "L0"),
        "all_new_run_rng_and_mode_preservation_checks_true": all(r["rng_and_modes_preserved_all_probes"] for r in runs if r["condition"] != "L0")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "training_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "training_analysis_ZH.md").write_text(make_markdown(result))
    print(f"Wrote {len(runs)} run aggregates; finite={all(row['all_epoch_history_and_probes_finite'] for row in runs)}")
    for row in runs:
        b, l = row["best"], row["last"]
        print(row['condition'],row['profile'],row['seed'],f"ep {b['epoch']}/{l['epoch']}",
              f"BCE {b['training_raw_terms']['task']:.4f}->{l['training_raw_terms']['task']:.4f}",
              f"NLL {b['validation_selection_nll']:.4f}->{l['validation_selection_nll']:.4f}",
              f"clip {row['overall_clipped_step_fraction']:.3f}")


if __name__ == "__main__":
    main()
