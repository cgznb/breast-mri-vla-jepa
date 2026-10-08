"""Aggregate saved L-suite world reports without loading patients or checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("t0", "dynamic")
SEEDS = (17, 43)
HORIZONS = ("1", "2", "3", "all")
LABELS = {"L0": "complete", "L1": "pcr_only", "L2": "no_reconstruction",
          "L3": "no_jepa", "L4": "patient_axis_variance"}


def read_reports():
    records = []
    for loss in LABELS:
        for profile in PROFILES:
            for seed in SEEDS:
                for kind in ("best", "last"):
                    if loss == "L0":
                        control = "a1" if profile == "t0" else "a2"
                        path = ROOT / "reports/l0_world_20261004" / f"{control}_{profile}_seed{seed}_{kind}.json"
                    else:
                        path = ROOT / "runs/registered_roi32_l1_l4_20261004" / f"{loss.lower()}_{profile}_seed{seed}" / f"world_{kind}.json"
                    data = json.loads(path.read_text())
                    if not data["checkpoint_file_unchanged"] or data["synthetic"]:
                        raise ValueError(f"Checkpoint or dataset validation failed: {path}")
                    if loss != "L0" and not data["runtime_source_matches_checkpoint"]:
                        raise ValueError(f"Current suite source identity failed: {path}")
                    record = {"loss": loss, "loss_label": LABELS[loss], "profile": profile,
                              "seed": seed, "checkpoint": kind, "epoch": data["checkpoint_epoch"],
                              "steps": data["checkpoint_steps"], "status": data["status"],
                              "source": str(path.relative_to(ROOT)),
                              "checkpoint_file_unchanged": data["checkpoint_file_unchanged"],
                              "runtime_source_matches_checkpoint": data["runtime_source_matches_checkpoint"]}
                    if loss in {"L1", "L3"}:
                        if data["status"] != "unavailable" or data["world_head_trained"]:
                            raise ValueError(f"Disabled world branch incorrectly evaluated: {path}")
                        record["reason"] = data["reason"]
                    else:
                        if data["status"] != "evaluated" or data["patients"] != 102:
                            raise ValueError(f"Invalid full validation result: {path}")
                        record.update(metrics=data["metrics"],
                                      teacher_forcing_all_adjacent=data["teacher_forcing_all_adjacent"]["metrics"],
                                      legal_prefixes=data["legal_prefixes"],
                                      precision=data["precision"])
                        for values in record["teacher_forcing_all_adjacent"].values():
                            error = values["teacher_forcing_l1"]
                            copy = values["teacher_forcing_copy_l1"]
                            if not all(math.isfinite(v) and v >= 0 for v in (error, copy)) or copy <= 1e-8:
                                raise ValueError(f"Invalid adjacent loss: {path}")
                            if abs(error / copy - values["relative_to_copy"]) > 1e-12 or abs(1 - values["relative_to_copy"] - values["skill_vs_copy"]) > 1e-12:
                                raise ValueError(f"Adjacent ratio replay failed: {path}")
                            if values["near_zero_copy_patients"] != 0:
                                raise ValueError(f"Unexpected zero-copy adjacent patient: {path}")
                        for horizon in HORIZONS:
                            values = record["metrics"][f"horizon_{horizon}"]
                            for task in ("autonomous", "teacher_forcing"):
                                error = values[f"{task}_l1"]
                                copy = values[f"{task}_copy_l1"]
                                ratio = values[f"{task}_relative_to_copy"]
                                skill = values[f"{task}_skill_vs_copy"]
                                if not all(math.isfinite(v) and v >= 0 for v in (error, copy)):
                                    raise ValueError(f"Invalid finite loss: {path}")
                                if copy <= 1e-8 or ratio is None or abs(error / copy - ratio) > 1e-12 or abs(1 - ratio - skill) > 1e-12:
                                    raise ValueError(f"Ratio or skill replay failed: {path}")
                                if values[f"{task}_near_zero_copy_patients"] != 0:
                                    raise ValueError(f"Unexpected zero-copy patient: {path}")
                            values["autonomous_relative_to_teacher_forcing_same_chain"] = values["autonomous_l1"] / values["teacher_forcing_l1"]
                    records.append(record)
    return records


def seed_means(records):
    result = []
    for loss in ("L0", "L2", "L4"):
        for profile in PROFILES:
            for kind in ("best", "last"):
                selected = [r for r in records if (r["loss"], r["profile"], r["checkpoint"]) == (loss, profile, kind)]
                summary = {"loss": loss, "profile": profile, "checkpoint": kind,
                           "reduction": "arithmetic mean of two seed-level ratios; not an ensemble or pooled patient ratio",
                           "horizons": {}}
                for horizon in HORIZONS:
                    rows = [r["metrics"][f"horizon_{horizon}"] for r in selected]
                    summary["horizons"][f"horizon_{horizon}"] = {
                        "autonomous_ratio_mean": mean(r["autonomous_relative_to_copy"] for r in rows),
                        "autonomous_skill_mean": mean(r["autonomous_skill_vs_copy"] for r in rows),
                        "teacher_forcing_ratio_mean": mean(r["teacher_forcing_relative_to_copy"] for r in rows),
                        "same_chain_autonomous_to_teacher_forcing_mean": mean(r["autonomous_relative_to_teacher_forcing_same_chain"] for r in rows),
                        "autonomous_beats_copy_seeds": sum(r["autonomous_relative_to_copy"] < 1 for r in rows),
                        "teacher_forcing_beats_copy_seeds": sum(r["teacher_forcing_relative_to_copy"] < 1 for r in rows)}
                summary["all_adjacent_tf_ratio_mean"] = mean(r["teacher_forcing_all_adjacent"]["all"]["relative_to_copy"] for r in selected)
                summary["all_adjacent_tf_skill_mean"] = 1 - summary["all_adjacent_tf_ratio_mean"]
                summary["adjacent_tf_ratio_mean_by_transition"] = {
                    transition: mean(r["teacher_forcing_all_adjacent"][transition]["relative_to_copy"] for r in selected)
                    for transition in ("T0_to_T1", "T1_to_T2", "T2_to_T3", "all")}
                result.append(summary)
    return result


def comparisons(records):
    indexed = {(r["loss"], r["profile"], r["seed"], r["checkpoint"]): r for r in records}
    rows = []
    for loss in ("L2", "L4"):
        for profile in PROFILES:
            for kind in ("best", "last"):
                for seed in SEEDS:
                    current = indexed[(loss, profile, seed, kind)]
                    control = indexed[("L0", profile, seed, kind)]
                    rows.append({"loss": loss, "profile": profile, "seed": seed, "checkpoint": kind,
                                 "autonomous_ratio_delta_vs_l0_by_horizon": {
                                     f"horizon_{h}": current["metrics"][f"horizon_{h}"]["autonomous_relative_to_copy"] - control["metrics"][f"horizon_{h}"]["autonomous_relative_to_copy"] for h in HORIZONS},
                                 "all_adjacent_tf_ratio_delta_vs_l0": current["teacher_forcing_all_adjacent"]["all"]["relative_to_copy"] - control["teacher_forcing_all_adjacent"]["all"]["relative_to_copy"]})
    return rows


def fmt(value):
    return f"{value:.4f}"


def markdown(report):
    records = report["records"]
    means = report["seed_means"]
    lines = ["# L0-L4 世界预测与 Copy 结果分析", "",
             "本报告只解析已有 JSON，没有运行训练、推理、加载模型权重或患者数据。24 份世界报告有效，16 份 L1/L3 报告按设计不可用。所有有效报告为 102 名开发验证患者、FP32 独立评价。", "",
             "## 评价含义", "",
             "表格中的 ratio 是同一 checkpoint 的预测 latent L1 / copy latent L1；小于 1 才优于 copy，skill=1-ratio。copy 对 teacher forcing 使用真实前一期状态，对自主推演则始终保持起点真实状态。先在每名患者内平均合法 prefix-target，再平均患者。两种 copy 的定义不同，不能直接比较它们各自的 ratio 来评价累计误差。", "",
             "每个 checkpoint 的 EMA 教师定义其坐标，所以原始 latent L1 不能跨 checkpoint 当作共同尺度排名。即使使用 copy 归一化，ratio 也只描述各模型自己坐标中的相对预测能力，不能证明其表示学到更多临床信息。best 由 pCR NLL 选出，last 并非按世界预测选模。", "",
             "固定 T0 的连续 1/2/3 步患者数分别为 100/92/84，目标数也是 100/92/84；汇总连续链为 100 名患者、276 个目标。动态版分别为 102/92/84 名患者、278/176/84 个目标，汇总 102 名患者、538 个目标。固定/动态汇总口径不同，不将两个 profile 直接当配对优劣比较。", "",
             "全部合法相邻 teacher forcing：T0→T1 为 100 名患者、100 个 prefix-pair；T1→T2 为 92 名患者、固定 92/动态 184 个 prefix-pair；T2→T3 为 86 名患者、固定 86/动态 256 个 prefix-pair；汇总 102 名患者、固定 278/动态 540 个 prefix-pair。动态数据同一转移会在不同合法前缀条件下重复评价，prefix-pair 不是独立患者数。缺随访不会拼接成跨阶段转移；缺 T1 后的合法 T2→T3 纳入相邻 TF，但不纳入从 T0 出发的连续三步。", "",
             "## Best 每个种子", "",
             "| 损失 | 版本 | 种子 | Epoch | 自主 1 步 ratio | 2 步 | 3 步 | 全步 | 全步 skill | 全部相邻 TF ratio | 自主/TF L1 同链全步 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["status"] != "evaluated" or r["checkpoint"] != "best":
            continue
        values = r["metrics"]["horizon_all"]
        ratios = [fmt(r["metrics"][f"horizon_{h}"]["autonomous_relative_to_copy"]) for h in HORIZONS]
        lines.append(f"| {r['loss']} | {r['profile']} | {r['seed']} | {r['epoch']} | " + " | ".join(ratios) + f" | {100 * values['autonomous_skill_vs_copy']:+.2f}% | {fmt(r['teacher_forcing_all_adjacent']['all']['relative_to_copy'])} | {fmt(values['autonomous_relative_to_teacher_forcing_same_chain'])} |")
    lines += ["", "## Last 每个种子", "",
              "| 损失 | 版本 | 种子 | Epoch | 自主 1 步 ratio | 2 步 | 3 步 | 全步 | 全步 skill | 全部相邻 TF ratio | 自主/TF L1 同链全步 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["status"] != "evaluated" or r["checkpoint"] != "last":
            continue
        values = r["metrics"]["horizon_all"]
        ratios = [fmt(r["metrics"][f"horizon_{h}"]["autonomous_relative_to_copy"]) for h in HORIZONS]
        lines.append(f"| {r['loss']} | {r['profile']} | {r['seed']} | {r['epoch']} | " + " | ".join(ratios) + f" | {100 * values['autonomous_skill_vs_copy']:+.2f}% | {fmt(r['teacher_forcing_all_adjacent']['all']['relative_to_copy'])} | {fmt(values['autonomous_relative_to_teacher_forcing_same_chain'])} |")
    lines += ["", "## 两种子均值", "",
              "这是两种子比值的算术平均，不是集成模型，也不是合并预测。", "",
              "| 权重 | 损失 | 版本 | 自主 1 步 ratio | 2 步 | 3 步 | 全步 | 全步 skill | 全部相邻 TF ratio | 全步超过 copy 种子数 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in means:
        values = r["horizons"]["horizon_all"]
        ratios = [fmt(r["horizons"][f"horizon_{h}"]["autonomous_ratio_mean"]) for h in HORIZONS]
        lines.append(f"| {r['checkpoint']} | {r['loss']} | {r['profile']} | " + " | ".join(ratios) + f" | {100 * values['autonomous_skill_mean']:+.2f}% | {fmt(r['all_adjacent_tf_ratio_mean'])} | {values['autonomous_beats_copy_seeds']}/2 |")
    lines += ["", "## 相邻转移 TF 两种子均值", "",
              "| 权重 | 损失 | 版本 | T0→T1 ratio | T1→T2 | T2→T3 | 全部相邻 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in means:
        values = r["adjacent_tf_ratio_mean_by_transition"]
        ratios = [fmt(values[key]) for key in ("T0_to_T1", "T1_to_T2", "T2_to_T3", "all")]
        lines.append(f"| {r['checkpoint']} | {r['loss']} | {r['profile']} | " + " | ".join(ratios) + " |")
    lines += ["", "## 同 checkpoint 尺度记录", "",
              "原始值保留用于审计分子和分母，只在每个 checkpoint 内解释，不用它们跨模型排列表示质量。", "",
              "| 权重 | 损失 | 版本 | 种子 | 自主全步 L1 | 起点 copy 全步 L1 | TF 全步 L1 | 真实前一期 copy 全步 L1 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["status"] != "evaluated":
            continue
        values = r["metrics"]["horizon_all"]
        lines.append(f"| {r['checkpoint']} | {r['loss']} | {r['profile']} | {r['seed']} | {fmt(values['autonomous_l1'])} | {fmt(values['autonomous_copy_l1'])} | {fmt(values['teacher_forcing_l1'])} | {fmt(values['teacher_forcing_copy_l1'])} |")
    lines += ["", "## 结论", "",
              "1. 动态 L2 去重构是唯一 best 权重下自主全步两种子均值优于 copy 的配置：ratio=0.9689、skill=+3.11%。seed17=0.9269（+7.31%），seed43=1.0110（-1.10%），所以方向比 L0 两种子均改善，但尚未形成两种子都超过 copy 的稳健结论。其全部相邻 TF 均值=1.0094，teacher forcing 也没有整体稳健超过 copy。", "",
              "2. 固定 T0 L2 的 best 全步均值从 L0 1.0450 降至 1.0375，但 seed17 改善、seed43 变差，且两种子均未超过 copy。固定 T0 去重构没有证实稳定提高世界预测。", "",
              "3. L4 患者轴方差的 best 结果几乎复现 L0：固定 T0 全步均值=1.0434（L0=1.0450），动态=1.0532（L0=1.0469）。动态两种子均略差 L0，方差轴更合理本身不等于预报效果更好。固定 T0 seed43 三步及全步略胜 copy，但全步仅 skill=+0.13%，不能据此声称稳健改进。", "",
              "4. best 的自主单步仅动态 L2 seed17 超过 copy。其余 11 个 best 单步 ratio 均大于 1。更远期某些自主 ratio 接近或低于 1，部分因为保持起点的 copy 随距离变差；应同时看同链自主/TF L1，best 全步均为 1.0411-1.0694，说明推演仍有额外误差。", "",
              "5. 同一训练的 last 自主全步 ratio 在全部 12 个有效运行中都低于 best；动态 L0/L2/L4 的 last 两种子都超过起点 copy，均值分别为 0.9216/0.9162/0.9199。世界相对任务继续改善可与 pCR 后期过拟合同时存在，不能由原始 JEPA loss 后期变大直接断言世界预测也变差。这些结果只说明各 checkpoint 坐标中相对 copy 改善，不能独立确认共同坐标上的世界泛化提高。EMA 教师坐标一直变化，需结合固定探针的教师尺度、漂移和患者方差；本报告不据 copy 差值独自判定表示坍塌。", "",
              "6. 所有 copy 分母有限且大于 1e-8，每个 horizon 的近零 copy 患者数为 0；不存在该保护阈值下的除零或虚假无限 skill。非零 copy 不证明表示质量充分。manifest 缺少逐转移配准 QC，因此无法对经过空间验证子集给出单独结论。", "",
              "7. L1/L3 的 JEPA 与 flow 权重都为零，世界预测器未受训练，16 份 best/last 明确不可用。这是消融设计结果，不是评估失败，不应将随机初始化世界头拿来与有监督世界模型排序。", "",
              "8. L0 复用历史 A1 固定 T0、A2 动态控制，其源码快照与当前版本不同（8 份 source-match 为 false 已公开），原始权重未更新；已核对数据分区及标准化，编码器、预测器及推演结构保持原定义。L2 最佳 epoch 也与 L0 不同，当前 world 比较对应各自 pCR 选模，不能分离训练时长和损失直接影响。", "",
              "102 人都是反复使用的开发验证集，只有两个种子，没有独立 test。没有逐患者世界误差导出，本次不能计算配对置信区间或显著性，也没有 QC 子组。", "",
              "详细精度数值、全部相邻各阶段 TF、各 horizon 两种子均值与对 L0 的逐种子变化见 world_analysis.json。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/l_loss_analysis_20261004")
    args = parser.parse_args()
    records = read_reports()
    for profile in PROFILES:
        selected = [r for r in records if r["status"] == "evaluated" and r["profile"] == profile]
        expected = {"1": (100, 100), "2": (92, 92), "3": (84, 84), "all": (100, 276)} if profile == "t0" else {"1": (102, 278), "2": (92, 176), "3": (84, 84), "all": (102, 538)}
        for r in selected:
            if r["legal_prefixes"] != (102 if profile == "t0" else 382) or r["precision"] != "fp32":
                raise ValueError(f"Evaluation profile or precision differs: {r['source']}")
            for h, counts in expected.items():
                values = r["metrics"][f"horizon_{h}"]
                if (values["patients"], values["prefix_targets"]) != counts:
                    raise ValueError(f"Continuous-chain coverage differs: {r['source']}")
            expected_adjacent = {"T0_to_T1": (100, 100), "T1_to_T2": (92, 92), "T2_to_T3": (86, 86), "all": (102, 278)} if profile == "t0" else {"T0_to_T1": (100, 100), "T1_to_T2": (92, 184), "T2_to_T3": (86, 256), "all": (102, 540)}
            for transition, counts in expected_adjacent.items():
                values = r["teacher_forcing_all_adjacent"][transition]
                if (values["patients"], values["prefix_pairs"]) != counts:
                    raise ValueError(f"Adjacent coverage differs: {r['source']}")
    report = {"dataset_role": "development_validation_not_independent_test", "validation_patients": 102,
              "report_count": len(records), "evaluated_count": sum(r["status"] == "evaluated" for r in records),
              "unavailable_count": sum(r["status"] == "unavailable" for r in records),
              "raw_l1_cross_checkpoint_comparability": False,
              "no_training_inference_or_checkpoint_load": True,
              "records": records, "seed_means": seed_means(records), "comparisons_vs_l0": comparisons(records)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "world_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.output / "world_analysis_ZH.md").write_text(markdown(report))
    print(json.dumps({"reports": report["report_count"], "evaluated": report["evaluated_count"],
                      "unavailable": report["unavailable_count"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
