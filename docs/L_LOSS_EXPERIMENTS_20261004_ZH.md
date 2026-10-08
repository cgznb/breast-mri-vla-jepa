# L1–L4 损失消融

L1–L4四种设置，各固定T0/动态两个版本、seed17/43，共16次训练。L0复用已有对照：固定T0=A1（dropout0.1、Plateau），动态=A2（dropout0.3、恒定LR）。完整28次实验见[矩阵](EXPERIMENTS_ZH.md)，结果见[聚合报告](../results_summary/README.md)。

| 设置 | pCR | JEPA | 观察重构 | 方差 | 方差定义 |
|---|---:|---:|---:|---:|---|
| L0 | 1 | 0.5 | 0.1 | 0.01 | 原定义 |
| L1 | 1 | 0 | 0 | 0 | 关闭 |
| L2 | 1 | 0.5 | 0 | 0.01 | 原定义 |
| L3 | 1 | 0 | 0.1 | 0.01 | 原定义 |
| L4 | 1 | 0.5 | 0.1 | 0.01 | 同阶段/token/channel的患者轴FP32 |

各版本从相应种子重新初始化。同版本相对L0只改变表中损失设置并增加共用诊断，MRI/临床/Arm输入、网络及优化/选模规则不变。L1是监督目标消融，仍输入MRI；实验编号与JEPA采用的latent L1绝对误差不同。

## 损失与梯度路径

`compute_loss`仅执行权重非零的分支。pCR BCE经分类头与共享主干更新表示；JEPA经独立世界预测器及共享转移条件特征回传，EMA教师的真实前驱和目标均无梯度。观察重构和方差发生在MRI encoder输出，未经过共享fusion。重构是`2×4×4`特征经1×1×1卷积与三线性上采样，对已观察MRI全图计算MSE，未使用ROI/support作为损失mask。

原方差将有效检查与空间tokens合并，计算各通道标准差。L4将观察特征恢复为`[B,4,N,C]`，保持同阶段、同token、同channel，只对该阶段有效患者计算FP32方差。采用`unbiased=false`、epsilon1e-4、std目标1；有效患者少于2的阶段跳过，对其余阶段平均。它避免空间多样性替代患者差异，性能收益需以结果判断。

L1/L3不读取未来目标MRI，独立world predictor没有受监督训练，世界指标标为不可用。所有组关闭MRI flow，观察重构不等于未来MRI生成。

## 诊断与世界评价

所有新实验每5epoch使用固定种子20261004选取32名训练患者、固定合法阶段、B16做eval模式诊断；保存/恢复RNG及模块模式，不更新参数、EMA或原梯度，也不参与选模。

`diagnostics.json`记录训练探针NLL、分项加权梯度/方向、表示方差与有效秩、教师尺度/漂移。逐轮记录裁剪比例；真实AdamW参数更新范数仅采样诊断轮第一批。小型训练探针不代表全验证集。

世界评价冻结每个checkpoint自己的EMA教师坐标，对连续链计算teacher-forcing与自主1/2/3步预测，并与同坐标copy比较。自主copy始终保持起点真实状态，teacher-forcing copy使用真实前一期状态。全部合法相邻转移另行报告，缺T1后存在的T2→T3不被丢弃或拼成T0→T2。按患者先平均再跨患者平均，`ratio=模型误差/copy误差`、`skill=1-ratio`；ratio小于1才优于copy。不同checkpoint的raw latent误差不属于统一特征坐标。

loss队列为各run的best/last分别生成`world_best.json`、`world_last.json`，L1/L3返回不可评价。普通pCR选模仍使用验证NLL，不据世界评价重新选择epoch。

## 命令

先按[README](../README.md)准备manifest及缓存，再运行：

```bash
python scripts/run_registered_roi32_experiments.py --suite losses --manifest data/registered_roi32/manifest.json --max-concurrent 1 --detach
python scripts/run_registered_roi32_experiments.py --suite losses --status
```

默认输出为`runs/registered_roi32_l1_l4_20261004`。恢复使用原命令加`--resume`。增加并发时按实际显存设`--max-concurrent`及可选`--gpu-memory-budget-mib`，训练、诊断和世界评价都计入峰值。改变配置、源码或输入时使用新run。
