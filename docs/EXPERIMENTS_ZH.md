# 全部实验与差异

2026年10月3–4日的正式队列包含7种实际训练设置、固定T0/动态两版本、seed17/43，共28次训练。L0是已有训练的别名。结果来自102名开发验证患者，没有独立测试成绩；聚合数值见[结果表](../results_summary/README.md)。

## 固定与动态

固定T0只看基线MRI及当时可见临床/Arm，在T0预测最终pCR。动态模型在每位患者截至采样landmark的合法真实历史上预测同一个最终标签，完整随访时可分别使用T0、T0–T1、T0–T2、T0–T3。四次预测不是四个不同临床终点，也不是把未来生成MRI直接输入分类器。

匹配的A0/A1/A2固定/动态配置只有`landmarks`不同；L系列采用每版选定的策略，固定/动态还有dropout和scheduler差异。固定T0选模NLL只含T0；动态选择NLL先患者内平均合法阶段再跨患者平均，二者不能直接相减并归因为随访收益。

## A系列：训练策略

| 设置 | Dropout | LR调度 | 训练次数 | 研究问题 |
|---|---:|---|---:|---|
| A0 | 0.1 | 恒定1e-4 | 4 | 原完整损失基线 |
| A1 | 0.1 | 验证NLL驱动Plateau | 4 | 验证停滞后减小更新是否改善选模/后期表现 |
| A2 | 0.3 | 恒定1e-4 | 4 | 增强全局dropout是否有效 |

完整损失为`pCR BCE + 0.5×JEPA latent L1 + 0.1×观察重构MSE + 0.01×原方差惩罚`。A1 Plateau为factor0.5、patience5、minLR1e-6、threshold0/abs；A2 dropout共同影响encoder、fusion和world predictor的Transformer。

A1/A2还共同增加缓存/预取和传输优化，保持图像内容与患者/阶段采样顺序，详见[数据管线](A1_A2_PIPELINE_20261004_ZH.md)。实际耗时是工程性能与训练轮数的组合，不能代替预测指标。

## L系列：训练目标

固定L0复用A1，动态L0复用A2。L1–L4各训练4次，共16次；MRI/临床/Arm输入不变。

| 设置 | 相对L0变化 | pCR / JEPA / 重构 / 方差 | 世界头受监督 |
|---|---|---|---|
| L0 | 原完整损失 | 1 / 0.5 / 0.1 / 0.01 | 是 |
| L1 | 同时去掉全部辅助目标 | 1 / 0 / 0 / 0 | 否 |
| L2 | 去观察重构 | 1 / 0.5 / 0 / 0.01 | 是 |
| L3 | 去JEPA | 1 / 0 / 0.1 / 0.01 | 否 |
| L4 | 仅改方差统计轴与FP32计算 | 1 / 0.5 / 0.1 / 0.01 | 是 |

L1回答辅助目标整体是否有利，变化不能只归因到其中一项。L2/L3分别隔离重构/JEPA贡献；L4避免空间差异满足本应约束患者差异的方差项。四项目标与梯度路径见[架构](ARCHITECTURE_ZH.md)和[L系列执行](L_LOSS_EXPERIMENTS_20261004_ZH.md)。

## 共用条件与配置

模型dim128、token grid2×4×4、encoder/fusion/predictor各2层4头、每目标阶段8个转移queries、4个pCR queries。AdamW、WD0.01、clip1.0、B16、accumulation1、BF16、max200epoch、早停50、seed17/43、EMA decay0.995。全部关闭flow，训练数据与信息边界相同。

A0配置：`registered_roi32_epoch200_{t0,dynamic}_seed{17,43}.yaml`；其他组：`registered_roi32_epoch200_{a1,a2,l1,l2,l3,l4}_{t0,dynamic}_seed{17,43}.yaml`。`configs/`与包内`resources/configs/`保存相同正式配置。

旧6000-step配置保留作通用入口，CPU smoke另使用小型合成数据，均不属于上述矩阵。A3、L5、JEPA权重扫描、MRI-only及临床+Arm输入对照没有在这28次中训练，不能视为已有结果。

## 评价解释

pCR使用NLL-selected `best.pt`，报告全部种子/阶段的AUROC/AUPRC/NLL/Brier；种子均值是指标平均，不是预测集成。`last.pt`用于过拟合诊断和恢复。世界任务独立比较同教师坐标下的模型/copy，不能仅按移动目标的raw latent误差或pCR-selected epoch判断全部世界能力。

缺失随访导致各阶段患者集合不同，比较阶段增益需同一患者子群。开发集用于选epoch与选设置，患者配对统计也不能替代独立患者测试。
