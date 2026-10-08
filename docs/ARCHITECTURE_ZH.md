# 网络与训练逻辑

## 1. 网络连接

```text
真实观察历史 MRI + 当时可用的临床/Arm
                    ↓
          逐次检查独立 3D MRI encoder
                    ↓
              共享 Fusion 主干
                    ↓
          ┌─────────┴─────────┐
          │                   │
  状态转移查询 z         pCR 查询 c（可读 z）
          │                   │
          ↓                   ↓
  独立时间因果世界模型     mean → LN → Linear
          ↑                   ↓
  真实前序状态【仅训练】    当前时点的最终 pCR 概率
          ↓                   ↓
     下一状态预测         同一个最终标签的 BCE
          ↓
  真实下一状态的 JEPA L1
```

VLA-JEPA 的两类主干查询分别用于状态转移学习和任务预测。本改版保留这一关系，用 3D MRI 观察编码器/共享 Transformer 替换 RGB VLM，用二分类头替换连续机器人动作头。状态目标采用 B 路线的医学 MRI EMA 编码器，而非原版冻结的预训练 V-JEPA2；没有加载 Qwen3-VL 或 V-JEPA2 权重。

每个 MRI 检查有三个 DCE 相位，输入为 `[3,D,H,W]`。四次纵向检查对应 T0–T3，输入容器为 `[B,4,3,D,H,W]`。默认编码网格 `2×4×4`、维度 128，每次检查得到 `32×128` 空间 tokens。

共享主干只编码 `observed_mask=True` 的扫描。MRI tokens 加阶段嵌入，临床数值与可见性 mask 投影为一个 token，Arm 为一个条件 token。查询排列为 `[观察 context, 转移 queries, pCR queries]`。观察 context 内可双向融合，context 不能读查询；查询按排列因果读取，因此 pCR 可以利用前面的转移特征。

## 2. S、z 和 c 的含义

`S0…S3` 是各真实检查经独立目标编码器得到的状态，教师参数不接受反向传播。`z` 是观察主干产生的转移条件特征，`c` 是 pCR 读出特征。

模型保留四个目标阶段查询槽：

```text
dynamics_features[:,0]：无下一阶段任务，屏蔽
dynamics_features[:,1]：0→1 的转移条件
dynamics_features[:,2]：1→2 的转移条件
dynamics_features[:,3]：2→3 的转移条件
```

世界模型的三个输入块是 `[z0→1,S0]`、`[z1→2,S1]`、`[z2→3,S2]`。块内空间/state-query tokens 全双向，块间仅能读取当前及过去；输出状态分别对齐 S1、S2、S3。始终可见的 root token 保证被屏蔽块的 attention 不产生全 mask NaN。

## 3. 训练样本与损失

假设四次真实检查均存在且按阶段可用：

| 采样阶段 | 共享主干真实 MRI | Teacher-forcing 源状态 | 对齐目标 | pCR |
|---|---|---|---|---|
| T0 | T0 | S0、S1、S2 | S1、S2、S3 | 一个最终概率 p0 |
| T1 | T0、T1 | S1、S2 | S2、S3 | 一个最终概率 p1 |
| T2 | T0、T1、T2 | S2 | S3 | 一个最终概率 p2 |
| T3 | T0–T3 | 无未来对 | 无未来对 | 一个最终概率 p3 |

三个转移可在一次因果世界模型前向中训练。预测 Ŝ1 的位置只能读 S0；预测 Ŝ2 可读真实 S1，因此是 teacher forcing。真实 S1/S2 只在独立世界模型损失路径中出现，不进入 T0 的主干或 pCR。

正式 epoch 训练每轮随机打乱患者并各使用一次，再固定 T0 或均匀随机采各自的合法阶段。它在已有病例上遮罩未来，模拟不同观察条件。所有预测阶段监督同一个最终 pCR 标签；所有前缀保持在同一患者数据划分中。

主损失为：

\[
L=L_{pCR}^{BCE}+\lambda_{WM}L_{WM}^{L1}
  +\lambda_{rec}L_{observed\ reconstruction}+\lambda_{var}L_{variance}.
\]

`task_weight` 对应 pCR，`jepa_weight` 对应 λWM。正式28次实验采用一阶段训练并关闭 MRI flow。不同损失消融的权重见[实验矩阵](EXPERIMENTS_ZH.md)；L1/L3不训练独立世界预测器，不能把其随机预测当作有效世界模型。

JEPA误差在空间token和channel上平均绝对差，再在患者内平均合法相邻pair，最后对有合法pair的患者平均。真实前驱和下一期状态均由无梯度EMA教师产生；梯度经world predictor及转移特征回到共享backbone，不更新教师。pCR分类不读取世界预测结果，但辅助训练可改变共享表示。

观察重构从在线encoder输出、fusion之前分支：`[有效检查,128,2,4,4]`特征经1×1×1卷积投影为三通道，再三线性上采样到`[32,128,128]`，对已观察MRI全图计算MSE。它不预测未来MRI，也不使用ROI/support作为loss mask；梯度进入encoder与重构层。

方差惩罚为`mean(relu(1-sqrt(var+1e-4)))`，作用于encoder输出。原定义合并有效检查和空间token后按channel统计，空间差异可满足约束。L4保持同stage/token/channel，在该阶段的有效患者轴用FP32统计；每阶段至少2人才贡献，对有效阶段平均。方差项未经过fusion或世界预测器。

WM 损失只计算两端真实存在、目标被请求的相邻阶段对，并在患者内先平均有效对。缺失 T1 时，0→1 和 1→2 均无该相邻对监督，2→3 可保留；固定阶段槽不会压缩成伪造的一步转移。T3 的 WM loss 为图连接的零，pCR/观察表征仍可训练。目标是否存在只决定损失 mask，不决定主干未来查询。

## 4. 分类与状态推演的推理接口

`forward(inp)` / `backbone(inp)` 只接受输入对象，返回 `pcr_logit`、`dynamics_features`、`pcr_features`。普通 pCR 推理不执行世界预测器、teacher forcing、生成器或 ODE。

动态模型的同一个 checkpoint：

```text
输入 T0       → p0
输入 T0,T1    → p1
输入 T0,T1,T2 → p2
输入 T0…T3    → p3
```

新真实 MRI 到来后只是更新输入并重算，不更新参数，不需要把旧 pCR 概率喂回网络。固定 T0 模型拒绝非 T0 的预测/评估请求，避免把只训练基线的权重当作已验证的动态模型。

如果请求 `forecast_states(inp)`，以真实当前 MRI 的状态为起点，使用预测 Ŝ1 继续预测 Ŝ2，再预测 Ŝ3。自主推演不读真实未来监督。过去及未请求槽返回零。若请求未来但当前阶段没有真实 MRI，则明确拒绝状态推演；pCR 分类仍可使用合法历史。T3 无未来请求，状态结果为空。

Teacher-forcing 状态误差和自主推演误差不是同一个指标；前者低不能证明长期 rollout 准确。可选 MRI flow 若开启，条件必须使用自主推演状态，不能接含真实未来上下文的 teacher-forced 预测。

## 对应代码

`model.py` 提供 `backbone`、`teacher_forcing_loss`、`compute_loss` 和 `forecast_states`。`training.py:train` 组织患者/阶段采样、优化与 EMA；`data.py:RawMRIStore.batch` 拆分输入和未来监督。

匹配的A0/A1/A2固定与动态配置只改变`landmarks`。L系列固定版继承A1（dropout0.1、Plateau），动态版继承A2（dropout0.3、恒定LR），因此两版还有优化策略差异。全部共用同一模型实现。
