# Breast MRI VLA-JEPA

乳腺纵向 MRI 的 pCR（病理完全缓解）预测与未来特征状态学习代码。固定 T0 模型只使用基线信息；动态模型在 T0–T3 各阶段使用截至当时的真实 MRI、可用临床信息与治疗 Arm，预测同一个最终 pCR 终点。

本仓库提供独立 Python 包、训练配置、CPU 合成示例、测试及已有实验的[聚合结果](results_summary/README.md)。真实 MRI、临床表、患者 manifest、逐患者预测和模型权重需自行准备。

模型从随机初始化训练，采用独立 3D MRI encoder、共享 Transformer、pCR queries、转移 queries 和 EMA 目标编码器，没有加载 Qwen3-VL 或 V-JEPA2 预训练权重。pCR 直接读取合法历史的共享表示；JEPA 的真实未来只作为辅助监督。全部 28 次正式实验关闭 MRI flow，未来状态预测输出潜变量，观察影像重构是另一项辅助损失。详见[架构](docs/ARCHITECTURE_ZH.md)与[来源](docs/SOURCES.md)。

## 安装与 CPU 示例

需要 Python 3.11+。在仓库根目录执行；PyTorch 的 CPU/CUDA 版本需适合实际硬件：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[raw,test,analysis]'
python -m mri_vla_jepa --help
python -m pytest -q

python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_t0.yaml --output runs/smoke_t0
python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_dynamic.yaml --output runs/smoke_dynamic
```

`raw` 提供 NIfTI/临床表依赖，`test` 提供 pytest，`analysis` 提供绘图依赖。CPU smoke 使用小型虚构数据，执行四次更新并保存权重、pCR 评价及状态推演。输出目录须为新目录；它验证流程，不复现真实队列效果。安装后 `python train.py`、`python -m mri_vla_jepa` 和 `mri-vla-jepa` 使用同一个 CLI。

本次发布副本的实测环境与验证结果见[验证记录](docs/VALIDATION_ZH.md)。

## 准备数据

正式实验使用已经配准、裁剪并统一标准化的三相 ROI32 NPZ：每次检查的 `image` 为 float32 `[3,32,128,128]`，患者保留 T0–T3 四槽，缺失检查为 `null`。网络输入为 `[B,4,3,32,128,128]`，不使用 VQ latent，也不重新 z-score 或 resize。

已有 ROI32 inventory 和官方临床表时：

```bash
python scripts/prepare_registered_roi32.py --inventory /path/to/registered_roi32/inventory.json --clinical /path/to/official_clinical.xlsx --output data/registered_roi32/manifest.json --report reports/registered_roi32_preflight.json
python scripts/prepare_roi32_image_cache.py --manifest data/registered_roi32/manifest.json --output data/registered_roi32/image_cache_20261004 --report reports/roi32_image_cache.json
```

替换 `/path/to/` 为自己的数据位置。准备脚本读取已有 ROI32 inventory，不包含 DICOM 整理、配准或上游 ROI 提取。A1/A2/L 系列配置中的 `image_cache_20261004` 相对 manifest 所在目录解析，缓存与源图像逐值一致。

历史队列为 764 名训练患者、102 名开发验证患者、3039 次检查，独立测试为空。Arm 按 `assigned_arm_scenario` 视为 T0 已知的官方分配类别，这是一项实验条件。输入结构、患者划分、临床可见性与原始 NPY/NIfTI 入口见[数据格式](docs/DATA_ZH.md)和[ROI32 接入](docs/REGISTERED_ROI32_ZH.md)。

## 正式 28 次实验

正式协议是**最多 200 epoch、早停 patience 50、物理 batch 16、梯度累积 1、seed 17/43**，采用 AdamW、CUDA BF16。每轮随机打乱训练患者并各使用一次；动态版为每名患者均匀抽取一个合法 landmark。固定版按 T0 验证 NLL 选模；动态版先患者内平均合法阶段 NLL，再跨患者平均。

| 设置 | 研究变量 | Dropout | 学习率 | pCR / JEPA / 重构 / 方差权重 |
|---|---|---:|---|---|
| A0 | 初始完整损失基线 | 0.1 | 恒定 | 1 / 0.5 / 0.1 / 0.01 |
| A1 | 验证 NLL 驱动 Plateau | 0.1 | Plateau | 1 / 0.5 / 0.1 / 0.01 |
| A2 | 增大全局 dropout | 0.3 | 恒定 | 1 / 0.5 / 0.1 / 0.01 |
| L1 | 仅 pCR 损失，仍输入 MRI/临床/Arm | 继承 L0 | 继承 L0 | 1 / 0 / 0 / 0 |
| L2 | 去观察影像重构 | 继承 L0 | 继承 L0 | 1 / 0.5 / 0 / 0.01 |
| L3 | 去 JEPA | 继承 L0 | 继承 L0 | 1 / 0 / 0.1 / 0.01 |
| L4 | 同阶段/token/channel 的患者轴 FP32 方差 | 继承 L0 | 继承 L0 | 1 / 0.5 / 0.1 / 0.01 |

每种设置均有固定 T0/动态两个版本与两个种子，共 `7×2×2=28` 次训练。**L0 是别名：固定 T0 复用 A1，动态复用 A2，没有额外训练。** L 系列固定版使用 dropout 0.1＋Plateau，动态版使用 dropout 0.3＋恒定 LR；两版差异不只有 MRI 历史长度。

A0 配置为 `configs/registered_roi32_epoch200_{t0,dynamic}_seed{17,43}.yaml`；其余组在 `epoch200_` 后增加 `{a1,a2,l1,l2,l3,l4}_`。旧 `raw_vla_jepa_*.yaml` 和 `registered_roi32_{t0,dynamic}.yaml` 保留 6000-step 接口，与本次 epoch 实验不同。详见[实验矩阵](docs/EXPERIMENTS_ZH.md)和[训练协议](docs/TRAINING_EPOCH200_ZH.md)。

单次 A0 动态训练示例：

```bash
python train.py train-vla-jepa --config configs/registered_roi32_epoch200_dynamic_seed17.yaml --manifest data/registered_roi32/manifest.json --output runs/a0_dynamic_seed17
```

复现全部矩阵，可依次运行三套队列。以下以单路为例，按实际显存调整并发数：

```bash
python scripts/run_registered_roi32_experiments.py --suite baseline --manifest data/registered_roi32/manifest.json --max-concurrent 1
python scripts/run_registered_roi32_experiments.py --suite a1_a2 --manifest data/registered_roi32/manifest.json --max-concurrent 1
python scripts/run_registered_roi32_experiments.py --suite losses --manifest data/registered_roi32/manifest.json --max-concurrent 1
```

队列分别运行 4、8、16 次训练。当前代码复现公开配置与流程，CUDA BF16 允许数值波动，不承诺逐值重现历史权重。源码、配置与数据须保持不变，才可用同一命令加 `--resume` 恢复对应 run/队列。L1/L3 没有训练世界预测头，其世界评价明确标为不可用。

## 评估与随访推理

```bash
python train.py evaluate-vla-jepa --checkpoint runs/a0_dynamic_seed17/best.pt --manifest data/registered_roi32/manifest.json --split val --device cuda --output results/dynamic_val.json
python train.py predict-vla-jepa --checkpoint runs/a0_dynamic_seed17/best.pt --manifest data/registered_roi32/manifest.json --patient-key PATIENT_ID --landmark 1 --device cuda --output results/patient_t1.json
python scripts/evaluate_registered_roi32_world.py --checkpoint runs/a0_dynamic_seed17/best.pt --manifest data/registered_roi32/manifest.json --split val --device cuda --batch-size 16 --output results/world_val.json
```

`PATIENT_ID` 替换为 manifest 内患者 ID；`--landmark 0/1/2/3` 对应 T0/T1/T2/T3。新 MRI 到来后更新合法输入并重算 pCR，不更新模型参数。pCR 报告使用 NLL 选出的 `best.pt`；`last.pt` 用于恢复与末期诊断。有独立测试患者时使用 `--split test`。

预测加 `--forecast-states` 才执行自主未来状态推演；正式配置不能用 `--generate` 生成未来 MRI。评价 JSON 与训练产物可能包含患者记录；发布实验结论时使用[聚合表](results_summary/README.md)。

## 代码导航

`contracts.py` 定义输入/监督边界；`data.py` 与 `data_pipeline.py` 处理合法历史和缓存；`encoder.py`、`model.py` 实现 MRI 编码、共享主干和世界预测；`training.py` 管理采样、优化、EMA、checkpoint 与选模；`diagnostics.py` 提供梯度/表示诊断与世界/copy 对照。CLI 在 `src/mri_vla_jepa/cli.py`，数据准备及队列工具在 `scripts/`。
