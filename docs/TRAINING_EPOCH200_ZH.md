# 200-epoch 训练协议

本页描述 A0/A1/A2/L1–L4 的正式协议，共28次训练。矩阵与差异见[实验说明](EXPERIMENTS_ZH.md)，已有结果见[聚合结果](../results_summary/README.md)。

## 患者与优化

每个 epoch 随机打乱训练患者并各使用一次。历史队列有764名训练患者，形成47个B16批次及一个12人末批，共48次优化更新；不丢弃末批。动态版每名患者均匀随机选择自己的一个合法观察阶段，固定版始终使用T0。历史验证集102人，动态评价382个合法前缀；这些人数描述历史数据，不是代码对新数据的规模要求。

正式配置采用一阶段训练：`training_unit=epochs`、`max_epochs=200`、`early_stopping_patience=50`、`early_stopping_min_delta=0`、`batch_size=16`、`accumulation=1`。AdamW初始LR为1e-4、weight decay为0.01；全局梯度范数裁剪阈值1.0。CUDA BF16用于训练，验证概率与指标以FP32/NumPy计算。每步优化后更新EMA教师，decay为0.995；教师始终eval且不接受梯度。

固定T0按T0验证NLL保存best；动态先在每位患者内平均合法阶段NLL，再跨患者平均。严格改善才重置早停计数并保存新的`best.pt`；连续50轮未改善则停止，最多200轮。pCR选模不按最高AUROC，也不按世界误差。历史队列若运行完整200轮，为9600次更新。

A1以及L系列固定版使用ReduceLROnPlateau：`mode=min, factor=0.5, patience=5, min_lr=1e-6, threshold=0, threshold_mode=abs`。每轮以同一选模NLL调用scheduler。其patience与50轮早停不同，下降时间遵循PyTorch的坏轮计数。

## 单次训练与恢复

在仓库根目录执行：

```bash
python train.py train-vla-jepa --config configs/registered_roi32_epoch200_t0_seed17.yaml --manifest data/registered_roi32/manifest.json --output runs/a0_t0_seed17
```

恢复同一run使用原命令加`--resume`。配置、manifest、训练统计、源文件身份和核心源码必须匹配。不要通过恢复旧run来改变损失、dropout或数据；新实验从头训练并使用新目录。历史checkpoint可按兼容规则只读评估，但移动、整理或修改代码不保证可继续其训练。

run中的`progress.json`记录优化步/epoch进度，`epoch_history.json`记录逐轮训练损失和验证指标。`best.pt`是最佳验证NLL权重；`last.pt`保存最近模型、优化器、EMA、scheduler、患者排列/游标、阶段采样器、随机状态和早停状态。

## 完整实验队列

队列工具面向Linux/CUDA环境，在仓库根目录使用：

```bash
python scripts/run_registered_roi32_experiments.py --suite baseline --manifest data/registered_roi32/manifest.json --max-concurrent 1 --detach
python scripts/run_registered_roi32_experiments.py --suite baseline --status
```

`baseline`运行A0四次，默认目录为`runs/registered_roi32_epoch200_20261003`；`a1_a2`运行八次，目录为`runs/registered_roi32_a1_a2_20261004`；`losses`运行十六次，目录为`runs/registered_roi32_l1_l4_20261004`。可通过`--output`指定新的目录，查看状态时需使用同一目录。日期后缀只是历史默认目录名。

`--max-concurrent`支持1–6；可另设`--gpu-memory-budget-mib`作为准入预算。内置任务显存预估来自历史硬件，并非所有GPU的保证；复现时按自己的完整尺寸训练/诊断峰值调整并发。队列的`controller.json`记录任务状态，拒绝重复控制器。

中断后用同一队列命令加`--resume`。控制器会核验并接管仍存活的匹配子进程，其他未完成训练从`last.pt`恢复；`losses`另评价best/last世界任务。训练与评价进程都占并发槽位。CPU合成测试可检查采样/RNG与恢复逻辑，CUDA BF16仍可能出现数值波动。
