# A1/A2 与 ROI32 数据缓存

A1在A0完整损失上增加验证NLL驱动的Plateau调度；A2将全局dropout从0.1提高到0.3，并保留恒定LR。每组固定T0/动态两版，各seed17/43，共八次训练。正式200-epoch协议见[训练说明](TRAINING_EPOCH200_ZH.md)，全部设置与结果见[实验矩阵](EXPERIMENTS_ZH.md)及[聚合结果](../results_summary/README.md)。

## 共同的数据管线变化

A1/A2相对A0共同启用NPY缓存、两批预取、两工作线程、pinned memory与非阻塞CUDA传输：

```yaml
image_cache: image_cache_20261004
prefetch_batches: 2
loader_workers: 2
pin_memory: true
non_blocking_transfer: true
```

缓存只保存原ROI32 NPZ的`image`数组，准备时逐值核对，要求float32 `[3,32,128,128]`，保持三相顺序、训练标准化与零背景。缓存路径相对manifest所在目录解析。索引绑定源文件/缓存文件的路径、大小、修改时间和预处理声明；访问时核验身份，首次映射时检查形状、dtype及有限值。它不进行新的图像变换或数据筛选。

后台工作线程按计划准备完整批次，以FIFO顺序消费并与主训练任务核对。预取计划使用采样器副本，只有已消费批次推进主采样器与checkpoint游标；恢复时丢弃未消费预取并重新规划。模型输入/监督边界仍进行检查。

这些设置减少重复解压与载入等待。墙钟时间还包含实际训练轮数、验证、保存和并发差异，不能据此判断收敛或预测效果。

## 准备并运行

```bash
python scripts/prepare_roi32_image_cache.py --manifest data/registered_roi32/manifest.json --output data/registered_roi32/image_cache_20261004 --report reports/roi32_image_cache.json
python scripts/run_registered_roi32_experiments.py --suite a1_a2 --manifest data/registered_roi32/manifest.json --max-concurrent 1 --detach
python scripts/run_registered_roi32_experiments.py --suite a1_a2 --status
```

各run的`epoch_history.json`记录实际使用的`lr`、下一轮`next_lr`及是否下降。A1 scheduler状态保存在checkpoint；恢复同一队列使用原命令加`--resume`。缓存等值与采样一致不意味着不同GPU训练权重逐位相同。
