# 发布整理与验证

验证日期：2026-10-08。本次整理来自已有乳腺 MRI VLA-JEPA 工程的独立源码副本。

## 本次检查

环境为 Python 3.12.13、PyTorch 2.5.1+cpu，使用合成数据与临时目录：

```bash
python -m pytest -q
python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_t0.yaml --output runs/smoke_t0
python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_dynamic.yaml --output runs/smoke_dynamic
python -m build
python tools/check_publication.py
```

- 现有测试 **208 passed**，覆盖输入/未来信息边界、患者/阶段采样、损失开关、梯度、EMA、续训、缓存、调度器与队列。
- 两版 CPU/FP32 smoke 均完成 4 次更新，并保存 checkpoint、pCR 评价及自主 latent 状态推演。
- 根目录与包内资源的 34 对 YAML 配置逐字一致；依赖兼容性检查通过。
- 七份 CSV 独立核对原始聚合结果，保留指标精度，移除运行路径；没有逐患者预测或真实患者记录。

CPU BF16 的两项测试用于验证 scatter 与梯度的数据类型处理。测试局部关闭 MHA fused fastpath 和 oneDNN，并在结束时恢复：PyTorch 2.5 的 fused eval 路径可能漏判 CPU autocast，BF16 反向内核也依赖 CPU 指令集。这不修改模型实现或正式 CUDA BF16 设置。普通 CPU 配置使用 FP32。

源码包与 wheel 的构建和内容检查在发布前执行。GitHub CI 使用 Python 3.11、CPU PyTorch 2.5.1，执行相同测试、两个 smoke、发布内容检查与打包。CI 运行状态以 GitHub Actions 为准。

## 整理范围

保留网络与训练实现、全部实验配置和上游版权声明。数据准备入口改为显式指定 inventory/clinical，补充 `analysis` 绘图依赖，清理不适用的许可证组件说明与历史文档引用；增加完整源码包清单、CI 和提交内容检查。旧 6000-step 配置与实际 200-epoch 矩阵在文档中分别说明。

发布内容仅为源码、配置、测试、虚构格式示例和汇总指标。患者数据、manifest、逐患者预测、缓存、日志、checkpoint 与凭据不进入 Git。`results_summary/` 是历史实验的聚合快照，不依赖本次合成验证产生。

本次没有重新运行完整真实队列、真实 MRI 数据预检或 CUDA/BF16 门禁。CPU 合成验证说明工程流程可运行，不能替代真实数据复现或独立临床验证。历史结果使用 102 人开发验证集，没有独立测试集。
