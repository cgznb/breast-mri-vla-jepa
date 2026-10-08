# 已配准三相 ROI32 接入

本入口读取已有的配准/裁剪ROI32数组，用于固定T0及动态pCR预测和JEPA未来状态学习。准备脚本不执行DICOM整理、配准或ROI提取。模型与输入边界见[架构](ARCHITECTURE_ZH.md)和[完整数据格式](DATA_ZH.md)。

## 数据与预处理

`prepare_registered_roi32.py`读取inventory的`visits`，按`patient_id`分组，解析`visit_id`阶段后缀，并相对inventory目录解析`file`。每名患者保留四槽，缺失检查写`null`，不把T0/T2/T3压缩成连续三次检查。

输入模式必须显式声明：

```json
{
  "image_preprocessing": {"mode": "preprocessed_roi32"},
  "phase_order": ["pre_aqc0", "first_post_aqc1", "metadata_late"]
}
```

每个非空visit指向一个NPZ。运行时只读取float32 `image[3,32,128,128]`，不读取VQ `latent`、`roi`或`support`。上游数组采用训练集DCE0共同均值/标准差，背景与填充保持零；本模式不再次标准化或插值。`image_normalization`随manifest及checkpoint保存，推理时必须匹配。原始NPY/NIfTI模式则采用T0 z-score/resize，两种处理不能在同一manifest中混用。

准备时检查形状、有限值、support/roi类型及维度、采集范围之外零背景、重复资产、阶段槽位、患者划分和临床联结。support仅为采集覆盖审核，不能等同于肿瘤mask，也不改变全图重构损失。上游已完成的图像统计依据声明校验，本包不会从缺失的原始前景mask重新拟合。

## 临床与信息边界

官方表按规范化患者ID联结，检查pCR/分配Arm冲突。保留年龄、HR、HER2、MP，缺失为`null`；该入口默认这四项基线字段T0已知，临床缩放只从训练患者拟合。Arm采用`assigned_arm_scenario`，把官方分配类别视为T0已知场景；未提供剂量、周期、阶段间治疗变化或分配时间证明。

模型共享主干只读取当前合法历史，未来MRI位于独立监督对象。普通pCR不读取EMA未来状态或world预测。`--forecast-states`另执行自主特征推演；正式28次实验MRI flow关闭。

历史队列764 train / 102 development val / 0 test、3039次检查，动态验证382个合法前缀。准备脚本对应历史inventory协议；接入其他队列需明确自己的患者划分和预处理来源，不能将开发验证称为独立测试。

## 准备、训练与评价

在仓库根目录执行，并替换自己的数据位置：

```bash
python scripts/prepare_registered_roi32.py --inventory /path/to/registered_roi32/inventory.json --clinical /path/to/official_clinical.xlsx --output data/registered_roi32/manifest.json --report reports/registered_roi32_preflight.json
python scripts/prepare_roi32_image_cache.py --manifest data/registered_roi32/manifest.json --output data/registered_roi32/image_cache_20261004 --report reports/roi32_image_cache.json
python train.py train-vla-jepa --config configs/registered_roi32_epoch200_dynamic_seed17.yaml --manifest data/registered_roi32/manifest.json --output runs/a0_dynamic_seed17
python train.py evaluate-vla-jepa --checkpoint runs/a0_dynamic_seed17/best.pt --manifest data/registered_roi32/manifest.json --split val --device cuda --output results/dynamic_val.json
```

正式epoch配置为B16、accumulation1、最多200轮、早停50。旧`registered_roi32_{t0,dynamic}.yaml`保留6000次更新、B1/accumulation4协议，不属于此次28次矩阵。正式配置文件与差异见[实验矩阵](EXPERIMENTS_ZH.md)。

评价提供T0–T3的AUROC、AUPRC、NLL、Brier。固定模型只接受T0；动态使用患者等权多阶段NLL选模。新v2 checkpoint保存配置、规范化状态、患者分区、manifest结构、源码文本快照和源文件元数据；严格恢复要求匹配，旧v1支持只读评估而不续训。文件元数据不是完整影像内容校验。

## 工程验证

```bash
python -m pytest -q
python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_t0.yaml --output runs/smoke_t0
python train.py smoke-vla-jepa --config configs/raw_vla_jepa_smoke_dynamic.yaml --output runs/smoke_dynamic
python scripts/verify_registered_roi32.py --manifest data/registered_roi32/manifest.json --output runs/registered_roi32_verification --steps 4
```

前两个smoke使用虚构CPU数据；最后命令需要真实ROI32输入，检查完整尺寸前向/训练和阶段评价。短程checkpoint仅用于工程验证。公开副本的当前验证见[验证记录](VALIDATION_ZH.md)，历史正式指标见[聚合结果](../results_summary/README.md)。真实manifest、缓存、患者预测及checkpoint不随源码发布。
