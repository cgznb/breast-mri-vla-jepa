# 来源与提取范围

本实现来源于 `cgznb/breast-world-model-v3-pcr` 中新增的 raw VLA-JEPA 医学适配代码：
https://github.com/cgznb/breast-world-model-v3-pcr

独立包于2026-10-03从当时的开发代码中提取，随后加入ROI32接入、200-epoch协议、A1/A2数据管线与L1–L4消融/诊断。本次发布整理已有源码、配置、测试和聚合结果。上游GitHub仓库不一定包含当时尚未提交的新增文件；该链接说明项目来源，不指定可逐值复现本包的上游commit。

参考思想来自VLA-JEPA：
https://github.com/ginwind/VLA-JEPA

医学适配保留共享主干中的状态转移查询与任务查询，以及独立世界预测器中的真实状态右移训练。观察输入是3D MRI，目标是EMA MRI encoder状态，任务为最终pCR。包内没有原版Qwen3-VL/V-JEPA2实现或预训练权重；原项目legacy codec、DiT、MONAI与旧V3模型也不在本包中。

许可和版权声明见仓库根目录[LICENSE](../LICENSE)。PyTorch等依赖保留各自许可。真实I-SPY2影像、临床表及其数据使用条件不由代码许可代替，本包仅提供虚构格式示例及不含患者记录的聚合结果。

数据、配置和checkpoint保留`responsewm_*` schema字符串以兼容已有格式；运行模块命名空间是`mri_vla_jepa`。源码快照、数据身份与配置用于严格恢复校验。旧项目其他模型的checkpoint不能直接用于本实现；本模型历史checkpoint按兼容规则可只读评估，修改或移动源码后的训练恢复需另行核验，不能把新配置接到旧run续训。
