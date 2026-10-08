# MRI 数据格式与信息可见性

本包直接读取已整理的三相位 3D DCE MRI。一个患者保留 T0、T1、T2、T3 四个纵向槽；每个槽是一整次检查，不是单张切片。数据入口是 `src/mri_vla_jepa/data.py` 的 `RawMRIStore`，输入与监督合同定义在 `src/mri_vla_jepa/contracts.py`。

已配准 ROI32 NPZ 使用显式 `image_preprocessing.mode="preprocessed_roi32"`：只读取 `image`，要求 float32 `[3,32,128,128]`，保留已完成的训练集 DCE0 统一标准化，跳过本页原始 NPY/NIfTI 路径的 T0 z-score 和 resize。具体合同见 [ROI32 适配说明](REGISTERED_ROI32_ZH.md)。未声明该模式的旧 manifest 继续使用原始影像处理。

## 1. MRI 索引、患者划分与训练 manifest

准备流程为：

```text
MRI index：患者 ID、源网格声明、四槽 MRI 路径
                    +
患者 split：明确的 train / val / test ID 列表
                    +
官方临床 CSV/XLSX：临床值、Arm、最终 pCR
                    ↓
scripts/prepare_raw_mri_jepa.py
                    ↓
responsewm_raw_mri_v1 manifest
                    ↓
RawMRIStore.batch([(patient_index, landmark), ...])
                    ↓
RawMRIInput / RawMRISupervision
```

保留 `responsewm_...` schema 名称是为了兼容已有数据文件，运行代码只依赖本包 `mri_vla_jepa`。

MRI 索引的 schema 为 `responsewm_raw_mri_index_v1`。顶层包含 `schema`、可选 `synthetic` 和 `patients`；每个患者至少提供 `patient_key`、`geometry`、`visits`。准备脚本加入 `split`、`clinical`、`arm` 和 `target`。示例见 [MRI 索引](examples/raw_mri_index.example.json) 与 [患者划分](examples/patient_split.example.json)，二者使用同一组虚构患者。

split 文件必须恰好包含 `train`、`val`、`test` 三个键，值为患者 ID 列表；开发时 `test` 可以为空。三个列表不能重叠，其并集必须与 MRI 索引的患者集合完全一致。同一患者的所有随访始终属于同一 split。准备脚本按规范化 ID 联结临床表：例如 `ISPY2-123456` 和 `123456` 视为同一个患者；数值 ID `123456.0` 规范化为 `123456`。重复 ID、跨患者重复使用同一 MRI 路径会被拒绝。

训练 manifest 的顶层字段如下。

| 字段 | 确切内容 |
|---|---|
| `schema` | `"responsewm_raw_mri_v1"` |
| `time_basis` | `"stage_index"` |
| `canonical_stages` | `[0, 1, 2, 3]` |
| `phase_order` | `["pre_aqc0", "first_post_aqc1", "metadata_late"]` |
| `clinical_features` | `["age_at_screening", "hr_positive", "her2_positive", "mammaprint_binary"]` |
| `synthetic` | 合成数据标记；为 `true` 时须显式允许合成数据 |
| `patients` | 非空患者对象列表 |
| `provenance` | 准备脚本记录源文件路径、大小、修改时间，以及时间可用性和 Arm 场景声明；新产物不记录 checksum 值 |

`time_basis="stage_index"` 表示时间由 0–3 的阶段编号表达。当前数据合同没有日期、实际间隔天数、剂量或治疗周期字段。

## 2. 三个 DCE phase 与原始 MRI 文件

每次检查的三个输入通道顺序固定为：

| 通道 | 名称 | 含义 |
|---|---|---|
| 0 | `pre_aqc0` | 本次检查的增强前相位 |
| 1 | `first_post_aqc1` | 本次检查的第一个增强后相位 |
| 2 | `metadata_late` | 由上游元数据确定的晚期相位 |

这三个通道属于同一次 MRI；四个 T0–T3 槽表示四次纵向检查。`metadata_late` 的选取需要上游正确整理，本包不会从 DICOM 自动寻找晚期序列。上述字符串的拼写必须与 `constants.py` 一致。

每个非空 visit 选择以下一种路径表示，`image` 与 `phases` 不能同时存在：

```json
{"stage": 0, "available_at": 0, "grid_id": "EXAMPLE_T0_RAS", "image": "MRI/T0.npy"}
```

或：

```json
{
  "stage": 0,
  "available_at": 0,
  "grid_id": "EXAMPLE_T0_RAS",
  "phases": ["MRI/T0_pre.nii.gz", "MRI/T0_first_post.nii.gz", "MRI/T0_late.nii.gz"]
}
```

| 表示 | 文件要求 |
|---|---|
| `image` | 一个 `.npy`，数组形状严格为 `[3,D,H,W]` |
| `phases` + NPY | 三个 `.npy`，每个为 `[D,H,W]`，三相位空间形状一致 |
| `phases` + NIfTI | 三个 `.nii` / `.nii.gz`，每个为独立三维 volume；不接受四维 DCE 文件 |

读取时转换为 float32，要求实际读取到的 MRI 强度有限；NPY 禁用 pickle。同一检查不能混用 NIfTI 与 NPY，同一患者的纵向检查也不能混用这两种几何表示。本包不含 DICOM 序列发现、DICOM 转 NIfTI、相位选择或 VQ 编码流程。

相对路径以 manifest 所在目录解析。准备脚本则先以 MRI index 所在目录解析原路径，再把绝对 MRI 路径写入生成的 manifest。因此移动准备脚本生成的 manifest 时，应同步维护其中的图像路径。

## 3. T0 source grid、ROI、RAS 与重采样

每个患者的 `geometry` 要求如下，所有非空 visit 的 `grid_id` 必须与其一致。

| 字段 | 必须满足 |
|---|---|
| `source_stage` | 整数 `0` |
| `source_only` | `true` |
| `grid_id` | 非空的患者源网格标识 |
| `orientation` | `"RAS"` |
| `roi_source` | `"T0"` |
| `resampling` | `"source_grid"` |

这里规定的是**上游数据整理的信息边界**：目标网格、裁剪区域若存在、ROI 都应仅依据 T0 定义。不能先查看 T3 肿瘤、利用未来 mask 决定 T0 裁剪，再把它声明为 source-only。

本包校验这些声明及网格 ID 的一致性，但不能从 JSON 证明 ROI 的真实来源。准备脚本只联结表格、整理路径、检查 manifest 结构并构造临床标准化状态；它不会打开 MRI 做裁剪、检查 NIfTI header，或重采样并另存整批图像。图像存在性、实际形状和强度在相应 `.batch()` 读取时检查。

对 NIfTI，`RawMRIStore._read_scan` 执行：

1. 使用 `nibabel.as_closest_canonical` 将各相位转为接近 RAS 的轴顺序。
2. 以重定向后的 **T0 第一个相位**的 shape 和 affine 作为源参考网格。
3. 其他相位或后续检查若 shape/affine 不同，通过 `resample_from_to(..., order=1)` 映射到该源网格。
4. NIfTI 的数组轴从 `[X,Y,Z]` 转为模型使用的 `[D,H,W]`，堆叠为三个通道。

RAS 重定向不等于轴向去斜或解剖配准。这里的重采样使用文件已有的 affine；代码不估计刚性、仿射或非刚性纵向变换，也不校正患者体位变化。共享网格不自动保证各阶段的肿瘤和组织逐体素对齐。

NPY 没有 NIfTI affine。NPY 的纵向检查必须已在上游整理为同一 T0 网格，代码检查其空间形状相同；相同的数组尺寸和 `grid_id` 仍不能证明物理位置对齐。

最后，MRI 经标准化后通过 trilinear interpolation 缩放到训练配置 `image_shape=[D,H,W]`。正式配置为 `[32,128,128]`。这是进入网络前的固定尺寸调整，不是纵向配准，也不输出具有原始空间 affine 的医学图像。准备脚本的 `--image-shape` 不会离线改写 MRI；实际训练尺寸由训练配置决定。

## 4. 阶段可用性、临床与 Arm

`visits` 必须是长度为 4 的列表。位置 0、1、2、3 分别对应 T0、T1、T2、T3；缺失检查保留 `null`，不能将后面的检查前移。T0 必须真实存在且 `available_at=0`。

每个非空 visit 的 `stage` 必须是等于槽位的整数；`available_at` 是该检查何时可以进入预测输入的整数阶段，满足 `stage <= available_at <= 3`。例如 T1 检查若延迟到 T2 才可用，可写 `stage=1, available_at=2`：T1 预测输入不能读取它，T2 才可以读取它。

在预测阶段 `k`，可见 MRI 满足：

```text
visit 存在，并且 stage <= k，并且 available_at <= k
```

动态训练的 `allowed_landmarks` 返回实际非空检查的 `available_at` 去重排序结果；标准及时可用的完整随访为 `[0,1,2,3]`，缺失 T2 的及时随访为 `[0,1,3]`。推理可显式指定某个阶段，即使该阶段没有当前 MRI，也可用合法既往历史做 pCR；若请求未来自主状态推演，则必须在当前 landmark 有真实 MRI。T3 没有未来状态需要推演。

用于前瞻 T3-pCR 预测时，还需要在真实病历中核实 T3 MRI 的时间早于最终 pCR 病理终点；阶段编号本身不能验证日期。

临床表必须包含以下列，列名区分大小写：

```text
Patient_ID, Arm, HR, HER2, MP, pCR, Age_at_Screening
```

准备后每个患者的 `clinical` 包含两个等长四元素列表：

| `values` / `known_at` 位置 | 字段 |
|---|---|
| 0 | `age_at_screening`，来自 `Age_at_Screening` |
| 1 | `hr_positive`，来自 `HR` |
| 2 | `her2_positive`，来自 `HER2` |
| 3 | `mammaprint_binary`，来自 `MP` |

HR、HER2、MP 保持官方 0/1 编码，不能把“没有记录”编码为 0。缺失值用 `null`，对应的 `known_at` 也必须为 `null`；存在的值需要明确整数阶段 0–3。预测阶段 `k` 只显示 `known_at <= k` 的字段。

`--clinical-known-at-stage` 会给**所有患者的全部非空临床字段**赋同一个阶段。它不会从表格推断实际日期。若各患者或字段的可用阶段不同，应在正式 manifest 中逐项填写真实的 `clinical.known_at`。

`arm` 恰好包含 `label`、`known_at_stage`、`semantics`：

- `label` 必须匹配 `arm.py` 的 13 个官方字符串之一；缺失用 `null`。
- `known_at_stage` 是该 Arm 当时已知的整数阶段；未知 Arm 必须用 `null`。
- `semantics` 是 `"prospective_verified"` 或 `"assigned_arm_scenario"`。

前者表示已核实在所声明阶段可获得该信息；后者表示实验把官方分配 Arm 作为给定治疗场景。官方表没有给出 Arm 分配时间、剂量、周期或阶段间治疗变化，准备脚本不推断这些信息。`--arm-known-at-stage` 与 `--arm-semantics` 是调用者的明确声明。

manifest 存 Arm 字符串；模型输入转换为 ID 1–13。未知或截至 `k` 尚不可用时，输入为 `arm_id=0, arm_mask=false, arm_known_at=-1`。

## 5. 标准化只使用何时可见的信息

临床标准化只从 **train split 且 T0 已知的临床值**拟合。年龄使用该训练人群的均值和总体标准差；HR、HER2、MP 保持 0/1，均值/尺度固定为 0/1。缺失值不参与统计；标准差很小时使用 1。验证集、测试集和更晚才可用的值不参与拟合。

`normalization_state()` 返回 `raw_mri_normalization_v2`，其中包含字段顺序、临床均值/标准差、有效计数和显式影像预处理模式。训练 checkpoint 保存这组坐标，评估和预测复用它。旧 v1 状态支持只读加载。仅含 val/test 或新患者的 manifest 不能自行拟合训练标准化，需要传入 checkpoint 保存的 `normalization_state`。

MRI 强度标准化按患者、按相位计算：

```text
mean_p / std_p = 该患者 T0 相位 p 的全部已整理体素的均值 / 标准差
normalized_I_t,p = (I_t,p - mean_p) / std_p
```

标准差小于 `1e-6` 时设为 1。同一患者 T1–T3 使用同一组 T0 参数；代码不利用未来扫描重新估计标准化，也不对每次扫描独立做 z-score。这里的统计范围是输入的整个 volume；若上游已裁剪，则是该 T0 裁剪区域，不是代码自动生成的肿瘤 mask。

`image_shape` resize 在上述强度标准化之后执行。导出的可选生成数组处于这套 T0 标准化和网络固定尺寸坐标中，需要额外流程才能恢复为具有医学空间信息的 MRI 文件。

## 6. 一个合法的虚构 patient manifest 对象

下面是 `patients` 列表中的一个对象，文件路径和患者 ID 全部虚构。它有真实 T0、T1、T3，缺失 T2；HER2 到 T1 才已知，MP 缺失，Arm 在 T1 作为给定场景可用。

```json
{
  "patient_key": "EXAMPLE-001",
  "split": "train",
  "geometry": {
    "source_stage": 0,
    "source_only": true,
    "grid_id": "EXAMPLE-001_T0_RAS",
    "orientation": "RAS",
    "roi_source": "T0",
    "resampling": "source_grid"
  },
  "visits": [
    {
      "stage": 0,
      "available_at": 0,
      "grid_id": "EXAMPLE-001_T0_RAS",
      "image": "MRI/EXAMPLE-001/T0.npy"
    },
    {
      "stage": 1,
      "available_at": 1,
      "grid_id": "EXAMPLE-001_T0_RAS",
      "image": "MRI/EXAMPLE-001/T1.npy"
    },
    null,
    {
      "stage": 3,
      "available_at": 3,
      "grid_id": "EXAMPLE-001_T0_RAS",
      "image": "MRI/EXAMPLE-001/T3.npy"
    }
  ],
  "clinical": {
    "values": [42, 1, 0, null],
    "known_at": [0, 0, 1, null]
  },
  "arm": {
    "label": "Paclitaxel",
    "known_at_stage": 1,
    "semantics": "assigned_arm_scenario"
  },
  "target": {
    "pcr": 1,
    "label_source": "synthetic_example_endpoint"
  }
}
```

将该对象放入第 1 节的完整顶层 manifest，并提供实际的虚构测试数组，才能调用 `.batch()`。虚构队列应设置 `synthetic=true` 并显式允许合成数据。该例展示结构，不代表真实临床证据。

有监督训练时 `target.pcr` 是最终二分类终点 0/1，非空标签必须有非空字符串 `label_source`。缺失终点可以写 `pcr=null`；纯输入推理也允许省略 `target`。同一患者 T0–T3 的预测都监督同一个最终 pCR，不使用每次 MRI 各自的 pCR 标签。

## 7. `inp`、`sup` 与 mask 的精确含义

`store.batch(tasks, supervised=True)` 返回 `(inp, sup)`；`tasks` 中的 `patient_index` 是 manifest 的患者列表下标，`landmark` 是 0–3。模型的普通 `forward` 只接受 `inp`。

| `RawMRIInput` 字段 | 形状 / 类型 | 内容 |
|---|---|---|
| `images` | float32 `[B,4,3,D,H,W]` | 合法可见 MRI；其余槽全 0 |
| `observed_mask` | bool `[B,4]` | 真实存在且截至 landmark 已可用的检查 |
| `landmark` | int64 `[B]` | 每名患者的当前预测阶段 |
| `clinical` | float32 `[B,4]` | 标准化年龄、二元临床值；不可用值为 0 |
| `clinical_mask` | bool `[B,4]` | 区分真实 0 和未知/尚不可用的值 |
| `arm_id` | int64 `[B]` | 已知官方类别 1–13；未知/不可用为 0 |
| `arm_mask` | bool `[B]` | 截至 landmark 是否已知 Arm |
| `arm_known_at` | int64 `[B]` | 可见 Arm 的已知阶段；不可见为 -1 |
| `query_mask` | bool `[B,4]` | 请求哪些未来阶段；默认所有 `stage > landmark` |

| `RawMRISupervision` 字段 | 形状 / 类型 | 内容 |
|---|---|---|
| `future` | float32 `[B,4,3,D,H,W]` | 独立的真实未来 MRI 监督 |
| `future_mask` | bool `[B,4]` | 请求的未来阶段中，哪些有真实监督检查 |
| `label` | float32 `[B]` | 同一个最终 pCR；只有 label_mask=true 才有效 |
| `label_mask` | bool `[B]` | 是否有最终标签 |

`query_mask` 由预测协议决定，**不能由未来是否有检查决定**。否则模型会从查询槽位获知未来缺失模式。`future_mask` 则只决定损失可以使用哪些真实目标；它必须是 `query_mask` 的子集。

对第 6 节的虚构患者，在 T0：

```text
observed_mask = [1,0,0,0]
query_mask    = [0,1,1,1]
future_mask   = [0,1,0,1]
clinical_mask = [1,1,0,0]
arm_id=0, arm_mask=0, arm_known_at=-1
```

在 T1，`observed_mask=[1,1,0,0]`、`query_mask=[0,0,1,1]`、`future_mask=[0,0,0,1]`，HER2 与 Arm 已可见。T3 的 `query_mask` 与 `future_mask` 均全 false，pCR 任务仍有效。

缺失 T2 不会把 T1→T3 压缩成一个相邻转移。世界模型的损失还要求每一相邻对两端真实存在；本例 T0 只有 T0→T1 有效，T1 阶段没有有效未来相邻对。真实未来状态只进入独立 teacher-forcing 世界模型监督，不进入共享主干或 pCR 头。

合同要求所有 `inp` 值有限且隐藏输入槽全 0。`sup` 的有效目标/标签也必须有限；被 mask 排除的位置允许 NaN，损失先按 mask 选择再读取或归约。数据 loader 默认将无效监督位置填 0，但不能把这个 0 当作真实检查或阴性标签。

## 8. 缺失随访与输入推理的读取隔离

```python
from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import load_checkpoint

saved_normalization = load_checkpoint("runs/vla_dynamic/best.pt")["normalization"]
store = RawMRIStore(
    "data/new_patients/manifest.json",
    image_shape=(32, 128, 128),
    normalization_state=saved_normalization,
)
inp, sup = store.batch([(0, 0)], supervised=False)
assert sup is None
```

这里的 `saved_normalization` 来自训练 checkpoint；命令行 `predict-vla-jepa` 自动加载并复用它。

`batch(..., supervised=False)` 先构造合法历史、临床和 Arm，随后立即返回：不打开未来阶段 MRI 文件，也不访问 `target` 的 pCR。推理 manifest 仍需保留合法的四槽结构和路径/几何声明，未来文件可尚不存在，也可以用 `null` 表示尚无该次检查；两种情况下未来查询协议相同。

检查若已在当前阶段合法可见，推理需要其真实文件；不能将本应可见的错误路径当作缺失检查。训练或有监督 `.batch()` 需要实际未来目标文件，训练的 `signature()` 还会对所有已声明文件读取文件大小和修改时间。这些训练检查不属于普通输入推理的读取路径。

`supervised=True` 仅在独立的 `sup.future` 中读取 `stage > landmark` 且实际存在的检查，最终标签仅用于 loss/评估；共享主干只编码 `inp.observed_mask` 选中的 MRI。新随访到来后更新对应 visit 与可用阶段，再使用同一 checkpoint 重新预测即可，无需在线更新权重。
