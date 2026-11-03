# PatchCore 项目优化计划（面试导向）

> 目标：把这个项目从「能跑出数字的复现」升级成「**能经得住追问、有差异化、能现场演示**」的面试作品。
> 本文件是执行清单，按优先级排序，每项都标注了**缺什么、怎么验、验收标准**。
>
> 生成时间：2026-11-01　｜　基线版本：`3ba6936`　｜　目标版本：`v1.0-interview`

---

## 0. 现状盘点（已核实）

### 0.1 代码与仓库

| 项 | 状态 |
|---|---|
| 仓库 | `makahua/patchcore-mvtec`，10 commits，MIT，最新 `3ba6936` |
| 工作区 | 干净（`git status` 无改动），origin 已指向 GitHub |
| 代码 | `patchcore_demo.py` 364 行、`classify.py` 536 行、`analyze.py` 94 行、`tools/` 2 个脚本、`make_synthetic.py` |
| 结果 | `experiment_data/00_baselines/results_best/` 17 个文件（15 类 scores + experiment_data/00_baselines/results_best/results.csv/json），共约 139 KB；另有 6 个历史实验目录。**experiment_data/00_baselines/results_best/results.csv 已加配置列**（`size` / `neighbors` / `imagenet_norm` / `coreset` / `coreset_k` / `coreset_ratio` / `calib_ratio` / `thr_pct` / `seed` / `weights`），新跑的结果可自证配置；历史 7 个目录的 CSV 仍是旧列，需要靠资源指标反推 |
| 文档 | 有 `README.md` 和本计划 `docs/OPTIMIZATION_PLAN.md`；尚无 `vs-upstream.md`、`experiments.md`、`interview-qa.md`、`reproduce.md` |
| 测试 | 有 `tests/test_metrics.py`（11 条，纯 numpy，覆盖 AUROC 并列/单调性/定义式对拍、PR-AUC、记忆库维度自检）；**仍无 CI** |

### 0.2 计划盘点时的环境（2026-11-01）

> README 中的实验环境（Python 3.12、torch 2.7.0、RTX 3050 Laptop）是生成已提交结果时记录的环境；下表是计划盘点时检查到的本机环境。两者不是同一环境快照，重跑实验前需确认当前解释器与依赖，并将实际运行环境写入每次结果元数据。

| 项 | 实测值 | 影响 |
|---|---|---|
| Python | `E:\APP\Anaconda3\python.exe` → **3.13.9** | ⚠️ 与原实验的 3.12 不同 |
| conda 环境 | **只有 base**（无 `ad` 环境） | README 里 `conda create -n ad` 那套还没建 |
| numpy | 2.3.5 ✅ | 够跑 `classify.py report` |
| pillow | 12.0.0 ✅ | — |
| matplotlib | 3.10.6 ✅ | 可画图 |
| scipy | 1.16.3 ✅ | 可做 PRO / 高斯（比自写快） |
| scikit-learn | 1.7.2 ✅ | 可做 PRO / ROC |
| tqdm | 4.67.1 ✅ | — |
| **torch** | ❌ **未安装** | 跑不了建库/推理 |
| **torchvision** | ❌ **未安装** | 同上 |
| timm / onnx / onnxruntime / faiss / opencv | ❌ 未安装 | 见 §2 |
| GPU | RTX 3050 Laptop **4096 MiB**，driver 617.42 | 4 GB 显存是硬约束 |
| 磁盘 | C: 剩 67 GB / E: 剩 **355.7 GB** / D: **无此盘** | 数据放 E 盘 |
| git-lfs | 3.7.1 ✅ 已装 | 可拉上游预训练 bank |

### 0.3 缺失的数据与权重（关键）

| 缺什么 | 大小 | 现状 | 来源 |
|---|---|---|---|
| **MVTec AD 数据集** | 5.27 GB zip → 解压约 5 GB | 计划盘点时**未在约定路径找到**（当时检查了 `E:/MVTec`、`E:/MVTec_dl`；本机无 D 盘）。后续由 `experiment_data/06_demo_outputs/demo_out/scores.json` 的路径反查确认实际位置在 `E:\MVTec_dl\x\mvtech_anomaly_detection`（注意上游拼写是 mvtech） | 官方 / hf-mirror / OpenDataLab |
| **wide_resnet50_2 权重** | 275 MB（safetensors） | **计划盘点时未找到**（`torch_cache/` 被 .gitignore 排除） | hf-mirror `timm/wide_resnet50_2.racm_in1k` |
| **上游官方预训练 bank** | 6 套 × 15 类 × 13 MB ≈ 数 GB（LFS） | 没有 | `git lfs` 拉 `amazon-science/patchcore-inspection` |

> ⚠️ **这三样都不在仓库里**（MVTec 许可证禁止再分发、权重超 GitHub 单文件上限），所以**换台机器就得重新准备**。这是复现计划里最耗时的一环（约 1~3 小时，取决于网速）。

---

## 1. P0：先修「会被当场问穿」的问题（约 2 小时，零风险）

### 1.1 补复现来源与引用声明 ⭐ 最高优先级

**问题**：全仓库 grep `amazon / arxiv / 2106.08265 / Roth` **零命中**。README 第 309 行只写"论文里 PatchCore…"，没说这是谁的论文、官方实现在哪。

**风险**：面试官问「这是你自己写的还是复现的？」，或自己去搜到 `amazon-science/patchcore-inspection` 发现你没提。

**改法**：`README.md` 第 4 行后插入

```markdown
> **复现来源**：本仓库复现自 AWS 官方实现
> [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)（Apache-2.0），
> 对应论文 *Towards Total Recall in Industrial Anomaly Detection*（arXiv:2106.08265, CVPR 2022）。
> 代码为**按论文独立重写**（未使用其 `src/` 包、未引入 faiss/sklearn 依赖），
> 与官方实现的逐项差异见 [docs/vs-upstream.md](vs-upstream.md)。
> 官方 WR50 基线：image 99.2%（IM224）/ 99.30%（IM320）、pixel 98.1%（IM224）/ 97.80%（IM320）、PRO 94.4%；
> 本项目 320 配置为 98.05 / 97.09，对**同分辨率 IM320** 差 **1.25 / 0.71** 个点。PRO 未实现，不做对比。
> （99.6% 是官方 ensemble 配置，不能当单模型基线。）
```

**验收**：`grep -ri "amazon-science\|2106.08265" README.md` 有命中。

### 1.2 修掉 99.6% 的口径错配

**问题**：README 曾把 **99.6%** 写成普通 WR50 配置的对照。官方仓库中 99.6% 是最高配置结果；WR50 单模型的具体结果随配置变化，例如 IM224 约 99.2%、IM320 约 99.3%。比较时必须指明配置，不能将最高结果、单模型基线和不同输入尺寸混成一个数字。

**改法**：两处统一改成

```
官方 WR50 示例：IM224 image 99.2% / pixel 98.1% / PRO 94.4%；IM320 image 99.3% / pixel 97.8% / PRO 94.3%
官方最高配置：image 99.6%（配置不同，不作为 WR50 单模型直接对照）
本项目：image 98.05% / pixel 97.09%（实现与评测条件仍需逐项对齐后再归因）
```

**验收**：README 全文搜索 `99.6`，每一处都带了「Ensemble」限定词。

### 1.3 补 LICENSE 说明

上游 Apache-2.0，本项目 MIT。**独立重写用 MIT 合法**，但必须在 README「许可」节（第 412 行）补充：

```markdown
本项目为独立重写实现，未复制 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)
（Apache-2.0）的源代码；算法思想来自该论文。若后续引入其任何代码片段，将同步改为 Apache-2.0 并保留 NOTICE。
```

**验收**：README 许可节有这段。

---

## 2. P0：缺失的工具清单（补齐才能跑实验）

> 这台机器**目前跑不了任何建库/推理实验**。下表是完整清单，按「必须 / 该装 / 可选」分级。

### 2.1 必须装（跑通主流程的最小集）

```powershell
# 1) 建 conda 环境（README 里写了 ad，但本机只有 base）
#    版本与 README 对齐用 3.12（本机 base 是 3.13.9，torch 对 3.13 支持不完整，别在 base 上装）
conda create -n ad python=3.12 -y
conda activate ad

# 2) PyTorch（4 GB 卡用 cu121 就够，别追最新）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) 基础依赖（numpy/pillow 已有，但新环境要重装）
pip install numpy pillow matplotlib tqdm
```

| 包 | 版本要求 | 本机现状 | 用途 |
|---|---|---|---|
| torch | ≥2.0（建议 2.7.0+cu121） | ❌ 未装 | 建库、推理、训练 |
| torchvision | ≥0.15 | ❌ 未装 | `wide_resnet50_2` |
| numpy | ≥1.24 | ✅ 2.3.5 | 全程 |
| pillow | ≥9.5 | ✅ 12.0.0 | 读图 |
| matplotlib | 可选 | ✅ 3.10.6 | 画曲线 |

> ⚠️ **Python 3.13 兼容性风险**：本机 base 是 3.13.9，部分 torch 版本对 3.13 支持不完整。**建议新建 3.12 环境**（与 README 记录的实验环境一致），不要直接在 base 上装。

### 2.2 该装（做进阶实验才需要）

| 包 | 用途 | 触发任务 |
|---|---|---|
| `scipy` ✅ 已装 1.16.3 | 高斯滤波、PRO 积分 | P3 PRO 指标 |
| `scikit-learn` ✅ 已装 1.7.2 | `roc_auc_score`、`precision_recall_curve` | P3 PRO 指标、PaDiM 对比 |
| `timm` | 加载 `wide_resnet50_2.racm_in1k` 原始权重 | 权重准备（也可用现成 state_dict 绕过） |
| `onnx` + `onnxruntime` | 导出与延迟对比 | P2 部署实验 |
| `opencv-python` | 图像扰动（旋转/模糊/JPEG） | P1 鲁棒性压测 |
| `fastapi` + `uvicorn` + `python-multipart` | 服务化 demo | P2 工程包装 |
| `pytest` | 单元测试 | P1 工程质量 |

### 2.3 可选（对齐上游 / 加分项）

| 包 | 用途 | 是否必要 |
|---|---|---|
| `faiss-cpu` | 对齐上游 `IndexFlatL2` / `IndexIVFPQ` | ❌ 不必要，`torch.cdist` 在 2000 patch 下足够快 |
| `pretrainedmodels` | 上游 40+ backbone 之一（inception 等） | ❌ 不必要 |
| `wandb` | 在线日志 | ❌ 不必要，本地 CSV 更实在 |
| `pandas` | 结果表处理 | ⚠️ 可选，纯 numpy 也行 |

> **刻意不装的**：`anomalib`（本项目卖点就是"不依赖 anomalib 自己实现"）、`faiss`（会掩盖 coreset 的作用）、`wandb`（面试现场用不上）。**这三条要写进 README，是加分项。**

---

## 3. P0：缺失的数据集与权重（准备步骤）

### 3.1 MVTec AD

**磁盘预算**：E 盘剩 355.7 GB，**充足**（需要 11 GB：zip 5.27 GB + 解压 5 GB + 余量）。

**推荐路径**（按本机网络实测，见 README 第 55~82 行）：

```powershell
# 方案① hf-mirror 单文件直链（最省事，断点续传）
New-Item -ItemType Directory -Force E:\MVTec_dl | Out-Null
curl.exe -L -C - -o E:\MVTec_dl\mvtec.zip `
  "https://hf-mirror.com/datasets/hdtech/mvtech_anomaly_detection/resolve/main/mvtech_anomaly_detection.zip"

# 解压（有 7-Zip 用 7z x，比 Expand-Archive 快很多）
Expand-Archive E:\MVTec_dl\mvtec.zip -DestinationPath E:\MVTec_dl\x
```

> ⚠️ **注意**：盘点机器上 `D:` 盘不存在。README 与脚本示例已统一使用 `E:/MVTec`；换机器时请按实际数据路径修改。

**验收结构**（必须长这样，否则 `--data` 指错层）：

```
E:\MVTec\
└── bottle\
    ├── train\good\*.png          (209 张)
    ├── test\good\*.png           (20 张)
    ├── test\broken_large\*.png   (20 张)
    ├── test\broken_small\*.png
    ├── test\contamination\*.png
    └── ground_truth\broken_large\*_mask.png
```

**只要 bottle 一类就能跑通**（解压后只留 `bottle/`，省 4 GB 和时间）。

### 3.2 权重（wide_resnet50_2）

**为什么不直接用 torchvision**：本机实测（README 第 105~118 行）`download.pytorch.org` 走不通，改从 hf-mirror 拿 timm 版权重。

```powershell
# 1) 下 timm 权重（safetensors，275 MB，断点续传）
python tools\dl_mvtec.py `
  "https://hf-mirror.com/timm/wide_resnet50_2.racm_in1k/resolve/main/model.safetensors" `
  torch_cache\wr50.safetensors

# 2) 转成 torchvision 能吃的 state_dict（自解析，无 safetensors 依赖）
python tools\safetensors_to_pth.py torch_cache\wr50.safetensors torch_cache\wide_resnet50_2_timm.pth
```

**验收**：加载时打印 `[weights] 命中 318/320 层`（fc 类被丢，正常）。命中率 <50% 脚本会报错退出，不会静默用错权重。

> **坑（已记录）**：`BobMcDear/wide_resnet50_2` 那个 `.pickle` 是 **Flax/JAX** 权重（键只有 params/batch_stats），torchvision 只命中 53/320，**别用**。

### 3.3 （可选）上游官方预训练 bank

只有想做「同一批图，官方 bank vs 我的 bank 直接对比」时才需要：

```powershell
cd E:\tmp
git lfs install
git clone https://github.com/amazon-science/patchcore-inspection.git upstream
# models/ 下 6 套 × 15 类的 .faiss + .pkl 会一并拉下来（数 GB）
```

> ⚠️ 上游是 **Apache-2.0**，只能本地对照，**不能把它的 bank 或代码提交进你的仓库**。

---

## 4. P0：先建立可比较的技术基线（约 1 天，需 GPU + 数据）

> 目标不是先追分，而是确认当前实现与论文/官方实现的差异，并建立后续实验可复用的评测协议。上游结果只有在 backbone、输入尺寸、coreset、预处理、后处理和指标口径匹配时才可直接比较。

### 4.0 实现对齐审计与实验协议 ⭐ 新增门槛

#### A. 建立实现差异表

逐项记录“官方实现 / 当前实现 / 差异 / 是否影响指标 / 后续验证方式”，至少覆盖：

| 模块 | 核对内容 | 需要固定或记录的配置 |
|---|---|---|
| Backbone | 具体 WideResNet50-2 权重来源、权重文件哈希、分类头处理 | 权重名、SHA256、torch/torchvision 版本 |
| 输入 | RGB/BGR、像素范围、ImageNet mean/std、resize/crop、插值方式 | 输入尺寸、变换顺序、插值和 antialias |
| 特征 | 提取层、层间空间对齐、通道映射/降维、patch 化与邻域聚合 | layer2/layer3、各阶段张量形状 |
| Memory bank | 候选池采样、随机投影、coreset 算法及选择空间 | 候选数、保留数/比例、随机种子 |
| 检索与打分 | 距离类型、最近邻数、分块方式、图像级打分及是否重加权 | k、距离、score 定义 |
| 像素图 | patch 网格到输入分辨率的插值、平滑先后顺序与核参数 | 输出尺寸、插值、sigma、kernel size |
| 评测 | image/pixel AUROC、PR-AUC、PRO 的实现、macro/pooled 聚合 | 正负样本定义、mask 对齐、聚合方式 |

官方 PatchCore 的 image-level scoring 包含基于最近邻的 reweighting；当前实现如使用 patch 距离最大值直接作为图像分数，应记录为算法差异并单独做消融，不把当前分数直接称为与官方完全同口径。参考官方 [PatchCore scoring 实现](https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/common.py) 与 [特征提取实现](https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/patchcore.py)。

#### B. 固定数据与评测协议

1. **数据角色分开**：`train/good` 只用于建 memory bank；从 good 图中独立划分 calibration 集，只用于定阈值；如需选择阈值策略/成本参数，使用 validation 集；最终 test 集只用于一次最终汇报。
2. **分割单位是图像**：同一张图及其 mask 不得跨集合；保存各类别的文件清单与划分种子，避免不同实验悄悄更换样本。
3. **少样本口径明确**：定义 N 为“建库图数”，calibration 图另计；从训练候选图中先划分 calibration，再按 N 取建库图，并保存实际图像清单。否则不同 N 会同时改变建库量与标定量。
4. **低误报目标报告不确定性**：每类给出 calibration 样本数、目标分位数、实际误报数/样本数；0.5% 误报目标若样本量不足以分辨，应标记为“无法可靠估计”，不得把目标值写成实测保证。阈值结果跨多个固定种子重复，报告均值、标准差或置信区间。
5. **成本参数不在 test 上调优**：`C_fn/C_fp/C_review` 和 `thr_ok/ng_ratio` 的选择只使用 validation 标签；test 只用于最终成本与检测指标报告。明确成本是示例假设，不代表通用产线成本。
6. **扰动实验报告两种工作方式**：A. 原阈值冻结，测分布漂移影响；B. 使用独立良品重新标定后再测，衡量重新标定的恢复能力。不能在扰动后的 test 集上重新挑阈值。
7. **性能测量固定条件**：预热后重复测量，记录 batch size、输入尺寸、设备、同步方式、吞吐量、延迟中位数/p95 和峰值显存；分开报告建库与推理耗时。

**本节验收**：生成配置清单（建议 JSON）、数据划分清单、上游差异表；同一配置重复运行可复现同一分数，或能量化随机波动。未通过前，不将后续结果称为“提分”或“上游对齐”。

---

### 4.1 像素图后处理：复现对照后再扫 sigma

**位置**：`patchcore_demo.py:116-124` `gaussian_blur(k=5, sigma=2.0)`

**待验证假设**：当前实现先在 /8 patch 网格上以 `sigma=2.0, k=5` 平滑，随后用于像素评估/热图；官方实现是在上采样到输入分辨率后再平滑。两者的插值顺序、核截断和边界行为不同，不能仅按网格步长将 sigma 换算后断言“平滑半径是官方 4 倍”。先核对两条完整处理链，再通过实验验证过度平滑是否导致小缺陷定位下降。

将当前结果和上游后处理流程作为两个明确的配置对照，记录输入尺寸、patch stride、插值模式、平滑分辨率、sigma、kernel size 及边界策略。初始关注 wood/tile/screw，但最终只报告观测结果，不预设一定改善。

初始关注 wood/tile/screw 的原因是当前结果显示这些类可能存在定位差距。正式比较前，必须在此表补齐本项目和上游的完整配置（权重、尺寸、coreset、预处理、后处理、指标聚合）；如果配置不匹配，数字只作为线索，不称为同口径差值。

**改法**：把 `sigma` 参数化，扫一遍

```python
ap.add_argument('--blur-sigma', type=float, default=2.0,
                help='patch 分数图高斯平滑 sigma（作用在 /8 网格上，1.0≈原图 8px）')
```

**实验设计**：只跑 wood / tile / screw 三类，sigma ∈ {0.5, 1.0, 1.5, 2.0}

**验收标准**：输出逐类 pixel AUROC、PRO（若实现）、异常区域可视化及延迟对照；若有提升，需跨种子或重复运行仍成立。若未提升，保留负结果并撤回“平滑过强”结论。不能只用一类的单次 +1pt 作为默认改动依据。

**成本**：改代码 10 分钟 + 跑实验 20 分钟。

### 4.2 9 邻域显存优化：比较不同 coreset 空间 ⭐ 候选研究点

**位置**：`patchcore_demo.py:217` 与 `:220-223`

**问题**：现在顺序是「先展开 9 倍 → 再 coreset 抽样」

```python
# 当前（:217 + :220-223）
emb, _ = embed(feat, args.neighbors)      # 先展开成 9 倍（13824 维）
mem = coreset_greedy(bank, args.greedy)   # 再抽样到 2000
```

> **⚠ 实测更正（2026-11）**：`experiment_data/00_baselines/results_neigh/` 已经用这条「先展开再 coreset」的现有顺序在 grid 上**完整跑通，没有 OOM**
> （18,944 patch、峰值 2,353 MB、360 ms）。所以「直接 OOM」这个说法只适用于某些更重的类别/配置，不能当成本项目的既定结论。
> 真实代价是：与同配置无邻域的 grid（`experiment_data/00_baselines/results_fix/`，0.9373 / 30.0 ms / 449.4 MB）相比，
> 精度 **−1.8 点**、时延 **12×**、显存 **5.2×**。也就是说这条优化要解决的不是「跑不跑得起来」，而是「跑起来了也不划算」。

**候选优化**：如果先在原始描述子空间压缩到 2000，再对保留 patch 展开邻域，理论上可以减少展开阶段的内存。但这会改变 coreset 所处的特征空间和选点结果，**不等价于**在 9 邻域拼接后的描述子空间执行 coreset；因此应作为新的近似策略做精度/资源权衡实验，而不是宣称保留官方算法结果。

粗略张量预算（只计 float32 描述子，不含距离矩阵、特征、临时张量和框架缓存）：

```
2000 patch × 13824 维 × 4 B ≈ 105 MB
查询侧 2048 × 13824 × 4 B ≈ 113 MB
合计约 220 MB；这是描述子张量估算，不是 CUDA 峰值显存保证，实际峰值需实测。
```

**改法**：

```python
# 目标顺序
emb, _ = embed(feat, False)                    # 先在 1536 维上取特征，不展开
idx = coreset_indices(bank, args.greedy)       # 返回索引而非向量
mem = expand_neighbors(bank[idx])              # 只对 2000 个 patch 做 9 邻域展开
```

需要把 `coreset_greedy`（`:152-164`）拆成「返回索引」和「按索引取向量」两步。

**对照组**：无邻域；先扩展后 coreset（若可运行）；原维度 coreset 后展开。三者固定候选池、保留 patch 数、图像划分和种子，并同时测 image AUROC、pixel AUROC/PRO、推理延迟、建库时间与实测峰值显存。不同特征空间选点的 coreset 不应只通过单一 image AUROC 判定。

**验收标准**：在目标硬件上记录峰值显存且可稳定完成；相对无邻域基线的指标变化通过多次运行报告。官方结果仅作参考，不把“+0.3~1 个点”作为预期或验收门槛。

**面试话术**（这段要背）：
> 面试表述应以实测为准：说明原始实现在哪一步 OOM、候选策略改变了 coreset 特征空间、实际峰值显存和精度变化。不要在实验完成前把“显存降 9 倍”或“精度保持”写成结论。

**成本**：改代码 2 小时 + 验证 1 小时。

### 4.3 直接 resize → Resize + CenterCrop

**位置**：`patchcore_demo.py:51-54` `load_img`

**待验证假设**：直接 `Image.resize((size, size))` 会改变长宽比，可能影响纹理与缺陷形状；其是否解释 wood/tile 的 pixel AUROC 差距，需要在变换参数及 mask 对齐后通过消融确认。

**上游做法**：上游配置会先 resize 再 center crop；需要从目标配置及数据 transform 代码确认 resize 的尺寸语义、插值和 crop 参数后再复刻，不能只写成泛化的“保比例、裁边”。

**改法**：加 `--preprocess {stretch, center-crop}` 开关，做消融

```python
ap.add_argument('--preprocess', default='stretch', choices=['stretch', 'center-crop'],
                help='stretch=直接拉成正方形（旧行为）；center-crop=Resize+CenterCrop（上游做法）')
```

> ⚠️ 改预处理后**历史结果不可比**，必须整类重跑并单独存目录（如 `results_cc/`）。

**验收标准**：图像和 ground-truth mask 使用空间一致的变换；报告逐类 image/pixel 指标、被裁掉的边缘比例和延迟。若 crop 截掉缺陷或没有稳定收益，保留 stretch 基线，不以 wood/tile 必须提升作为预设结论。

**成本**：改代码 3 小时 + 跑全类 1 小时。

---

## 5. P1：能拉开差距的实验（2~3 天）

> 数字提 1 个点面试官记不住；**这三个实验才是让他记住你的东西**。

### 5.1 少样本曲线 ⭐ 最该补

**问题**：项目定位是「少样本无监督」，但 README 里**没有任何少样本实验**。面试官必问「产线只有 5 张良品图怎么办？」

**缺什么**：`--train-limit N` 参数（`patchcore_demo.py` 现在没有）

**改法**：

```python
ap.add_argument('--train-limit', type=int, default=0,
                help='只用前 N 张 train/good 建库（少样本实验）；0=全部')
```

在 `run_category` 里，`train_imgs` 划分出 calib 之后，再对剩余做 `[:N]` 截断。

**实验设计**：N ∈ {5, 10, 20, 50, 100, 全部}，跑 **bottle（快）/ screw（难）/ grid（纹理）** 三类。N 表示建库图数，calibration 图不计入 N；每个 N 使用固定的候选训练集和多个预先声明的随机种子。不要用“前 N 张”作为唯一选择方式，需用种子抽样并保存图片清单，避免排序偏差。

**预期产出**：

| N | bottle | screw | grid |
|---|---|---|---|
| 5 | ? | ? | ? |
| 10 | ? | ? | ? |
| ... | | | |

**待验证问题**：不同类别的性能是否随正常样本数增加而提升？提升是否在少量样本阶段更明显？不预填结果，结合跨种子波动讨论趋势和适用边界。

**成本**：3 类 × 6 档 = 18 次建库，每次 1~3 分钟，**约一下午**。

**验收标准**：一张 `docs/figs/fewshot.png` 曲线图，展示跨种子均值与波动范围，并附各 N 实际可用于建库/标定的图数；README 只写曲线支持的结论，不预设“10 张到 0.99”等结果。

### 5.2 鲁棒性压测 ⭐ 产线视角杀手锏

**问题**：MVTec 是实验室打光，产线不是。你现在所有数字都是"理想条件"。

**缺什么**：图像扰动模块 + `opencv-python`（或纯 numpy 实现）

**扰动设计**：

| 扰动 | 参数 | 考察什么 |
|---|---|---|
| 高斯噪声 | σ ∈ {5, 10, 20} | 相机噪声 |
| 亮度漂移 | ×{0.8, 0.9, 1.1, 1.2} | 光源衰减 |
| 高斯模糊 | σ ∈ {1, 2} | 失焦 |
| 旋转 | ±2°, ±5° | 上料偏移 |
| JPEG 压缩 | Q ∈ {90, 70, 50} | 传输压缩 |

**改法**：加可复现的扰动配置（类型、参数、随机种子），只作用于评测图像；mask 保持与几何变换一致。旋转等几何扰动必须同步变换 mask，亮度/噪声/模糊/JPEG 不改 mask。对每个扰动同时报告“冻结原阈值”和“独立良品重新标定”两种结果，绝不在扰动后的测试标签上调阈值。

**预期产出**：冻结阈值与重新标定两种条件下，逐扰动报告 AUROC、固定误报率下召回、阈值变化及置信区间；不预设模型一定会明显退化。

**分析要求**：把性能变化与输入归一化、亮度、噪声等因素分别做对照；扰动结果不能单独证明某个机制（例如归一化）是原因。先写假设，实验后再归因；可能出现性能稳定，也应如实报告。

**成本**：半天。

**验收标准**：`docs/robustness.md` + README 一节。

### 5.3 灰区成本最优化

**问题**：现在灰区是写死的 `thr_ng = thr_ok × 1.2`（真正生效的代码在 `classify.py:183-186` 的 `build` 与 `classify.py:362` 的 `report`；`classify.py:19-22` 只是说明口径的 docstring），**拍脑袋的常数**。

**改法**：引入成本模型。成本值是可配置的业务假设，不应把示例权重写成行业通用常数。

```
总成本 = 漏检数 × C_fn + 过杀数 × C_fp + 复核数 × C_review
```

在独立 validation 集上，给定示例成本（例如 `C_fn=100, C_fp=10, C_review=1`），**扫 `thr_ok` 分位 × `ng_ratio` 二维网格，找 validation 总成本最低点**；锁定策略后，在 test 集只评一次。另做成本权重敏感性分析，展示结论对假设变化是否稳健。

**改法**：`classify.py report` 加 `--cost fn,fp,review` 模式。

**预期产出**：热力图 + validation 选出的工作点 + test 集上的最终成本和置信区间。结论需明确成本假设、样本数和数据划分，例如“在给定成本假设下，validation 选出的策略在锁定 test 集上的成本变化为……”。

**成本**：半天（只需改 `report`，纯 numpy，本机就能跑）。

**验收标准**：`docs/cost-model.md` + README 一段。

---

## 6. P1：工程化与文档（2 天）

### 6.1 `docs/vs-upstream.md` ⭐ 差异化最大

**内容**：逐项对账表（上游官方 vs 本项目），包括：

1. **仓库层面**：star / commit / 许可证 / 维护状态
2. **代码层面**：backbone / 特征层 / 维度 / 邻域 / 归一化 / 预处理 / coreset / 检索 / 图像分数 / 像素后处理 / 指标 / ensemble / 工程形态（12 项逐条对比）
3. **数字层面**：macro 同口径逐类对比表（15 类 × image/pixel）
4. **差距归因**：5 个原因按影响排序
5. **各自独有**：你有他没有的（三档判定、标定集、双口径、逐图分数）、他有你没有的（PRO、ensemble、faiss、40+ backbone）

**为什么值钱**：市面上 99% 的"复现项目"只有一句「我复现了 PatchCore，AUROC 0.98」。能拿出**逐类逐指标的差异归因**，是研究能力 + 工程严谨性的双重证明。

**成本**：3 小时（数据已齐，见本文档 §7）。

### 6.2 README 重构

**问题**：414 行，信息密度过高，重点被埋。面试官/HR 只给 3 分钟。

**目标结构**：

```
README.md（精简到 ~120 行）
├── 一句话定位 + 3 个核心数字 + 复现来源引用
├── 一张架构图（backbone → patch 描述子 → 记忆库 → 最近邻 → 三档判定）
├── 结果表：官方 vs 本机（并列，差距可见）
├── 三档判定 + 产线曲线（独家亮点，放前面）
├── 3 个"我踩的坑"（自匹配偏差 / 9邻域显存 / 高斯平滑过强）
└── 链接到 docs/

docs/
├── vs-upstream.md      ← 逐项对账
├── experiments.md      ← 少样本曲线 + 鲁棒性压测 + 成本优化
├── interview-qa.md     ← 20 个问题的答案
└── reproduce.md        ← 从零复现（现 README 第 2-3 节内容）
```

**成本**：半天。

### 6.3 工程质量

| 项 | 现状 | 目标 |
|---|---|---|
| 测试 | ✅ `tests/test_metrics.py`（11 条，纯 numpy） | 补 `tests/test_rel_path.py`；对拍 sklearn 需要额外依赖，改为自实现定义式对拍 |
| CI | ❌ 无 | GitHub Actions：pytest + flake8（只跑不需要 torch 的部分） |
| 类型/风格 | 无 | 可选 ruff |

**验收标准**：`pytest` 全绿，GitHub 上有绿色 CI 徽章。

**成本**：半天。

---

## 7. 现成的对比数据（写文档时直接抄）

### 7.1 同口径（macro）总体对比

| 指标 | 上游 WR50@320 | 上游 WR50@224 | 本项目 | 备注 |
|---|---|---|---|---|
| image AUROC | **0.9930** | 0.9916 | **0.9805** | 差值仅作参考，输入/权重/实现细节尚未完全对齐 |
| pixel AUROC | **0.9780** | 0.9811 | **0.9709** | 差值仅作参考，后处理与 mask 流程需核对 |
| PRO | 0.9432 | 0.9437 | 无 | 本项目尚未实现 |

### 7.2 逐类对比（本项目 − 上游@320，正数=本项目更好）

| 类 | image 差 | pixel 差 |
|---|---|---|
| toothbrush | **−5.28** | +0.22 |
| pill | **−3.59** | +0.78 |
| screw | **−3.32** | −1.15 |
| zipper | **−2.97** | −1.27 |
| grid | **−2.84** | −3.75 |
| capsule | −1.76 | −0.03 |
| tile | +1.01 | **−4.51** |
| wood | −0.27 | **−5.19** |
| leather | +0.03 | −0.66 |
| carpet | −0.05 | −0.52 |
| transistor | +0.29 | **+5.00** ✅ |
| 其余 4 类 | ±0.15 内 | ±0.5 内 |

**关键洞察**：差距**高度集中**——图像级在细粒度小目标（toothbrush/pill/screw/zipper/grid），像素级在纹理类（wood/tile）。**这个分布本身就指向了原因**（见 §4.1 / §4.3）。

### 7.3 差距候选解释（待验证，不代表因果结论）

以下因素是实现审计与消融的候选项，当前不能按影响大小排序，也不能仅凭逐类相关性断言它们造成了指标差距：

1. coreset 保留数量、候选池采样及是否使用随机投影；
2. 当前 3×3 邻域平均与上游 patch/邻域聚合策略的差异；
3. ImageNet 输入归一化与 backbone 权重版本；
4. 直接拉伸、resize/crop 的变换差异；
5. 像素图上采样与高斯平滑顺序、尺度及核参数。

只有在固定其他配置、逐项消融并重复验证后，才把某个因素写成差距归因。

---

## 8. P2：可选加分项（时间充裕再做）

| 项 | 内容 | 成本 | 收益 |
|---|---|---|---|
| **PRO 指标** | 补 `full_pro` / `anomaly_pro`，与上游同表对比。`scipy`/`sklearn` 本机已装，几十行 | 半天 | 补齐 MVTec 标准列 |
| **PaDiM 对比** | 多元高斯 + 马氏距离，~50 行。讲"为什么 PatchCore 比 PaDiM 强" | 半天 | 多一个基线 |
| **FastAPI 服务** | `POST /detect` → `{score, verdict, heatmap_b64, latency}`；复用 yolo11 项目的 Vue3 骨架 | 2 天 | **现场演示** |
| **ONNX 导出** | `torch.onnx.export` + onnxruntime 延迟对比 | 半天 | 答"上线用什么跑" |
| **coreset 1% + 投影** | 对齐论文口径 | 半天 | image +0.3 点 |
| **换真实工件** | 拍 30 张正常螺栓/PCB 跑同一套流程 | 1 天 | 「公开数据集」→「真实工件」，杀伤力翻倍 |

---

## 9. 执行顺序与里程碑

> 技术顺序调整：先完成 §4.0 的上游差异审计和数据/评测协议，再执行单变量消融；完成后才讨论“提分”与最终工作点。所有实验记录配置、数据划分清单、随机种子、代码版本和资源测量方式。

### 里程碑 1：可信基线（1 天，无 GPU 可完成文档与协议部分）

- [ ] §1.1 补引用声明
- [ ] §1.2 修 99.6% 口径
- [ ] §1.3 补 LICENSE 说明
- [ ] §4.0 上游差异表、数据划分和指标协议
- [ ] §4.1 高斯平滑流程对齐与参数化（**改代码**，验证需 GPU）
- [ ] §6.1 `docs/vs-upstream.md`（含具体配置和差异归因，不只罗列数字）

**产出**：`v0.9-safe` —— 不会再被问穿。

### 里程碑 2：消融版（2 天，需 GPU + 数据）

- [ ] §3.1 下载 MVTec（E 盘）
- [ ] §3.2 准备权重
- [ ] §2.1 建 conda `ad` 环境
- [ ] §4.1 验证 sigma 消融
- [ ] §4.2 对比不同邻域描述子与 coreset 顺序（按新策略做实验，不视为等价重排）
- [ ] §4.3 输入预处理消融（图像与 mask 同步变换）

**产出**：`v0.95` —— 获得可复算的消融结果；无论提升、持平或下降，都记录配置与结果，不预设指标必须提升。

### 里程碑 3：泛化与决策版（2~3 天）

- [ ] §5.1 多种子少样本曲线（固定数据角色与样本清单）
- [ ] §5.2 鲁棒性压测（冻结阈值与重新标定分别报告）
- [ ] §5.3 validation 选择成本策略，test 锁定评估并做成本敏感性分析
- [ ] §6.2 README 重构
- [ ] §6.3 测试 + CI

**产出**：`v1.0-interview` —— 可以拿去面试。

### 里程碑 4（可选）：演示版

- [ ] §8 FastAPI + 前端
- [ ] §8 ONNX 延迟
- [ ] §8 真实工件

---

## 10. 风险与注意事项

| 风险 | 说明 | 应对 |
|---|---|---|
| **Python 3.13 兼容** | 本机 base 是 3.13.9，torch 对 3.13 支持不完整 | **必须新建 3.12 环境**（与 README 对齐），别在 base 上装 |
| **4 GB 显存** | 9 邻域、大 coreset 都可能 OOM | 实测 9 邻域在 grid 上**跑通了**（峰值 2353 MB，无 OOM），但精度 −1.8 点、时延 12×，见 README 结论 3；真正的收益需按 §4.2 重新实验 |
| **历史结果不可比** | 改预处理 / 邻域 / 归一化后，旧数字全废 | 每个实验**单独存目录**（`results_sigma1/`、`results_cc/`…），README 里注明配置 |
| **网络** | `download.pytorch.org`、`raw.githubusercontent.com` 本机不通 | 走 hf-mirror；curl 加 `-C -` 断点续传 |
| **MVTec 许可证** | CC BY-NC-SA 4.0，禁止再分发 | **绝不能提交数据进仓库**；`.gitignore` 已排除 |
| **上游许可证** | Apache-2.0 | 可以对照学习，**不能复制代码进 MIT 仓库** |
| **口径混用** | macro 0.9805 vs pooled 0.8812 差 9.9 个点 | 所有数字必须标口径，`analyze.py` 已强制同时打印 |

---

## 11. 三个必背的面试深挖点

1. **自匹配偏差**（README `:165-171`，埋在 165 行，**必须提到前面**）
   > 用建库图自评分定阈值 → 最近邻距离趋近 0 → 阈值系统性偏低 → 误报 28%。修法：留出 30% 良品只做标定。

2. **9 邻域显存**（§4.2）
   > OOM 出现在哪一步、改变 coreset 特征空间带来什么精度/资源取舍，以及实测是否值得采用。

3. **screw 为什么漏检 76.5%**（README `:289-291`）
   > /8 特征图上螺纹细划痕只占 1~2 个 patch → 被 3×3 平均抹平 → 验证方式（关掉平均 / 提分辨率 / 换 9 邻域）。

---

## 12. 三个必答的送命题

| 问题 | ❌ 错误答法 | ✅ 正确答法 |
|---|---|---|
| 「这是你自己写的还是抄的？」 | 「参考了论文」 | 「按论文独立重写的，没用官方 `src/` 包；逐项差异在 `docs/vs-upstream.md`，差异和指标影响按实现审计与消融结果说明」 |
| 「AUROC 0.98 为什么不如论文最高结果？」 | 「我显卡不行」 | 「先说明对比配置和指标口径；最高结果与 WR50 单模型配置不同。再依据逐项实现审计和同配置消融解释差异，避免把尚未验证的原因当结论」 |
| 「这个能上线吗？」 | 「能，AUROC 0.98」 | 「不能直接用。0.5% 过杀下漏检还有 20%，工业线要求 <1%。要么放宽过杀到 10%（漏检 4.1%），要么靠双阈值+复核兜底，要么上更高分辨率——三条路我都量化了成本和收益」 |

---

## 附录 A：本机环境快照（2026-11-01）

```
OS        : Windows
Python    : 3.13.9 (Anaconda, E:\APP\Anaconda3\python.exe)
conda env : 仅 base
已装      : numpy 2.3.5 / pillow 12.0.0 / matplotlib 3.10.6 / scipy 1.16.3 /
            scikit-learn 1.7.2 / tqdm 4.67.1 / openpyxl 3.1.5
缺失      : torch / torchvision / timm / onnx / onnxruntime / faiss / opencv
GPU       : NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 617.42
磁盘      : C: 67 GB free / E: 355.7 GB free / D: 不存在
git-lfs   : 3.7.1
```

## 附录 B：数据与权重清单（换机器必看）

| 项 | 放哪 | 大小 | 是否入仓 |
|---|---|---|---|
| MVTec AD | `E:\MVTec\` | 5 GB（zip 5.27 GB） | ❌ 许可证禁止 |
| timm 权重 | `torch_cache\wide_resnet50_2_timm.pth` | 275 MB | ❌ .gitignore |
| 记忆库 bank | `experiment_data/04_memory_banks/banks\<类>.npz` | 每类几 MB | ❌ .gitignore |
| 逐图分数 | `experiment_data/00_baselines/results_best\scores\*.json` | 139 KB | ✅ **已入仓**（唯一放行的产物） |
| 热力图 | `experiment_data/00_baselines/results*/heatmaps/` | 每张 ~175 KB | ❌ .gitignore |

## 附录 C：常用命令速查

```powershell
# 0) 只复算三档表（只要有 numpy，1 秒）
cd E:\面试项目文件夹\patchcore-mvtec
python classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2

# 1) 单类跑通（bottle，约 3 分钟）
python patchcore_demo.py --data E:/MVTec --category bottle --size 320 --greedy 2000 `
  --calib-ratio 0.3 --thr-pct 99 --weights torch_cache\wide_resnet50_2_timm.pth

# 2) 全 15 类
python patchcore_demo.py --data E:/MVTec --category all --size 320 --greedy 2000 `
  --calib-ratio 0.3 --weights torch_cache\wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/results_v2

# 3) 三档判定
python classify.py build --data E:/MVTec --category bottle --weights torch_cache\wide_resnet50_2_timm.pth
python classify.py eval  --bank experiment_data/04_memory_banks/banks/bottle.npz --data E:/MVTec --category bottle --sweep

# 4) 离线分析
python analyze.py experiment_data/01_mvtec_gpu/results_v2
```

> ⚠️ 盘点机器上 `D:` 盘不存在；README 和代码示例现已统一到 `E:/MVTec`，迁移到其他机器时需替换为实际路径。
