# PatchCore 项目优化计划（面试导向）

> 目标：把这个项目从「能跑出数字的复现」升级成「**能经得住追问、有差异化、能现场演示**」的面试作品。
> 本文件是执行清单，按优先级排序，每项都标注了**缺什么、怎么验、验收标准**。
>
> 生成时间：2026-11-01　｜　首次基线 `3ba6936`；**最近复核 2026-11-04，当前提交 `d8c5c4e`**　｜　目标版本：`v1.0-interview`

---

> **读法**：§0 是当前实况（每次复核重写）；§1–§8 保留计划原文，只在标题下加「状态」行；§9 起为里程碑，附录 A/B/C 已按实况更新。

## 0.0 执行状态总览（2026-11-04 复核）

图例：✅ 已完成并验证　⚠️ 部分完成　❌ 未开始　🚫 已作废（有实测反证）

| 计划项 | 状态 | 证据 / 缺口 |
|---|---|---|
| §1.1 复现来源与引用 | ✅ | `README.md:11`、`README.md:321-322` 写明论文、arXiv 编号、AWS 官方实现 |
| §1.2 99.6% 口径 | ✅ | `README.md:291`、`:314-316` 明确标注 ensemble，并给出 1.25 / 1.55 两种差值 |
| §1.3 LICENSE 说明 | ✅ | README「许可」节已补：独立重写、未复制上游 Apache-2.0 代码；若引入则同步转 Apache-2.0 并保留 NOTICE |
| §2 依赖与运行环境 | ✅ | `.venv`：Python 3.13.9 + torch 2.7.0+cu128 + torchvision 0.22.0+cu128，CUDA 可用 |
| §3 数据与权重 | ✅ | MVTec 15 类在 `MVTec\mvtech_anomaly_detection\`；权重 `torch_cache\wide_resnet50_2_timm.pth` |
| §4.0 实现审计与评测协议 | ✅ | `docs/EXPERIMENT_PROTOCOL.md` + `docs/VISA.md` + 新增 `docs/vs-upstream.md`（10 项差异、2 项待量化、1 项实现缺陷） |
| §4.1 高斯平滑 sigma 消融 | ✅ | `--blur-sigma` 已参数化；`results_gpu_blur`、`results_gpu_blur_confirm` 已跑，结论写入 `docs/RESULTS_OPTIMIZATION_PLAN.md` |
| §4.2 coreset 顺序重构 | ✅ | 已加 `--neighbor-coreset base`（先选点再展开）+ `--neighbor-pad {zero,replicate,roll}`；grid 同预算：zero 比 roll +1.93 点，base 与 expanded 精度持平而建库 194→22 s、候选池 2.1 GB→236 MB；全量 185 图跑通（517 MB），9 邻域仍比不开差 0.34 点、时延 2.9×，默认关（`docs/neighbor-ablation.md`） |
| §4.3 resize → Resize+CenterCrop | ✅ | `--preprocess {stretch,letterbox}` + `--aspect-crop`，**GT mask 已用同一套几何变换**（`load_mask`，NEAREST + 灰边填 0）；六组重跑后 letterbox 的 pixel AUROC 三类全不差（+2.43 / +0.65 / +6.19 点），旧版「灰边污染定位」是 mask 未同步的假象 |
| §5.1 少样本曲线 | ✅ | `--train-limit` 已加；5/10/20/50/100/224 六档 + 固定 `split_seed=42` 复验 |
| §5.2 鲁棒性压测 | ✅ | 亮度×0.8 / 噪声 σ=0.05,0.10 / 模糊 1px 各跑「冻结阈值 + 重标定」两组；汇总 `docs/robustness.md`：亮度可救、噪声伤 max 聚合（image AUROC 0.9553→0.4782）、模糊不可救（0.4583）；旋转、JPEG 仍未实现 |
| P0-3 预处理对照 | ✅ | `--preprocess {stretch,letterbox}` + `--aspect-crop`；MVTec 全方图下两者逐位等价，4:3 模拟视野下（mask 已同步）letterbox 的 pixel AUROC +2.43/+0.65/+6.19 点、screw image +1.45 点，见 `docs/preprocessing-ablation.md` |
| P0-2 阈值统计保证 | ✅ | `tools/conformal_threshold.py` + `classify.py --conformal`；0/15 类可达 0.5%（需 199 张，实际 18–117 张），见 `docs/conformal-thresholds.md` |
| P1-1 端到端时延拆解 | ✅ | `tools/latency_breakdown.py` + `docs/latency-breakdown.md`；端到端 **35.5 ms @900²**（报告 19.7 ms 只含算法段），解码占 36.7%、CPU/IO 占 52%；分辨率曲线 700² 37.1 / 900² 42.2 / 1024² 62.4 / 2048² 130 / 4096² 447 ms（后两档为合成外推） |
| P1-2 CI + pytest + pyproject | ✅ | 新增 `pyproject.toml`、`.github/workflows/ci.yml`（metrics matrix + docs 断链）、`tools/check_links.py`；18 篇 md 28 条相对链接全有效 |
| P1-3 PRO/AUPRO | ✅ | `tools/pro_aupro.py` + `tests/test_pro.py`（8/8）+ `docs/pro-aupro.md`；AUPRO@FPR≤30% **0.9211 ± 0.0398**（@10% 0.8068）；tile image AUROC 1.0000 但 AUPRO 0.8181；与 image AUROC 相关系数 −0.019 |
| §5.3 灰区成本最优化 | ✅ | `report --cost FN,FP,REV --prevalence P`：validation 选点 5% × 1.5×，test 0.1814/件，比默认 0.5% × 1.2×（0.2206/件）省 22%；先验降到 0.1% 时选点退回 0.2% 过杀 |
| §6.1 `docs/vs-upstream.md` | ✅ | 已建，含 10 行差异表、缺陷认定、与官方数字的正确比较方式 |
| §6.2 README 瘦身到 ~150 行 | ✅ | 现 **125 行**；安装/数据下沉 `docs/install.md`，坑/简历/问答/复现/自检下沉 `docs/interview-notes.md` |
| §6.3 测试 + CI | ✅ | `pyproject.toml` + `.github/workflows/ci.yml`（py3.10/3.12 跑 23 条测试与 report 冒烟、md 断链检查）；本机无 PyPI 访问，pytest 只在 CI 里装 |
| §8 P2 加分项 | ⚠️ | PRO/AUPRO 已在 P1-3 补齐；PaDiM / FastAPI / ONNX / coreset 1% 投影 / 真实工件仍未做 |
| 计划外额外完成 | ✅ | `EXPERIMENT_PROTOCOL.md`、`FAILURE_ANALYSIS.md`、`REPRODUCE.md`、`VISA.md`、`INTERVIEW_DEMO.md`、`RESULTS_OPTIMIZATION_PLAN.md`、`tools/threshold_stability.py`、VisA 全流程 |

**一句话现状**：P0 三项、P1 四项（时延拆解、CI/pyproject、PRO/AUPRO、成本模型）、P2 两项（扰动补全、README 瘦身）全部完成；剩下的都是按需加分项（PaDiM / FastAPI / ONNX / coreset 1% 投影），不影响面试主线。

---

## 0. 当前实况（2026-11-04 复核）

> 本节每次复核整体重写。计划生成时（2026-11-01）的旧盘点已作废，需要回看可 `git show 3ba6936:docs/OPTIMIZATION_PLAN.md`。

### 0.1 代码与仓库

| 项 | 实况 |
|---|---|
| 仓库 | `makahua/patchcore-mvtec`，11 commits，MIT；本地 HEAD `d8c5c4e` 领先 `origin/main` 1 个提交 |
| 工作区 | 本次复核产生的文档改动尚未提交（`M README.md`、`M docs/*.md`、新增 `experiment_data/README.md`）；`mvtec.zip` 5.27 GB、`VisA_20220922.tar` 1.93 GB 未入库（`.gitignore` 覆盖） |
| 代码 | `patchcore_demo.py` 461 行、`classify.py` 548 行、`analyze.py`、`pyqt_demo.py`、`make_synthetic.py`、`tools/` 4 个脚本、`tests/test_metrics.py` |
| 结果 | `experiment_data/` 21.6 MB：00_baselines 7 组、01_mvtec_gpu 7 组、02_visa 5 组、03_triage 4 份、04_memory_banks 4 套、05_indexes 3 份索引（`all_results.csv` 170 行 × 38 列，由 `tools/index_results.py` 生成）、06_demo_outputs 2 组 |
| 配置自证 | `results_best/results.csv` 每行带 `size / neighbors / imagenet_norm / coreset / coreset_k / coreset_ratio / calib_ratio / thr_pct / seed / weights`；`05_indexes/all_results.csv` 由 `tools/index_results.py` 扫全部 86 个结果目录生成，列取并集（38 列，缺列留空） |
| 文档 | 18 篇 `docs/*.md` + README 125 行（`install.md`、`interview-notes.md` 已补齐原 README 的长尾内容） |
| 测试 | `tests/test_metrics.py` 15 条纯 numpy，`python tests/test_metrics.py` → 15/15 通过；CI 见 `.github/workflows/ci.yml`（本机 `.venv` 仍未装 pytest，因 PyPI 被代理阻断） |
| 已知缺陷 | ~~README 链接 `experiment_data/README.md` 断链~~ 已于 2026-11-04 新建该文件修复，待提交 |
| 磁盘 | `.venv` 6.2 GB、`MVTec` 5.0 GB、`VisA_raw` 1.8 GB、`VisA_pytorch` 1.8 GB、`torch_cache` 0.5 GB，全部被 `.gitignore` 排除 |

### 0.2 当前环境（2026-11-04 实测）

| 项 | 实测值 | 说明 |
|---|---|---|
| 解释器 | `.venv\Scripts\python.exe` → **Python 3.13.9** | 由 Anaconda base 3.13.9 创建；README 原写 3.12，已更正 |
| torch | **2.7.0+cu128** | `torch.cuda.is_available()` 为真 |
| torchvision | **0.22.0+cu128** | — |
| numpy / pillow | 2.3.5 / 12.0.0 | 够跑 `classify.py report` |
| GPU | RTX 3050 Laptop，**4096 MiB** | 4 GB 显存是硬约束，仍是所有设计的前提 |
| 权重 | `torch_cache\wide_resnet50_2_timm.pth`（约 267 MB） | timm `wide_resnet50_2.racm_in1k` 转换，加载命中 318/320 层 |
| 缺 | pytest、timm、onnx/onnxruntime、opencv、faiss | 只有跑 CI / 导出 / 旋转扰动才需要 |
| 其它 | 系统 `python` 是 `E:\APP\Anaconda3\python.exe` base 3.13.9 | 本项目一律用 `.venv\Scripts\python.exe`，base 不装 torch |

### 0.3 数据与权重（已就位）

| 项 | 实际位置 | 大小 | 入仓 |
|---|---|---|---|
| MVTec AD（15 类） | `MVTec\mvtech_anomaly_detection\` | 5.0 GB | ❌ 许可证禁止再分发 |
| timm WR50 权重 | `torch_cache\wide_resnet50_2_timm.pth` | 267 MB | ❌ 超单文件上限，`.gitignore` |
| 记忆库 | `experiment_data/04_memory_banks/banks*/<类>.npz` | 每类数 MB | ❌ `.gitignore` |
| 逐图分数 | `experiment_data/00_baselines/results_best/scores/*.json` | 139 KB | ✅ 唯一放行的产物 |
| VisA | `VisA_pytorch\`（整理后）、`VisA_raw\`（原始包） | 1.8 + 1.8 GB | ❌ `.gitignore` |

> ⚠️ **这三样（数据集、权重、记忆库）都不在仓库里**（MVTec 许可证禁止再分发、权重超单文件上限），换台机器就要重新准备，约 1~3 小时，取决于网速。

---

## 1. P0：先修「会被当场问穿」的问题（约 2 小时，零风险）

### 1.1 补复现来源与引用声明 ⭐ 最高优先级

> **状态（2026-11-04）：✅ 已完成。** `README.md:11` 与 `README.md:321-322` 已写清 PatchCore 论文、arXiv:2106.08265 与 `amazon-science/patchcore-inspection`，并注明「按论文独立重写、未复制上游代码」。

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

> **状态：✅ 已完成。** `README.md:291`、`:314-316` 已标注 ensemble，并给出与论文最高结果的 1.25 / 1.55 两种差值，不再出现「超过论文」的说法。

**问题**：README 曾把 **99.6%** 写成普通 WR50 配置的对照。官方仓库中 99.6% 是最高配置结果；WR50 单模型的具体结果随配置变化，例如 IM224 约 99.2%、IM320 约 99.3%。比较时必须指明配置，不能将最高结果、单模型基线和不同输入尺寸混成一个数字。

**改法**：两处统一改成

```
官方 WR50 示例：IM224 image 99.2% / pixel 98.1% / PRO 94.4%；IM320 image 99.3% / pixel 97.8% / PRO 94.3%
官方最高配置：image 99.6%（配置不同，不作为 WR50 单模型直接对照）
本项目：image 98.05% / pixel 97.09%（实现与评测条件仍需逐项对齐后再归因）
```

**验收**：README 全文搜索 `99.6`，每一处都带了「Ensemble」限定词。

### 1.3 补 LICENSE 说明

> **状态：✅ 已完成（2026-11-04）。** README「许可」节末尾已补独立重写声明与 Apache-2.0 回退条款。

上游 Apache-2.0，本项目 MIT。**独立重写用 MIT 合法**，但必须在 README「许可」节（第 412 行）补充：

```markdown
本项目为独立重写实现，未复制 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)
（Apache-2.0）的源代码；算法思想来自该论文。若后续引入其任何代码片段，将同步改为 Apache-2.0 并保留 NOTICE。
```

**验收**：README 许可节有这段。

---

## 2. P0：缺失的工具清单（补齐才能跑实验）

> **状态（2026-11-04）：✅ 已就绪** —— `.venv` 内 Python 3.13.9 + torch 2.7.0+cu128 + torchvision 0.22.0+cu128 已装好且 CUDA 可用，主流程可跑；唯一仍缺的测试依赖是 pytest。下表保留为完整清单，按「必须 / 该装 / 可选」分级。

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

> **状态：✅ 已就位。** 数据集与权重都已下载到位，实际路径见 §0.3；下面各小节保留原始准备步骤，路径按本机实测标注。

### 3.1 MVTec AD

> **状态：✅ 已下载并解压**，实际位置 `MVTec\mvtech_anomaly_detection\`（15 类齐全、5.0 GB），**不是**下文假设的 `E:\MVTec\`。

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

> ⚠️ **注意**：盘点机器上 `D:` 盘不存在。**2026-11-04 复核**：本机数据实际在 `E:\面试项目文件夹\patchcore-mvtec\MVTec\mvtech_anomaly_detection\`（上游目录名拼写是 `mvtech`），下文命令中的 `E:/MVTec` 只是「数据集根目录」的旧写法占位，执行前请替换；`python` 也请用 `.venv\Scripts\python.exe`。

**验收结构**（`--data` 必须直接指向含类别目录的这一层；下例是旧机器的目录名，本机该层为 `E:\面试项目文件夹\patchcore-mvtec\MVTec\mvtech_anomaly_detection\`）：

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

> **状态：✅ 已就位。** `torch_cache\wide_resnet50_2_timm.pth`（267 MB，timm `racm_in1k` 转 state_dict，加载命中 318/320 层）。

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

> **状态：✅ 完成（2026-11-04）。** 数据角色与评测协议见 `docs/EXPERIMENT_PROTOCOL.md`、`docs/VISA.md`；逐项差异见 `docs/vs-upstream.md`（10 行表 + 缺陷认定），其中「预处理」「权重」「PRO」三项仍待量化。

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

> **状态：✅ 已完成。** `--blur-sigma` 已参数化，`results_gpu_blur` / `results_gpu_blur_confirm` 已跑：screw 上 σ=0/0.5/1/2 → image AUROC 0.9158/0.9326/0.9543/0.9553（`docs/RESULTS_OPTIMIZATION_PLAN.md`）。

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
> **状态：✅ 已完成（2026-11-04，P3）。** 顺序重构已实现为 `--neighbor-coreset base`，填充缺陷已改为 `--neighbor-pad zero`；grid 上四组同预算对照见 `docs/neighbor-ablation.md`。结论：修完 bug + 换顺序之后 9 邻域「不再更差」，但仍不划算，默认保持关闭。

**位置**：`patchcore_demo.py:216-233`（`coreset_greedy`）+ 建库段（`expand_selected`，`:231-258`）

**问题（重构前）**：旧顺序是「先展开 9 倍 → 再 coreset 抽样」

```python
# 重构前（旧顺序）
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
> 面试表述应以实测为准：**本项目实测并没有 OOM**——「先 9 邻域展开再 coreset」在 grid 上跑通（18,944 patch、峰值 2,353 MB、360 ms），代价是精度 −1.8 点、时延 12×。正确说法是「候选策略改变的是 coreset 的特征空间，不只是省内存」；「显存降 9 倍」「精度保持」都还没验证，不能当结论。

**成本**：改代码 2 小时 + 验证 1 小时。

### 4.3 直接 resize → Resize + CenterCrop

> **状态：✅ 已完成（2026-11-04，v2）。** `load_img(path, size, preprocess='stretch', aspect_crop=0.0)` 支持 `stretch` / `letterbox` 与中心裁剪；`load_mask(path, size, preprocess, aspect_crop)` 对 GT mask 做**完全相同**的裁剪与补边（NEAREST、灰边填 0），像素指标口径已与图像一致；实测 MVTec 15 类全为方图，两模式逐位等价，故用 `--aspect-crop 0.75` 制造非方视野做对照（见 `docs/preprocessing-ablation.md`）。**没做的部分**：letterbox 仍不等于本计划设想的 Resize+CenterCrop（缺「短边缩放到 S 后中心裁」这一种），但它已不是「不可比的实验」，而是「另一种保比例策略」。

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

> **状态：✅ 已完成。** `--train-limit` 已加；5/10/20/50/100/224 六档 + 固定 `split_seed=42` 复验（`docs/RESULTS_OPTIMIZATION_PLAN.md`）。

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

> **状态：⚠️ 部分完成。** `--perturb-brightness / --perturb-noise / --perturb-blur / --perturb-calibration` 四个开关已实现；实测只跑了亮度（×0.8）与模糊（σ=1px），旋转与 JPEG 未实现。

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

> **状态：✅ 已完成（P1-4）。** `classify.py report --cost FN,FP,REV --prevalence P` 已实现：validation 选点、test 只报一次、先验扫描与权重敏感度；结论与热力图见 `docs/cost-model.md`。

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

### 5.4 端到端时延拆解

> **状态：✅ 完成（2026-11-04）。** 新增 `tools/latency_breakdown.py` 与 `docs/latency-breakdown.md`；端到端 **35.5 ms @900²**（1024² 62.4 ms、合成 4096² 447 ms），报告里的 19.7 ms 只含算法段；README 的时延列已统一标注「算法段」。

**问题**：`results_best/results.csv` 的 `latency_ms` 在 `patchcore_demo.py:314-325` 计时，只包住 `patch_scores()`——图早已解码、resize、搬进显存。拿它当产线节拍报，会低估 1.8 倍。

**做法**：单张拆 8 段（read / decode / resize / h2d / forward / embed / nn / post），bottle 83 张轮转 200 次、warmup 10、GPU 段前后 `synchronize()`；另测 batch 1/4/8 与一次性成本（权重载入、库载入、冷启动首张）。

**结果**：decode **12.99 ms（36.7%）** > forward 12.64（35.5%）> resize 4.40（12.7%）> nn 3.38（9.5%）> 其余 1.93；合计 **35.53 ms**（p95 36.86、抖动 1.33）。**CPU/IO 18.39 ms（52%）> GPU 17.15 ms（48%）**。单流 28.1 张/s；batch 8 的 GPU 段 14.85 ms/张（67 张/s），但串行解码把端到端卡在 ~36 张/s（估算）。一次性：权重 1.74 s、库 49 ms、冷启动首张 360 ms、稳态显存 354 MB。

**派生结论**：优化顺序要反着来——GPU 只占一半，ONNX/INT8 最多 −26%，而「多线程解码 + 预取队列」能到 ~20 ms 量级且实现成本低；`--prefetch` 线程池列入待办。

**验收标准**：`docs/latency-breakdown.md` + README 一节 ✅。

---

## 6. P1：工程化与文档（2 天）

### 6.1 `docs/vs-upstream.md` ⭐ 差异化最大

> **状态：✅ 完成（2026-11-04）。** `docs/vs-upstream.md` 已建：10 项差异、影响方向、是否已量化、核对方式（代码 / 论文）；并写明「不能拿 0.9805 直接对 0.996」的正确比较方式。

**内容**：逐项对账表（上游官方 vs 本项目），包括：

1. **仓库层面**：star / commit / 许可证 / 维护状态
2. **代码层面**：backbone / 特征层 / 维度 / 邻域 / 归一化 / 预处理 / coreset / 检索 / 图像分数 / 像素后处理 / 指标 / ensemble / 工程形态（12 项逐条对比）
3. **数字层面**：macro 同口径逐类对比表（15 类 × image/pixel）
4. **差距归因**：5 个原因按影响排序
5. **各自独有**：你有他没有的（三档判定、标定集、双口径、逐图分数）、他有你没有的（PRO、ensemble、faiss、40+ backbone）

**为什么值钱**：市面上 99% 的"复现项目"只有一句「我复现了 PatchCore，AUROC 0.98」。能拿出**逐类逐指标的差异归因**，是研究能力 + 工程严谨性的双重证明。

**成本**：3 小时（数据已齐，见本文档 §7）。

### 6.2 README 重构

> **状态：✅ 已完成（2026-11-04）。** README 由 600 行压到 **125 行**：
> - 安装、权重、MVTec 下载与目录 → `docs/install.md`（83 行）
> - 六个坑、简历写法、面试问答、30 分钟复现、自检与 CI → `docs/interview-notes.md`（118 行）
> - 主 README 只留：指标表、快速开始、六条关键结论、设计决策、诚实的边界、文件与材料索引。
> - 全部相对链接经 `tools/check_links.py` 校验通过（20 篇 md / 26 条）。

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

> **状态：✅ 完成（2026-11-04）。** `tests/test_metrics.py` 15 条全绿；新增 `pyproject.toml`（依赖/可选 extras/ruff/pytest 配置）、`.github/workflows/ci.yml`（`metrics` 在 py3.10 与 3.12 上跑测试 + `classify.py report` 冒烟、`docs` 跑断链检查）、`tools/check_links.py`。

| 项 | 现状 | 目标 |
|---|---|---|
| 测试 | ✅ `tests/test_metrics.py`（15 条，纯 numpy，含成本模型 4 条） | 补 `tests/test_rel_path.py`；对拍 sklearn 需要额外依赖，改为自实现定义式对拍 |
| CI | ✅ `.github/workflows/ci.yml`：install 只装 numpy+pytest → `pytest tests -q` → `classify.py report` 冒烟（grep 0.9805 / 0.8812）→ 断链检查 | 保持「不需要 torch 也能进 CI」，整套十几秒 |
| 类型/风格 | ✅ `pyproject.toml` 里配了 ruff（line-length 120、E/F/W/I，未接入 CI 阻塞） | 可选：接入 ruff 成为硬门禁 |

**验收标准**：`pytest` 全绿 ✅（本机以 `python tests/test_metrics.py` 验证 15/15；CI 里由 pytest 跑），GitHub 上有绿色 CI 徽章（**待推送后点亮**）。

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

> **状态：❌ 全未做。** PRO / PaDiM / FastAPI / ONNX / coreset 1% 投影 / 真实工件，六项都没有。

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

- [x] §1.1 补引用声明
- [x] §1.2 修 99.6% 口径
- [x] §1.3 补 LICENSE 说明
- [x] §4.0 数据划分、指标协议与上游差异表全部完成（`docs/EXPERIMENT_PROTOCOL.md`、`docs/vs-upstream.md`）
- [x] §4.1 高斯平滑参数化并验证（σ=0/0.5/1/2 已跑）
- [x] §6.1 `docs/vs-upstream.md`（10 行差异表 + 缺陷认定 + 正确比较方式）

**产出**：`v0.9-safe` —— 不会再被问穿。

### 里程碑 2：消融版（2 天，需 GPU + 数据）

- [x] §3.1 下载 MVTec（在 `MVTec\mvtech_anomaly_detection\`）
- [x] §3.2 准备权重
- [x] §2.1 建环境（实际用 `.venv`，不是 conda `ad`）
- [x] §4.1 验证 sigma 消融
- [x] §4.2 对比不同邻域描述子与 coreset 顺序（已实测：`docs/neighbor-ablation.md`）
- [x] §4.3 输入预处理消融（图像与 mask 同步变换）

**产出**：`v0.95` —— 获得可复算的消融结果；无论提升、持平或下降，都记录配置与结果，不预设指标必须提升。

### 里程碑 3：泛化与决策版（2~3 天）

- [x] §5.1 少样本曲线（六档 + 固定 `split_seed=42` 复验）
- [x] §5.2 鲁棒性压测（亮度/噪声/模糊已做并写成 `docs/robustness.md`；旋转、JPEG 待补）
- [x] §5.3 validation 选择成本策略，test 锁定评估并做成本敏感性分析
- [ ] §6.2 README 重构
- [x] §6.3 测试 15 条已绿，CI 已建（push 后点亮）

**产出**：`v1.0-interview` —— 可以拿去面试。

### 里程碑 4（可选）：演示版

- [ ] §8 FastAPI + 前端
- [ ] §8 ONNX 延迟
- [ ] §8 真实工件

---

## 10. 风险与注意事项

| 风险 | 说明 | 应对 |
|---|---|---|
| **Python 版本** | 原文档按 3.12 编写；实测 `.venv` 是 **3.13.9**，torch 2.7.0+cu128 / torchvision 0.22.0+cu128 在 3.13 上工作正常 | 已实测可用，无需降级；但 torch 只装在 `.venv`，base 不装（README 已更正为 3.13.9） |
| **4 GB 显存** | 9 邻域、大 coreset 都可能 OOM | 9 邻域已跑通并做同预算对照：修掉环绕填充（+1.9 点）后仍比不开差 0.34 点、时延 2.9×、峰值 517 MB；expanded 顺序在 185 图上需 ≈16 GB 候选池，已用 base 顺序绕开，见 `docs/neighbor-ablation.md` |
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
   > 实测：grid 上跑通（18,944 patch、2,353 MB、360 ms），但 image AUROC 0.9373 → 0.9190。要说清「改变的是 coreset 特征空间」，以及为什么默认关掉。

3. **screw 为什么漏检 73.1%**（README `:289-291`；实测 87/119，`results_best/results.csv` 的 `recall_at_thr=0.2689`）
   > /8 特征图上螺纹细划痕只占 1~2 个 patch → 被 3×3 平均抹平 → 验证方式（关掉平均 / 提分辨率 / 换 9 邻域）。

---

## 12. 三个必答的送命题

| 问题 | ❌ 错误答法 | ✅ 正确答法 |
|---|---|---|
| 「这是你自己写的还是抄的？」 | 「参考了论文」 | 「按论文独立重写的，没用官方 `src/` 包；逐项差异在 `docs/vs-upstream.md`，差异和指标影响按实现审计与消融结果说明」 |
| 「AUROC 0.98 为什么不如论文最高结果？」 | 「我显卡不行」 | 「先说明对比配置和指标口径；最高结果与 WR50 单模型配置不同。再依据逐项实现审计和同配置消融解释差异，避免把尚未验证的原因当结论」 |
| 「这个能上线吗？」 | 「能，AUROC 0.98」 | 「不能直接用。0.5% 过杀下漏检还有 20%，工业线要求 <1%。要么放宽过杀到 10%（漏检 4.1%），要么靠双阈值+复核兜底，要么上更高分辨率——三条路我都量化了成本和收益」 |

---

## 附录 A：环境快照（2026-11-04 实测）

```
OS         : Windows
解释器     : E:\面试项目文件夹\patchcore-mvtec\.venv\Scripts\python.exe
Python     : 3.13.9（venv；基于 E:\APP\Anaconda3 的 base 3.13.9 创建）
torch      : 2.7.0+cu128   (torch.cuda.is_available() = True)
torchvision: 0.22.0+cu128
numpy      : 2.3.5          pillow: 12.0.0
GPU        : NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB
数据集     : MVTec\mvtech_anomaly_detection\  (15 类, 5.0 GB)
权重       : torch_cache\wide_resnet50_2_timm.pth (267 MB, timm racm_in1k 转换, 命中 318/320 层)
缺失       : pytest / timm / onnx / onnxruntime / faiss / opencv
```

## 附录 B：数据与权重清单（换机器必看）

| 项 | 放哪 | 大小 | 是否入仓 |
|---|---|---|---|
| MVTec AD | `MVTec\mvtech_anomaly_detection\` | 5.0 GB（zip 5.27 GB） | ❌ 许可证禁止 |
| timm 权重 | `torch_cache\wide_resnet50_2_timm.pth` | 267 MB | ❌ .gitignore |
| 记忆库 bank | `experiment_data/04_memory_banks/banks*/<类>.npz`（含 VisA 的 3 套） | 每类几 MB | ❌ .gitignore |
| VisA 数据 | `VisA_pytorch\`（整理后）、`VisA_raw\`（原始包） | 1.8 + 1.8 GB | ❌ .gitignore |
| 逐图分数 | `experiment_data/00_baselines/results_best\scores\*.json` | 139 KB | ✅ **已入仓**（唯一放行的产物） |
| 热力图 | `experiment_data/00_baselines/results*/heatmaps/` | 每张 ~175 KB | ❌ .gitignore |

## 附录 C：常用命令速查

```powershell
# 0) 只复算三档表（只要有 numpy，1 秒）
cd E:\面试项目文件夹\patchcore-mvtec
.venv\Scripts\python.exe classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2

# 1) 单类跑通（bottle，约 3 分钟）
.venv\Scripts\python.exe patchcore_demo.py --data MVTec/mvtech_anomaly_detection --category bottle --size 320 --greedy 2000 `
  --calib-ratio 0.3 --thr-pct 99 --weights torch_cache\wide_resnet50_2_timm.pth

# 2) 全 15 类
.venv\Scripts\python.exe patchcore_demo.py --data MVTec/mvtech_anomaly_detection --category all --size 320 --greedy 2000 `
  --calib-ratio 0.3 --weights torch_cache\wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/results_v2

# 3) 三档判定
.venv\Scripts\python.exe classify.py build --data MVTec/mvtech_anomaly_detection --category bottle --weights torch_cache\wide_resnet50_2_timm.pth
.venv\Scripts\python.exe classify.py eval  --bank experiment_data/04_memory_banks/banks/bottle.npz --data MVTec/mvtech_anomaly_detection --category bottle --sweep

# 4) 离线分析
.venv\Scripts\python.exe analyze.py experiment_data/01_mvtec_gpu/results_v2
```

> ⚠️ **2026-11-04 更正**：本机数据集实际在 `MVTec\mvtech_anomaly_detection\`（上游拼写是 mvtech），命令里的路径均为本机实测值；换机器时替换为实际路径。另外 `python` 一律用 `.venv\Scripts\python.exe`（base 没装 torch），中文路径下 `cd E:\面试项目文件夹\patchcore-mvtec` 可原样使用。

---

## 附录 D：本次复核改动清单（2026-11-04）

| 位置 | 改了什么 |
|---|---|
| 头部 | 标注「最近复核 2026-11-04，当前提交 `d8c5c4e`」；新增「读法」 |
| §0.0 | 新增执行状态总览表（18 项，✅/⚠️/❌ + 证据行号） |
| §0.1–0.3 | 整体重写为实况：11 commits、`d8c5c4e` 领先 origin 1 个提交、`patchcore_demo.py` 461 行 / `classify.py` 548 行、`experiment_data/` 21.6 MB、Python 3.13.9 |
| §1–§8 | 每个小节标题下加「状态」行；保留原计划正文作为历史记录 |
| §2 | 「这台机器跑不了任何建库/推理实验」→ 环境已就绪 |
| §4.2 | 面试话术与同节实测更正自相矛盾，改为「实测没有 OOM，但精度 −1.8 点、时延 12×」 |
| §9 | 里程碑复选框按实际完成情况更新 |
| §10 | 「Python 3.13 必须降级到 3.12」→ 已在 3.13.9 上实测可用 |
| §11 | screw 漏检 **76.5% → 73.1%**（87/119，与 `results_best/results.csv` 的 `recall_at_thr=0.2689` 一致） |
| 附录 A | 环境快照按 2026-11-04 重新实测（`.venv` 3.13.9 / torch 2.7.0+cu128 / torchvision 0.22.0+cu128 / RTX 3050 4096 MiB） |
| 附录 B | 数据集真实路径 `MVTec\mvtech_anomaly_detection\`；补 VisA 行；权重 275 → 267 MB |
| 附录 C | 命令改用 `.venv\Scripts\python.exe` 与本机数据路径 |
| §1.3 状态 | 由 ⚠️ 改为 ✅（README 许可节已补独立重写声明） |
| P3 邻域消融 | 新增 `docs/neighbor-ablation.md`；`torch.roll` → zero padding（+1.93 点）、`--neighbor-coreset base`（建库 8.7×、内存 8.9×）；9 邻域仍差 0.34 点、时延 2.9×，默认关 |
| P0-2 | 新增 `docs/conformal-thresholds.md`、`tools/conformal_threshold.py`；`classify.py` 加 `conformal_quantile()` 与 build/report 的 `--conformal` |
| P0-3 | `patchcore_demo.py` 加 `--preprocess` / `--aspect-crop`；新增 `docs/preprocessing-ablation.md` 与 `results_preproc_{mode}_{类别}/` 六组结果 |
| P1-1 | 新增 `tools/latency_breakdown.py`、`docs/latency-breakdown.md`；README 加「端到端时延」段；计划加 §5.4 |
| P1-2 | 新增 `pyproject.toml`、`.github/workflows/ci.yml`、`tools/check_links.py`；§6.3 状态改 ✅；README §9 补 CI 说明 |
| P1-3 | 新增 `tools/pro_aupro.py`、`tests/test_pro.py`、`docs/pro-aupro.md`；`--dump-pixel` 落盘逐图像素分数图；§0.0 增行、§8 由 ❌ 改 ⚠️；README 新增「区域级指标」一节 |
| P1-4 | `report --cost FN,FP,REV --prevalence P`（validation 选点 / test 只报一次 / 先验扫描 / 权重敏感度）；新建 `docs/cost-model.md`；§5.3 ❌→✅、§9 勾选、§0.0 增行；README 新增「成本模型」小节 |
| 链接修复 | 新建 `experiment_data/README.md`，修掉 README 里的断链 |
| P2-① 扰动补全 | 噪声 σ=0.05/0.10 冻结+重标定 4 次运行，写成 `docs/robustness.md`，README 增表 |
| P2-② README 瘦身 | 600 → 125 行；拆出 `docs/install.md`、`docs/interview-notes.md`；§6.2 ❌→✅、§0.0/§0.1 计数更新 |
| P0-1 | 新建 `docs/vs-upstream.md`；§4.0 / §6.1 状态改 ✅ |
| §4.3 mask 同步 | 新增 `load_mask(path,size,preprocess,aspect_crop)`（与 `load_img` 同一套裁剪/补边，NEAREST、灰边填 0）；六组预处理实验重跑，transistor letterbox pixel 0.9206→0.9576、stretch 0.9363→0.8957；§4.3 ⚠️→✅ |
