# PatchCore 少样本无监督缺陷检测（MVTec AD）

给机器视觉/工业质检岗的一个**能跑出真实数字**的项目：不依赖 anomalib，只用 torch + torchvision 独立实现 PatchCore（预训练特征 → patch 记忆库 → 最近邻距离当异常分）。

> 环境：Python 3.13.9（`.venv`）+ torch 2.7.0+cu128 + torchvision 0.22.0+cu128，单卡 RTX 3050 Laptop（4 GB）。
> 算法来自 PatchCore 论文（Roth et al., CVPR 2022，[arXiv:2106.08265](https://arxiv.org/abs/2106.08265)），参考 [AWS 官方实现](https://github.com/amazon-science/patchcore-inspection) 的思路**独立重写**；与官方评测口径未完全对齐，逐项差异见 [vs-upstream](docs/vs-upstream.md)。

## 30 秒看懂

| 项 | 数字 |
|---|---|
| image AUROC（15 类 macro，3 seed） | **0.9830 ± 0.0008** |
| 同批图像 pooled | 0.8564 ± 0.0008（按类换层后跨类尺度更不可比，只作混合流参考） |
| pixel AUROC | 0.9711 ± 0.0005 |
| 区域级 AUPRO@FPR≤30% | **0.9229 ± 0.0012**（@10% 0.8160） |
| 端到端时延（PNG→判定） | 900² **39.7 ms**（35.5–42.2）｜ 1024² **64.9 ms**（62.4–64.9）｜ 算法段 22.1 ms |
| 吞吐（预取，bottle 900²） | fp32 50.3 → fp16 63.1 → fp16+batch8 **101.4 张/s** |
| 显存 / 记忆库 | 190 MB（b8 峰值 345 MB）｜ 2,000 patch/类 |
| 三档判定（P99.5、灰区 1.2×，3 seed） | 漏检 **12.8%** ｜ 过杀 0.2% ｜ 复核 27.0% |
| 成本模型（macro 对账，先验 1%） | 0.5%×1.5× **0.1902** /件 ｜ 5%×1.5× 0.2082 /件（[Phase 1](docs/phase1-notes.md)） |

> **主表口径（2026-11-06 起）**：`docs/OPTIMIZATION_PLAN.md` 冻结协议——timm `racm_in1k`、320 stretch 不做 ImageNet 归一化、legacy greedy 2000、`2+3` 为默认、**screw/capsule 按类改 `1+2`**（[configs/patchcore_profile.json](configs/patchcore_profile.json)，证据见 [分辨率与频率](docs/screw-resolution.md)）、阈值 P99.5、`split-seed 42`、coreset seed 0/1/2，共 90 个 run + 6 个按类覆盖 run。上表是这 15 类 × 3 seed 的均值；机器可读版本见 [pair-baseline.json](docs/pair-baseline.json)，逐 run 产物在 [10_pair](experiment_data/06_optimization/10_pair/README.md)。**MVTec test 为 development-only**。

### 0) 逐类主表（P99.5，三 seed 均值）

| 类别 | train/calib/test | image AUROC | pixel AUROC | AUPRO@30 | 阈值 | 召回@thr | 过杀 | 算法段 ms | GPU 峰 MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 147/62/83 | 1.0000 | 0.9855 | 0.9454 | 39.67 | 0.958 | 0.000 | 20.3 | 190.0 |
| cable | 157/67/150 | 0.9856 | 0.9804 | 0.9268 | 47.49 | 0.815 | 0.017 | 23.7 | 190.0 |
| capsule | 154/65/132 | 0.9904 | 0.9835 | 0.9607 | 32.57 | 0.746 | 0.000 | 26.2 | 230.8 |
| carpet | 196/84/117 | 0.9845 | 0.9859 | 0.9423 | 35.43 | 0.925 | 0.036 | 23.3 | 189.6 |
| grid | 185/79/78 | 0.9708 | 0.9586 | 0.8622 | 35.75 | 0.912 | 0.016 | 19.4 | 190.0 |
| hazelnut | 274/117/110 | 0.9998 | 0.9871 | 0.9440 | 51.73 | 0.976 | 0.000 | 23.2 | 189.6 |
| leather | 172/73/124 | 1.0000 | 0.9882 | 0.9725 | 31.76 | 1.000 | 0.031 | 22.2 | 189.6 |
| metal_nut | 154/66/115 | 0.9979 | 0.9832 | 0.9407 | 43.88 | 0.986 | 0.030 | 18.7 | 189.6 |
| pill | 187/80/167 | 0.9519 | 0.9764 | 0.9508 | 41.77 | 0.671 | 0.000 | 18.4 | 190.0 |
| screw | 224/96/160 | 0.9737 | 0.9897 | 0.9523 | 33.26 | 0.781 | 0.041 | 24.6 | 230.8 |
| tile | 161/69/117 | 1.0000 | 0.9197 | 0.8177 | 40.94 | 0.964 | 0.000 | 19.9 | 190.0 |
| toothbrush | 42/18/42 | 0.9426 | 0.9903 | 0.9076 | 34.77 | 0.978 | 0.305 | 22.4 | 189.6 |
| transistor | 150/63/100 | 0.9958 | 0.9480 | 0.9109 | 43.86 | 0.900 | 0.017 | 23.4 | 189.6 |
| wood | 173/74/79 | 0.9866 | 0.9106 | 0.8749 | 41.28 | 0.967 | 0.158 | 26.0 | 190.0 |
| zipper | 168/72/151 | 0.9659 | 0.9801 | 0.9348 | 35.36 | 0.860 | 0.062 | 19.7 | 189.6 |

最差三类 image：toothbrush 0.943 / pill 0.952 / zipper 0.966——**难类没有因为换协议消失**。`1+2` 全类推广会把 cable 拉低 5 个点、15 类均值时延 22.1→24.6 ms，所以只在 screw/capsule 上按类启用（这两类 1+2 的收益是 +5.06 / +0.81 点 image，漏检 −14.6 / −31.5 点；见 [10_pair 判定](experiment_data/06_optimization/10_pair/README.md) 与 [分辨率与频率](docs/screw-resolution.md)）。

一句话：**AUROC 0.98 不等于能上线**——固定划分下 0.5% 目标过杀仍有 12.8% 漏检（只靠阈值能压到 6–7%，代价是复核翻倍）；区域级 AUPRO 与 image AUROC 几乎不相关（tile：1.0000 vs 0.8177）；噪声扰动会击穿 `max` 聚合。所有数字都能离线复算。

## 快速开始

最短路径（只需 numpy，不用 GPU、数据集、权重，1 秒）：

```powershell
# 注意：这条命令复算的是 2026-10 的旧快照（P99 口径，macro 0.9805 / pooled 0.8812），
# 不是上表的 P99.5 主表（0.9830 / 0.8564）。两个快照不可互证，原因见 docs/baseline-acceptance.md。
.venv\Scripts\python.exe classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2
```

要复算主表，用规范化 run（`06_optimization/10_pair` 与按类覆盖的 `16_screw12_p995`、`14_capsule12`）：

```powershell
.venv\Scripts\python.exe tools/build_pair_baseline.py --check
.venv\Scripts\python.exe tools/export_thresholds.py --check
```

完整跑（需要 torch 与本机数据；环境、权重来源、MVTec 下载见 [安装与数据准备](docs/install.md)）：

```powershell
.venv\Scripts\python.exe patchcore_demo.py --data MVTec/mvtech_anomaly_detection --category all --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce
.venv\Scripts\python.exe tests/test_metrics.py     # 15 条纯 numpy 自检
```

产出目录与逐类结果索引见 [experiment_data](experiment_data/README.md)。

## 关键结果

### 1) 五档配置：greedy coreset 是性价比最高的一步

> 下表是早期 P99、单 seed 的配置对照（历史口径，保留以便复算）；与上面新协议的数字不可直接相减。

| 配置 | image AUROC | 时延（算法段） | 显存 | 召回@0.5%名义过杀 |
|---|---:|---:|---:|---:|
| 随机 10% / 256 | 0.9662 ± 0.039 | 28.3 ms | 574 MB | 83.0% |
| **greedy 2000 / 256** | 0.9737 ± 0.027 | **16.4 ms** | **169 MB** | 83.4% |
| 随机 10% / 320 | 0.9763 ± 0.022 | 58.2 ms | 935 MB | 81.4% |
| **greedy 2000 / 320（旧 P99 主表）** | **0.9805 ± 0.019** | 19.7 ms | 190 MB | 82.8% |
| 9 邻域 / 256（旧实现，仅 grid） | 0.9190 | 360 ms | 2353 MB | — |

- **顺序很重要：先压 coreset，再提分辨率。** 随机抽样下 320 会让记忆库膨胀、时延 28→58 ms；定长 2000 后 320 只多 3 ms 就换 +0.68 点。
- **时延列是算法段**（`results.csv` 口径：`patch_scores()` 内部，图已解码、缩放、搬进显存），不含读图/解码；端到端见第 4 条。
- 9 邻域修掉两处实现问题：`torch.roll` 的边界环绕（同预算 image AUROC +1.93 点）与选点顺序（`--neighbor-coreset base`：建库 194→22 s、候选池内存 2.1→0.24 GB，精度持平）；修完仍比不开差 0.34 点、时延 2.9×（52 vs 18 ms），默认关 → `docs/neighbor-ablation.md`。
- 最终档最差三类：toothbrush 0.947 / screw 0.955 / pill 0.956。

> **pooled 能不能当结论**：不能当主结论。15 类的分数尺度不同（同一阈值在 bottle 是 39、在 V1 权重下是 2），pooled 把不同尺度的分数混进同一个排序里，只能当「混合流/单类产线」的参考；主口径必须是 macro，且报告时要写清是哪一个。
>
> **口径**：上表是**旧 P99 表**——image AUROC 是 macro（15 类各算再平均），同一批分数 pooled 是 0.8812；新协议（P99.5、三 seed）对应 macro **0.9791** / pooled **0.8799**，两套不可相减；按类换层后是 macro **0.9830** / pooled **0.8564**——**pooled 下降不代表变差**，它只是把两类 1+2 的分数尺度也混进去了。两个数都对，但对外必须写清是哪个——面试官拿仓库里的 scores 复算，写 0.98 得到 0.88 会当场翻车。

### 2) 阈值不是小事：从 28% 误报到可用工作点

初版把阈值取成「**建库图自评分** P99」，而记忆库就从这些图建出来 → 自匹配让分数偏低 → 阈值过低 → 良品误报 28%。修法是 `--calib-ratio 0.3`：留出 18–117 张良品**不参与建库**，只用来标定。这条最能证明「知道自己在哪一步作弊」。

### 3) 三档判定与成本：AUROC 之后，产线只问三件事

`thr_ok`（良品 P(1−fpr) 分位）与 `thr_ng`（×1.2）把画面分成 OK / REVIEW / NG；`--conformal` 换成有有限样本保证的分位——代价是 0.5% 需要 ≥199 张良品，本项目每类 18–117 张，**15 类全部不可达**（可达 0.85%–5.26%）。

新协议（P99.5、灰区 1.2×、三 seed、按类配置）：**漏检 12.8% ｜ 实际过杀 0.2% ｜ 复核 27.0%**（逐 seed 漏检 177/159/146 件）；把工作点放宽到 1% 目标过杀可再换来 6–7% 漏检，代价是复核量翻倍——这是工作点选择，不是模型能力。旧 P99 协议下是漏检 20.0% / 复核 22.7%——阈值从 P99 抬到 P99.5 换回 3.1 个点漏检，代价是多送 1.9 个点去复核。最差 screw 漏检 73.1%（87/119，旧协议）。

逐类的 `thr_ok` / `thr_ng`、所用层组合与三 seed 波动见 [thresholds.json](docs/thresholds.json)——**阈值与类别、层组合、权重都绑定**，跨类或换权重都不能直接搬（`racm_in1k` → torchvision V1 时同一配置的阈值从 38.18 掉到 1.99，19×）。

成本模型（漏检:过杀:复核 = 100:10:1，缺陷先验 1%，用先验纠偏测试集 72.9% 的缺陷富集）：**同一交替划分重算后**，0.5%×1.2× = 0.2862 /件、0.5%×1.5× = **0.1902** /件、5%×1.2× = 0.5156、5%×1.5× = 0.2082（早期 0.1814 是不同划分与口径，见 [Phase 1 对账](docs/phase1-notes.md)）——「5% 最优」站不住，最优工作点随先验走。注意这组数的「漏检」按灰区上界 `thr_ng` 计，比三档表的 `thr_ok` 更保守，两组数字不能直接相减。

命令：`classify.py report --conformal` / `--cost 100,10,1 --prevalence 0.01`。详见 [conformal 阈值](docs/conformal-thresholds.md)、[成本模型](docs/cost-model.md)。

### 4) 时延与分辨率：算法段 21.5 ms，端到端按相机尺寸另算

最终协议（P99.5、按类配置：默认 2+3、screw/capsule 1+2，15 类 × 3 seed）**算法段 22.09 ± 1.65 ms**（逐类 18–26 ms）——这是 `results.csv` 的口径，只包住 `patch_scores()`（图已解码、缩放、搬进显存）。端到端（读盘→PNG 解码→缩放→H2D→前向→匹配→后处理）在最终配置上实测 p50：**700² 40.9 ms ｜ 900² 39.7 ms ｜ 1024² 64.9 ms**；两次独立测量的区间是 37–41 / 35.5–42.2 / 62.4–64.9，跨批次别拿单次数相减。再往上用上采样合成图外推：**2048² 130 ms、4096² 447 ms**（只读趋势）。增长全在解码与缩放，GPU 段 18.3–22.0 ms 基本不变。**报节拍前先问相机出多大图。** 见 [时延拆解](docs/latency-breakdown.md)。

端到端还能再压：多线程解码 + 有界预取把 900² 从 25.2 抬到 **50.3 张/s**（三档被拉平到 GPU 上限）；可选 `--fp16`（默认关，闸门 AUPRO 最大差 0.0008）到 **63.1 张/s**，算法段 21.5→13.7 ms，显存 +17 MB；再叠 `--batch 8` 到 **101.4 张/s**（1024² 为 85.6；b4 即打满，显存 +150 MB，分数非逐位一致但 AUROC/AUPRO 四位小数不变）→ [端到端上限估算](docs/e2e-ceiling.md)。

### 5) 区域级 AUPRO：image AUROC 会骗人

PRO/AUPRO 按 GT 连通域等权，新协议 macro **AUPRO@30% 0.9229 ± 0.0012**、@10% 0.8160 ± 0.0006（旧 P99 单 seed 批次为 0.9211 / 0.8068）；与 image AUROC 相关系数 **−0.019**（旧批次测得；tile：image 1.0000 但区域级 0.8181）。pooled 0.7808 是陷阱，必须报 macro。见 [PRO/AUPRO](docs/pro-aupro.md)。

### 6) 扰动鲁棒性：三种失效形态，三种处置

| 扰动（screw，旧 P99 口径） | image AUROC | 冻结 召回/过杀 | 重标定 召回/过杀 | 处置 |
|---|---:|---:|---:|---|
| 无 | 0.9553 | 26.9% / 0.0% | — | — |
| 亮度 ×0.8 | 0.9492 | 23.5% / 0.0% | 26.9% / 0.0% | 重标定即可 |
| 噪声 σ=0.05 / 0.10 | 0.7387 / 0.4782 | 100% / 100% | 33.6% / 4.9% | 伤在 `max` 聚合 |
| 模糊 1 px | 0.4583 | 100% / 100% | 1.7% / 4.9% | 不可救 |

噪声下 pixel AUROC 只掉 2 点——patch 分数仍可信，坏在「取最大值」被噪声尾部支配；模糊则是信息删除。见 [扰动鲁棒性](docs/robustness.md)。

> **几何预处理对照**：MVTec 15 类全是方图，`letterbox` 与 `stretch` 逐位等价（maxdiff 0）；用 `--aspect-crop 0.75` 模拟 4:3 相机后才有区别——mask 与图像同步变换后，letterbox 的 pixel AUROC 三类**全部不差**（screw +2.43 / pill +0.65 / transistor +6.19 点），image 上 screw +1.45 点。旧文档里「transistor pixel −1.57、灰边污染定位」是 mask 未同步造成的假象，已更正。见 [预处理对照](docs/preprocessing-ablation.md)。

## 三个设计决策

1. **coreset 用 k-center 贪心，不用随机采样**：同等预算下跨类更稳（std 0.039→0.027），显存与时延降一个量级。
2. **阈值必须留出标定集**：自匹配偏差是这类方法最隐蔽的坑，也是产线误报的主要来源。
3. **灰区把「不确定」交给人工**：1.2×–1.5× 用 11%–23% 复核量兜回约 30% 缺陷；但复核权重一高就不划算——成本模型给出边界。
4. **配置按类下沉，不追求一套超参打天下**：`configs/patchcore_profile.json` 只给 screw/capsule 换层，其余不动；代价可算（+0.57 ms 均值、+41 MB 显存），收益可查（两类漏检减半）。

## 诚实的边界

- screw/capsule 的低召回定位到**特征频率**而不是像素数：640 输入（`2+3`）把 AUROC 抬到 0.953 但匹配 FPR 的召回不动；`1+2`（layer1 原生 /4）把 screw 漏检 36.4%→21.8%、capsule 56.9%→25.4%——**已按类启用**，并在 split-123 上复核（[split 敏感性](docs/split-seed-check.md)）。其余类别仍 2+3（cable 用 1+2 掉 5 点）。
- MVTec 已参与多轮开发，只能按开发集呈现；VisA PCB2 已揭盲一次外部检查，不再是干净 holdout（见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)）。
- 失败实验也留着：`--imagenet-norm` 让 screw 从 0.901 掉到 0.867；9 邻域修完 bug 仍不划算，所以默认关、只作为可复算的对照保留。
- 更多坑、简历写法、面试问答、30 分钟复现与 CI 说明见 [面试笔记](docs/interview-notes.md)。

## 文件结构

```
patchcore-mvtec/
  patchcore_demo.py    # 主实验：建库 → 评分 → AUROC/时延/显存（--greedy/--calib-ratio/--perturb-*）
  classify.py          # 三档判定：build/infer/eval/report（report 支持 --conformal / --cost）
  analyze.py           # 批量结果对照；pyqt_demo.py 单图 GUI
  configs/             # 按类覆盖配置（patchcore_profile.json：screw/capsule → 1+2）
  tools/               # pro_aupro / latency_breakdown / conformal_threshold / check_links / 数据准备
  tests/               # 35 条纯 numpy 测试（15 指标+成本、8 PRO/AUPRO、5 bank meta、7 稳健聚合）
  docs/                # 26 篇方法与实验文档 + pair-baseline.json（主表数字）+ thresholds.json（逐类阈值）
  experiment_data/     # 结果索引（大文件由 .gitignore 排除）
```

文档索引：[vs-upstream](docs/vs-upstream.md)｜[分辨率与频率](docs/screw-resolution.md)｜[Phase 1 笔记](docs/phase1-notes.md)｜[实验协议](docs/EXPERIMENT_PROTOCOL.md)｜[难类失败分析](docs/FAILURE_ANALYSIS.md)｜[演示流程](docs/INTERVIEW_DEMO.md)｜[复现说明](docs/REPRODUCE.md)｜[VisA](docs/VISA.md)｜[最新优化计划](docs/OPTIMIZATION_PLAN.md)｜[安装与数据](docs/install.md)｜[基线验收](docs/baseline-acceptance.md)｜[coreset 预算](docs/coreset-budget.md)｜[阈值稳定性](docs/threshold-stability.md)｜[特征层消融](docs/features-ablation.md)｜[成本模型](docs/cost-model.md)｜[PRO/AUPRO](docs/pro-aupro.md)｜[时延拆解](docs/latency-breakdown.md)｜[端到端上限估算](docs/e2e-ceiling.md)｜[鲁棒性](docs/robustness.md)｜[预处理对照](docs/preprocessing-ablation.md)｜[邻域填充](docs/neighbor-ablation.md)｜[鲁棒聚合](docs/robust-aggregation.md)｜[split 敏感性](docs/split-seed-check.md)｜[表征与几何试点](docs/representation-pilot.md)｜[阈值表](docs/thresholds.json)｜[面试笔记](docs/interview-notes.md)。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方：PyTorch / torchvision（BSD-3-Clause）、MVTec AD 数据集（CC BY-NC-SA 4.0，仅限非商业研究）。

本项目为独立重写实现，未复制 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)（Apache-2.0）的源代码；若后续引入其任何代码片段，将同步改为 Apache-2.0 并保留 NOTICE。
