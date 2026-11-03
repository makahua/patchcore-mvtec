# PatchCore 少样本无监督缺陷检测（MVTec AD）

给机器视觉/工业质检岗的一个**能跑出真实数字**的项目：不依赖 anomalib，只用 torch + torchvision 独立实现 PatchCore（预训练特征 → patch 记忆库 → 最近邻距离当异常分）。

> 环境：Python 3.13.9（`.venv`）+ torch 2.7.0+cu128 + torchvision 0.22.0+cu128，单卡 RTX 3050 Laptop（4 GB）。
> 算法来自 PatchCore 论文（Roth et al., CVPR 2022，[arXiv:2106.08265](https://arxiv.org/abs/2106.08265)），参考 [AWS 官方实现](https://github.com/amazon-science/patchcore-inspection) 的思路**独立重写**；与官方评测口径未完全对齐，逐项差异见 [vs-upstream](docs/vs-upstream.md)。

## 30 秒看懂

| 项 | 数字 |
|---|---|
| image AUROC（15 类 macro） | **0.9805 ± 0.0189** |
| 同批图像 pooled | 0.8812（口径不同，差 9.9 点） |
| pixel AUROC / image PR-AUC | 0.9709 / 0.993 |
| 区域级 AUPRO@FPR≤30% | **0.9211 ± 0.0398** |
| 端到端时延（PNG→判定） | 900² 42.2 ms ｜ 1024² 62.4 ms ｜ 算法段 19.7 ms |
| 显存 / 记忆库 | 190 MB ｜ 2,000 patch/类 |
| 三档判定（目标过杀 0.5%、灰区 1.2×） | 漏检 20.0% ｜ 过杀 0.2% ｜ 复核 22.7% |
| 成本模型（validation 选点 5%×1.5×，先验 1%） | 0.1814/件（默认工作点 0.2206） |

一句话：**AUROC 0.98 不等于能上线**——固定划分下 0.5% 目标过杀仍有 20% 漏检；区域级 AUPRO 与 image AUROC 几乎不相关（tile：1.0000 vs 0.8181）；噪声扰动会击穿 `max` 聚合。所有数字都能离线复算。

## 快速开始

最短路径（只需 numpy，不用 GPU、数据集、权重，1 秒）：

```powershell
.venv\Scripts\python.exe classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2
```

完整跑（需要 torch 与本机数据；环境、权重来源、MVTec 下载见 [安装与数据准备](docs/install.md)）：

```powershell
.venv\Scripts\python.exe patchcore_demo.py --data MVTec/mvtech_anomaly_detection --category all --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce
.venv\Scripts\python.exe tests/test_metrics.py     # 15 条纯 numpy 自检
```

产出目录与逐类结果索引见 [experiment_data](experiment_data/README.md)。

## 关键结果

### 1) 五档配置：greedy coreset 是性价比最高的一步

| 配置 | image AUROC | 时延（算法段） | 显存 | 召回@0.5%名义过杀 |
|---|---:|---:|---:|---:|
| 随机 10% / 256 | 0.9662 ± 0.039 | 28.3 ms | 574 MB | 83.0% |
| **greedy 2000 / 256** | 0.9737 ± 0.027 | **16.4 ms** | **169 MB** | 83.4% |
| 随机 10% / 320 | 0.9763 ± 0.022 | 58.2 ms | 935 MB | 81.4% |
| **greedy 2000 / 320（最终档）** | **0.9805 ± 0.019** | 19.7 ms | 190 MB | 82.8% |
| 9 邻域 / 256（旧实现，仅 grid） | 0.9190 | 360 ms | 2353 MB | — |

- **顺序很重要：先压 coreset，再提分辨率。** 随机抽样下 320 会让记忆库膨胀、时延 28→58 ms；定长 2000 后 320 只多 3 ms 就换 +0.68 点。
- **时延列是算法段**（`results.csv` 口径：`patch_scores()` 内部，图已解码、缩放、搬进显存），不含读图/解码；端到端见第 4 条。
- 9 邻域修掉两处实现问题：`torch.roll` 的边界环绕（同预算 image AUROC +1.93 点）与选点顺序（`--neighbor-coreset base`：建库 194→22 s、候选池内存 2.1→0.24 GB，精度持平）；修完仍比不开差 0.34 点、时延 2.9×（52 vs 18 ms），默认关 → `docs/neighbor-ablation.md`。
- 最终档最差三类：toothbrush 0.947 / screw 0.955 / pill 0.956。

> **口径**：上表 image AUROC 是 macro（15 类各算再平均）；同一批分数 pooled 是 0.8812。两个数都对，但对外必须写清是哪个——面试官拿仓库里的 scores 复算，写 0.98 得到 0.88 会当场翻车。

### 2) 阈值不是小事：从 28% 误报到可用工作点

初版把阈值取成「**建库图自评分** P99」，而记忆库就从这些图建出来 → 自匹配让分数偏低 → 阈值过低 → 良品误报 28%。修法是 `--calib-ratio 0.3`：留出 18–117 张良品**不参与建库**，只用来标定。这条最能证明「知道自己在哪一步作弊」。

### 3) 三档判定与成本：AUROC 之后，产线只问三件事

`thr_ok`（良品 P(1−fpr) 分位）与 `thr_ng`（×1.2）把画面分成 OK / REVIEW / NG；`--conformal` 换成有有限样本保证的分位——代价是 0.5% 需要 ≥199 张良品，本项目每类 18–117 张，**15 类全部不可达**（可达 0.85%–5.26%）。

全 15 类（目标过杀 0.5%、灰区 1.2×）：**过杀 0.2% / 良品进灰区 2.1% / 漏检 20.0% / 复核 22.7%**；最差 screw 漏检 73.1%（87/119）。

成本模型（漏检:过杀:复核 = 100:10:1，缺陷先验 1%，用先验纠偏测试集 72.9% 的缺陷富集）：默认工作点 0.2206/件；validation 选 5%×1.5× → test **0.1814/件**（省 22%，漏检 6.9%，产线复核负荷 11.2%）。**最优工作点由先验决定，不由模型好坏决定。**

命令：`classify.py report --conformal` / `--cost 100,10,1 --prevalence 0.01`。详见 [conformal 阈值](docs/conformal-thresholds.md)、[成本模型](docs/cost-model.md)。

### 4) 时延与分辨率：19.7 ms 只是算法段

19.7 ms 只包住 `patch_scores()`（图已解码、缩放、搬进显存）。端到端实测：700² 37.1 ms、900² 42.2 ms、**1024² 62.4 ms**；再往上用上采样合成图外推：**2048² 130 ms、4096² 447 ms**（decode 占 71%/74%，`19.7 ms` 分别失真 6.6×/22.7×）。增长全在 PNG 解码与缩放，GPU 段基本不变。**报节拍前先问相机出多大图。** 见 [时延拆解](docs/latency-breakdown.md)。

### 5) 区域级 AUPRO：image AUROC 会骗人

PRO/AUPRO 按 GT 连通域等权，macro **AUPRO@30% 0.9211 ± 0.0398**、@10% 0.8068；与 image AUROC 相关系数 **−0.019**（tile：image 1.0000 但区域级 0.8181）。pooled 0.7808 是陷阱，必须报 macro。见 [PRO/AUPRO](docs/pro-aupro.md)。

### 6) 扰动鲁棒性：三种失效形态，三种处置

| 扰动（screw） | image AUROC | 冻结 召回/过杀 | 重标定 召回/过杀 | 处置 |
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

## 诚实的边界

- 固定划分、单 seed 为主；screw/capsule 的低召回说明**分辨率与视野是硬约束**，不是调参能解决的。
- MVTec 已参与多轮开发，只能按开发集呈现；VisA PCB2 已揭盲一次外部检查，不再是干净 holdout（见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)）。
- 失败实验也留着：`--imagenet-norm` 让 screw 从 0.901 掉到 0.867；9 邻域修完 bug 仍不划算，所以默认关、只作为可复算的对照保留。
- 更多坑、简历写法、面试问答、30 分钟复现与 CI 说明见 [面试笔记](docs/interview-notes.md)。

## 文件结构

```
patchcore-mvtec/
  patchcore_demo.py    # 主实验：建库 → 评分 → AUROC/时延/显存（--greedy/--calib-ratio/--perturb-*）
  classify.py          # 三档判定：build/infer/eval/report（report 支持 --conformal / --cost）
  analyze.py           # 批量结果对照；pyqt_demo.py 单图 GUI
  tools/               # pro_aupro / latency_breakdown / conformal_threshold / check_links / 数据准备
  tests/               # 23 条纯 numpy 测试（15 指标+成本、8 PRO/AUPRO）
  docs/                # 17 篇方法与实验文档
  experiment_data/     # 结果索引（大文件由 .gitignore 排除）
```

文档索引：[vs-upstream](docs/vs-upstream.md)｜[实验协议](docs/EXPERIMENT_PROTOCOL.md)｜[难类失败分析](docs/FAILURE_ANALYSIS.md)｜[演示流程](docs/INTERVIEW_DEMO.md)｜[复现说明](docs/REPRODUCE.md)｜[VisA](docs/VISA.md)｜[优化计划](docs/OPTIMIZATION_PLAN.md)｜[结果优化记录](docs/RESULTS_OPTIMIZATION_PLAN.md)｜[安装与数据](docs/install.md)。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方：PyTorch / torchvision（BSD-3-Clause）、MVTec AD 数据集（CC BY-NC-SA 4.0，仅限非商业研究）。

本项目为独立重写实现，未复制 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)（Apache-2.0）的源代码；若后续引入其任何代码片段，将同步改为 Apache-2.0 并保留 NOTICE。
