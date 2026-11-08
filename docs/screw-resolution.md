# 分辨率与频率：screw 640 与 capsule 1+2 的因果对照

> development-only。MVTec test 已参与多轮开发，本文不构成独立评估。
> 协议：timm `racm_in1k`、stretch、legacy greedy 2000、阈值 P99.5（fpr 0.005）、灰区 ×1.2、`split-seed 42`、seed 0/1/2、FP32 batch=1。

## 0. 问题

执行计划把「screw 为什么会漏」拆成两个候选解释：

- **网格密度**：320 输入把 1024² 的螺丝压了 3.2 倍，细缺陷在 /8 网格上只剩 1–2 格；
- **特征频率**：`2+3` 的 layer3 本身是 /16，即使上采样到 /8 也没有高频内容。

640 输入 + `2+3` 与 320 输入 + `1+2` 都能把网格做到 80×80，但一个靠「更多像素」，一个靠「更浅、更细的层」。所以这两组正好构成一组对照。

## 1. 三配置（screw，3 seed 均值 ± 样本标准差）

| 配置 | 网格 | image AUROC | pixel AUROC | AUPRO@30 | 漏检@P99.5 | 过杀 | 算法段 | 建库 | GPU 峰 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 / `2+3`（默认） | 40×40 | 0.9231 ± 0.0065 | 0.9840 | 0.9293 ± 0.0056 | 36.4% | 3.3% | 18.9 ms | 204 s | 188.8 MB |
| 320 / `1+2` | 80×80 | **0.9737 ± 0.0031** | 0.9897 | **0.9523 ± 0.0035** | **21.8%** | 4.1% | 24.6 ms | 467 s | 229.6 MB |
| 640 / `2+3` | 80×80 | 0.9533 ± 0.0026 | 0.9884 | 0.9446 ± 0.0045 | 42.3% | 0.0% | 67.6 ms | 97 s* | 329.3 MB |

\* 640 建库用 `--pool-cap 150000`（候选池 1,433,600 → 149,856，抽样 10.5%），否则单精度池 8.8 GB、`cat` 峰值 16.4 GB，会换页到不可用。

**只看 AUROC 会得出「640 也有用」；看工作点则是反的**：640 的漏检比默认还差 5.9 个百分点。

## 2. 匹配 FPR 的召回（关键）

阈值都取「良品分数的 (1−fpr) 分位」，在同一批 test 图上算：

| 配置 | fpr=0 | fpr=0.5% | fpr=1% | fpr=3% |
|---|---:|---:|---:|---:|
| 320 / `2+3` | 55.5% | 59.7% | 62.7% | 68.6% |
| 320 / `1+2` | 53.8% | **68.9%** | **75.4%** | **86.3%** |
| 640 / `2+3` | 55.2% | 59.9% | 62.7% | 72.5% |

`1+2` 在 0.5%–3% 这一段全面提升（+9.2 / +12.7 / +17.7 点），而 640 与默认几乎重合。

## 3. 子类型：640 换来了什么

| 子类型 | 320 `2+3` | 640 `2+3` |
|---|---:|---:|
| thread_side（螺纹侧面） | 29% | 20% |
| thread_top（螺纹顶面） | 78% | **54%** |
| manipulated_front | 43% | 39% |
| scratch_head | 83% | 82% |
| scratch_neck | 83% | **91%** |

640 只改善了 `scratch_neck`，把最难的 `thread_top` 反而做坏了——上采样后的 layer3 在细长螺纹上产生了更平滑、更少判别性的响应。

## 4. capsule：同样的对照，幅度更大

| 配置 | image AUROC | pixel AUROC | AUPRO@30 | 漏检@P99.5 | 算法段 |
|---|---:|---:|---:|---:|---:|
| 320 / `2+3` | 0.9823 | 0.9890 | 0.9449 | 56.9% | 23.2 ms |
| 320 / `1+2` | **0.9904** | 0.9835 | **0.9607** | **25.4%** | 26.2 ms |

匹配 FPR 的召回：

| 配置 | fpr=0 | 0.5% | 1% | 3% |
|---|---:|---:|---:|---:|
| 320 / `2+3` | 37.6% | 44.6% | 55.7% | 69.1% |
| 320 / `1+2` | **67.9%** | **76.1%** | **83.2%** | **93.0%** |

子类型（recall@P99.5）：

| 子类型 | `2+3` | `1+2` |
|---|---:|---:|
| faulty_imprint | 11% | **70%** |
| poke | 35% | **83%** |
| scratch | 28% | **52%** |
| crack | 52% | **77%** |
| squeeze | 95% | 95% |

capsule 的漏检不是「阈值放歪」，是**小面积压印/戳伤在 /8 网格上几乎没有响应**；换成 /4 + layer1 后直接补回来。

## 5. 结论

1. **瓶颈是每格能看到的高频内容，不是像素总数。** 640 增加了像素，但 layer3 上采样后仍是低频；`1+2` 用 layer1 的原生 /4 响应，把 screw 0.5% 处召回 59.7%→68.9%、capsule 37.6%→67.9%（fpr=0）。
2. **AUROC 与工作点会分叉。** 640 的 image AUROC +3.0 点，但 P99.5 处漏检反而 +5.9 点；只报 AUROC 会把一个不该采用的配置说成「有收益」。
3. **screw 的 `thread_side` 仍是 20–29%。** 这一档属于「缺陷与良品纹理同分布」的重叠，换分辨率/层组合都动不了，下一棒应是 ROI 裁剪或区域判定，而不是继续加像素。

## 6. 决策

- **默认配置不变**（320 / `2+3` / P99.5）。单类改动不进入默认，除非满足「跨 seed 稳定 + 单类不退化 + 15 类均值 ≤25 ms」三条同时成立。
- **per-class `1+2` 是候选**，不是默认：screw（18.9→24.6 ms）与 capsule（23.2→26.2 ms）都只影响各自一类；若两类都切换，15 类均值 21.5→22.1 ms，仍在门槛内。
- **640 不采用**：算法段 67.6 ms（15 类均值会到 24.8 ms，贴线）、建库依赖抽样、低 FPR 召回没有改善。
- 未过门槛的候选仍保留在 `experiment_data/06_optimization/13_screw640`、`14_capsule12`、`16_screw12_p995`，可复算。

## 7. 局限

- 每类只有 3 个 coreset seed、一个 split-seed；timm 权重、P99.5、`max` 聚合都沿用基线，没有换权重复验。
- 640 的建库是抽样池（`--pool-cap 150000`），与 320 的完整池不是同一候选集；640 的 AUPRO 只覆盖 seed 0–2 的同协议分包。
- 阈值仍由每类 18–117 张良品估 P99.5，尾部误差未消除（见 [phase1 笔记](phase1-notes.md)）。
- 时延为单进程、算法段；端到端另算（见 [latency-breakdown](latency-breakdown.md)）。

## 9. split-123 复核（2026-11-09）

换一个划分（`--split-seed 123`）重跑两组，确认拆层收益不是 split-42 的噪声。基线 run 在
`12_split123/`，1+2 复核在 `17_split123_l12/`（两批都没有 pixel dump，故基线无 AUPRO）。

| 类别 | 配置 | image AUROC | 漏检@P99.5 | 算法段 | AUPRO@30 |
|---|---|---:|---:|---:|---:|
| screw | 2+3 | 0.9279 ± 0.0126 | 57.7% | 19.6 ms | — |
| screw | **1+2** | **0.9763 ± 0.0041** | **26.9%** | 26.3 ms | 0.9555 |
| capsule | 2+3 | 0.9525 ± 0.0118 | 30.9% | 24.5 ms | — |
| capsule | **1+2** | **0.9818 ± 0.0011** | **21.1%** | 25.8 ms | 0.9607 |

两个方向都成立：screw（+4.84 点 image、漏检 −30.8 点）、capsule（+2.93 点、−9.8 点）。注意
**同一配置的漏检率本身随划分变动**（screw 2+3：split42 36.4% vs split123 57.7%；capsule 2+3：
56.9% vs 30.9%）——这正是「阈值尾部估计不稳」的证据，也是不能只报单一划分的原因。

复现：

```powershell
.venv\Scripts\python.exe tools\run_experiment.py --run-dir experiment_data/06_optimization/17_split123_l12/12_screw_seed_000 -- `
  --category screw --layers 1+2 --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99.5 `
  --split-seed 123 --seed 0 --weights torch_cache/wide_resnet50_2_timm.pth `
  --data MVTec/mvtech_anomaly_detection
```

## 8. 复现

```powershell
# screw 640（抽样池）
.venv\Scripts\python.exe tools/run_experiment.py --run-dir experiment_data/06_optimization/13_screw640/23_screw640_seed_000 -- --category screw --layers 2+3 --size 640 --greedy 2000 --pool-cap 150000 --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 --split-seed 42 --seed 0 --weights torch_cache/wide_resnet50_2_timm.pth --data MVTec/mvtech_anomaly_detection --dump-pixel experiment_data/06_optimization/13_screw640/23_screw640_seed_000/pixel_dumps

# capsule 320 / 1+2
.venv\Scripts\python.exe tools/run_experiment.py --run-dir experiment_data/06_optimization/14_capsule12/12_capsule_seed_000 -- --category capsule --layers 1+2 --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 --split-seed 42 --seed 0 --weights torch_cache/wide_resnet50_2_timm.pth --data MVTec/mvtech_anomaly_detection --dump-pixel experiment_data/06_optimization/14_capsule12/12_capsule_seed_000/pixel_dumps
```
