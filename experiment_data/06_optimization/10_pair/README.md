# 10_pair —— 15 类 × 3 seed × 2 层组合的配对基线（协议：`docs/OPTIMIZATION_PLAN.md`）

## 协议固定项（与计划书 §1 一致，开跑后不改）

| 项目 | 值 |
|---|---|
| 权重 | `torch_cache/wide_resnet50_2_timm.pth`（SHA256 `1810845…5dc64`） |
| 输入 | 320×320，stretch，不做 ImageNet 归一化 |
| 记忆库 | legacy greedy：先随机取 10% 候选，再贪心选 2000 个中心 |
| 层组合 | `2+3`（1536 维 /40×40）对 `1+2`（768 维 /80×80） |
| 分数 | 高斯平滑 σ=2，图像聚合 `max` |
| 划分 | 70% 建库 / 30% calibration，`split-seed=42`；coreset seed 0/1/2 |
| 主阈值 | 经验 P99.5（`--thr-pct 99.5`），灰区 `thr_ng = thr_ok × 1.2` |
| 精度/批量 | FP32、`--batch 1` |
| 数据角色 | MVTec test 全部为 development-only |

目录命名：`<层组合>_<类别>_seed_<NNN>`，例如 `23_screw_seed_000`、`12_metal_nut_seed_001`。

## 每个 run 的产物

`config.json`（命令、环境、权重 SHA256、**源码 SHA256**、banks 清单）、`per_image.csv`、`metrics.json`、`summary.csv`、`timing.json`（含 **`peak_cpu_mb`**）、`run.log`、`banks/<类别>.npz`（记忆库 + 元数据）、`pixel_dumps/<类别>.npz`（不入库）、`aupro.json`。

## 第 0 步 smoke 结果（2026-11-06，seed 0）

| run | 层组合 | image AUROC | pixel AUROC | AUPRO@30 | 阈值 | 召回 | 过杀 | 算法段 ms | 建库 s | GPU 峰值 | CPU 峰值 | bank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `23_screw_seed_000` | 2+3 | 0.9240 | 0.9823 | 0.9270 | 38.623 | 63.0% | 0.0% | 19.77 | 252.0 | 189.6 MB | 3.5 GB | 9.5 MB |
| `12_screw_seed_000` | 1+2 | 0.9703 | 0.9910 | 0.9562 | 33.502 | 75.6% | 4.9% | 25.43 | 507.0 | 230.8 MB | 4.9 GB | 5.3 MB |
| `12_pill_seed_000` | 1+2 | 0.9602 | 0.9612 | — | 34.466 | 64.5% | 0.0% | 41.06 | 333.9 | 230.8 MB | 8.0 GB | 5.3 MB |

- 三份都通过产物验收：六件套 + `banks/` + `pixel_dumps/` + `aupro.json`，`tools/verify_protocol.py` 判定划分互斥且跨组合一致。
- **风险信号**：`1+2` 的算法段时延在 screw 上 25.4 ms、pill 上 41.1 ms，而计划书 §3 的时延门槛是「15 类均值 ≤ 25 ms」。按当前实现，`1+2` 很可能在时延门槛被否决——这一条要在正式 90 run 完成后按实测确认，不能提前下结论。
- `pill` 的 AUPRO 缺失是 smoke 未跑 `pro_aupro.py`（脚本只对前两个跑了）；正式批次会逐 run 补齐。
- CPU 峰值随候选池增长：`2+3` 3.5 GB（pill 预计 5 GB+）、`1+2` 8.0 GB。16 GB 机器可跑，但不能并行多进程。

## 复现

```bash
.venv\Scripts\python.exe tools/run_experiment.py \\
  --run-dir experiment_data/06_optimization/10_pair/12_screw_seed_000 -- \\
  --category screw --layers 1+2 --data MVTec/mvtech_anomaly_detection --size 320 \\
  --greedy 2000 --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 \\
  --split-seed 42 --seed 0 --weights torch_cache/wide_resnet50_2_timm.pth \\
  --dump-pixel experiment_data/06_optimization/10_pair/12_screw_seed_000/pixel_dumps
```

所有 MVTec test 指标均为 development（test 已反复查看），不是独立 holdout。

## 正式 90 run（2026-11-06 05:39–11:27，5.88 h，90/90 成功）

### 汇总（每 seed 先算 15 类 macro，再跨 3 seed 取均值 ± 样本标准差 ddof=1）

| 组 | image AUROC | pixel AUROC | AUPRO@30 | 算法段 ms |
|---|---:|---:|---:|---:|
| `2+3`（1536 维 /40×40） | **0.9791 ± 0.0010** | **0.9711 ± 0.0005** | 0.9203 ± 0.0015 | **21.52 ± 1.95** |
| `1+2`（768 维 /80×80） | 0.9822 ± 0.0009 | 0.9583 ± 0.0015 | **0.9211 ± 0.0020** | 24.60 ± 1.71 |
| 配对差（`1+2` − `2+3`） | +0.0031 | −0.0128 | +0.0008 | +3.08 |

逐 seed 的配对差：image `+0.0027 / +0.0031 / +0.0035`，pixel `−0.0143 / −0.0131 / −0.0111`，AUPRO `−0.0013 / +0.0000 / +0.0036`——**三个 seed 的符号完全一致**，不是单次抖动。

### 单类去向（`1+2` − `2+3`，三 seed 平均）

| 类别 | Δimage | ΔAUPRO | 读法 |
|---|---:|---:|---|
| cable | **−0.0500** | −0.0467 | 线缆的细长结构在高分辨率网格上被拆散，判定与定位同时掉 |
| carpet | **−0.0315** | +0.0018 | 判定掉、定位持平：纹理类不吃这个高分辨率 |
| transistor | −0.0179 | **−0.2308** | 定位崩掉，pixel AUROC 0.948 → 0.786 |
| tile | −0.0123 | +0.0120 | 判定小掉、定位小升 |
| screw | **+0.0506** | +0.0230 | 最大受益者 |
| toothbrush | +0.0324 | +0.0322 | 小缺陷受益 |
| grid | +0.0292 | +0.0689 | 定位受益最明显 |
| zipper | +0.0250 | +0.0318 | 受益 |
| pill | +0.0104 | +0.0154 | 时延 29.3 ms，超出单类预算 |

### 三档判定（候选组，`thr_ng = thr_ok × 1.2`）

| seed | 漏检 | 实际 NG 过杀 | 复核负荷 |
|---:|---:|---:|---:|
| 0 | 220/1258 = 17.5% | 0/467 = 0.0% | 638/1725 = 37.0% |
| 1 | 204/1258 = 16.2% | 0/467 = 0.0% | 643/1725 = 37.3% |
| 2 | 189/1258 = 15.0% | 0/467 = 0.0% | 662/1725 = 38.4% |

对照组 `2+3` 的复核负荷是 24.1% / 24.6% / 25.0%——`1+2` 的分数分布更分散，同样 1.2× 灰区要多送 **约 13 个百分点**的图去人工复核。

### 默认切换门槛

| 门槛 | 结果 | 实测 |
|---|---|---|
| 完整性：两组各 15 类 × 3 seed | PASS | 类别 15，共同 seed 3 |
| macro image AUROC 配对均值差 ≥ 0 | PASS | +0.0031 |
| macro AUPRO@30 配对均值差 ≥ 0 | PASS | +0.0008 |
| 单类退化：任一类三 seed 平均下降 ≤ 0.03 | **FAIL** | 最差 Δimage −0.0500（cable）；ΔAUPRO 最差 −0.2308（transistor） |
| 算法时延：15 类均值 ≤ 25 ms | PASS（贴线） | 24.60 ms |

**结论：默认保持 `2+3`。** `1+2` 的平均分更好看（image +0.31 点、AUPRO +0.08 点），但代价是三处不可接受：

1. **cable 判定掉 5.0 个点、transistor 定位掉 23.1 个点**——单类退化门槛直接否决，而且方向是「结构细节反而不见了」，不是噪声。
2. 整体 pixel AUROC 掉 1.28 点：`/4` 网格上的 patch 更小，缺陷内部反而更容易被背景稀释。
3. 复核负荷从 ~24% 涨到 ~37%，产线上等于多派 1.5 倍人力；时延 24.60 ms 也已贴住 25 ms 门槛。

`1+2` 只对 screw / toothbrush / grid / zipper 这类「细小、局部、纹理规则」的缺陷有稳定收益，作为**按类可选项**记录，不进默认；这条与 `docs/features-ablation.md` 的旧结论方向一致（旧实验是 320/单 seed，本次是三 seed、P99.5 口径复验）。

### 产物

- 每个 run：六件套 + `banks/<类别>.npz` + `pixel_dumps/<类别>.npz`（不入库，可由命令重建）+ `aupro.json`
- 汇总：`sweep_status.json`（90 行状态）、`pair_12_vs_23.json`（`tools/pair_stats.py --out` 的完整输出）
- 校验：`tools/verify_protocol.py --root experiment_data/06_optimization/10_pair` 通过；`tools/pair_stats.py` 覆盖四道门槛

### 复现

```bash
.venv\Scripts\python.exe tools/pair_stats.py \
  --root experiment_data/06_optimization/10_pair --baseline 23 --candidate 12 \
  --out experiment_data/06_optimization/10_pair/pair_12_vs_23.json
```
