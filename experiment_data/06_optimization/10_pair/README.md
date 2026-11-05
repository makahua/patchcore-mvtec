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
