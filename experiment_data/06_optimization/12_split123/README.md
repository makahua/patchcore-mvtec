# 12_split123 — split-seed 敏感性复验

`--split-seed 123` 下的 15 类 × 3 seed = 45 个 run，配置与 `10_pair/23_*`（split-seed 42）逐项相同：
`--size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 --weights torch_cache/wide_resnet50_2_timm.pth`。

因为 90 run 门槛判定后默认仍是 `2+3`，**最终配置 = 基线配置**，这一组同时充当两者的复验。

## 目录

每个 `<category>_seed_<NNN>/` 是标准六件套（`config.json` / `metrics.json` / `per_image.csv` / `summary.csv` / `timing.json` / `run.log`），
`native/` 下是 `results.csv`、`results.json`、`manifests/`、`scores/`、`heatmaps/`、`banks/`。

## 结果

| 指标 | split-seed 42 | split-seed 123 | 差值 |
|---|---:|---:|---:|
| image AUROC | 0.9791 ± 0.0010 | 0.9786 ± 0.0019 | -0.0005 |
| pixel AUROC | 0.9711 ± 0.0005 | 0.9714 ± 0.0003 | +0.0002 |
| 算法段时延 (ms) | 21.5153 ± 1.9477 | 21.3327 ± 1.0239 | -0.1827 |

逐类位移与判断见 [`docs/split-seed-check.md`](../../../docs/split-seed-check.md)。

## 复现

```bash
.venv\Scripts\python.exe tools\split_check.py \
  --a experiment_data/06_optimization/10_pair --b experiment_data/06_optimization/12_split123
```

> 数据角色：MVTec test 为 development-only，不构成 holdout 结论。