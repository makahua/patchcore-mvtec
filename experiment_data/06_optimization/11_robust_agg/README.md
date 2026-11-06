# 11_robust_agg：5 种聚合 × 5 组条件

第 ⑤ 项的产物目录。9 个 run（screw / pill / toothbrush × coreset seed 0/1/2），每次只建一次库，然后用同一个库对五组条件各 dump 一份 calib/test 分数图，离线比较 5 种图像聚合。

## 协议

- 记忆库：legacy greedy 10% → 2000 中心、`2+3`（1536 维）、`size=320`、stretch、无 ImageNet 归一化
- 阈值：经验 P99.5（fpr=0.005）、灰区 ×1.2；frozen（干净 calib 冻结）与 recal（同条件 calib 重标定）两套
- 条件：`clean`、`bright08`、`noise005`、`noise010`、`blur1`
- 聚合：`max`、`top1pct`、`top5pct`、`top10pct`、`trimmed`
- MVTec test **development-only**

## 目录

```
<cat>_seed_<NNN>/            六件套 + native/ + suite/<cat>_<cond>_{calib,test}.npz
agg_matrix.json              全部单元格（auroc / frozen / recal）
agg_matrix.md                同上，markdown
sweep_status.json            9 个 run 的状态
```

`suite/*.npz` 含 `scores (N,40,40) float32`、`masks`、`labels`、`paths`、`thr_pct`、`size`、`dtype`、`condition`。

## 结论（详见 docs/robust-aggregation.md）

| 条件 | 最好的聚合 | macro AUROC |
|---|---|---:|
| clean | `top5pct` | 0.9591（`max` 0.9392） |
| bright08 | `top1pct` | 0.9534（`max` 0.9438） |
| noise005 | `top5pct` | 0.7417（`max` 0.7388） |
| noise010 | `max` | 0.6415 |
| blur1 | `top5pct` | 0.8548（`max` 0.7875） |

筛查判定 **不通过**（σ=0.10 时没有一种聚合能比 `max` 提升 0.03），默认继续保持 `max`；噪声是特征退化，不是聚合问题。

## 复现

```powershell
.venv\Scripts\python.exe tools\run_experiment.py --run-dir experiment_data\06_optimization\11_robust_agg\screw_seed_000 -- `
  --category screw --seed 0 --data MVTec/mvtech_anomaly_detection --size 320 --greedy 2000 `
  --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 --split-seed 42 `
  --weights torch_cache/wide_resnet50_2_timm.pth --dump-suite <run>\suite --dump-dtype float32

.venv\Scripts\python.exe tools\agg_matrix.py --suite experiment_data\06_optimization\11_robust_agg
```
