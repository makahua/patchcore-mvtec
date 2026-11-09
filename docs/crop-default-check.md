# crop 是否可作为默认预处理：8 类 × 3 seed 配对判定

> 候选：现有按类 profile（screw/capsule 用 `1+2`，其余 `2+3`）+ `--preprocess crop --crop-zoom 1.144`。
> 基线：同 seed 同图、stretch 预处理（screw 用 `16_screw12_p995`，capsule 用 `14_capsule12`，其余 `10_pair/23_*`）。
> 产物：`experiment_data/06_optimization/23_crop_cand/`；协议与数字见 `docs/crop-default-check.json`。

## 1. 总体

| 指标 | 基线 | 候选 | Δ |
|---|---:|---:|---:|
| image AUROC（8 类 macro） | 0.9722 | 0.9842 | +0.0120 |
| pixel AUROC | 0.9810 | 0.9696 | -0.0114 |
| AUPRO@30 | 0.9314 | 0.9088 | -0.0227 |
| 算法段时延 | — | — | +0.8 ms |

## 2. 逐类

| 类别 | Δimage | 95% CI | Δpixel | ΔAUPRO | Δ时延 |
|---|---:|---|---:|---:|---:|
| toothbrush | +0.0518 | +0.0151, +0.1004 | -0.0029 | -0.0196 | -0.0 ms |
| zipper | +0.0236 | +0.0092, +0.0414 | -0.0109 | -0.0174 | +0.7 ms |
| grid | +0.0108 | +0.0032, +0.0212 | -0.0046 | -0.0229 | +1.1 ms |
| carpet | +0.0088 | +0.0012, +0.0188 | -0.0015 | -0.0183 | +0.1 ms |
| screw | +0.0083 | +0.0012, +0.0158 | -0.0139 | -0.0212 | +1.5 ms |
| pill | +0.0080 | +0.0000, +0.0181 | -0.0212 | +0.0001 | +2.5 ms |
| metal_nut | -0.0004 | -0.0032, +0.0016 | -0.0275 | -0.0715 | +0.9 ms |
| capsule | -0.0148 | -0.0400, +0.0030 | -0.0084 | -0.0104 | -0.3 ms |

## 3. 门槛

- 配对 image 均值 ≥ 0：**PASS**（+0.0120）
- 无单类退化超过 0.01：**FAIL**（最差 capsule -0.0148）
- AUPRO 不退化超过 0.01：**FAIL**（最差 metal_nut -0.0715）
- 15 类算法段均值 ≤ 25 ms：本实验只覆盖 8 类，粗略值为 **22.4 ms**（仅供估计）

## 4. 结论

- **不能作为默认：至少一项门槛失败，保留为可选开关**
- 逐类方向：image 升 6 类、降 2 类；AUPRO 升 1 类、降 7 类

## 5. 复现

```powershell
.venv\Scripts\python.exe tools\run_experiment.py --run-dir experiment_data/06_optimization/23_crop_cand/toothbrush_seed_000 -- --data MVTec/mvtech_anomaly_detection --category toothbrush --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 --split-seed 42 --seed 0 --weights torch_cache/wide_resnet50_2_timm.pth --preprocess crop --crop-zoom 1.144 --profile configs/patchcore_profile.json --dump-pixel experiment_data/06_optimization/23_crop_cand/toothbrush_seed_000/pixel_dumps
.venv\Scripts\python.exe tools\pro_aupro.py --dumps experiment_data/06_optimization/23_crop_cand/toothbrush_seed_000/pixel_dumps --data MVTec/mvtech_anomaly_detection --category toothbrush --out experiment_data/06_optimization/23_crop_cand/toothbrush_seed_000/aupro.json
```
