# 图像分数聚合的鲁棒性：5 种聚合 × 5 组条件

> 计划 §5。回答三个问题：换一种聚合能不能救噪声/模糊下的判定？对干净图有没有代价？
> 结论一句话：**噪声不是聚合问题**（所有聚合在 σ=0.10 都掉到 0.59–0.64），**干净图上 `top5pct` 更好**（macro 0.9591 对 0.9392），但**筛查判定不通过，默认保持 `max`**。

## 1. 协议

| 项 | 设置 |
|---|---|
| 类别 | screw、pill、toothbrush（3 类） |
| seed | coreset seed 0/1/2，`split-seed=42` |
| 记忆库 | legacy greedy 10% → 2000 中心，`2+3`（1536 维） |
| 图像 | `size=320`、stretch、不做 ImageNet 归一化 |
| 扰动 | `clean`、`bright08`（×0.8）、`noise005`（σ=0.05）、`noise010`（σ=0.10）、`blur1`（σ=1px） |
| 聚合 | `max`、`top1pct`、`top5pct`、`top10pct`、`trimmed`（去掉最高 1% 后取次 10% 均值） |
| 阈值 | 经验 P99.5（fpr=0.005），灰区 ×1.2；两套口径：**frozen**（干净 calib 标定后冻结）与 **recal**（同条件 calib 重标定） |
| 精度 | FP32、batch=1、`--dump-dtype float32` |
| 数据集角色 | MVTec test **development-only**，不宣称 holdout |

实现：`patchcore_demo.py --dump-suite DIR`（同一记忆库对五组条件各存 `calib`/`test` 分数图）+ `tools/agg_matrix.py`（离线算五种聚合）+ `tools/agg_robust.py`（聚合与判定原语）。

## 2. 图像 AUROC（3 类 × 3 seed 的 macro 均值）

| 条件 | max | top1pct | top5pct | top10pct | trimmed |
|---|---:|---:|---:|---:|---:|
| clean | 0.9392 | 0.9538 | **0.9591** | 0.9512 | 0.9385 |
| bright08 | 0.9438 | **0.9534** | 0.9499 | 0.9408 | 0.9250 |
| noise005 | **0.7388** | 0.7404 | 0.7417 | 0.7342 | 0.7191 |
| noise010 | **0.6415** | 0.6300 | 0.6068 | 0.5956 | 0.5891 |
| blur1 | 0.7875 | 0.8299 | **0.8548** | 0.8503 | 0.8423 |

## 3. 冻结阈值下的判定（macro 平均）

| 条件 | 聚合 | 召回 | 过杀 | 复核 |
|---|---|---:|---:|---:|
| clean | max | 0.763 | 2.78% | 38.2% |
| clean | top5pct | 0.792 | 0.00% | 43.3% |
| noise005 | max | 1.000 | **68.1%** | 21.1% |
| noise005 | top5pct | 1.000 | **100.0%** | 0.0% |
| noise010 | max | 1.000 | **100.0%** | 0.3% |
| blur1 | max | 0.890 | 62.0% | 16.6% |
| blur1 | top5pct | 0.964 | 66.7% | 23.0% |

重标定后（同条件 calib）噪声下召回只有 22.5%–31.0%，模糊 42.7%–48.0%，与聚合方式几乎无关——说明**信息已经丢了**，不是分数聚合能补回来的。

## 4. 逐类细节（AUROC，三 seed 均值）

| 条件 | 类别 | max | top1pct | top5pct | top10pct | trimmed |
|---|---|---:|---:|---:|---:|---:|
| clean | pill | 0.9520 | 0.9539 | 0.9493 | 0.9404 | 0.9264 |
| clean | screw | 0.9230 | 0.9326 | 0.9299 | 0.9214 | 0.9031 |
| clean | toothbrush | 0.9426 | 0.9750 | **0.9981** | 0.9917 | 0.9861 |
| noise010 | pill | 0.5104 | 0.4983 | 0.4856 | 0.4775 | 0.4765 |
| noise010 | screw | 0.5179 | 0.4890 | 0.4329 | 0.4112 | 0.3991 |
| noise010 | toothbrush | 0.8963 | 0.9028 | 0.9019 | 0.8981 | 0.8917 |
| blur1 | pill | 0.9305 | 0.9295 | 0.9160 | 0.9010 | 0.8796 |
| blur1 | screw | 0.4959 | 0.6138 | 0.6965 | **0.7019** | 0.7027 |
| blur1 | toothbrush | 0.9361 | 0.9463 | 0.9519 | 0.9481 | 0.9444 |

## 5. 五条判断

1. **噪声击穿的是特征，不是聚合。** σ=0.10 下最大值的 0.6415 已经是所有聚合里最好的；换成 top-k 均值平均还掉 1.1–5.2 个点（screw 掉得最多 −11.9 点）。这和 `docs/robustness.md` 的观察一致：噪声让 patch 分数整体抬升、缺陷与良品的**相对排序**被破坏，平滑聚合只会把抬升铺得更平。
2. **干净图上 `top5pct` 明显更好**：macro 0.9591 对 0.9392（+1.99 点），几乎全部来自 toothbrush（0.9426 → 0.9981）；screw +0.7 点、pill −0.3 点。这符合直觉：小缺陷只占少量 patch，最大值对单点噪声敏感，top-5% 均值更稳。
3. **模糊下 `top5pct` 帮 screw**：0.4959 → 0.6965（+20.1 点），macro +6.7 点；但冻结阈值下过杀从 62% 升到 67%，必须先重标定才能用。
4. **阈值比聚合更脆弱**：干净 calib 标定的 P99.5 遇到噪声直接 100% 过杀（noise010 全聚合），bright08 却几乎无损（0% 过杀）。原因是亮度扰动近似单调缩放（分数尺度变了但排序没变），噪声则是把分布整体抬高。
5. **筛查判定：不通过。** 计划 §5 要求「σ=0.10 时至少 2/3 类比 max 提升 ≥0.03、且干净图下降 ≤0.01」。实际所有 top-k 在 σ=0.10 都是负增益（pill −1.2～−3.4 点、screw −2.9～−11.9 点，仅 toothbrush 的 top1pct +0.6 点），**默认继续保持 `max`**。

## 6. 结论与建议

- **默认不变**（`max`）。理由不是它最好，而是它**没有单项显著更差**，且在噪声下退化最少。
- 若要吃 `top5pct` 在干净图的 +2 点，需要同时解决两件事：① 对噪声做前置抑制（中值/双边滤波或训练期增强），② 按批次重标定阈值；否则换聚合只会在产线上放大过杀。
- 模糊场景（镜头失焦）建议直接判为**不可用工况**：重标定后召回也只有 43%–48%，且过杀 0%，意味着「几乎全放行」。
- 噪声 0.10 的结论与 `docs/robustness.md` 一致：**报告失效形态，而不是只报 AUROC**。

## 7. 复现

```powershell
# 9 个 run（3 类 × 3 seed），每个建库一次、dump 五组条件
.venv\Scripts\python.exe tools\run_experiment.py --run-dir experiment_data\06_optimization\11_robust_agg\screw_seed_000 -- `
  --category screw --seed 0 --data MVTec/mvtech_anomaly_detection --size 320 --greedy 2000 `
  --calib-ratio 0.3 --thr-pct 99.5 --blur-sigma 2 --split-seed 42 `
  --weights torch_cache/wide_resnet50_2_timm.pth --dump-suite <run>\suite --dump-dtype float32

# 矩阵
.venv\Scripts\python.exe tools\agg_matrix.py --suite experiment_data\06_optimization\11_robust_agg
```

产物：`experiment_data/06_optimization/11_robust_agg/`（9 个六件套 + `suite/*.npz` + `agg_matrix.json` / `agg_matrix.md`）。

## 8. 局限

1. 只覆盖 3 类（screw/pill/toothbrush），且这三类在 `docs/features-ablation.md` 里就是「难类」，不代表 15 类平均。
2. 扰动是合成的（亮度乘法、高斯噪声、高斯模糊），不是真实相机退化（离焦、运动模糊、曝光漂移）。
3. 重标定用的是**同条件**的 calib 图；产线上没有「同条件良品」可标定时只能用冻结阈值，而过杀会到 60%–100%，这本身就是结论。
4. 没有做去噪/增强的前置处理，只在聚合端做文章。
5. 阈值口径仍是经验分位（P99.5），没有 conformal 有限样本保证；`docs/conformal-thresholds.md` 已说明 0.5% 在本数据量下不可达。
6. 单次运行、无 bootstrap 区间；聚合之间的差异（如 clean 上 +1.99 点）未做显著性检验。
