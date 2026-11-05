# 特征层消融（任务 D）阶段记录

> 目录用途：`experiment_data/06_optimization/03_features/` 下每次运行一个目录，命名 `<layer_combo>_<category>_seed_NNN/`，内含六件套（`config.json` / `per_image.csv` / `metrics.json` + `summary.csv` / `timing.json` / `run.log`）、`native/`（原始 results.csv、scores、manifests、heatmaps）、`pixel_dumps/<category>.npz`、`aupro.json`。
> 所有 MVTec test 指标均为 **development-only**，不是独立 holdout，不能作为生产泛化结论。
> 最终分析报告：[`docs/features-ablation.md`](../../../docs/features-ablation.md)。

## 1. 固定协议

| 项 | 值 |
|---|---|
| 数据根 | `MVTec/mvtech_anomaly_detection` |
| 特征骨干 | timm `wide_resnet50_2.racm_in1k`（`torch_cache/wide_resnet50_2_timm.pth`，SHA256 `1810845932…7dc64`） |
| 输入尺寸 | 320（/4 网格 80×80，/8 网格 40×40） |
| 记忆库 | coreset 贪心 `k=2000`（三种组合同一预算） |
| 标定 | `--calib-ratio 0.3`，阈值 `--thr-pct 99` |
| 划分 | `--split-seed 42`，运行 `--seed 0/1/2` |
| 类别 | screw / toothbrush / pill |
| 组合 | `--layers 2+3`（默认）、`1+2`、`1+2+3` |
| 像素级标注 | `--dump-pixel` 保留原始分数图，AUPRO 由 `tools/pro_aupro.py` 离线计算 |

## 2. 实现与网格（只读审计结论）

`patchcore_demo.py` 的 `Features` 无 forward hook，直接持 `stem / l1 / l2 / l3`：

| 组合 | 融合方式 | 特征维度 | 网格 | patch/图 |
|---|---|---|---|---|
| `2+3`（默认） | `f3` 双线性上采样到 `f2` 的 /8 网格后 concat | 512+1024 = **1536** | 40×40 | 1600 |
| `1+2` | `f2` 双线性上采样到 `f1` 的 /4 网格后 concat | 256+512 = **768** | 80×80 | 6400 |
| `1+2+3` | `f2`、`f3` 都上采样到 /4 后与 `f1` concat | 256+512+1024 = **1792** | 80×80 | 6400 |

**口径更正（2026-11-05）**：第一版实现把 layer1 用 `avg_pool2d(2,2)` 降到 /8，导致 `1+2` 与 `1+2+3` 仍是 40×40 网格，等于只做通道拼接、丢掉 layer1 的空间分辨率。该版两次试跑已移入 `_invalid_layer1_downsampled/`，**只作调试记录，不进结论**。现行实现保持 layer1 原网格 /4，把深层上采样到 /4。

## 3. smoke 阶段结果（toothbrush，seed=0）

| 组合 | 候选池 | 记忆库 | image AUROC | pixel AUROC | AUPRO@30% | 召回 | 过杀率 | 算法时延 | 建库 | GPU 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2+3 | (67200, 1536) | (2000, 1536) | 0.9417 | 0.9902 | 0.9063 | 0.9667 | 0.2500 | 24.45 ms | 38.8 s | 188.0 MB |
| 1+2 | (268800, 768) | (2000, 768) | 0.9583 | 0.9894 | 0.9423 | 1.0000 | 0.4167 | 26.36 ms | 75.4 s | 229.6 MB |
| 1+2+3 | (268800, 1792) | (2000, 1792) | 0.8806 | 0.9911 | 0.9328 | 1.0000 | 0.5000 | 34.42 ms | 174.0 s | 354.5 MB |

- 三份均六件套齐，`pixel_dumps/toothbrush.npz` 分别 109,307 / 429,294 / 410,745 字节。
- 候选池验证：42 张 × 1600 = 67,200；42 张 × 6400 = 268,800。

## 4. AUPRO 计算方式

```powershell
.\.venv\Scripts\python.exe tools\pro_aupro.py `
  --dumps experiment_data\06_optimization\03_features\<run>\pixel_dumps `
  --data MVTec\mvtech_anomaly_detection `
  --out experiment_data\06_optimization\03_features\<run>\aupro.json
```

- 分数图双线性上采样回原图尺寸，与原始 GT mask 比对；4 邻接连通域、每区域等权；200 个阈值；AUPRO 截断 FPR ≤ 30%，同时输出 @10% 与 PRO@5%FPR。
- 与主结果 `pro_aupro.json`（40×40 网格、AUPRO@30% 0.9211）**不同网格不可直接比较**。

## 5. 完成情况（27 个计划 run）

| 组合 | screw | toothbrush | pill |
|---|---|---|---|
| `2+3` | 3 seed ✅ | 3 seed ✅ | 3 seed ✅ |
| `1+2` | 3 seed ✅ | 3 seed ✅ | 3 seed ✅ |
| `1+2+3` | **1 seed**（seed0） | 3 seed ✅ | **0** |

- 成功 22 个 run（含复用的 3 个 smoke、2 个预检），全部六件套 + AUPRO。
- 未完成 5 个：`1+2+3` 的 screw seed1/2、pill seed0/1/2 —— 候选池 9.6 GB / 11.4 GB，本机物理内存 15.9 GB，实测余量不足。screw seed1 曾启动，16 h 墙钟仅 3503 s CPU、无进度（页文件 thrash），人工终止；其余未启动。
- 7 个曾因驱动脚本把 `--layers` 拼成 `1+23` 被 argparse 拒绝（13 s 内 rc=2），目录在 `_aborted/failed_layers_arg/`，**不计入数据**；修复后重跑。

## 6. 结论（详见 docs/features-ablation.md）

- `1+2`：image AUROC +5.1 / +3.2 / +1.0 点（screw / toothbrush / pill），AUPRO +2.3 / +3.2 / +1.5 点；代价建库 2–3×、显存 +41.6 MB。
- `1+2+3`：相对 `1+2` 无收益且在 toothbrush 上 image −9.3 点（pixel 不降），成本翻倍。
- 默认保持 `2+3`，`1+2` 作为有条件的可选方案写进文档。
- 提醒：pixel AUROC 与 AUPRO 在 pill 上排序相反，报指标必须同时给两者 + 截断说明。
