# 基线验收记录（任务 A）

> 日期：2026-11-05　｜　范围：计划 §9.3 任务 A「建立可复算基线」　｜　**本记录不覆盖任务 B–F**
>
> 所有 MVTec test 数字均为 **development-only**：该 test 已反复参与项目探索，不构成独立 holdout，不可对外宣称产线泛化。

## 1. 结论

1. **包内结果可复算。** `experiment_data/06_optimization/00_baseline/mvtec_seed_000` 六件套齐全；从 `per_image.csv` 复算的逐类 image AUROC 与 `metrics.json` 最大偏差 **4.7×10⁻⁵**（recall@thr、FPR 偏差 ≤5×10⁻⁴）；macro/pooled 复算一致。两次独立重跑与包内记录**逐位一致**。
2. **与 README 主指标有系统性差异。** macro image AUROC **0.9789 vs 0.9805**（−0.16 点）、pooled **0.8791 vs 0.8812**（−0.21 点）；逐类最大偏差 **screw −3.13 点**、**capsule +2.83 点**，其余 13 类 ≤0.6 点；15 类阈值全部不同。
3. **该差异不是重复运行波动。** 同命令重跑 σ = 0（三次调用得到同一组分数）；screw 换建库 seed 的跨 seed 标准差也只有 0.65 点。权重 SHA256、split 行数、预处理、评测口径都逐项排除后，仍无法归因——`results_best` 的 `scores/*.json` 只记录 `category/thr/thr_pct/calib/test`，没有 config、权重、环境，属于**不可追溯的早期快照**。
4. **按任务 A 第 4 步，已暂停后续 sweep。** 未改动 README 主表、未改默认配置、未重跑全量、未动历史结果。

## 2. 验收环境（与 config.json 一致）

| 项 | 值 |
| --- | --- |
| Python | 3.13.9 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| numpy | 2.5.4 |
| CUDA / GPU | 12.8 / NVIDIA GeForce RTX 3050 Laptop GPU |
| 权重 | `torch_cache/wide_resnet50_2_timm.pth`，SHA256 `18108459320717df6bf3f3673c0812203d1cffe979cfda43803389354947dc64` |
| git commit | `5d196f7`（运行时刻记录） |
| 命令 | `--data MVTec/mvtech_anomaly_detection --category all --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --seed 0 --split-seed 42 --weights torch_cache/wide_resnet50_2_timm.pth` |
| 耗时 | 2485.1 s（15 类，全量 test 1725 张） |

## 3. 六件套

| 文件 | 字节 | 作用 |
| --- | --- | --- |
| `config.json` | 3017 | 命令、参数、环境、GPU、权重路径 + SHA256、输入清单、development 声明 |
| `per_image.csv` | 175399 | 逐图分数/标签/类别/split（held_out 1085 + test 1725 = 2810 行） |
| `metrics.json` | 16327 | 逐类 + macro + pooled，含 `dataset_role` |
| `summary.csv` | 3140 | 与 `native/results.csv` 逐字节相同 |
| `timing.json` | 2629 | 总时长 + 逐类 latency/p95/build/peak |
| `run.log` | 15653 | 完整 stdout |

`native/` 另含 `results.csv`、`results.json`、`manifests/<cat>.json`、`scores/<cat>.json`、`heatmaps/`。

## 4. 审计项（198 项全部通过）

- **config 可追溯**：命令含 `--size 320 / --greedy 2000 / --calib-ratio 0.3 / --thr-pct 99 / --split-seed 42 / --seed 0 / --category all`；权重 SHA256 与磁盘文件一致；记录 python、git commit；note 声明 development-only。
- **六件套一致**：`summary.csv` == `native/results.csv`（逐行逐列）；metrics 15 类 == summary 15 类。
- **manifest vs 磁盘**（15 类逐一）：`train + calibration` 与 `train/good` 全集合相等且互不相交；`test` 缺陷清单与 `test/<defect>/` 相等；`test/good` 清单相等；manifest 里每个路径都存在于磁盘。
- **per_image vs manifest**：路径集合逐一相等（calib + test）；行数与 `n_calib/n_test` 一致；pooled test = 1725。
- **指标复算**：逐类 image AUROC（效应量法）、recall@thr、FPR 全部在容差内；macro/pooled 与 metrics.json 一致。
- **timing 一致**：逐类 `timing.json.latency_ms` == `metrics.latency_ms`；总时长一致。

复算方式：临时审计脚本从 `per_image.csv` 与 manifest 重建（脚本未入库，避免在仓库新增一次性代码；步骤见第 8 节）。

## 5. 复现实验（screw，同命令）

| 运行 | image AUROC | pixel AUROC | 阈值 | recall@thr | FPR | 建库 s |
| --- | --- | --- | --- | --- | --- | --- |
| 包内基线 `00_baseline/mvtec_seed_000` | 0.9240 | 0.9823 | 38.18 | 0.6723 | 0.0244 | 190.2 |
| 重跑 1 `05_audit/screw_repro_seed_000` | 0.9240 | 0.9823 | 38.18 | 0.6723 | 0.0244 | 208.7 |
| 重跑 2 `05_audit/screw_repro_seed_001` | 0.9240 | 0.9823 | 38.18 | 0.6723 | 0.0244 | 206.7 |
| 旧代码 `a38d1f7` 重跑 `05_audit/screw_oldcode_seed_000` | 0.9240 | 0.9823 | 38.18 | 0.6723 | 0.0244 | 206.0 |

结论：当前代码是确定性的；`a38d1f7`（P3 重构前）与 HEAD 在 legacy greedy 路径上给出同一组分数，说明 **coreset/邻域重构没有改变旧路径数值**。

## 6. 与 README 主指标（`results_best`）逐类对照

| 类别 | best img | base img | Δ img | best thr | base thr |
| --- | --- | --- | --- | --- | --- |
| bottle | 1.0000 | 1.0000 | +0.0000 | 37.99 | 38.63 |
| cable | 0.9843 | 0.9854 | +0.0011 | 47.93 | 47.45 |
| capsule | 0.9625 | 0.9908 | **+0.0283** | 36.36 | 37.30 |
| carpet | 0.9843 | 0.9835 | −0.0008 | 33.25 | 34.93 |
| grid | 0.9641 | 0.9641 | +0.0000 | 34.03 | 35.91 |
| hazelnut | 1.0000 | 1.0000 | +0.0000 | 49.24 | 49.67 |
| leather | 1.0000 | 1.0000 | +0.0000 | 30.82 | 32.25 |
| metal_nut | 0.9990 | 0.9961 | −0.0029 | 45.65 | 43.55 |
| pill | 0.9555 | 0.9498 | −0.0057 | 40.39 | 41.67 |
| screw | 0.9553 | 0.9240 | **−0.0313** | 42.98 | 38.18 |
| tile | 1.0000 | 1.0000 | +0.0000 | 40.42 | 40.23 |
| toothbrush | 0.9472 | 0.9417 | −0.0055 | 36.45 | 34.87 |
| transistor | 0.9958 | 0.9971 | +0.0013 | 45.68 | 44.15 |
| wood | 0.9912 | 0.9860 | −0.0052 | 47.11 | 40.33 |
| zipper | 0.9677 | 0.9651 | −0.0026 | 32.46 | 34.09 |

pixel AUROC 逐类最大偏差 0.0022（screw）；macro/pooled 见第 1 节。

## 7. 偏差诊断（逐项排除）

| 假设 | 检查 | 结果 |
| --- | --- | --- |
| 权重文件不同 | 重算 `torch_cache/wide_resnet50_2_timm.pth` SHA256 与 config 记录比对 | 一致，排除 |
| split 不同 | manifest 与 metrics 的 `n_train/n_calib/n_test` 对 `results_best.results.csv` | 15 类全一致（224/96/160 等），排除 |
| 预处理不同 | `scores/*.json` 记 `preprocess=stretch`、`aspect_crop=0`，与默认一致 | 排除 |
| 评测口径不同 | 由 per_image 复算 AUROC/recall/FPR 与 metrics 对比 | 偏差 ≤5e-4，排除 |
| 代码漂移 | `a38d1f7` 旧代码同命令重跑 | 与 HEAD 同分，排除 |
| 运行噪声 | 连续两次重跑 | 分数完全一致，排除 |

**剩余解释**：`results_best` 生成于某个未归档的代码/环境状态（权重文件曾被替换、或早期 torch 版本不同），其 artifacts 中没有 config/环境字段，**无法事后归因**；因此本记录不做"哪个更对"的判断，只记录两者不可互证。

## 8. 已知不可复算项与限制

1. **pixel AUROC / AUPRO 不在包内。** `per_image.csv` 只有图像级分数；像素级指标需 `--dump-pixel` 重跑才有原始图，基线包里的 pixel 数值只能追溯到 `metrics.json` 自身记录。
2. **timing 的百分位口径未定义。** 打包 `timing.json` 只写 `latency_ms` / `latency_p95_ms`；capsule 出现 mean 30.04 > p95 29.14（重尾或首图预热），说明两列不是同一统计量的不同分位。任务 E 统一定义前，这两列不可与端到端拆解直接并列。
3. **`00_protocol_smoke` 仍缺 5 件**，按"禁止覆盖/删除"保留原样。
4. 本记录只覆盖 15 类全量基线；未验证独立 validation（MVTec 无此划分）。

## 9. 复算与复核命令

```powershell
# 汇总 06_optimization 下已跑配置（只读）
.venv\Scripts\python.exe tools/summarize_runs.py --root experiment_data/06_optimization/00_baseline

# 单类重跑（期望 screw: img 0.9240 / px 0.9823 / thr 38.18）
.venv\Scripts\python.exe tools/run_experiment.py `
  --run-dir experiment_data/06_optimization/05_audit/screw_repro_seed_00X -- `
  --data MVTec/mvtech_anomaly_detection --category screw --size 320 --greedy 2000 `
  --calib-ratio 0.3 --thr-pct 99 --seed 0 --split-seed 42 `
  --weights torch_cache/wide_resnet50_2_timm.pth
```

审计产物：`05_audit/repro_summary.json`（三次运行摘要）、`05_audit/screw_oldcode_seed_000/`（旧代码对照）。

## 10. 待决

1. README 主表（`0.9805 / 0.8812`）是否改为引用本基线（`0.9789 / 0.8791`），或加"历史快照，配置未归档"标注——**本轮未改**，等确认。
2. 是否用全量重跑把 `00_baseline` 升格为唯一主基线（现包已满足任务 A，直接复用即可）。
3. 任务 B–F 是否继续（本记录结论：差异已排查并记录，可按计划推进，但主指标口径需先定）。
