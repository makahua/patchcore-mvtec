# experiment_data 索引

本目录集中存放**可复算的数字与索引**。数据集、权重、记忆库、热力图都很大且不可再分发，全部被 `.gitignore` 排除；换机器后需要重新生成（见 [`docs/REPRODUCE.md`](../docs/REPRODUCE.md)）。

## 目录结构

| 层 | 内容 | 子目录 | 入仓 |
|---|---|---|---|
| `00_baselines` | MVTec 早期基线 sweep（分辨率、coreset、邻域、归一化对照） | `results`、`results_320`、`results_best`、`results_fix`、`results_greedy`、`results_neigh`、`results_normtest` | ✅ 除 `heatmaps/`、`scores/`（`results_best/scores` 例外） |
| `01_mvtec_gpu` | GPU 上的主实验：平滑消融、标定、少样本曲线、扰动鲁棒性、PRO 落盘 | `results_gpu_blur`、`results_gpu_blur_confirm`、`results_gpu_calib`、`results_gpu_fewshot`、`results_gpu_fewshot_curve`、`results_gpu_fewshot_fixedcal`、`results_gpu_robust`（亮度/噪声/模糊 × 冻结+重标定）、`results_preproc_{stretch,letterbox}_{screw,pill,transistor}`、`results_pro_dump`、`results_p3_*`（9 邻域消融） | ✅ 同上 |
| `02_visa` | VisA 开发集（PCB1）与一次锁定配置的外部检查（PCB2） | `results_visa_pcb1`、`results_visa_smoke`、`results_visa_ui`、`results_visa_top5_ui`、`results_external_visa_pcb2` | ✅ |
| `03_triage` | 灰区/复核量 triage 结果 | `triage_visa_pcb1`、`triage_visa_pcb1_rebuilt`、`triage_visa_fpr05`、`triage_visa_top5_fpr05` | ✅ |
| `04_memory_banks` | 记忆库 `.npz`（MVTec 1 套 + VisA 3 套） | `banks`、`banks_visa_fpr05`、`banks_visa_pre_orderfix`、`banks_visa_top5_fpr05` | ❌ `.gitignore` |
| `05_indexes` | 汇总索引 | `all_results.csv`（**170 行 × 38 列，合并 86 个结果目录**，由 `tools/index_results.py` 生成，勿手改）、`file_inventory.csv`（文件校验清单）、`weights_sha256.txt`（权重来源与校验值）、`latency_*.json`（8 段时延：metal_nut 700² / bottle 900² / carpet 1024² 为真实分辨率，2048²/4096² 为 bicubic 上采样合成图、只读趋势）、`pro_aupro.json` + `pro_table.md`（PRO/AUPRO 逐类结果，`pixel_dumps/` 由 `--dump-pixel` 再生成故不入仓） | ✅ 除 `pixel_dumps/` |
| `06_demo_outputs` | 演示产物 | `demo_out`、`demo_out_interview`（`scores.json` + 热力图） | ✅ 仅分数 |
| `06_optimization` | 执行版计划的规范化实验包（每目录六件套：`config.json` / `per_image.csv` / `metrics.json` / `summary.csv` / `timing.json` / `run.log`；`per_image.csv` 的 `split` 为 `held_out`（标定留出）+ `test`） | `00_baseline`、`05_latency`（串行 vs 多线程解码+有界预取，见 [`docs/latency-breakdown.md`](../docs/latency-breakdown.md) §3.7）、`01_coreset_fixed`（screw k=500/1000/2000/4000）、`02_coreset_projected`（screw ratio=1% 与同预算 0.558%）、`03_features`（特征层消融 2+3 / 1+2 / 1+2+3，见 [`docs/features-ablation.md`](../docs/features-ablation.md)）、`03_smoke_coreset`、`04_representative`（bottle/grid）；`00_protocol_smoke` 缺 5 件，为早期协议冒烟，保留原样；`05_audit` 为 screw 复现探针与旧代码对照（结论见 `docs/baseline-acceptance.md`）、`06_threshold`（阈值工作点与 bootstrap 区间，见 [`docs/threshold-stability.md`](../docs/threshold-stability.md)）、`07_aupro_backfill`（coreset 8 个关键点补测 AUPRO，见 [`docs/coreset-budget.md`](../docs/coreset-budget.md) §8）、`08_fp16`（FP16 精度闸门与吞吐验证，见 [`docs/e2e-ceiling.md`](../docs/e2e-ceiling.md) §9–§10）、`09_batch`（批推理吞吐与 b8 精度闸门，见同文件 §11）、`10_pair`（新执行计划的 15 类 × 3 seed × 2 层组合配对基线，协议见 `docs/OPTIMIZATION_PLAN.md`）、`11_robust_agg`（5 种图像聚合 × 5 组扰动，见 [`docs/robust-aggregation.md`](../docs/robust-aggregation.md)）、`12_split123`（split-seed=123 的 45 个复验 run，见 [`docs/split-seed-check.md`](../docs/split-seed-check.md)）、`13_screw640`（640 输入 / `2+3`，必须 `--pool-cap`；见 [`docs/screw-resolution.md`](../docs/screw-resolution.md)）、`14_capsule12`（capsule 320 / `1+2`）、`15_phase1`（成本对账与阈值敏感性）、`16_screw12_p995`（screw 320 / `1+2` / P99.5）；汇总用 `tools/summarize_runs.py`（coreset）、`tools/summarize_features.py`（特征层）与 `tools/threshold_ci.py`（阈值） | ✅ 除 `native/` 下的权重与记忆库 |、`17_split123_l12`（screw/capsule 按类 1+2 的 split-123 复核）

## 当前主结果（`00_baselines/results_best`）

> 该表的运行产物（`scores/*.json`）生成于 config 尚未归档的早期状态，与任务 A 规范化基线存在系统性差异（screw −3.13 点、capsule +2.83 点，macro 0.9805 → 0.9789）。差异排查与结论见 [`docs/baseline-acceptance.md`](../docs/baseline-acceptance.md)。

配置：WR50 timm 权重、`--size 320`、`--greedy 2000`、先良品留出 `--calib-ratio 0.3` 标定、`--thr-pct 99`、score 聚合 `max`、高斯 `σ=2`。

- image AUROC **0.9805**（15 类 macro）／**0.8812**（同批图像 pooled）
- pixel AUROC **0.9709**，image-level PR-AUC **0.993**
- 推理与最近邻评分均值 **19.7 ms（算法段：`patch_scores()` 内部，不含读图解码）**；端到端 900² 为 42.2 ms，显存峰值约 **190 MB**，记忆库 **2,000 patch/类**
- 区域级 **AUPRO@FPR≤30% 0.9211 ± 0.0398**、@10% 0.8068（`05_indexes/pro_aupro.json`，见 [`docs/pro-aupro.md`](../docs/pro-aupro.md)）

## 复算方式

```powershell
cd E:\面试项目文件夹\patchcore-mvtec
.venv\Scripts\python.exe classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2
```

`report` 只读 `results_best/scores/`（唯一入仓的逐图分数，139 KB），不需要 GPU、不需要数据集。

重建索引（新增结果目录后必跑，CI 会校验一致性）：

```powershell
.venv\Scripts\python.exe tools/index_results.py            # 重建 all_results.csv
.venv\Scripts\python.exe tools/index_results.py --check    # 只校验
```

## 列约定

`results_best/results.csv` 与 `05_indexes/all_results.csv` 每行都带配置列：`size / neighbors / neighbor_coreset / neighbor_pad / preprocess / aspect_crop / imagenet_norm / coreset / coreset_k / coreset_ratio / calib_ratio / thr_pct / seed / weights`，新跑的结果可自证配置。`all_results.csv` 由 `tools/index_results.py` 扫全部 `results_*/results.csv` 生成，列取并集（当前 38 列，含 `train_limit / train_seed / split_seed / perturb_*`）；更早的目录缺的列留空，不做猜测填充。
