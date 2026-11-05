# 09_batch —— 批推理吞吐与 b8 精度闸门

## 为什么有这一组

`docs/e2e-ceiling.md` §10 测到 FP16 预取 67.8 ips 后判断「已 GPU-bound，只有批推理还有空间」。本目录是那一步的实测：吞吐提到 95.6 ips，并用带 dumps 的 b8 重跑验证指标不变。

## 内容

| 目录/文件 | 说明 |
|---|---|
| `screw_b1_fp32_seed_000` | b1 回归（验证 `--batch 1` 与历史逐位一致） |
| `screw_b8_fp32_seed_000` | screw b8（带 `pixel_dumps` + `aupro.json`） |
| `bottle_b8_fp16_seed_000` | bottle b8 + fp16（带 `pixel_dumps` + `aupro.json`） |
| `e2e_bottle_b8_fp16.json` | 预取 w4/d8 + batch8 + fp16：**95.55 ips** |
| `e2e_bottle_b4_fp16.json` | batch4 + fp16：95.82 ips（与 b8 打平） |
| `e2e_bottle_b8_fp32.json` | batch8 + fp32：57.04 ips |
| `batch_status.json` | 运行状态 |

## 结论

- 吞吐：b1 fp32 27.5（串行）/ 49.9（预取）→ fp16 67.8 → **b8+fp16 95.6 ips**；b4 已达 95.8，说明收益在 b4 吃满、再加 batch 只多占显存（259 → 345 MB）。
- 精度：screw b8 image AUROC 0.9238（b1 0.9240），pixel/AUPRO/召回/过杀/阈值全同；bottle b8 与 b1 完全一致。**通过闸门**。
- 代价：峰值显存 +150 MB；batch>1 时 cuDNN 换算法导致**分数不再逐位一致**（max|Δ|≈0.0088，0.03%），指标四位小数不变，但换阈值要重标定。
- 算法段时延：screw 18.23 → 15.84 ms，bottle(fp16) 13.74 → 9.79 ms（按张均摊）。

## 复现

```bash
.venv\Scripts\python.exe tools/run_experiment.py \\
  --run-dir experiment_data/06_optimization/09_batch/bottle_b8_fp16_seed_000 -- \\
  --category bottle --data MVTec/mvtech_anomaly_detection --size 320 --greedy 2000 \\
  --calib-ratio 0.3 --thr-pct 99 --blur-sigma 2 --split-seed 42 --seed 0 \\
  --weights torch_cache/wide_resnet50_2_timm.pth --batch 8 --fp16 \\
  --dump-pixel experiment_data/06_optimization/09_batch/bottle_b8_fp16_seed_000/pixel_dumps

.venv\Scripts\python.exe tools/e2e_bench.py --data MVTec/mvtech_anomaly_detection \\
  --category bottle --bank experiment_data/04_memory_banks/banks/bottle.npz \\
  --n 150 --warmup 10 --workers 4 --depth 8 --batch 8 --fp16 \\
  --out experiment_data/06_optimization/09_batch/e2e_bottle_b8_fp16.json
```

所有 MVTec test 结果均为 development（test 已被反复查看），不可当 holdout。
