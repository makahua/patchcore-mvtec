# 08_fp16 —— FP16 精度闸门与吞吐验证

## 为什么有这一组

`docs/e2e-ceiling.md` 的探针给出「FP16 前向 + 检索」的理论上限（约 109 ips），但探针用随机张量、不测精度。本目录记录**真实数据上的闸门实验**：AUPRO 掉超过 0.5 点就不启用 FP16。

## 协议

固定 `--size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --blur-sigma 2 --split-seed 42 --seed 0`，权重 `torch_cache/wide_resnet50_2_timm.pth`，f32 / fp16 在同一会话内成对跑，每个 run 加 `--dump-pixel` 后用 `tools/pro_aupro.py` 算 AUPRO。

| 目录 | 说明 |
|---|---|
| `screw_fp32_seed_000` / `screw_fp16_seed_000` | screw 闸门对照 |
| `bottle_fp32_seed_000` / `bottle_fp16_seed_000` | bottle 闸门对照 |
| `e2e_bottle_fp16.json` | 预取 w4/d8 + fp16 吞吐（67.76 ips） |
| `e2e_bottle_fp16_w6.json` | 预取 w6/d12 + fp16 吞吐（66.10 ips） |
| `gate_status.json` / `gate_summary.json` | 运行状态与汇总 |

## 结论

| 类别 | 精度 | image AUROC | pixel AUROC | AUPRO@30% | 召回 | 过杀 | 算法段时延 | GPU 峰值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| screw | fp32 | 0.9240 | 0.9823 | 0.9270 | 67.2% | 2.4% | 20.03 ms | 189.6 MB |
| screw | fp16 | 0.9240 | 0.9823 | 0.9263 | 67.2% | 2.4% | 13.26 ms | 207.0 MB |
| bottle | fp32 | 1.0000 | 0.9856 | 0.9463 | 96.8% | 0.0% | 19.94 ms | 190.0 MB |
| bottle | fp16 | 1.0000 | 0.9856 | 0.9463 | 96.8% | 0.0% | 13.74 ms | 205.6 MB |

- **通过闸门**：AUPRO 最大差 0.0008，image/pixel/召回/过杀四位小数完全一致，阈值只动 0.04。
- 代价：峰值显存 +17 MB（fp16 记忆库副本 + autocast 工作区），算法段时延 −1.45～1.51×。
- 吞吐：预取 + fp16 达 **67.8 ips**（fp32 49.9），但 w6 不再涨 → 已 GPU-bound（GPU p50 14.3–14.5 ms），下一步只能走批推理。
- 默认仍关 `--fp16`；这是「可选加速档」，不是默认配置变更。

## 复现

```bash
# 闸门（fp16 侧，fp32 侧去掉 --fp16）
.venv\Scripts\python.exe tools/run_experiment.py --run-dir experiment_data/06_optimization/08_fp16/bottle_fp16_seed_000 -- \\
  --category bottle --data MVTec/mvtech_anomaly_detection --size 320 --greedy 2000 \\
  --calib-ratio 0.3 --thr-pct 99 --blur-sigma 2 --split-seed 42 --seed 0 \\
  --weights torch_cache/wide_resnet50_2_timm.pth --fp16 \\
  --dump-pixel experiment_data/06_optimization/08_fp16/bottle_fp16_seed_000/pixel_dumps

.venv\Scripts\python.exe tools/pro_aupro.py \\
  --dumps experiment_data/06_optimization/08_fp16/bottle_fp16_seed_000/pixel_dumps \\
  --data MVTec/mvtech_anomaly_detection --category bottle \\
  --out experiment_data/06_optimization/08_fp16/bottle_fp16_seed_000/aupro.json

# 吞吐
.venv\Scripts\python.exe tools/e2e_bench.py --data MVTec/mvtech_anomaly_detection --category bottle \\
  --bank experiment_data/04_memory_banks/banks/bottle.npz --n 150 --warmup 10 \\
  --workers 4 --depth 8 --fp16 --out experiment_data/06_optimization/08_fp16/e2e_bottle_fp16.json
```

所有 MVTec test 结果均为 development（test 已被反复查看），不可当 holdout。
