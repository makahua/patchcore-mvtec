# 05_latency：端到端吞吐对照（串行 vs 多线程解码 + 有界预取）

> 对应计划 §4.1 的第一步（解码/预取）。**只测性能**：固定 bottle 记忆库（2000×1536），
> 不报分类指标，不改默认配置，也不参与任何 AUROC 结论。
> 数据角色：development only（与其它阶段一致，MVTec test 已被多轮查看）。

## 协议

| 项 | 设置 |
|---|---|
| 工具 | `tools/e2e_bench.py` |
| 机器 | RTX 3050 Laptop 4 GB，torch 2.7.0+cu128 |
| 模型 | WR50（timm `racm_in1k`，命中 318/320 层），输入 320×320 |
| 记忆库 | `experiment_data/04_memory_banks/banks/bottle.npz`，2000×1536 fp32 |
| 两条路径 | `sequential`：读→解码→缩放→上显存→前向→匹配→后处理，串行；`prefetch`：4 个 worker 做前三步，进深度 8 的有界队列，主线程只做 GPU 段 |
| 数据 | metal_nut 700²（115 张）、bottle 900²（83 张）、carpet 1024²（117 张），各 `--n 150 --warmup 10` |
| 统计 | 逐张 `perf_counter()`；预取额外报「提交→出分」驻留时间与其中排队时间 |

## 结果

| 原生尺寸 | 类别 | 串行 p50 | 串行 p95 | 串行吞吐 | 预取吞吐 | 加速 | 预取驻留 p50 | 排队 p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 700² | metal_nut | 40.74 ms | 44.76 ms | 25.45 张/s | 50.24 张/s | 1.97× | 237.73 ms | 202.99 ms |
| 900² | bottle | 36.28 ms | 37.68 ms | 27.45 张/s | 49.87 张/s | 1.82× | 240.02 ms | 200.86 ms |
| 1024² | carpet | 58.95 ms | 63.39 ms | 17.41 张/s | 49.85 张/s | 2.86× | 256.71 ms | 201.17 ms |

GPU 段 p50：串行 18.82 / 18.66 / 19.57 ms，预取 19.69 / 19.69 / 19.73 ms → 理论上限约 50.8 张/s。

## 结论

1. 预取把三档吞吐拉平到 **≈50 张/s**（= GPU 上限），档间差 <1%：解码不再是瓶颈，系统变成 GPU-bound。
2. 提速随原生分辨率变大（1.97× / 1.82× / 2.86×）——图越大，串行路径里 GPU 空转越多。
3. **吞吐是用驻留时间换的**：单图 238–257 ms 里约 201 ms 在排队（4 worker + 深度 8，约 12 张在飞）。
   在线检测看 p95 驻留，离线复检才按吞吐算产能；不要写成「时延降低」。
4. 抽样比对同一张图的分数，`max|Δscore| = 0.0`：预取没有改变计算。

## 文件

- `e2e_metal_nut.json` / `e2e_bottle.json` / `e2e_carpet.json`：每个含 `sequential`、`prefetch`、`speedup`、`score_max_abs_diff_sampled`、`peak_gpu_mb`。

## 复现

```powershell
.venv\Scripts\python.exe tools\e2e_bench.py `
  --data MVTec/mvtech_anomaly_detection --category carpet `
  --bank experiment_data/04_memory_banks/banks/bottle.npz `
  --n 150 --warmup 10 --workers 4 --depth 8 `
  --out experiment_data/06_optimization/05_latency/e2e_carpet.json
```

口径裁定（为什么曾有 35.5 与 42.2 两个端到端数）见 `docs/latency-breakdown.md` §2.1；预取对照的完整解读见 §3.7。
