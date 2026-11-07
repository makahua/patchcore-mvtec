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
- `ceiling_probe.json`：GPU 段拆解与后续手段的上限估算（`tools/ceiling_probe.py` 生成；前向 batch 1–16、FP16 autocast、FP16 检索、CPU 解码容量）。解读见 [`docs/e2e-ceiling.md`](../../../docs/e2e-ceiling.md)。

## 复现

```powershell
.venv\Scripts\python.exe tools\e2e_bench.py `
  --data MVTec/mvtech_anomaly_detection --category carpet `
  --bank experiment_data/04_memory_banks/banks/bottle.npz `
  --n 150 --warmup 10 --workers 4 --depth 8 `
  --out experiment_data/06_optimization/05_latency/e2e_carpet.json
```

口径裁定（为什么曾有 35.5 与 42.2 两个端到端数）见 `docs/latency-breakdown.md` §2.1；预取对照的完整解读见 §3.7。

---

## 最终配置复测（2026-11-08）

> 上一节的库是 `04_memory_banks` 的旧库。这里换成新协议的库
> （`10_pair/23_<cat>_seed_000/banks/*.npz`，2+3 / 1536 维 / thr_ok 39.21），
> 代码也含后来的 `--fp16` / `--batch`，用于回答「最终配置到底多快」。

### 串行分段（`tools/latency_breakdown.py`，`--n 200 --warmup 10`）

| 原生尺寸 | 类别 | read | decode | resize | forward | nn | total p50 | p95 | GPU 段 | CPU/IO | 串行吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 700² | metal_nut | 7.09 | 12.68 | 3.66 | 13.42 | 3.41 | **40.85 ms** | 47.42 | 18.25 | 22.35 | 24.5 张/s |
| 900² | bottle | 0.49 | 13.34 | 4.62 | 13.62 | 3.41 | **39.68 ms** | 47.61 | 18.56 | 20.85 | 25.2 张/s |
| 1024² | carpet | 8.18 | 30.05 | 6.29 | 16.38 | 4.26 | **64.87 ms** | 71.42 | 21.97 | 42.14 | 15.4 张/s |

- 700² 与 900² 只差 1 ms，是因为 metal_nut 这一轮 `read` 高达 7.09 ms（页缓存冷），解码本身 12.68 ms。
  **跨批次读 700² 请按 37–41 ms 区间报**，不要拿单次 p50 当定值。
- 与 2026-11-06 的旧构建相比：900² 35.53 / 42.18 → 本轮 39.68；1024² 62.35 → 64.87；
  差异来自 CPU/IO（PIL 单线程解码抖动 ±19%），GPU 段 18.6–22.0 ms 基本不变。

### 吞吐（`tools/e2e_bench.py`，`--n 150 --warmup 10 --workers 4 --depth 8`）

| 类别 | 精度 | batch | 串行 | 预取 | GPU p50 | 峰值显存 | max\|Δscore\| |
|---|---|---:|---:|---:|---:|---:|---:|
| bottle 900² | fp32 | 1 | 24.17 | **50.32 张/s** | 19.56 ms | 171.2 MB | 0 |
| bottle 900² | fp16 | 1 | 27.83 | **63.13 张/s** | 15.51 ms | 194.3 MB | 0 |
| bottle 900² | fp16 | 8 | 30.82 | **101.44 张/s** | 9.38 ms | 344.5 MB | 0.0071 |
| carpet 1024² | fp32 | 1 | 17.67 | **49.36 张/s** | 19.78 ms | 171.2 MB | 0 |
| carpet 1024² | fp16 | 8 | 21.93 | **85.57 张/s** | 9.31 ms | 344.5 MB | 0.0090 |

结论与 `docs/e2e-ceiling.md` 一致：批 8 把 GPU 段压到 9.4 ms，吞吐由解码/队列决定；
1024² 因解码更重，同样配置只到 85.6 张/s。batch>1 的分数非逐位一致（max|Δ|≈0.007–0.009），
AUROC/AUPRO 未变但换阈值需重标定。

