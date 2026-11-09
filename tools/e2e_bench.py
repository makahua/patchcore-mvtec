# -*- coding: utf-8 -*-
"""端到端吞吐对比：串行基线 vs 多线程解码 + 有界预取。

对应计划 §4.1 的第一步（解码/预取）。两条路径共享同一权重、同一记忆库、
同一样本集与同一 GPU 计算段，唯一变量是 CPU 侧 I/O 与预处理是否并行：

* ``sequential``：读 → 解码 → 缩放 → 上显存 → 前向 → 匹配 → 后处理，串行；
* ``prefetch``  ：N 个 worker 线程做「读 + 解码 + 缩放」，结果放进深度 D 的有界队列，
                  主线程只做 GPU 段；生产者在队列满时阻塞（背压明确）。

对 prefetch 路径同时报告：吞吐、单图「提交→出分」驻留时间（含排队）、
以及队列等待占比——排队时间不能从端到端里藏掉。

用法：
  .venv/Scripts/python.exe tools/e2e_bench.py --data MVTec/mvtech_anomaly_detection \
      --category bottle --bank experiment_data/04_memory_banks/banks/bottle.npz \
      --n 120 --warmup 10 --workers 4 --depth 8 --out experiment_data/06_optimization/05_latency/e2e_bottle.json

计时用固定记忆库（bottle，2000×1536）：只用于性能比较，不用于报分类指标。
"""
import argparse
import glob
import io
import json
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def weights_path():
    cand = os.path.join(ROOT, "torch_cache", "wide_resnet50_2_timm.pth")
    return cand if os.path.isfile(cand) else None


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="MVTec/mvtech_anomaly_detection")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--bank", default="experiment_data/04_memory_banks/banks/bottle.npz")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--fp16", action="store_true",
                    help="前向 autocast(fp16) + 检索 fp16（默认关，精度闸门见 docs/e2e-ceiling.md）")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=1, help="批推理大小（默认 1）")
    a = ap.parse_args()

    import torch
    import patchcore_demo as P
    from PIL import Image

    import classify as C

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(a.bank, allow_pickle=True)
    meta = C._meta_from_npz(z, a.bank)
    key = "bank" if "bank" in z.files else "mem"
    mem = torch.from_numpy(z[key]).float().to(device)
    model = P.Features(weights_path=weights_path(), imagenet_norm=bool(meta.get("imagenet_norm", False)),
                       layers=str(meta.get("layers", "2+3"))).to(device).eval()
    size = int(meta.get("size", 320))
    neighbors = bool(meta.get("neighbors", False))
    sigma = float(meta.get("blur_sigma", 2.0))
    agg = meta.get("score_agg", "max")
    pad = str(meta.get("neighbor_pad", "zero"))

    paths = sorted(glob.glob(os.path.join(a.data, a.category, "test", "**", "*.png"), recursive=True))
    if not paths:
        raise SystemExit(f"找不到测试图：{a.data}/{a.category}/test")
    print(f"device={device} size={size} mem={tuple(mem.shape)} 图 {len(paths)} 张")

    def load_cpu(path):
        t = time.perf_counter()
        with open(path, "rb") as f:
            buf = f.read()
        t_read = time.perf_counter() - t
        t = time.perf_counter()
        im = Image.open(io.BytesIO(buf)).convert("RGB")
        im.load()
        im = im.resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x, t_read, time.perf_counter() - t

    mem_h = mem.half() if a.fp16 else mem

    def score_one(feat):
        emb, (h, w) = P.embed(feat, neighbors, pad)
        emb = emb.squeeze(0)
        mins = []
        for i in range(0, emb.shape[0], 2048):
            if a.fp16:
                d = torch.cdist(emb[i:i + 2048].half(), mem_h)
                mins.append(d.min(dim=1).values.float())
            else:
                d = torch.cdist(emb[i:i + 2048], mem)
                mins.append(d.min(dim=1).values)
        patch = torch.cat(mins).reshape(h, w)
        if sigma > 0:
            patch = P.gaussian_blur(patch, sigma=sigma)
        if agg == "max":
            sc = patch.max()
        else:
            k = max(1, int(np.ceil(patch.numel() * 0.05)))
            sc = patch.flatten().topk(k).values.mean()
        return float(sc.item())

    def infer_batch(xs):
        """xs: list[CHW] -> list[score]；一次前向，逐图算分。"""
        with torch.no_grad():
            xb = torch.stack(xs).to(device)
            if a.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feat = model(xb)
                feat = feat.float()
            else:
                feat = model(xb)
            outs = [score_one(feat[k:k + 1]) for k in range(feat.shape[0])]
            if device.type == "cuda":
                torch.cuda.synchronize()
        return outs

    def infer(x):
        return infer_batch([x])[0]

    def run_sequential(n, warmup, bs):
        for i in range(warmup):
            x, _, _ = load_cpu(paths[i % len(paths)])
            infer(x)
        rows = []
        t_start = time.perf_counter()
        for start in range(0, n, bs):
            k = min(bs, n - start)
            t0 = time.perf_counter()
            xs, reads, cpus = [], [], []
            for j in range(k):
                x, t_read, t_cpu = load_cpu(paths[(start + j) % len(paths)])
                xs.append(x); reads.append(t_read); cpus.append(t_cpu)
            t1 = time.perf_counter()
            scores = infer_batch(xs)
            t2 = time.perf_counter()
            for j in range(k):
                rows.append({"path": paths[(start + j) % len(paths)],
                             "read_ms": reads[j] * 1000, "decode_resize_ms": cpus[j] * 1000,
                             "gpu_ms": (t2 - t1) * 1000 / k,
                             "total_ms": (t2 - t0) * 1000 / k, "score": scores[j]})
        wall = time.perf_counter() - t_start
        return rows, wall

    def run_prefetch(n, warmup, workers, depth, bs):
        for i in range(warmup):
            x, _, _ = load_cpu(paths[i % len(paths)])
            infer(x)

        q = queue.Queue(maxsize=depth)
        stop = threading.Event()
        counter = {"next": 0}
        lock = threading.Lock()

        def producer():
            while not stop.is_set():
                with lock:
                    idx = counter["next"]
                    if idx >= n:
                        break
                    counter["next"] = idx + 1
                t0 = time.perf_counter()
                x, t_read, t_cpu = load_cpu(paths[idx % len(paths)])
                item = {"idx": idx, "x": x, "t_submit": t0, "t_ready": time.perf_counter(),
                        "read_ms": t_read * 1000, "decode_resize_ms": t_cpu * 1000}
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                else:
                    return
            q.put(None)

        rows = []
        threads = [threading.Thread(target=producer, daemon=True) for _ in range(workers)]
        t_start = time.perf_counter()
        for th in threads:
            th.start()
        done = 0
        while done < n:
            item = q.get()
            if item is None:
                continue
            group = [item]
            while len(group) < bs and done + len(group) < n:
                try:
                    nxt = q.get(timeout=0.02)
                except queue.Empty:
                    break
                if nxt is None:
                    continue
                group.append(nxt)
            t0 = time.perf_counter()
            scores = infer_batch([g["x"] for g in group])
            t1 = time.perf_counter()
            for g, score in zip(group, scores):
                rows.append({"path": paths[g["idx"] % len(paths)], "read_ms": g["read_ms"],
                             "decode_resize_ms": g["decode_resize_ms"],
                             "gpu_ms": (t1 - t0) * 1000 / len(group),
                             "queue_ms": (t0 - g["t_ready"]) * 1000,
                             "total_ms": (t1 - g["t_submit"]) * 1000,
                             "score": score})
                done += 1
        wall = time.perf_counter() - t_start
        stop.set()
        for th in threads:
            th.join(timeout=2)
        return rows, wall

    bs = max(1, int(a.batch))
    result = {"category": a.category, "size": size, "mem_patches": int(mem.shape[0]),
              "n": a.n, "warmup": a.warmup, "workers": a.workers, "depth": a.depth,
              "batch": bs, "fp16": bool(a.fp16),
              "device": str(device), "mode": "performance only; fixed bottle bank"}

    seq_rows, seq_wall = run_sequential(a.n, a.warmup, bs)
    seq_total = [r["total_ms"] for r in seq_rows]
    result["sequential"] = {
        "throughput_ips": round(a.n / seq_wall, 2),
        "total_p50_ms": round(pct(seq_total, 50), 2),
        "total_p95_ms": round(pct(seq_total, 95), 2),
        "gpu_p50_ms": round(pct([r["gpu_ms"] for r in seq_rows], 50), 2),
        "cpu_p50_ms": round(pct([r["read_ms"] + r["decode_resize_ms"] for r in seq_rows], 50), 2),
        "wall_s": round(seq_wall, 2),
    }
    print(f"sequential: {result['sequential']}")

    pre_rows, pre_wall = run_prefetch(a.n, a.warmup, a.workers, a.depth, bs)
    pre_total = [r["total_ms"] for r in pre_rows]
    result["prefetch"] = {
        "throughput_ips": round(a.n / pre_wall, 2),
        "resident_p50_ms": round(pct(pre_total, 50), 2),
        "resident_p95_ms": round(pct(pre_total, 95), 2),
        "queue_p50_ms": round(pct([r["queue_ms"] for r in pre_rows], 50), 2),
        "gpu_p50_ms": round(pct([r["gpu_ms"] for r in pre_rows], 50), 2),
        "wall_s": round(pre_wall, 2),
    }
    print(f"prefetch(w={a.workers},d={a.depth}): {result['prefetch']}")

    result["speedup"] = round(result["prefetch"]["throughput_ips"] / result["sequential"]["throughput_ips"], 3)
    # 分数一致性：同图同库必须得到同一分数（抽样），保证预取没有改变计算
    seq_map = {r["path"]: r["score"] for r in seq_rows}
    diffs = [abs(seq_map[r["path"]] - r["score"]) for r in pre_rows if r["path"] in seq_map]
    result["score_max_abs_diff_sampled"] = round(max(diffs), 6) if diffs else None

    if device.type == "cuda":
        result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)

    print(f"speedup={result['speedup']}x  score_max_abs_diff={result['score_max_abs_diff_sampled']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("写入", a.out)


if __name__ == "__main__":
    main()
