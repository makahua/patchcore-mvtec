# -*- coding: utf-8 -*-
"""端到端时延拆解：把「一张图 19.7 ms」拆到每个阶段，产线节拍才有意义。

阶段划分（单张、batch=1、与 classify.py infer 同一路径）：
    read     : 打开文件读字节                    （磁盘 IO）
    decode   : PNG 解码 + convert('RGB')         （CPU，单线程）
    resize   : resize 到 S×S + 转 tensor          （CPU）
    h2d      : CPU → GPU 拷贝                     （PCIe）
    forward  : WR50 stem+layer1..3 + 上采样拼接    （GPU）
    embed    : 3×3 邻域平均 + reshape             （GPU）
    nn       : 分块 cdist 到记忆库 + 取 min        （GPU）
    post     : 高斯平滑 + 图像分数聚合 + 阈值判定   （GPU）

用法：
    python tools/latency_breakdown.py --bank experiment_data/04_memory_banks/banks/bottle.npz \
        --data MVTec/mvtech_anomaly_detection --category bottle --n 100

只测「已经建好库之后」的推理路径；建库/标定是一次性成本，单列输出。
最后附 batch 1/4/8 的吞吐对比（产线多相机/多工位场景要用）。
"""
import argparse
import glob
import io
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

STAGES = ['read', 'decode', 'resize', 'h2d', 'forward', 'embed', 'nn', 'post']


def weights_path():
    cand = os.path.join(ROOT, 'torch_cache', 'wide_resnet50_2_timm.pth')
    return cand if os.path.isfile(cand) else None


def pct(x, q):
    return float(np.percentile(x, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True, help='classify.py build 产出的 .npz')
    ap.add_argument('--data', default='MVTec/mvtech_anomaly_detection')
    ap.add_argument('--category', default='bottle')
    ap.add_argument('--n', type=int, default=100, help='计时张数（按 test 目录轮转）')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    import torch
    import patchcore_demo as P
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.perf_counter()
    model = P.Features(weights_path=weights_path(), imagenet_norm=False).to(device).eval()
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    z = np.load(a.bank, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    mem = torch.from_numpy(z['bank']).to(device)
    t_bank = time.perf_counter() - t0

    size = int(meta.get('size', 320))
    neighbors = bool(meta.get('neighbors', False))
    agg = meta.get('score_agg', 'max')
    sigma = float(meta.get('blur_sigma', 2.0))
    thr_ok = float(meta.get('thr_ok', 0.0))

    paths = sorted(glob.glob(os.path.join(a.data, a.category, 'test', '**', '*.png'), recursive=True))
    if not paths:
        raise SystemExit(f'找不到测试图：{a.data}/{a.category}/test')
    print(f'device={device} size={size} neighbors={neighbors} agg={agg} sigma={sigma} '
          f'mem={tuple(mem.shape)} 图 {len(paths)} 张')

    def score_of(x):
        """返回 batch 内逐张分数（batch>1 时 embed 给 (B, N, C)，逐张聚合）。"""
        with torch.no_grad():
            feat = model(x)
            emb, (h, w) = P.embed(feat, neighbors)
            b, n, c = emb.shape
            flat = emb.reshape(b * n, c)
            mins = []
            for i in range(0, flat.shape[0], 2048):
                d = torch.cdist(flat[i:i + 2048], mem)
                mins.append(d.min(dim=1).values)
            dist = torch.cat(mins)
            out = []
            for j in range(b):
                patch = dist[j * n:(j + 1) * n].reshape(h, w)
                if sigma > 0:
                    patch = P.gaussian_blur(patch, sigma=sigma)
                if agg == 'max':
                    sc = patch.max()
                else:
                    k = max(1, int(np.ceil(patch.numel() * 0.05)))
                    sc = patch.flatten().topk(k).values.mean()
                out.append(float(sc.item()))
        return out

    def run_one(p):
        s = {}
        t = time.perf_counter()
        with open(p, 'rb') as f:
            buf = f.read()
        s['read'] = time.perf_counter() - t
        t = time.perf_counter()
        im = Image.open(io.BytesIO(buf)).convert('RGB')
        im.load()
        s['decode'] = time.perf_counter() - t
        t = time.perf_counter()
        im = im.resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        s['resize'] = time.perf_counter() - t
        t = time.perf_counter()
        x = x.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        s['h2d'] = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            feat = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        s['forward'] = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            emb, (h, w) = P.embed(feat, neighbors)
            emb = emb.squeeze(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        s['embed'] = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            mins = []
            for i in range(0, emb.shape[0], 2048):
                d = torch.cdist(emb[i:i + 2048], mem)
                mins.append(d.min(dim=1).values)
            patch = torch.cat(mins).reshape(h, w)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        s['nn'] = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            if sigma > 0:
                patch = P.gaussian_blur(patch, sigma=sigma)
            if agg == 'max':
                sc = patch.max()
            else:
                k = max(1, int(np.ceil(patch.numel() * 0.05)))
                sc = patch.flatten().topk(k).values.mean()
            sc = float(sc.item()) > thr_ok
            if device.type == 'cuda':
                torch.cuda.synchronize()
        s['post'] = time.perf_counter() - t
        s['total'] = sum(s.values())
        return s

    # 冷启动首张（含首次显存分配/权重页换入/首张解码）
    cold = run_one(paths[0])

    for i in range(a.warmup):
        run_one(paths[i % len(paths)])

    rows = [run_one(paths[i % len(paths)]) for i in range(a.n)]
    ms = {k: [r[k] * 1000 for r in rows] for k in STAGES + ['total']}

    print(f'\n模型加载 {t_model * 1000:.0f} ms | 记忆库加载 {t_bank * 1000:.1f} ms（一次性，开机/换型时）')
    print(f'冷启动首张 total {cold["total"] * 1000:.1f} ms（含首次显存分配与权重建页）\n')
    print(f'{"阶段":<10}{"中位":>8}{"均值":>8}{"p95":>8}{"占比":>7}')
    print('-' * 41)
    tot = ms['total']
    for k in STAGES:
        v = ms[k]
        print(f'{k:<10}{pct(v, 50):>8.2f}{statistics.fmean(v):>8.2f}{pct(v, 95):>8.2f}'
              f'{statistics.fmean(v) / statistics.fmean(tot):>7.1%}')
    print('-' * 41)
    print(f'{"合计":<10}{pct(tot, 50):>8.2f}{statistics.fmean(tot):>8.2f}{pct(tot, 95):>8.2f}')
    gpu_rows = [r['forward'] + r['embed'] + r['nn'] + r['post'] for r in rows]
    cpu_rows = [r['read'] + r['decode'] + r['resize'] + r['h2d'] for r in rows]
    gpu = [v * 1000 for v in gpu_rows]
    cpu = [v * 1000 for v in cpu_rows]
    print(f'\n吞吐 {1000 / pct(tot, 50):.1f} 张/s | jitter(p95-p50) {pct(tot, 95) - pct(tot, 50):.2f} ms')
    print(f'GPU 段合计 {pct(gpu, 50):.2f} ms ({statistics.fmean(gpu) / statistics.fmean(tot):.0%}) | '
          f'CPU/IO 段合计 {pct(cpu, 50):.2f} ms ({statistics.fmean(cpu) / statistics.fmean(tot):.0%})')

    # batch 吞吐：把解码+resize 排除，只比 GPU 计算段（多工位复用同一张卡时才关心）
    print('\nbatch 吞吐（只含 forward+embed+nn+post，已预取并 resize 好）')
    print(f'{"batch":>6}{"每张":>10}{"张/s":>10}')
    tens = []
    for i in range(max(8, 8)):
        p = paths[i % len(paths)]
        im = Image.open(p).convert('RGB').resize((size, size), Image.BILINEAR)
        tens.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1))
    batch_out = {}
    for b in (1, 4, 8):
        xs = torch.stack(tens[:b]).to(device)
        with torch.no_grad():
            for _ in range(3):
                score_of(xs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t = time.perf_counter()
            reps = 10
            for _ in range(reps):
                score_of(xs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / reps
        batch_out[b] = round(dt * 1000, 2)
        print(f'{b:>6}{dt * 1000 / b:>10.2f}{b / dt:>10.1f}')

    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f'\n显存峰值（含权重与记忆库）{peak:.0f} MB')
    else:
        peak = 0.0

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(dict(category=a.category, device=str(device), size=size, mem_patches=int(mem.shape[0]),
                           neighbors=neighbors, score_agg=agg, blur_sigma=sigma, n=a.n,
                           model_load_ms=round(t_model * 1000, 1), bank_load_ms=round(t_bank * 1000, 1),
                           cold_first_ms=round(cold['total'] * 1000, 1),
                           stages={k: dict(p50=round(pct(v, 50), 2), mean=round(statistics.fmean(v), 2),
                                           p95=round(pct(v, 95), 2)) for k, v in ms.items()},
                           gpu_ms=round(pct(gpu, 50), 2), cpu_io_ms=round(pct(cpu, 50), 2),
                           throughput_ips=round(1000 / pct(tot, 50), 2),
                           batch_ms=batch_out, peak_gpu_mb=round(peak, 1)), f, ensure_ascii=False, indent=2)
        print('写入', a.out)


if __name__ == '__main__':
    main()
