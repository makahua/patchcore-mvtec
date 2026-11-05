# -*- coding: utf-8 -*-
"""Probe the GPU-bound pipeline: where the remaining milliseconds are and what each step could buy."""
import os
import statistics as st
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import patchcore_demo as P  # noqa: E402

DEV = 'cuda'
RES = {}


def sync():
    torch.cuda.synchronize()


def timeit(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return st.median(ts), st.mean(ts), np.percentile(ts, 95)


def build_model(weights, fp16=False):
    m = P.Features(pretrained=False).to(DEV).eval()
    sd = torch.load(weights, map_location='cpu')
    sd = {k: v for k, v in sd.items() if k in m.state_dict()}
    m.load_state_dict(sd, strict=False)
    if fp16:
        m.half()
    return m


def main():
    weights = os.path.join(ROOT, 'torch_cache', 'wide_resnet50_2_timm.pth')
    print('weights:', os.path.getsize(weights), 'bytes')

    x32 = torch.randn(1, 3, 320, 320, device=DEV)
    model = build_model(weights)

    # ---- 1) forward at several batch sizes (fp32) ----
    print('\n== forward fp32 (ms/img) ==')
    for b in (1, 2, 4, 8, 16):
        x = torch.randn(b, 3, 320, 320, device=DEV)
        with torch.no_grad():
            med, mean, p95 = timeit(lambda: model(x))
        RES['fwd32_b%d' % b] = med / b
        print('  batch %-3d total %7.2f ms  per-image %6.2f ms (p95 %6.2f)' % (b, med, med / b, p95 / b))

    # ---- 2) forward fp16 autocast ----
    print('\n== forward fp16 autocast (ms/img) ==')
    for b in (1, 8):
        x = torch.randn(b, 3, 320, 320, device=DEV)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            med, mean, p95 = timeit(lambda: model(x))
        RES['fwd16_b%d' % b] = med / b
        print('  batch %-3d total %7.2f ms  per-image %6.2f ms' % (b, med, med / b))

    # ---- 3) embed + kNN + post at the real shapes ----
    print('\n== per-image stages (screw-like: 40x40 grid, 2000x1536 bank) ==')
    feat = torch.randn(1, 1536, 40, 40, device=DEV)
    with torch.no_grad():
        med, mean, p95 = timeit(lambda: P.embed(feat, False))
    RES['embed'] = med
    print('  embed         %6.3f ms' % med)

    mem = torch.randn(2000, 1536, device=DEV)
    q = torch.randn(1600, 1536, device=DEV)

    def knn(m, qq):
        d = torch.cdist(qq, m)
        return d.min(dim=1).values

    with torch.no_grad():
        med, mean, p95 = timeit(lambda: knn(mem, q))
    RES['knn_fp32'] = med
    print('  knn fp32      %6.3f ms  (2000x1536 exact cdist)' % med)

    mem16, q16 = mem.half(), q.half()
    with torch.no_grad():
        try:
            med, mean, p95 = timeit(lambda: knn(mem16, q16))
            RES['knn_fp16'] = med
            print('  knn fp16      %6.3f ms' % med)
        except Exception as e:  # noqa: BLE001
            print('  knn fp16 failed:', repr(e)[:120])

    schunk = q[:490]
    with torch.no_grad():
        med, mean, p95 = timeit(lambda: knn(mem, schunk))
    print('  knn fp32 @490 patches (bottle-like) %6.3f ms' % med)

    sc = torch.rand(1600, device=DEV)
    def post():
        return P.gaussian_blur(sc.reshape(40, 40))
    with torch.no_grad():
        med, mean, p95 = timeit(post)
    RES['post'] = med
    print('  post (blur+max) %6.3f ms' % med)

    # ---- 4) CPU-only capacity: read+decode+resize with 4 threads ----
    print('\n== CPU-only decode capacity (4 threads) ==')
    data = os.path.join(ROOT, 'MVTec', 'mvtech_anomaly_detection')
    for cat, px in (('bottle', 900), ('carpet', 1024)):
        imgs = []
        for root, _, files in os.walk(os.path.join(data, cat, 'test', 'good')):
            for f in files:
                if f.lower().endswith('.png'):
                    imgs.append(os.path.join(root, f))
        imgs = imgs[:60]
        if not imgs:
            continue
        from concurrent.futures import ThreadPoolExecutor
        def job(p):
            im = Image.open(p)
            im.load()
            im = im.convert('RGB').resize((320, 320), Image.BILINEAR)
            a = np.asarray(im)
            return a.sum()
        for _ in range(5):
            job(imgs[0])
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(job, imgs))
        dt = time.perf_counter() - t0
        RES['cpu_ips_%d' % px] = len(imgs) / dt
        print('  %s (%dpx, %d imgs): %6.1f img/s -> %5.2f ms/img' % (cat, px, len(imgs), len(imgs) / dt, dt * 1000 / len(imgs)))

    print('\n== derived ==')
    gpu_parts = ['fwd32_b1', 'embed', 'knn_fp32', 'post']
    t_gpu32 = sum(RES[k] for k in gpu_parts)
    print('  GPU work per image (fp32, batch1): %.2f ms -> %.1f img/s ceiling' % (t_gpu32, 1000 / t_gpu32))
    t_gpu_mixed = RES['fwd32_b8'] + RES['embed'] + RES.get('knn_fp16', RES['knn_fp32']) + RES['post']
    print('  GPU work per image (fwd batch8 + fp16 knn): %.2f ms -> %.1f img/s' % (t_gpu_mixed, 1000 / t_gpu_mixed))
    t_gpu_fp16 = RES['fwd16_b8'] + RES['embed'] + RES.get('knn_fp16', RES['knn_fp32']) + RES['post']
    print('  GPU work per image (fwd fp16 + fp16 knn): %.2f ms -> %.1f img/s' % (t_gpu_fp16, 1000 / t_gpu_fp16))

    import json
    out = os.path.join(ROOT, 'experiment_data', '06_optimization', '05_latency', 'ceiling_probe.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({k: round(v, 4) for k, v in RES.items()}, f, ensure_ascii=False, indent=2, sort_keys=True)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
