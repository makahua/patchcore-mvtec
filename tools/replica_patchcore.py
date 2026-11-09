# -*- coding: utf-8 -*-
"""官方 PatchCore 配方的独立复刻（试点用，不改动主流程）。

对应 amazon-science/patchcore-inspection 的 IM320 设置：
  Resize(366) -> CenterCrop(320) -> ToTensor -> ImageNet Normalize
  layer2+3, Unfold(3) 邻域, MeanMapper 到 1024 维, 跨层聚合到 1024 维
  approx_greedy_coreset, 保留比例 1%, image score = max patch distance

用法：
  python tools/replica_patchcore.py --categories toothbrush,pill --pct 0.01
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


def build_model(weights, device):
    m = models.wide_resnet50_2(weights=None)
    sd = torch.load(weights, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k: v.float() for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m.to(device)


def make_tf(resize, imagesize, stretch=False, normalize=True, antialias=True):
    ops = []
    if stretch:
        ops.append(transforms.Resize((imagesize, imagesize), antialias=antialias))
    else:
        ops.append(transforms.Resize(resize, antialias=antialias))
        ops.append(transforms.CenterCrop(imagesize))
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(mean=MEAN, std=STD))
    return transforms.Compose(ops)


@torch.no_grad()
def embed(model, x):
    """x: (B,3,S,S) -> (B, L, 1024)

    与官方一致：先在各自网格上 Unfold(3)，再把 C*9 当作通道双线性插值到 layer2 网格，
    然后每层 MeanMapper 到 1024、跨层聚合到 1024。
    """
    h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    f2 = model.layer2(model.layer1(h))
    f3 = model.layer3(f2)
    ref = f2.shape[-2:]
    vecs = []
    for f in (f2, f3):
        b, c, hh, ww = f.shape
        u = F.unfold(f, kernel_size=3, padding=1).reshape(b, c, 3, 3, hh, ww)
        if (hh, ww) != ref:
            u = u.reshape(b, c * 9, hh, ww)
            u = F.interpolate(u, size=ref, mode='bilinear', align_corners=False)
            u = u.permute(0, 2, 3, 1).reshape(b, ref[0] * ref[1], c * 9)
        else:
            u = u.permute(0, 4, 5, 1, 2, 3).reshape(b, hh * ww, c * 9)
        p = F.adaptive_avg_pool1d(u.reshape(b * u.shape[1], 1, c * 9), 1024)
        vecs.append(p.squeeze(1).reshape(b, u.shape[1], 1024))
    stacked = torch.stack(vecs, dim=1)                      # (B, 2, L, 1024)
    b, _, l, _ = stacked.shape
    flat = stacked.reshape(b * l, 1, 2048)
    return F.adaptive_avg_pool1d(flat, 1024).squeeze(1).reshape(b, l, 1024)


@torch.no_grad()
def approx_greedy(feats, pct, device, seed, proj=128, starts=10):
    """近似贪心 coreset：128 维投影上做 mm 距离（比 cdist 逐次调用快两个数量级）。"""
    n = feats.shape[0]
    k = max(1, int(n * pct))
    g = torch.Generator(device='cpu').manual_seed(seed)
    sub = torch.randperm(n, generator=g)[:min(starts, n)].to(device)
    torch.manual_seed(seed)
    lin = torch.nn.Linear(feats.shape[1], proj, bias=False).to(device)
    r = lin(feats).contiguous()
    r2 = (r * r).sum(dim=1, keepdim=True)

    def d2_to(idx):
        v = r[idx]
        return (r2 + (v * v).sum() - 2.0 * (r @ v).unsqueeze(1)).clamp_min_(0)

    anchor = torch.stack([d2_to(i) for i in sub.tolist()], dim=1).mean(dim=1)
    sel = []
    for _ in range(k):
        i = int(anchor.argmax().item())
        sel.append(i)
        anchor = torch.minimum(anchor, d2_to(i))
    return torch.tensor(sel, device=device, dtype=torch.long)


@torch.no_grad()
def scores_for(model, bank, paths, tf, device, has_label=None):
    """返回 (scores, labels)"""
    scores, labels = [], []
    batch = 4
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        xs = torch.stack([tf(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        feats = embed(model, xs)                            # (b, L, 1024)
        b, l, d = feats.shape
        dist = torch.cdist(feats.reshape(b * l, d), bank)   # (b*L, k)
        dmin = dist.min(dim=1).values.reshape(b, l)
        scores.extend(dmin.max(dim=1).values.tolist())
        if has_label:
            labels.extend(has_label[i:i + batch])
    return scores, labels


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([neg, pos]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[len(neg):]
    return float((r_pos.mean() - (len(pos) + 1) / 2.0) / len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'MVTec', 'mvtech_anomaly_detection'))
    ap.add_argument('--weights', default=os.path.join(ROOT, 'torch_cache', 'wide_resnet50_2_v1.pth'))
    ap.add_argument('--categories', default='bottle')
    ap.add_argument('--resize', type=int, default=366)
    ap.add_argument('--imagesize', type=int, default=320)
    ap.add_argument('--pct', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=22)
    ap.add_argument('--limit-train', type=int, default=0)
    ap.add_argument('--stretch', action='store_true')
    ap.add_argument('--no-norm', action='store_true')
    ap.add_argument('--no-antialias', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: %s%s' % (device, ' ' + torch.cuda.get_device_name(0) if device == 'cuda' else ''), flush=True)
    model = build_model(a.weights, device)
    tf = make_tf(a.resize, a.imagesize, stretch=a.stretch, normalize=not a.no_norm, antialias=not a.no_antialias)
    cats = [c.strip() for c in a.categories.split(',') if c.strip()]
    if cats == ['all']:
        cats = CATS

    results = {}
    macro = []
    for cat in cats:
        t0 = time.time()
        base = os.path.join(a.data, cat)
        train = sorted(glob.glob(os.path.join(base, 'train', 'good', '*.png')))
        if a.limit_train:
            train = train[:a.limit_train]
        test = sorted(glob.glob(os.path.join(base, 'test', '**', '*.png'), recursive=True))
        labels = [0 if os.sep + 'good' + os.sep in p else 1 for p in test]

        feats = []
        with torch.no_grad():
            for i in range(0, len(train), 4):
                xs = torch.stack([tf(Image.open(p).convert('RGB')) for p in train[i:i + 4]]).to(device)
                f = embed(model, xs)
                feats.append(f.reshape(-1, f.shape[-1]))
        feats = torch.cat(feats, 0)
        n_patch = feats.shape[0]
        t_embed = time.time()
        idx = approx_greedy(feats, a.pct, device, a.seed)
        bank = feats[idx]
        del feats
        torch.cuda.empty_cache()
        t_greedy = time.time()

        sc, lb = scores_for(model, bank, test, tf, device, labels)
        au = auc(sc, lb)
        pos = [s for s, l in zip(sc, lb) if l == 1]
        neg = [s for s, l in zip(sc, lb) if l == 0]
        thr = float(np.percentile(neg, 99.5)) if neg else float('nan')
        rec = float(np.mean([1.0 if s >= thr else 0.0 for s in pos])) if pos else float('nan')
        macro.append(au)
        results[cat] = {
            'image_auroc': round(au, 4),
            'recall_at_p995': round(rec, 4),
            'n_train': len(train), 'n_test': len(test),
            'n_patch': int(n_patch), 'bank': int(bank.shape[0]),
            'seconds': round(time.time() - t0, 1),
        }
        print('[%s] AUROC %.4f  bank %d/%d  test %d  %.0fs (embed %.0f / greedy %.0f / score %.0f)' % (
            cat, au, bank.shape[0], n_patch, len(test), time.time() - t0,
            t_embed - t0, t_greedy - t_embed, time.time() - t_greedy), flush=True)

    m = float(np.mean(macro)) if macro else float('nan')
    print('macro image AUROC %.4f over %d classes' % (m, len(macro)))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'macro': round(m, 4), 'per_class': results,
                       'config': vars(a)}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

