"""PRO / AUPRO（per-region overlap）评测工具。

PRO（Per-Region Overlap）是 MVTec 官方在 PatchCore / PaDiM 里报的区域级指标：
把 GT 掩码按连通域切成若干区域，统计阈值 t 下每个区域被"命中"的像素比例，对所有区域取平均；
再对像素级 FPR 积分得到 AUPRO（通常截到 FPR ≤ 0.3 再除以 0.3 归一）。

本工具读取 `patchcore_demo.py --dump-pixel DIR` 产出的 `<类别>.npz`：
    scores: (N, h, w) float16   patch 级分数图（预处理坐标系，h=w=size/8）
    masks : (N, h, w) bool      GT 掩码（同坐标系，最近邻降采样）
    paths : (N,) str            测试图相对路径
    labels: (N,) int            0=良品、1=缺陷
做法：

1. 把 patch 分数图双线性上采样回**原图分辨率**，GT 掩码也按原图重新读取，两者严格对齐
   （等价于 PatchCore 把 anomaly map 上采样到原图再算指标）；
2. 在原图上做 4-邻接连通域切分（不引入 scipy，用行游程 + 并查集）；
3. 用全局等距阈值网格建直方图，一次算出 PRO(t) 与像素级 FPR(t)，阈值语义为「分数 > t 判异常」；
4. AUPRO = PRO-FPR 曲线在 [0, limit] 上的梯形积分 / limit；另给 PRO@FPR=5%/10%。

用法：
    python tools/pro_aupro.py --dumps experiment_data/05_indexes/pixel_dumps \
        --data MVTec/mvtech_anomaly_detection --category all \
        --out experiment_data/05_indexes/pro_aupro.json --markdown docs/_pro_table.md

只依赖 numpy + Pillow（和 tests/test_metrics.py 一样不需要 torch）。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# numpy 2.0 把 trapz 改名为 trapezoid；这里保留回退，numpy<2 环境不会直接 AttributeError
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def _pil():
    """Pillow 只在真正读图时引入——纯 numpy 的曲线/连通域函数与 CI 都不需要它。"""
    from PIL import Image
    return Image


# --------------------------------------------------------------------------- #
# 连通域标记（行游程 + 并查集；900x900 掩码约几毫秒）
# --------------------------------------------------------------------------- #
def label_regions(mask):
    """4-邻接连通域。返回 (labels int32[h, w], n_regions)，编号从 1 开始，0 为背景。"""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    if not mask.any():
        return lab, 0

    parent = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    nxt = 0
    prev = []  # 上一行游程 [(start, end, rid)]
    for i in range(h):
        row = mask[i]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate(([False], row, [False])).astype(np.int8))
        starts = np.flatnonzero(d == 1).tolist()
        ends = np.flatnonzero(d == -1).tolist()
        cur = []
        for s, e in zip(starts, ends):
            rid = None
            for ps, pe, prid in prev:
                if ps < e and s < pe:  # 与上一行有列重叠 -> 同一区域
                    if rid is None:
                        rid = prid
                    else:
                        union(rid, prid)
            if rid is None:
                nxt += 1
                rid = nxt
                parent[rid] = rid
            cur.append((s, e, rid))
        for s, e, rid in cur:
            lab[i, s:e] = rid
        prev = cur

    lut = np.zeros(nxt + 1, np.int32)
    roots = sorted({find(r) for r in parent})
    remap = {r: k + 1 for k, r in enumerate(roots)}
    for r in parent:
        lut[r] = remap[find(r)]
    return lut[lab], len(roots)


# --------------------------------------------------------------------------- #
# PRO 曲线
# --------------------------------------------------------------------------- #
def pro_curve(score_maps, gt_masks, num_th=200, nbins=1024):
    """同尺寸分数图 + GT 掩码 -> PRO(t) / FPR(t)。用直方图实现，复杂度与像素数同阶。"""
    if len(score_maps) != len(gt_masks):
        raise ValueError('score_maps 与 gt_masks 长度不一致')
    lo = min(float(np.min(m)) for m in score_maps)
    hi = max(float(np.max(m)) for m in score_maps)
    if hi <= lo:
        raise ValueError('分数图没有动态范围')
    scale = (hi - lo) / float(nbins)

    region_hists, region_areas = [], []
    neg_hist = np.zeros(nbins, np.int64)
    neg_total = 0
    for m, g in zip(score_maps, gt_masks):
        m = np.asarray(m, np.float64)
        g = np.asarray(g, bool)
        idx = np.clip(((m - lo) / scale).astype(np.int64), 0, nbins - 1)
        if g.any():
            lab, nreg = label_regions(g)
            ids = lab[g].astype(np.int64)
            flat = ids * nbins + idx[g]
            hist = np.bincount(flat, minlength=(nreg + 1) * nbins).reshape(nreg + 1, nbins)
            for r in range(1, nreg + 1):
                area = int(hist[r].sum())
                if area:
                    region_hists.append(hist[r])
                    region_areas.append(area)
        keep = ~g
        if keep.any():
            neg_hist += np.bincount(idx[keep], minlength=nbins)
            neg_total += int(keep.sum())

    if not region_hists:
        raise ValueError('没有可用的缺陷连通域（GT 全空？）')
    region_hists = np.stack(region_hists)
    region_areas = np.asarray(region_areas, np.float64)

    ths = np.linspace(lo, hi, num_th + 2)[1:-1]
    ks = np.clip(((ths - lo) / scale).astype(np.int64), 0, nbins)
    cdf_r = np.concatenate([np.zeros((len(region_hists), 1), np.int64),
                            np.cumsum(region_hists, axis=1)], axis=1)[:, ks]
    pro = (1.0 - cdf_r / region_areas[:, None]).mean(axis=0)

    cdf_n = np.concatenate([[0], np.cumsum(neg_hist)])[ks]
    fpr = (neg_total - cdf_n) / float(max(neg_total, 1))

    return dict(thresholds=ths, pro=pro, fpr=fpr,
                n_regions=int(len(region_areas)), n_images=len(score_maps),
                region_areas=region_areas)


def aupro(fpr, pro, limit=0.3):
    """PRO-FPR 曲线在 [0, limit] 上的归一化面积（1 为完美）。

    只对曲线上真实出现的 (FPR, PRO) 点做梯形积分；FPR=0 处用最小可达 FPR 的 PRO 值
    向前平推（阈值再抬高到全图正常时 PRO 也只会更低，所以这不会虚高）。
    若整条曲线的 FPR 都高于 limit（检测器在低过杀区完全不可用），返回 0。
    """
    x = np.asarray(fpr, np.float64)
    y = np.asarray(pro, np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = x <= limit
    if not keep.any():
        return 0.0
    x, y = x[keep], y[keep]
    xs = np.concatenate([[0.0], x, [limit]]) if x[0] > 0 else np.concatenate([x, [limit]])
    ys = np.concatenate([[y[0]], y, [np.interp(limit, x, y)]]) if x[0] > 0 else np.concatenate([y, [np.interp(limit, x, y)]])
    return float(_TRAPZ(ys, xs)) / limit


def pro_at(fpr, pro, target):
    """PRO 在指定 FPR 处的值（曲线线性插值）。"""
    x = np.asarray(fpr, np.float64)
    y = np.asarray(pro, np.float64)
    order = np.argsort(x)
    return float(np.interp(target, x[order], y[order]))


def pixel_auroc(scores, masks, nbins=1024):
    """像素级 AUROC。scores/masks 为同形状数组列表（patch 级即可，与 results.csv 同口径）。"""
    lo = min(float(np.min(m)) for m in scores)
    hi = max(float(np.max(m)) for m in scores)
    if hi <= lo:
        # 常数分数图（模型完全退化）：排序无信息，AUROC 按 0.5 处理而不是除零崩溃
        return 0.5
    scale = (hi - lo) / float(nbins)
    pos = np.zeros(nbins, np.int64)
    neg = np.zeros(nbins, np.int64)
    for m, g in zip(scores, masks):
        idx = np.clip(((np.asarray(m, np.float64) - lo) / scale).astype(np.int64), 0, nbins - 1)
        g = np.asarray(g, bool)
        pos += np.bincount(idx[g], minlength=nbins)
        neg += np.bincount(idx[~g], minlength=nbins)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # AUROC = P(score_pos > score_neg) + 0.5 * P(相等)，逐 bin 用均匀分布近似相等部分
    neg_below = np.concatenate([[0], np.cumsum(neg)])[:-1]  # 严格小于当前 bin 的负数个数
    lucky = (pos.astype(np.float64) * neg_below).sum() + 0.5 * (pos.astype(np.float64) * neg).sum()
    return float(lucky / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# 数据装载
# --------------------------------------------------------------------------- #
def _upsample(m, size):
    """patch 分数图 -> (w, h)：线性拉伸到 0..255 再双线性放大，最后映射回原值域。"""
    arr = np.asarray(m, np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo if hi > lo else 1.0
    Image = _pil()
    img = Image.fromarray(((arr - lo) / span * 255.0).astype(np.uint8))
    up = np.asarray(img.resize(size, Image.BILINEAR), np.float32)
    return up / 255.0 * span + lo


def mask_path(data_root, cat, rel):
    rel = str(rel).replace('\\', '/')
    d, base = os.path.split(rel)
    stem = os.path.splitext(base)[0]
    return os.path.join(data_root, cat, 'ground_truth', d, stem + '_mask.png')


def load_class(dump_dir, data_root, cat, max_images=0):
    """返回 (原图分辨率分数图列表, 原图分辨率 GT 列表, patch 级 (scores, masks), meta)。"""
    fp = os.path.join(dump_dir, cat + '.npz')
    z = np.load(fp, allow_pickle=False)
    scores, masks = z['scores'], z['masks']
    paths, labels = z['paths'], z['labels']
    smaps, gmaps, used, missing = [], [], 0, 0
    Image = _pil()
    for i in range(len(paths)):
        rel = str(paths[i])
        if labels[i] == 1:
            mp = mask_path(data_root, cat, rel)
            if not os.path.exists(mp):
                missing += 1
                continue
            g = np.asarray(Image.open(mp).convert('L')) > 128
            h, w = g.shape
        else:
            with Image.open(os.path.join(data_root, cat, 'test', rel)) as im:
                w, h = im.size
            g = np.zeros((h, w), bool)
        smaps.append(_upsample(scores[i], (w, h)))
        gmaps.append(g)
        used += 1
        if max_images and used >= max_images:
            break
    meta = dict(category=cat, images=used, missing_masks=missing, total=int(len(paths)),
                defect_images=int((labels == 1).sum()))
    return smaps, gmaps, (scores, masks, paths, labels), meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description='PRO/AUPRO 评测（读 patchcore_demo --dump-pixel 产物）')
    ap.add_argument('--dumps', default='experiment_data/05_indexes/pixel_dumps')
    ap.add_argument('--data', default='MVTec/mvtech_anomaly_detection')
    ap.add_argument('--category', default='all')
    ap.add_argument('--num-th', type=int, default=200, help='PRO 曲线的阈值个数')
    ap.add_argument('--limit', type=float, default=0.3, help='AUPRO 的 FPR 截断（官方 0.3）')
    ap.add_argument('--max-images', type=int, default=0, help='每类最多多少张（调试）')
    ap.add_argument('--out', default='', help='逐类结果 JSON')
    ap.add_argument('--markdown', default='', help='把表格写到这个 md 文件（覆盖）')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dumps):
        print('dump 目录不存在：%s' % args.dumps, file=sys.stderr)
        return 2
    cats = sorted(os.path.splitext(f)[0] for f in os.listdir(args.dumps) if f.endswith('.npz'))
    if args.category != 'all':
        cats = [c for c in cats if c == args.category]
    if not cats:
        print('没有可用的 dump 文件：%s' % args.dumps, file=sys.stderr)
        return 2

    rows, pooled_s, pooled_g, metas = [], [], [], []
    t0 = time.time()
    for cat in cats:
        try:
            smaps, gmaps, (ps, pm, paths, labels), meta = load_class(
                args.dumps, args.data, cat, args.max_images)
        except Exception as e:  # noqa: BLE001
            print('[%s] 跳过：%s' % (cat, e), file=sys.stderr)
            continue
        if not smaps or not any(g.any() for g in gmaps):
            print('[%s] 跳过：没有缺陷区域' % cat, file=sys.stderr)
            continue
        c = pro_curve(smaps, gmaps, num_th=args.num_th)
        pxl = pixel_auroc(list(ps), list(pm))
        r = dict(category=cat, aupru=aupro(c['fpr'], c['pro'], args.limit),
                 aupru_01=aupro(c['fpr'], c['pro'], 0.1),
                 pro_at_005=pro_at(c['fpr'], c['pro'], 0.05),
                 pro_at_01=pro_at(c['fpr'], c['pro'], 0.10),
                 pixel_auroc=pxl, n_regions=c['n_regions'], n_images=c['n_images'])
        rows.append(r)
        metas.append(meta)
        pooled_s.extend(smaps)
        pooled_g.extend(gmaps)
        print('[%-11s] AUPRO@30%% = %.4f | AUPRO@10%% = %.4f | PRO@5%%FPR = %.4f | 区域 %4d | %.1fs'
              % (cat, r['aupru'], r['aupru_01'], r['pro_at_005'], r['n_regions'], time.time() - t0),
              flush=True)

    if not rows:
        print('没有任何类别跑出结果', file=sys.stderr)
        return 2

    pooled = None
    if pooled_s:
        c = pro_curve(pooled_s, pooled_g, num_th=args.num_th)
        pooled = dict(aupru=aupro(c['fpr'], c['pro'], args.limit),
                      aupru_01=aupro(c['fpr'], c['pro'], 0.1),
                      pro_at_005=pro_at(c['fpr'], c['pro'], 0.05),
                      pro_at_01=pro_at(c['fpr'], c['pro'], 0.10),
                      n_regions=c['n_regions'], n_images=c['n_images'])

    aup = np.array([r['aupru'] for r in rows])
    summary = dict(classes=len(rows), aupru_macro=float(aup.mean()),
                   aupru_std=float(aup.std(ddof=0)),
                   aupru_min=float(aup.min()), aupru_max=float(aup.max()),
                   worst=sorted(rows, key=lambda z: z['aupru'])[:3],
                   pooled=pooled, per_class=rows, images=metas,
                   protocol=dict(metric='PRO/AUPRO', fpr_limit=args.limit, num_th=args.num_th,
                                 connectivity=4, weight='每个 GT 连通域等权（官方口径）',
                                 upsample='双线性 patch->原图', dumps=args.dumps))
    print('\n%d 类：AUPRO@30%% macro **%.4f ± %.4f**（min %.4f / max %.4f）'
          % (len(rows), summary['aupru_macro'], summary['aupru_std'],
             summary['aupru_min'], summary['aupru_max']))
    if pooled:
        print('pooled AUPRO@30%% = %.4f（%d 图 / %d 区域）'
              % (pooled['aupru'], pooled['n_images'], pooled['n_regions']))

    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print('已写入 %s' % args.out)

    if args.markdown:
        lines = ['| 类别 | AUPRO@30% | AUPRO@10% | PRO@5%FPR | pixel AUROC | 区域数 |',
                 '|---|---|---|---|---|---|']
        for r in sorted(rows, key=lambda z: -z['aupru']):
            lines.append('| %s | %.4f | %.4f | %.4f | %.4f | %d |'
                         % (r['category'], r['aupru'], r['aupru_01'],
                            r['pro_at_005'], r['pixel_auroc'], r['n_regions']))
        lines.append('| **macro 均值** | **%.4f ± %.4f** | — | — | %.4f | %d |'
                     % (summary['aupru_macro'], summary['aupru_std'],
                        float(np.mean([r['pixel_auroc'] for r in rows])),
                        sum(r['n_regions'] for r in rows)))
        if pooled:
            lines.append('| **pooled** | **%.4f** | %.4f | %.4f | — | %d |'
                         % (pooled['aupru'], pooled['aupru_01'], pooled['pro_at_005'],
                            pooled['n_regions']))
        d = os.path.dirname(args.markdown)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.markdown, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print('表格已写入 %s' % args.markdown)
    return 0


if __name__ == '__main__':
    sys.exit(main())
