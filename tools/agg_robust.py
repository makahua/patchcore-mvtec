# -*- coding: utf-8 -*-
"""鲁棒聚合：同一份 patch 分数图算五种图像分数聚合，比较判定稳定性。

只做聚合，不改特征/记忆库/阈值规则。五种聚合：
- max        最大 patch 分数（当前默认）
- top1pct    最高 1% patch 的均值（向上取整，至少 1 个）
- top5pct    最高 5% patch 的均值
- top10pct   最高 10% patch 的均值
- trimmed    去掉最高 1% patch 后，取下一个 10% 的均值（截尾均值）

阈值规则与主协议一致：calibration 良品分数的 (1-fpr) 分位数；灰区 thr_ng = thr_ok × ng_ratio。
用法：
    python tools/agg_robust.py --calib calib.npz --test test.npz --fpr 0.005 --out agg.json
两种 npz 都由 patchcore_demo.py 的 dump 生成，需含 `scores`(N,h,w) 与 `labels`(N,)（calib 全良品可缺）。
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODES = ('max', 'top1pct', 'top5pct', 'top10pct', 'trimmed')


def _top_mean(flat, frac):
    """最高 frac 比例的均值；个数向上取整且至少 1。"""
    n = flat.size
    k = max(1, int(np.ceil(n * frac)))
    part = np.partition(flat, -k)[-k:]
    return float(part.mean())


def agg_image(flat, mode):
    flat = np.asarray(flat, dtype=np.float64).ravel()
    if mode == 'max':
        return float(flat.max())
    if mode == 'top1pct':
        return _top_mean(flat, 0.01)
    if mode == 'top5pct':
        return _top_mean(flat, 0.05)
    if mode == 'top10pct':
        return _top_mean(flat, 0.10)
    if mode == 'trimmed':
        n = flat.size
        drop = max(1, int(np.ceil(n * 0.01)))
        keep = max(1, int(np.ceil(n * 0.10)))
        desc = np.sort(flat)[::-1]
        part = desc[drop:drop + keep]
        return float(part.mean()) if part.size else float(desc[-1])
    raise ValueError(f'未知聚合 {mode}')


def agg_all(maps, modes=MODES):
    """(N,h,w) 分数图 → {mode: (N,) 图像分数}。"""
    maps = np.asarray(maps)
    out = {m: np.empty(maps.shape[0], dtype=np.float64) for m in modes}
    for i in range(maps.shape[0]):
        for m in modes:
            out[m][i] = agg_image(maps[i], m)
    return out


def auc(y, s):
    """图像级 AUROC（秩和法，处理并列）。"""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def threshold_of(calib_scores, fpr):
    return float(np.percentile(np.asarray(calib_scores, dtype=np.float64), 100 * (1 - fpr)))


def evaluate(scores, labels, thr_ok, ng_ratio):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    thr_ng = thr_ok * ng_ratio
    ok = s < thr_ok
    ng = s >= thr_ng
    bad = y == 1
    good = y == 0
    miss = float((bad & ok).sum() / max(1, bad.sum()))
    over = float((good & ng).sum() / max(1, good.sum()))
    review = float(((s >= thr_ok) & (s < thr_ng)).sum() / max(1, len(s)))
    return dict(n=int(len(s)), n_bad=int(bad.sum()), n_good=int(good.sum()),
                auroc=auc(y, s), miss=miss, over=over, review=review,
                thr_ok=float(thr_ok), thr_ng=float(thr_ng))


def _load(path):
    with np.load(path, allow_pickle=True) as z:
        d = dict((k, z[k]) for k in z.files)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description='五种图像分数聚合对比（只读打分图）')
    ap.add_argument('--calib', required=True, help='calibration 分数图 npz（scores 或 pixel_scores）')
    ap.add_argument('--test', required=True, help='test 分数图 npz（含 scores 与 labels）')
    ap.add_argument('--fpr', type=float, default=0.005, help='目标误报率（分位数口径）')
    ap.add_argument('--ng-ratio', type=float, default=1.2)
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--markdown', action='store_true', help='打印 markdown 表')
    ap.add_argument('--out', default=None, help='JSON 输出路径')
    a = ap.parse_args(argv)

    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    cal = _load(a.calib)
    tst = _load(a.test)
    cal_maps = cal.get('scores', cal.get('pixel_scores'))
    tst_maps = tst.get('scores', tst.get('pixel_scores'))
    if cal_maps is None or tst_maps is None:
        raise SystemExit('npz 里没有 scores/pixel_scores')
    labels = np.asarray(tst.get('labels'))
    if labels.size != tst_maps.shape[0]:
        raise SystemExit(f'labels 长度 {labels.size} 与分数图 {tst_maps.shape[0]} 不一致')

    cal_agg = agg_all(cal_maps, modes)
    tst_agg = agg_all(tst_maps, modes)
    rows = {}
    for m in modes:
        thr = threshold_of(cal_agg[m], a.fpr)
        rows[m] = evaluate(tst_agg[m], labels, thr, a.ng_ratio)

    if a.markdown:
        print('| 聚合 | image AUROC | 漏检 | 实际 NG 过杀 | 复核 | thr_ok |')
        print('|---|---:|---:|---:|---:|---:|')
        for m in modes:
            r = rows[m]
            print(f'| {m} | {r["auroc"]:.4f} | {r["miss"]:.2%} | {r["over"]:.2%} '
                  f'| {r["review"]:.2%} | {r["thr_ok"]:.2f} |')
    else:
        for m in modes:
            print(m, rows[m])

    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(dict(fpr=a.fpr, ng_ratio=a.ng_ratio, modes=modes, rows=rows,
                           calib_n=int(cal_maps.shape[0]), test_n=int(tst_maps.shape[0])),
                      f, ensure_ascii=False, indent=2)
        print('已写入', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
