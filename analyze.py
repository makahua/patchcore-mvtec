# -*- coding: utf-8 -*-
"""离线分析 PatchCore 结果：从 scores/*.json 重算「固定过杀率下的召回」与 PR-AUC。

用法: python analyze.py results_fix [results_greedy ...]
"""
import csv, glob, json, os, sys
import numpy as np


def auc(y, s):
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return float('nan')
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(y, s):
    order = np.argsort(-np.asarray(s, dtype=np.float64)); yy = np.asarray(y)[order]
    tp = np.cumsum(yy); npos = max(1, int(yy.sum()))
    prec = tp / np.arange(1, len(yy) + 1); rec = tp / npos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'scores', '*.json'))):
        j = json.load(open(f, encoding='utf-8'))
        calib = np.array(j['calib'], dtype=np.float64)
        y = np.array([t['label'] for t in j['test']])
        s = np.array([t['score'] for t in j['test']], dtype=np.float64)
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())

        def at(pct):
            thr = float(np.percentile(calib, pct))
            pred = s > thr
            return (int((pred & (y == 1)).sum()) / max(1, npos),
                    int((pred & (y == 0)).sum()) / max(1, nneg))
        r15, f15 = at(98.5)
        r05, f05 = at(99.5)
        r01, f01 = at(99.9) if len(calib) >= 30 else (float('nan'), float('nan'))
        rows.append(dict(cat=j['category'], img=auc(y, s), pr=auprc(y, s), _y=y, _s=s,
                         r15=r15, f15=f15, r05=r05, f05=f05, r01=r01, f01=f01,
                         ngood=nneg, ndef=npos))
    px = {}
    c = os.path.join(d, 'results.csv')
    if os.path.exists(c):
        for r in csv.DictReader(open(c, encoding='utf-8-sig')):
            px[r['category']] = (float(r['pixel_auroc']), float(r['latency_ms']),
                                 int(r['mem_patches']), float(r.get('peak_gpu_mb') or 0))
    for r in rows:
        if r['cat'] in px:
            r['px'], r['ms'], r['mem'], r['gpu'] = px[r['cat']]
    return rows


def show(tag, rows):
    print(f"\n### {tag}  ({len(rows)} 类)")
    print('| 类别 | image AUROC | pixel AUROC | PR-AUC | 召回@1.5%过杀 | 召回@0.5%过杀 | 过杀率 | 单张ms | 记忆库 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['cat']} | {r['img']:.3f} | {r.get('px', 0):.3f} | {r['pr']:.3f} | "
              f"{r['r15']:.1%} | {r['r05']:.1%} | {r['f05']:.1%} | {r.get('ms', 0):.1f} | {r.get('mem', 0)} |")
    m = lambda k: np.mean([r[k] for r in rows if k in r and not np.isnan(r[k])])
    sd = lambda k: np.std([r[k] for r in rows if k in r and not np.isnan(r[k])])
    print(f"| **平均（macro）** | **{m('img'):.4f} ± {sd('img'):.4f}** | **{m('px'):.4f} ± {sd('px'):.4f}** | "
          f"**{m('pr'):.3f}** | **{m('r15'):.1%}** | **{m('r05'):.1%}** | {m('f05'):.1%} | {m('ms'):.1f} | {m('mem'):.0f} |")
    worst = sorted(rows, key=lambda r: r['img'])[:3]
    print(f"最低三类： " + "、".join(f"{r['cat']} {r['img']:.3f}" for r in worst))

    # ---- 同一批分数换一种口径：把所有类的图混在一起算一次（pooled）----
    # macro 每类权重相同；pooled 每张图权重相同，缺陷图多的难类（screw/capsule）会拉低它。
    # 两个数都对，差的不是精度而是口径——对外只报一个时必须标注。
    all_y = [yy for r in rows for yy in r['_y']]
    all_s = [ss for r in rows for ss in r['_s']]
    pooled = auc(all_y, all_s)
    print(f"**口径**：macro image AUROC **{m('img'):.4f}**（15 类平均）vs pooled **{pooled:.4f}**"
          f"（{len(all_y)} 张混池），差 {abs(m('img') - pooled) * 100:.1f} 个点。")


for d in sys.argv[1:]:
    show(os.path.basename(d), load(d))
