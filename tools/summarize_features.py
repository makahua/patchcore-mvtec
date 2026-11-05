# -*- coding: utf-8 -*-
r"""只读汇总 03_features 下的特征层消融结果，输出 markdown 表。

用法：
  .\.venv\Scripts\python.exe tools\summarize_features.py
  .\.venv\Scripts\python.exe tools\summarize_features.py --root experiment_data/06_optimization/03_features --md out.md

不写任何结果目录，只读 metrics.json / aupro.json / config.json。
"""
import argparse
import io
import json
import os
import statistics as st

COMBOS = [('layer23', '2+3'), ('layer12', '1+2'), ('layer123', '1+2+3')]
CATS = ['screw', 'toothbrush', 'pill']


def load(run_dir):
    mp = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(mp):
        return None
    m = json.load(io.open(mp, encoding='utf-8'))
    a = m['aggregate']['macro']
    c = m['categories'][0]
    rec = {
        'layers': c.get('layers') or '',
        'feat_dim': c.get('feat_dim'),
        'mem': c.get('mem_patches'),
        'n_train': c.get('n_train'),
        'n_calib': c.get('n_calib'),
        'n_test': c.get('n_test'),
        'img': a['img_auroc'], 'px': a['pixel_auroc'],
        'recall': a['recall_at_thr'], 'fpr': a['false_positive_rate'],
        'lat': a['latency_ms'], 'build': a['build_seconds'], 'peak': a['peak_gpu_mb'],
    }
    ap = os.path.join(run_dir, 'aupro.json')
    if os.path.exists(ap):
        try:
            j = json.load(io.open(ap, encoding='utf-8'))
            pc = (j.get('per_class') or [{}])[0]
            rec['aupro30'] = pc.get('aupru')
            rec['aupro10'] = pc.get('aupru_01')
            rec['pro05'] = pc.get('pro_at_005')
            rec['n_regions'] = pc.get('n_regions')
        except Exception:  # noqa: BLE001
            pass
    return rec


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiment_data/06_optimization/03_features')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()

    lines = []
    lines.append('| 组合 | 类别 | n(seed) | image AUROC | pixel AUROC | AUPRO@30% | AUPRO@10% | PRO@5% | 召回 | 过杀 | 算法时延 | 建库 | GPU 峰值 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for tag, label in COMBOS:
        for cat in CATS:
            recs = []
            for seed in (0, 1, 2):
                rd = os.path.join(a.root, '%s_%s_seed_%03d' % (tag, cat, seed))
                r = load(rd) if os.path.isdir(rd) else None
                if r:
                    recs.append(r)
            if not recs:
                lines.append('| `%s` | %s | 0 | — | — | — | — | — | — | — | — | — | — |'
                             % (label, cat))
                continue
            def fmt(k, nd=4, scale=1.0):
                m, s = mean_std([r.get(k) for r in recs])
                if m is None:
                    return '—'
                if nd == 0:
                    return '%.0f' % (m * scale)
                return ('%.' + str(nd) + 'f') % (m * scale)
            lines.append('| `%s` | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                label, cat, len(recs),
                fmt('img'), fmt('px'), fmt('aupro30'), fmt('aupro10'), fmt('pro05'),
                fmt('recall'), fmt('fpr'), fmt('lat', 2), fmt('build', 0), fmt('peak', 1)))
    text = '\n'.join(lines)
    print(text)
    if a.md:
        io.open(a.md, 'w', encoding='utf-8', newline='\n').write(text + '\n')


if __name__ == '__main__':
    main()
