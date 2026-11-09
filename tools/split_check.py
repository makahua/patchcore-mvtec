# -*- coding: utf-8 -*-
"""对比两个 split-seed 下的基线（2+3 / legacy greedy 2000 / P99.5）。

用法：
  python tools/split_check.py --a experiment_data/06_optimization/10_pair --b experiment_data/06_optimization/12_split123
"""
import argparse
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_group(root, slug):
    out = {}
    pattern = f'{slug}_*_seed_*' if slug else '*_seed_*'
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        p = os.path.join(d, 'metrics.json')
        if not os.path.exists(p):
            continue
        m = json.load(open(p, encoding='utf-8'))
        name = os.path.basename(d)
        seed = int(name.rsplit('_seed_', 1)[1])
        for c in m.get('categories', []):
            name_c = c.get('category') if isinstance(c, dict) else c
            out[(name_c, seed)] = dict(image=c.get('img_auroc'), pixel=c.get('pixel_auroc'),
                                       lat=c.get('latency_ms'), thr=c.get('threshold'),
                                       rec=c.get('recall_at_thr'), fpr=c.get('false_positive_rate'))
    return out


def macro(group, seeds, cats, key):
    per_seed = [st.mean(group[(c, s)][key] for c in cats if (c, s) in group) for s in seeds
                if any((c, s) in group for c in cats)]
    return st.mean(per_seed), (st.stdev(per_seed) if len(per_seed) > 1 else 0.0), per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=os.path.join(ROOT, 'experiment_data/06_optimization/10_pair'))
    ap.add_argument('--b', default=os.path.join(ROOT, 'experiment_data/06_optimization/12_split123'))
    ap.add_argument('--slug-a', default='23')
    ap.add_argument('--slug-b', default='')
    a = ap.parse_args()

    ga = load_group(a.a, a.slug_a)
    gb = load_group(a.b, a.slug_b)
    cats = sorted({c for c, _ in gb}) or sorted({c for c, _ in ga})
    seeds = sorted({s for _, s in gb}) or sorted({s for _, s in ga})

    print(f'class count: {len(cats)}  seeds: {seeds}')
    rows = []
    for key in ('image', 'pixel', 'lat', 'thr'):
        if key in ('thr',):
            ma = st.mean(ga[(c, s)][key] for c in cats for s in seeds if (c, s) in ga)
            mb = st.mean(gb[(c, s)][key] for c in cats for s in seeds if (c, s) in gb)
            print(f'{key:6s} split42={ma:.4f} split123={mb:.4f}  d={mb - ma:+.4f}')
            continue
        ma, sa, _ = macro(ga, seeds, cats, key)
        mb, sb, _ = macro(gb, seeds, cats, key)
        print(f'{key:6s} split42={ma:.4f}±{sa:.4f}  split123={mb:.4f}±{sb:.4f}  d={mb - ma:+.4f}')
        rows.append(dict(metric=key, split42=ma, split42_std=sa, split123=mb, split123_std=sb, delta=mb - ma))

    print('\nper-class image AUROC (split42 -> split123, 3-seed mean)')
    worst = []
    for c in cats:
        va = [ga[(c, s)]['image'] for s in seeds if (c, s) in ga]
        vb = [gb[(c, s)]['image'] for s in seeds if (c, s) in gb]
        if not va or not vb:
            continue
        da, db = st.mean(va), st.mean(vb)
        worst.append((abs(db - da), c, da, db))
        print(f'  {c:11s} {da:.4f} -> {db:.4f}  d={db - da:+.4f}')

    worst.sort(reverse=True)
    print('\nlargest shift:', ', '.join(f'{c} {db - da:+.4f}' for _, c, da, db in worst[:3]))
    out = os.path.join(a.b, 'split_check.json')
    json.dump(dict(categories=cats, seeds=seeds, macro=rows,
                   per_class={c: dict(split42=da, split123=db) for _, c, da, db in worst}),
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written', out)


if __name__ == '__main__':
    raise SystemExit(main())
