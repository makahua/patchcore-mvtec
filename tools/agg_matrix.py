# -*- coding: utf-8 -*-
"""第 ⑤ 项：5 种聚合 × 5 组条件的离线对比（读 --dump-suite 产物）。

两种阈值口径：
  frozen —— 用干净 calibration 标定，阈值冻结后应用到扰动图；
  recal  —— 用同条件 calibration 重新标定（P99.5）。

用法：
  python tools/agg_matrix.py --suite experiment_data/06_optimization/11_robust_agg \
      --out experiment_data/06_optimization/11_robust_agg/agg_matrix.json
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.agg_robust import MODES, agg_all, auc, threshold_of, evaluate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS = ('clean', 'bright08', 'noise005', 'noise010', 'blur1')


def load_pair(suite, cat, cond):
    """返回 (calib_scores_by_mode, test_scores_by_mode, test_labels)。"""
    cal = np.load(os.path.join(suite, f'{cat}_{cond}_calib.npz'))
    tst = np.load(os.path.join(suite, f'{cat}_{cond}_test.npz'))
    cs = agg_all(cal['scores'])
    ts = agg_all(tst['scores'])
    return cs, ts, np.asarray(tst['labels']).astype(int)


def main():
    ap = argparse.ArgumentParser(description="聚合 × 条件矩阵")
    ap.add_argument('--suite', default=os.path.join(ROOT, 'experiment_data', '06_optimization', '11_robust_agg'))
    ap.add_argument('--fpr', type=float, default=0.005)
    ap.add_argument('--ng-ratio', type=float, default=1.2)
    ap.add_argument('--md', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(a.suite, '*_seed_*')))
    pairs = {}
    for d in runs:
        name = os.path.basename(d)
        cat, seed_txt = name.rsplit('_seed_', 1)
        seed = int(seed_txt)
        pairs.setdefault((cat, seed), os.path.join(d, 'suite'))

    if not pairs:
        raise SystemExit('[error] 没有找到 *_seed_* 目录')

    cats = sorted({c for c, _ in pairs})
    seeds = sorted({s for _, s in pairs})
    rec = {}
    for (cat, seed), suite in pairs.items():
        for cond in CONDITIONS:
            cs, ts, labels = load_pair(suite, cat, cond)
            clean_cs = cs if cond == 'clean' else load_pair(suite, cat, 'clean')[0]
            for mode in MODES:
                thr_f = threshold_of(clean_cs[mode], a.fpr)
                thr_r = threshold_of(cs[mode], a.fpr)
                rec[(cat, seed, cond, mode)] = dict(
                    auroc=auc(labels, ts[mode]),
                    frozen=evaluate(ts[mode], labels, thr_f, a.ng_ratio),
                    recal=evaluate(ts[mode], labels, thr_r, a.ng_ratio),
                )

    out = {'protocol': dict(fpr=a.fpr, ng_ratio=a.ng_ratio, thr_pct=99.5,
                            conditions=list(CONDITIONS), modes=list(MODES),
                            categories=cats, seeds=seeds),
           'cells': {'|'.join(map(str, k)): v for k, v in rec.items()}}

    lines = ['# 聚合 × 条件矩阵', '',
             f'- 阈值口径：主阈值 P99.5（fpr={a.fpr}），灰区 ×{a.ng_ratio}',
             f'- 类别：{", ".join(cats)}；seed：{seeds}', '',
             '## 图像 AUROC（冻结阈值不影响 AUROC，两套口径同一列）', '',
             '| 条件 | ' + ' | '.join(MODES) + ' |',
             '|---|' + '---:|' * len(MODES)]
    for cond in CONDITIONS:
        cells = []
        for mode in MODES:
            vals = [rec[(c, s, cond, mode)]['auroc'] for c in cats for s in seeds
                    if (c, s, cond, mode) in rec]
            cells.append(f'{st.mean(vals):.4f}' if vals else '—')
        lines.append(f'| {cond} | ' + ' | '.join(cells) + ' |')

    lines += ['', '## 冻结阈值下的召回 / 过杀（macro 平均）', '',
              '| 条件 | 聚合 | 召回 | 过杀 | 复核 |', '|---|---|---:|---:|---:|']
    for cond in CONDITIONS:
        for mode in MODES:
            vals = [rec[(c, s, cond, mode)]['frozen'] for c in cats for s in seeds
                    if (c, s, cond, mode) in rec]
            if not vals:
                continue
            rec_ = st.mean(1 - v['miss'] for v in vals)
            fpr_ = st.mean(v['over'] for v in vals)
            rev_ = st.mean(v['review'] for v in vals)
            lines.append(f'| {cond} | {mode} | {rec_:.3f} | {fpr_:.4f} | {rev_:.3f} |')

    lines += ['', '## 筛查判定（计划 §5）', '']
    base = {(c, s): rec[(c, s, 'noise010', 'max')]['auroc'] for c in cats for s in seeds}
    improved = []
    for mode in MODES:
        if mode == 'max':
            continue
        gain = [rec[(c, s, 'noise010', mode)]['auroc'] - base[(c, s)] for c in cats for s in seeds]
        clean_drop = [rec[(c, s, 'clean', 'max')]['auroc'] - rec[(c, s, 'clean', mode)]['auroc']
                      for c in cats for s in seeds]
        per_cat = {}
        for c in cats:
            g = [rec[(c, s, 'noise010', mode)]['auroc'] - base[(c, s)] for s in seeds]
            per_cat[c] = st.mean(g)
        n_cat_ok = sum(1 for c in cats if per_cat[c] >= 0.03)
        worst_clean = max(clean_drop)
        ok = n_cat_ok >= 2 and worst_clean <= 0.01
        improved.append((mode, per_cat, st.mean(gain), worst_clean, ok,
                         'PASS' if ok else 'FAIL'))
        lines.append(f'- `{mode}`：σ=0.10 提升 ' +
                     '，'.join(f'{c} {v:+.4f}' for c, v in per_cat.items()) +
                     f'；合格类别 {n_cat_ok}/3；干净最大下降 {worst_clean:+.4f} → {improved[-1][5]}')

    if a.md:
        with open(a.md, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
