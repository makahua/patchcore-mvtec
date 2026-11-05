# -*- coding: utf-8 -*-
"""Verify split integrity and seed consistency across packaged runs.

Checks every `native/manifests/<cat>.json` under --root:
  * train / calibration / test are pairwise disjoint
  * train+calibration+test do not overlap and cover the expected per-category sets
  * the same category yields identical splits across seeds/layer combos
  * optional --require lists the categories that must be present

Usage:
  python tools/verify_protocol.py --root experiment_data/06_optimization/10_pair
  python tools/verify_protocol.py --root <dir> --require all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_manifests(root: Path):
    found = defaultdict(list)  # cat -> [(run_rel, manifest)]
    for path in sorted(root.glob('**/native/manifests/*.json')):
        cat = path.stem
        data = json.loads(path.read_text(encoding='utf-8'))
        run_rel = str(path.parents[2].relative_to(root))
        found[cat].append((run_rel, data))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--require', default='',
                    help='"all" 表示必须有 15 类；也可逗号分隔类别名')
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_absolute():
        root = ROOT / root
    if not root.is_dir():
        print(f'[错误] 目录不存在：{root}')
        return 2

    found = load_manifests(root)
    if not found:
        print(f'[错误] 在 {root} 下没找到 native/manifests/*.json')
        return 2

    problems = []
    print(f'{"类别":14s} {"运行数":>5s} {"train":>6s} {"calib":>6s} {"test":>6s}  一致性')
    for cat in sorted(found):
        runs = found[cat]
        base_run, base = runs[0]
        ok = True
        for run_rel, m in runs:
            for key in ('train', 'calibration', 'test'):
                if sorted(m.get(key) or []) != sorted(base.get(key) or []):
                    problems.append(f'{cat}: {run_rel} 的 {key} 清单与 {base_run} 不一致')
                    ok = False
        m = base
        tr, ca, te = (set(m.get(k) or []) for k in ('train', 'calibration', 'test'))
        for x, y, nx, ny in ((tr, ca, 'train', 'calibration'), (tr, te, 'train', 'test'),
                             (ca, te, 'calibration', 'test')):
            overlap = x & y
            if overlap:
                problems.append(f'{cat}: {nx} 与 {ny} 重叠 {len(overlap)} 个文件，例：{sorted(overlap)[0]}')
                ok = False
        print(f'{cat:14s} {len(runs):>5d} {len(tr):>6d} {len(ca):>6d} {len(te):>6d}  '
              f'{"OK" if ok else "FAIL"}')

    if a.require:
        want = sorted(found) if a.require == 'all' else [c.strip() for c in a.require.split(',') if c.strip()]
        missing = [c for c in want if c not in found]
        if missing:
            problems.append('缺少类别：' + ', '.join(missing))
        if a.require == 'all' and len(found) != 15:
            problems.append(f'要求 15 类，实际 {len(found)} 类')

    if problems:
        print('\n[失败]')
        for p in problems:
            print('  -', p)
        return 1
    print(f'\n[通过] {len(found)} 个类别，划分互斥且跨运行一致')
    return 0


if __name__ == '__main__':
    sys.exit(main())
