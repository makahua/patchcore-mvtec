#!/usr/bin/env python
"""重建 experiment_data/05_indexes/all_results.csv —— 扫描每个含 results.csv 的结果目录。

用法：
    python tools/index_results.py                 # 重建索引
    python tools/index_results.py --check         # 只校验索引是否与磁盘一致（CI 用）

为什么要有这个工具：结果目录是实验的「事实来源」，索引是给人看的视图。
视图靠手抄一定会烂（137 行里缺掉新跑的预处理/邻域/噪声/PRO 共 30 多组），
而烂索引比没有索引更糟——它会让决策基于过期数字。
"""
import argparse
import csv
import io
import os
import sys

# 逻辑顺序；实际列先按这个排，再补其余列（按字母序）
CANON = [
    'category', 'n_train', 'n_calib', 'n_test', 'size',
    'neighbors', 'neighbor_coreset', 'neighbor_pad',
    'preprocess', 'aspect_crop', 'imagenet_norm',
    'coreset', 'coreset_k', 'coreset_ratio', 'score_agg', 'blur_sigma',
    'train_limit', 'train_seed', 'split_seed',
    'perturb_brightness', 'perturb_noise', 'perturb_blur', 'perturb_calibration',
    'calib_ratio', 'thr_pct', 'seed', 'weights',
    'mem_patches', 'img_auroc', 'pixel_auroc', 'threshold',
    'recall_at_thr', 'false_positive_rate',
    'latency_ms', 'latency_p95_ms', 'build_seconds', 'peak_gpu_mb',
]
HEAD = ['archive_path']


def scan(exp_root):
    rows = []
    seen = []
    for dirpath, _dirnames, filenames in os.walk(exp_root):
        if 'results.csv' not in filenames:
            continue
        fp = os.path.join(dirpath, 'results.csv')
        rel = os.path.relpath(dirpath, exp_root).replace(os.sep, '/')
        with io.open(fp, 'r', encoding='utf-8-sig', newline='') as f:
            rd = csv.DictReader(f)
            for r in rd:
                r = {(k or '').strip(): (v or '').strip() for k, v in r.items()}
                r['archive_path'] = rel + '/results.csv'
                rows.append(r)
                for k in rd.fieldnames:
                    k = (k or '').strip()
                    if k and k not in seen:
                        seen.append(k)
    return rows, seen


def columns(seen):
    cols = list(HEAD)
    cols += [c for c in CANON if c in seen]
    cols += sorted(c for c in seen if c not in CANON)
    return cols


def render(rows, cols):
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\r\n')
    w.writerow(cols)
    for r in sorted(rows, key=lambda r: (r['archive_path'], r.get('category', ''))):
        w.writerow([r.get(c, '') for c in cols])
    return out.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description='重建实验结果索引 all_results.csv')
    ap.add_argument('--root', default='.', help='仓库根目录')
    ap.add_argument('--out', default='experiment_data/05_indexes/all_results.csv')
    ap.add_argument('--check', action='store_true', help='只校验，不写盘；不一致则 exit 1')
    a = ap.parse_args(argv)

    exp_root = os.path.join(a.root, 'experiment_data')
    if not os.path.isdir(exp_root):
        print('找不到 experiment_data/：%s' % exp_root, file=sys.stderr)
        return 2
    rows, seen = scan(exp_root)
    cols = columns(seen)
    text = render(rows, cols)

    out_path = os.path.join(a.root, a.out)
    if a.check:
        old = io.open(out_path, encoding='utf-8', newline='').read() if os.path.exists(out_path) else ''
        # 行尾可能被 git 归一（core.autocrlf）：Windows 工作区 CRLF、CI 检出 LF，
        # 比较内容而不比较换行符，否则同一个提交在 Linux 上会误报过期。
        norm = lambda t: t.replace('\r\n', '\n').replace('\r', '\n')
        if norm(old) == norm(text):
            print('索引一致：%d 行 / %d 列' % (len(rows), len(cols)))
            return 0
        old_n = norm(old).count('\n')
        print('索引过期：磁盘 %d 行，文件 %d 行（先跑 python tools/index_results.py 重建）' % (len(rows), old_n),
              file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, 'wb') as f:
        f.write(text.encode('utf-8'))
    runs = len(set(r['archive_path'] for r in rows))
    print('写入 %s：%d 行 / %d 个结果目录 / %d 列' % (a.out, len(rows), runs, len(cols)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
