# -*- coding: utf-8 -*-
"""五种图像分数聚合的算术定义必须可复算：数量取整规则、截尾区间、阈值与三档判定。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.agg_robust as A  # noqa: E402


def test_max_is_max():
    flat = np.array([0.1, 0.9, 0.5])
    assert abs(A.agg_image(flat, 'max') - 0.9) < 1e-12


def test_top_k_ceiling_rule():
    # 1000 个 patch、top1% → 10 个；数值 0..999，前 10 个是 990..999
    flat = np.arange(1000, dtype=np.float64)
    assert abs(A.agg_image(flat, 'top1pct') - np.mean(np.arange(990, 1000))) < 1e-9
    # 33 个 patch、top5% → ceil(1.65)=2 个
    small = np.arange(33, dtype=np.float64)
    assert abs(A.agg_image(small, 'top5pct') - np.mean([32, 31])) < 1e-9


def test_trimmed_drops_top_one_percent():
    flat = np.arange(1000, dtype=np.float64)
    # 去掉最高 10 个（990..999），再取下一个 100 个（890..989）的均值
    assert abs(A.agg_image(flat, 'trimmed') - np.mean(np.arange(890, 990))) < 1e-9


def test_trimmed_handles_tiny_maps():
    flat = np.array([1.0, 2.0])
    # drop = 1，keep = 1 → 只剩 1.0
    assert abs(A.agg_image(flat, 'trimmed') - 1.0) < 1e-12


def test_threshold_is_upper_quantile():
    calib = np.arange(100, dtype=np.float64)
    thr = A.threshold_of(calib, 0.05)
    assert 94 <= thr <= 96


def test_evaluate_counts_three_zones():
    scores = np.array([1.0, 5.0, 10.5, 30.0])
    labels = np.array([0, 1, 0, 1])
    r = A.evaluate(scores, labels, thr_ok=10.0, ng_ratio=1.2)  # thr_ng = 12.0
    # 1.0 → OK（良品 OK）；5.0 → OK（缺陷漏检）；10.5 → REVIEW（良品复核）；30.0 → NG（缺陷命中）
    assert r['n_bad'] == 2 and r['n_good'] == 2
    assert abs(r['miss'] - 0.5) < 1e-12
    assert abs(r['over'] - 0.0) < 1e-12
    assert abs(r['review'] - 0.25) < 1e-12


def test_agg_all_shapes():
    maps = np.random.RandomState(0).rand(3, 4, 4)
    out = A.agg_all(maps, ['max', 'top10pct'])
    assert out['max'].shape == (3,)
    assert out['top10pct'].shape == (3,)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
