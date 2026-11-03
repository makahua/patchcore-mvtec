# -*- coding: utf-8 -*-
"""PRO / AUPRO 自检：只依赖 numpy + Pillow，不需要 torch / GPU / MVTec 数据。

跑法：
    python -m pytest tests/ -q
    python tests/test_pro.py

为什么值得有：PRO 的坑全在细节里——连通域用 4 邻接还是 8 邻接、区域等权还是按面积加权、
FPR 截断后面积除 limit 还是除实际宽度、FPR=0 处怎么取锚点。任何一种写歪了，
AUPRO 会稳定地偏 3~8 个点而且不报错，只有合成用例能钉住。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from pro_aupro import aupro, label_regions, pixel_auroc, pro_curve, pro_at  # noqa: E402


def _canvas(h=48, w=48):
    return np.zeros((h, w), bool), np.zeros((h, w), np.float64)


# ---------------------------------------------------------------- 连通域
def test_label_regions_two_blocks():
    """两块不相邻的方块必须是 2 个区域。"""
    g, _ = _canvas()
    g[2:8, 2:8] = True
    g[20:26, 20:26] = True
    lab, n = label_regions(g)
    assert n == 2, n
    assert lab[3, 3] != lab[21, 21]
    assert lab[3, 3] != 0 and lab[10, 10] == 0


def test_label_regions_diagonal_is_two():
    """对角相接（只共享一个顶点）在 4 邻接下是两个区域，不是一块。"""
    g, _ = _canvas()
    g[2:6, 2:6] = True
    g[6:10, 6:10] = True
    lab, n = label_regions(g)
    assert n == 2, n
    assert lab[3, 3] != lab[7, 7]


def test_label_regions_l_shape_is_one():
    """L 形拐角相接是同一个区域。"""
    g, _ = _canvas()
    g[4:12, 4:8] = True
    g[8:12, 8:16] = True
    lab, n = label_regions(g)
    assert n == 1, n
    assert lab[5, 5] == lab[10, 12]


# ---------------------------------------------------------------- AUPRO
def test_aupro_perfect_detector_is_one():
    """GT 内部分数高、外部分数低：低 FPR 区 PRO 恒为 1，AUPRO 应≈1。"""
    g, _ = _canvas()
    g[10:20, 10:20] = True
    m = np.where(g, 10.0, 0.0)
    c = pro_curve([m], [g], num_th=200)
    a = aupro(c['fpr'], c['pro'], 0.3)
    assert a > 0.98, a
    assert pro_at(c['fpr'], c['pro'], 0.05) > 0.98


def test_aupro_inverted_detector_is_zero():
    """分数完全反了（缺陷低、背景高）：FPR 一直很高，AUPRO 必须是 0。"""
    g, _ = _canvas()
    g[10:20, 10:20] = True
    m = np.where(g, 0.0, 10.0)
    c = pro_curve([m], [g], num_th=200)
    assert aupro(c['fpr'], c['pro'], 0.3) == 0.0


def test_aupro_half_regions_is_half():
    """两个等大的缺陷区域只检出一个：区域等权时 PRO=0.5，AUPRO≈0.5。"""
    g, _ = _canvas(48, 64)
    g[5:15, 5:15] = True       # 区域 A
    g[5:15, 40:50] = True      # 区域 B
    m = np.zeros((48, 64))
    m[5:15, 5:15] = 10.0       # 只给 A 高分
    c = pro_curve([m], [g], num_th=200)
    a = aupro(c['fpr'], c['pro'], 0.3)
    assert abs(a - 0.5) < 0.02, a


def test_aupro_reports_region_count():
    """区域数要如实统计（2 块 -> 2）。"""
    g, _ = _canvas()
    g[2:6, 2:6] = True
    g[30:34, 30:34] = True
    c = pro_curve([np.where(g, 5.0, 0.0)], [g])
    assert c['n_regions'] == 2, c['n_regions']


# ---------------------------------------------------------------- 像素 AUROC
def test_pixel_auroc_perfect_and_reversed():
    g, _ = _canvas()
    g[10:20, 10:20] = True
    pos = np.where(g, 3.0, 0.0)
    assert pixel_auroc([pos], [g]) > 0.999
    neg = np.where(g, 0.0, 3.0)
    assert pixel_auroc([neg], [g]) < 0.001


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'  FAIL  {name}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
