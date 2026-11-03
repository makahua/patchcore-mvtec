# -*- coding: utf-8 -*-
"""指标与记忆库自检：只依赖 numpy，不需要 torch / GPU / MVTec 数据。

跑法：
    python -m pytest tests/ -q          # 有 pytest
    python tests/test_metrics.py        # 没 pytest 也能直接跑

为什么值得有这组测试：README 里那些「AUROC 0.9805 / 0.9709」是面试官唯一会当场
复算的数字。auc() 是手写的 rank 法（为了不依赖 sklearn），一旦并列值处理写错，
所有指标会整体偏一点而不会报错——这种 bug 靠肉眼看不出来，只能靠测试钉住。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyze import auc, auprc  # noqa: E402


# ---------------------------------------------------------------- AUROC
def test_auc_perfect_separation():
    """完全可分时 AUROC 必须是 1.0。"""
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert auc(y, s) == 1.0


def test_auc_reversed_is_zero():
    """完全反序时 AUROC 必须是 0.0。"""
    y = [0, 0, 1, 1]
    s = [0.9, 0.8, 0.2, 0.1]
    assert auc(y, s) == 0.0


def test_auc_all_ties_is_half():
    """全并列时任何排序都无信息，AUROC = 0.5。

    这是 rank 法最容易写错的地方：直接按 argsort 的名次算，并列值会被
    当成有先后顺序，结果随输入排列漂移。正确做法是把并列名次取平均。
    """
    y = [0, 1, 0, 1]
    s = [0.5, 0.5, 0.5, 0.5]
    assert auc(y, s) == 0.5


def test_auc_ties_are_order_invariant():
    """有并列值时，打乱输入顺序结果不能变。"""
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * 50)
    s = np.round(rng.rand(100), 1)  # 只保留 1 位小数 → 大量并列
    base = auc(y, s)
    for _ in range(5):
        idx = rng.permutation(len(s))
        assert abs(auc(y[idx], s[idx]) - base) < 1e-12


def test_auc_matches_pairwise_definition():
    """和「正负样本两两比较」的定义式对拍（小样本，暴力枚举）。"""
    rng = np.random.RandomState(42)
    for _ in range(20):
        n = 12
        y = rng.randint(0, 2, size=n)
        if y.sum() in (0, n):
            continue
        s = rng.rand(n)
        pos, neg = s[y == 1], s[y == 0]
        # 定义式：正样本分数 > 负样本记 1，相等记 0.5
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        expect = wins / (len(pos) * len(neg))
        assert abs(auc(y, s) - expect) < 1e-12


def test_auc_single_class_is_nan():
    """只有一类时无定义，返回 nan 而不是抛异常或瞎给个数。"""
    assert np.isnan(auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(auc([1, 1], [0.1, 0.2]))


# ---------------------------------------------------------------- PR-AUC
def test_auprc_perfect_is_one():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert abs(auprc(y, s) - 1.0) < 1e-12


def test_auprc_between_zero_and_one():
    rng = np.random.RandomState(1)
    for _ in range(10):
        y = rng.randint(0, 2, size=30)
        if y.sum() == 0:
            continue
        s = rng.rand(30)
        v = auprc(y, s)
        assert 0.0 <= v <= 1.0


def test_auprc_rewards_better_ranking():
    """排序更好的分数必须拿到更高的 PR-AUC（单调性自检）。"""
    rng = np.random.RandomState(7)
    y = rng.randint(0, 2, size=200)
    if y.sum() in (0, len(y)):  # 万一抽成单类，PR-AUC 无定义
        y[:1] = 1 - y[:1]
    good = y + rng.rand(200) * 0.1   # 几乎就是标签本身
    bad = rng.rand(200)              # 纯随机
    assert auprc(y, good) > auprc(y, bad)


# ---------------------------------------------------------------- 记忆库维度自检
def test_bank_dims_accepts_consistent_meta():
    """维度与 neighbors 一致时不应报错。"""
    from classify import check_bank_dims

    check_bank_dims(np.zeros((10, 1536)), {'neighbors': False})
    check_bank_dims(np.zeros((10, 13824)), {'neighbors': True})


def test_bank_dims_rejects_mismatch():
    """这是 BUG-1 的回归测试：建库恒 1536 维却记 neighbors=True 时必须立刻报错，
    而不是等到推理阶段抛一个跟根因无关的 cdist 维度错误。"""
    from classify import check_bank_dims

    try:
        check_bank_dims(np.zeros((10, 1536)), {'neighbors': True})
    except SystemExit as e:
        assert '1536' in str(e) and '13824' in str(e)
    else:
        raise AssertionError('维度不自洽时应当报错')


# ---------------------------------------------------------------- 手动跑
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
