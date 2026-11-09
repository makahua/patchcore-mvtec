# -*- coding: utf-8 -*-
"""图像级指标的唯一实现（AUROC / PR-AUC）。

所有脚本（analyze / classify / patchcore_demo / tools/*）都从这里取，避免同一口径有 7 份实现。
纯 numpy，不依赖 torch —— 这样 classify.py report 与 CI 的 numpy-only 环境也能直接调用。
"""
import numpy as np


def auc(y_true, y_score):
    """Rank-sum AUROC（并列按平均秩处理）。

    y_true 里 1 = 缺陷（正类）、0 = 良品；y_score 越大越像缺陷。
    退化情形（只有一类）返回 nan。
    """
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc(scores, labels):
    """auc 的参数顺序别名（部分工具按 (scores, labels) 调用）。"""
    return auc(labels, scores)


def auprc(y_true, y_score):
    """PR-AUC：按分数降序的阶梯积分（与 analyze.py 原实现逐位一致）。

    注意沿用的是「平均精度」的离散写法：sum(d_recall * precision_at_k)。
    """
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))
    yy = np.asarray(y_true)[order]
    tp = np.cumsum(yy)
    n_pos = max(1, int(yy.sum()))
    prec = tp / np.arange(1, len(yy) + 1)
    rec = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
