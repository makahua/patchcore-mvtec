# -*- coding: utf-8 -*-
"""patch 打分、图像聚合与 AUROC（从 patchcore_demo.py 拆出）。"""
import numpy as np
import torch
import torch.nn.functional as F

from .features import apply_pca, embed

def gaussian_blur(x, k=5, sigma=2.0):
    """对 patch 分数图做高斯平滑（PatchCore 用它抑制单点噪声）。"""
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.device)
    x = x.unsqueeze(0).unsqueeze(0)                          # 1,1,h,w
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    x = F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return x.squeeze(0).squeeze(0)


def auc(y_true, y_score):
    """rank 法 AUROC，避免依赖 sklearn。"""
    y_true = np.asarray(y_true).astype(np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))



def score_from_feat(feat, mem, neighbors=False, score_agg='max', blur_sigma=2.0,
                    neighbor_pad='zero', fp16=False):
    """由单图特征图算 patch 分数图与图像分数（最近邻 + 可选模糊 + 聚合）。"""
    emb, (h, w) = embed(feat, neighbors, neighbor_pad)
    emb = emb.squeeze(0)                                    # N, C
    emb = apply_pca(emb)                                    # --pca-dim 时降到低维再算距离
    # 分块算最近邻距离，避免显存炸
    mins = []
    if fp16:
        mem_h = mem.half()
        for i in range(0, emb.shape[0], 2048):
            d = torch.cdist(emb[i:i + 2048].half(), mem_h)
            mins.append(d.min(dim=1).values.float())
    else:
        for i in range(0, emb.shape[0], 2048):
            d = torch.cdist(emb[i:i + 2048], mem)
            mins.append(d.min(dim=1).values)
    patch = torch.cat(mins).reshape(h, w)
    patch = gaussian_blur(patch, sigma=blur_sigma) if blur_sigma > 0 else patch
    if score_agg == 'max':
        image_score = patch.max()
    elif score_agg == 'top5':
        k = max(1, int(np.ceil(patch.numel() * 0.05)))
        image_score = torch.topk(patch.flatten(), k).values.mean()
    else:
        raise ValueError(f'unknown image score aggregation: {score_agg}')
    return patch, float(image_score.item())


@torch.no_grad()
def patch_scores(model, mem, x, device, neighbors=False, score_agg='max', blur_sigma=2.0,
                 neighbor_pad='zero', fp16=False):
    """单图打分（历史路径，数值与之前逐位一致）。

    fp16=True 时：前向走 autocast(fp16)，最近邻在 fp16 下算、距离再转回 fp32。
    默认 False，与历史结果逐位一致（见 docs/e2e-ceiling.md 的精度闸门）。
    """
    xb = x.to(device)
    if fp16:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feat = model(xb)
        feat = feat.float()
    else:
        feat = model(xb)
    return score_from_feat(feat, mem, neighbors, score_agg, blur_sigma, neighbor_pad, fp16)


def patch_scores_batch(model, mem, xs, device, neighbors=False, score_agg='max', blur_sigma=2.0,
                       neighbor_pad='zero', fp16=False):
    """多图一次前向，再逐图算最近邻与聚合；返回 [(patch, score), ...]。

    batch>1 只改变前向的并行度（eval 模式下 BN 逐样本等价），分数与单图路径一致。
    这里用 no_grad：batch 下保留计算图会线性放大显存；单图路径保持原样不改。
    """
    xb = torch.cat([x for x in xs], dim=0).to(device)   # load_img 返回 [1,3,S,S]
    with torch.no_grad():
        if fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feat = model(xb)
            feat = feat.float()
        else:
            feat = model(xb)
        return [score_from_feat(feat[k:k + 1], mem, neighbors, score_agg, blur_sigma,
                                neighbor_pad, fp16) for k in range(feat.shape[0])]


