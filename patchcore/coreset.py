# -*- coding: utf-8 -*-
"""coreset 采样：随机、贪心与近似贪心（从 patchcore_demo.py 拆出）。"""
import numpy as np
import torch

def corset_random(feats, ratio, seed=0, return_idx=False):
    n = feats.shape[0]
    keep = max(1, int(n * ratio))
    idx = np.sort(np.random.RandomState(seed).choice(n, size=min(keep, n), replace=False))
    return (feats[idx], torch.from_numpy(idx)) if return_idx else feats[idx]


def coreset_greedy(feats, k, seed=0, ratio=0.1, return_idx=False):
    """论文的 k-center 贪心选点（先随机降采样再贪心），k 为保留 patch 数。

    return_idx=True 时返回的 idx 是**相对传入 feats 的行号**（已把随机降采样的行号映射回去），
    这样调用方可以直接用它去原始候选池里取点（例如先选点、再展开 9 邻域）。
    """
    if return_idx:
        feats, sub_idx = corset_random(feats, ratio, seed, return_idx=True)
    else:
        feats, sub_idx = corset_random(feats, ratio, seed), None
    n = feats.shape[0]
    k = min(k, n)
    picked = [int(np.random.RandomState(seed).randint(n))]
    dist = torch.cdist(feats, feats[picked[0]].unsqueeze(0)).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(dist).item())
        picked.append(i)
        d = torch.cdist(feats, feats[i].unsqueeze(0)).squeeze(1)
        dist = torch.minimum(dist, d)
    sel = torch.tensor(sorted(set(picked)))
    if return_idx:
        return feats[sel], sub_idx[sel]
    return feats[sel]


def coreset_projected_greedy(feats, ratio, seed=0, projection_dim=128,
                             starting_points=10, device=None):
    """Approximate greedy coreset using a random low-dimensional projection.

    Selection happens in projected space; returned bank vectors remain in the
    original feature space. Unlike legacy ``coreset_greedy``, every candidate
    participates and the target size is a percentage of the full candidate set.
    """
    if not 0 < ratio <= 1:
        raise ValueError('projected greedy ratio must be in (0, 1]')
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError('features must be a non-empty [N, D] tensor')
    n, dim = feats.shape
    k = min(n, max(1, int(n * ratio)))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    projection = torch.randn((dim, projection_dim), generator=generator,
                             dtype=torch.float32) / np.sqrt(projection_dim)
    projection = projection.to(device)
    projected = torch.empty((n, projection_dim), dtype=torch.float32, device=device)
    chunk = max(1, min(16384, n))
    with torch.no_grad():
        for start in range(0, n, chunk):
            projected[start:start + chunk] = feats[start:start + chunk].to(device) @ projection
        rng = np.random.RandomState(seed)
        starts = rng.choice(n, size=min(max(1, starting_points), n), replace=False)
        nearest = torch.cdist(projected, projected[torch.as_tensor(starts, device=device)]).mean(dim=1)
        selected = torch.zeros(n, dtype=torch.bool, device=device)
        indices = []
        for _ in range(k):
            nearest.masked_fill_(selected, -1)
            idx = int(torch.argmax(nearest).item())
            selected[idx] = True
            indices.append(idx)
            distance = torch.cdist(projected, projected[idx:idx + 1]).squeeze(1)
            nearest = torch.minimum(nearest, distance)
    idx = torch.tensor(indices, dtype=torch.long)
    return feats[idx], idx


def expand_selected(base_maps, idx, pad='zero'):
    """按「先在基础描述子上选点」的顺序补回 9 邻域：只在选中的 2000 个位置上展开。

    base_maps: List[(C,h,w)]，每张训练图的基础描述子
    idx:      LongTensor(N)，全局 patch 下标（image_idx * h*w + spatial_idx）
    """
    hh, ww = base_maps[0].shape[-2:]
    per = hh * ww
    outs = []
    for g in idx.tolist():
        img, pos = divmod(g, per)
        m = base_maps[img]
        i, j = divmod(pos, ww)
        parts = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < hh and 0 <= jj < ww:
                    parts.append(m[:, ii, jj])
                elif pad == 'replicate':
                    parts.append(m[:, min(max(ii, 0), hh - 1), min(max(jj, 0), ww - 1)])
                else:  # zero：与 embed(pad='zero') 一致
                    parts.append(torch.zeros(m.shape[0], dtype=m.dtype))
        outs.append(torch.cat(parts))
    return torch.stack(outs)


