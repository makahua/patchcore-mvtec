# -*- coding: utf-8 -*-
"""骨干特征、patch 描述子与可选 PCA 投影（从 patchcore_demo.py 拆出）。"""
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from .imaging import MEAN, STD

class Features(torch.nn.Module):
    """wide_resnet50_2 的多层特征拼接，输出网格由 --layers 决定（2+3 → /8，1+2/1+2+3 → /4）。

    --layers 2+3（默认）：layer2(512,/8) + layer3(1024,/16→上采样) = 1536 通道，/8 网格（320 输入 = 40×40 = 1600 patch）
    --layers 1+2      ：layer1(256,/4) + layer2(512,/8→双线性上采样到 /4) = 768 通道，/4 网格（80×80 = 6400 patch）
    --layers 1+2+3    ：256 + 512 + 1024 = 1792 通道，/4 网格（6400 patch）
    1+2 / 1+2+3 保留 layer1 的 /4 分辨率（不做降采样），深层统一双线性上采样对齐。
    """

    def __init__(self, pretrained=True, weights_path=None, imagenet_norm=False, layers='2+3'):
        super().__init__()
        if weights_path:
            # 本地权重文件（.pth/.pickle/.bin），支持 torchvision 原始 state_dict 与 Lightning 封装
            m = wide_resnet50_2(weights=None)
            sd = torch.load(weights_path, map_location='cpu')
            for key in ('state_dict', 'model', 'net'):
                if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
            clean = {}
            for k, v in sd.items():
                for pre in ('module.', 'model.'):
                    if k.startswith(pre):
                        k = k[len(pre):]
                if k.startswith('fc.'):        # 分类头不用，直接丢
                    continue
                clean[k] = v
            missing, unexpected = m.load_state_dict(clean, strict=False)
            hit = len(m.state_dict()) - len(missing)
            print(f'[weights] 载入 {weights_path}：命中 {hit}/{len(m.state_dict())} 层，'
                  f'多余 {len(unexpected)} 键')
            if hit < len(m.state_dict()) * 0.5:
                raise SystemExit('这个权重文件和 wide_resnet50_2 对不上，换一个或去掉 --weights')
        else:
            m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3 = m.layer1, m.layer2, m.layer3
        if layers not in ('2+3', '1+2', '1+2+3'):
            raise SystemExit(f'--layers 只支持 2+3 / 1+2 / 1+2+3，收到 {layers!r}')
        self.layers = layers
        # ImageNet 归一化做成开关：默认关（保持历史结果可复现）。
        # 实测在 screw 上打开反而更差（image AUROC 0.901→0.867），推测 BN 已吸收输入尺度差异，
        # 但只验过一个类，别当结论用。要开就加 --imagenet-norm。
        self.imagenet_norm = imagenet_norm
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)

    def forward(self, x):
        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        f1 = self.l1(self.stem(x))                          # B,256,H/4,W/4
        f2 = self.l2(f1)                                    # B,512,H/8,W/8
        f3 = self.l3(f2)                                    # B,1024,H/16,W/16
        if self.layers == '2+3':
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
            return torch.cat([f2, f3], dim=1)               # B,1536,H/8,W/8
        # 1+2 / 1+2+3：保持 layer1 的 /4 网格，较深层双线性上采样到 /4
        size = f1.shape[-2:]
        f2u = F.interpolate(f2, size=size, mode='bilinear', align_corners=False)
        if self.layers == '1+2':
            return torch.cat([f1, f2u], dim=1)              # B,768,H/4,W/4
        f3u = F.interpolate(f3, size=size, mode='bilinear', align_corners=False)
        return torch.cat([f1, f2u, f3u], dim=1)             # B,1792,H/4,W/4


EMBED_STATE = {'mode': 'avgpool', 'dim': 1024}


def unfold_pool(feat, dim):
    """unfold 3x3 -> C*9 每 patch -> adaptive_avg_pool1d 到 dim（对齐上游 MeanMapper）。"""
    b, c, h, w = feat.shape
    u = F.unfold(feat, kernel_size=3, padding=1)          # (B, C*9, h*w)
    u = u.transpose(1, 2).contiguous()                    # (B, h*w, C*9)
    return F.adaptive_avg_pool1d(u, int(dim))             # (B, h*w, dim)


def base_desc(feat):
    """3x3 邻域平均后的基础 patch 描述子：(B,C,h,w)，不做 9 邻域拼接。"""
    return F.avg_pool2d(feat, 3, stride=1, padding=1)


def embed(feat, neighbors=False, pad='zero'):
    """把特征图转成 patch 描述子：(B, C, h, w) -> (B, h*w, C[*9])

    neighbors=True 时拼接 3x3 邻域描述子。pad 决定边界填充方式：
      zero   —— 与上游 Unfold(padding=1) 一致，越界位置补 0（默认）
      replicate —— 镜像/复制边界，避免 0 被当成真实特征
      roll   —— 旧实现，环形平移（会让左边缘看到右边缘，属于实现缺陷，仅用于复现旧结果）
    """
    if EMBED_STATE.get('mode') == 'unfold_pool' and not neighbors:
        p = unfold_pool(feat, EMBED_STATE.get('dim', 1024))
        return p.contiguous(), (int(feat.shape[-2]), int(feat.shape[-1]))
    p = base_desc(feat)                                    # 3x3 邻域平均，压噪
    if neighbors:
        outs = []
        if pad == 'roll':
            outs = [torch.roll(p, shifts=(i, j), dims=(2, 3))
                    for i in (-1, 0, 1) for j in (-1, 0, 1)]
        else:
            mode = 'replicate' if pad == 'replicate' else 'constant'
            q = F.pad(p, (1, 1, 1, 1), mode=mode)
            hh, ww = p.shape[-2:]
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    outs.append(q[:, :, 1 + i:1 + i + hh, 1 + j:1 + j + ww])
        p = torch.cat(outs, dim=1)
    b, c, h, w = p.shape
    return p.reshape(b, c, h * w).permute(0, 2, 1).contiguous(), (h, w)


PCA_STATE = {}


def fit_pca(feats, dim, whiten=False):
    """在良品 patch 上拟合 PCA，返回 (mean (C,), comp (C, dim))。只看训练集，不看 test。"""
    x = feats.float()
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.t() @ xc) / max(1, x.shape[0])
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    dim = max(1, min(int(dim), evecs.shape[1]))
    comp = evecs[:, :dim]
    if whiten:
        comp = comp / torch.sqrt(evals[:dim].clamp_min(1e-8)).unsqueeze(0)
    return mean, comp.contiguous()


def apply_pca(emb):
    """(N, C) -> (N, dim)。未启用时原样返回。"""
    mean, comp = PCA_STATE.get('mean'), PCA_STATE.get('comp')
    if mean is None or comp is None:
        return emb
    if mean.device != emb.device:                       # PCA 在 CPU 上拟合，打分时搬到当前设备
        mean, comp = mean.to(emb.device), comp.to(emb.device)
        PCA_STATE['mean'], PCA_STATE['comp'] = mean, comp
    return (emb - mean) @ comp


