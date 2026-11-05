# -*- coding: utf-8 -*-
"""PatchCore 少样本无监督缺陷检测 demo（MVTec AD）

只依赖 torch / torchvision / numpy / pillow / matplotlib，不需要 anomalib。
思路与 PatchCore 论文一致：ImageNet 预训练特征 -> patch 级记忆库 -> 最近邻距离评分。

用法（先在 MVTec AD 的 bottle 类别上跑通，再上全类别）：
    python patchcore_demo.py --data E:/MVTec --category bottle
    python patchcore_demo.py --data E:/MVTec --category all
    python patchcore_demo.py --data E:/MVTec --category bottle --limit 20   # 冒烟测试

产出：results/results.csv、results/<类别>_summary.json、results/heatmaps/
"""
import argparse, csv, json, os, random, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rel_path(p, data_root, cat):
    """把图片绝对路径转成 <类别>/<train|test>/... 的相对形式。

    落盘的 scores/*.json 要进公开仓库，写绝对路径会泄漏本机目录结构，
    所以这里统一按数据根目录做相对化，换台机器复算也不受影响。
    """
    try:
        rel = os.path.relpath(os.path.abspath(p), os.path.abspath(data_root))
        rel = rel.replace(os.sep, '/')
        if rel.startswith('..'):
            raise ValueError
        return rel
    except Exception:
        return '/'.join([cat] + os.path.abspath(p).replace('\\', '/').split('/')[-3:])


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_img(path, size, preprocess='stretch', aspect_crop=0.0):
    """读图 → [1,3,S,S]。

    preprocess:
      stretch   —— 直接 resize 到 S×S（会改变长宽比；MVTec 全为方图，故与 letterbox 等价）
      letterbox —— 等比缩放到长边 = S，再用均值灰 (114) 居中补成 S×S（保住长宽比）
    aspect_crop:
      模拟非方形视野：先中心裁剪成 h/w = 该比例（如 0.75 ≈ 4:3 相机），再走上面的预处理。
      数据集本身全是方图，只有裁剪之后「拉伸畸变」才真正被引入，这就是本开关的用途。
    """
    im = Image.open(path).convert('RGB')
    if aspect_crop and aspect_crop > 0:
        w, h = im.size
        ch = min(h, max(1, int(round(w * aspect_crop))))
        top = (h - ch) // 2
        im = im.crop((0, top, w, top + ch))
    if preprocess == 'letterbox':
        w, h = im.size
        s = size / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (size, size), (114, 114, 114))
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        im = canvas
    else:
        im = im.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x


def load_mask(path, size, preprocess='stretch', aspect_crop=0.0):
    """与 load_img 完全相同的几何变换，但用 NEAREST、灰边填 0，返回 S×S 的 'L' 图。

    只有几何变换（裁剪 / letterbox）需要同步到 mask；亮度、噪声、模糊不动 mask。
    """
    im = Image.open(path).convert('L')
    if aspect_crop and aspect_crop > 0:
        w, h = im.size
        ch = min(h, max(1, int(round(w * aspect_crop))))
        top = (h - ch) // 2
        im = im.crop((0, top, w, top + ch))
    if preprocess == 'letterbox':
        w, h = im.size
        s = size / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = im.resize((nw, nh), Image.NEAREST)
        canvas = Image.new('L', (size, size), 0)
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        im = canvas
    else:
        im = im.resize((size, size), Image.NEAREST)
    return im


def perturb_img(x, brightness=1.0, noise_std=0.0, blur_sigma=0.0, seed=0):
    """可复现的非几何采集扰动；x 为 [1,C,H,W]、范围 [0,1]。"""
    x = (x * brightness).clamp(0, 1)
    if blur_sigma > 0:
        radius = max(1, int(round(3 * blur_sigma)))
        ax = torch.arange(-radius, radius + 1, dtype=x.dtype)
        kernel = torch.exp(-(ax ** 2) / (2 * blur_sigma ** 2))
        kernel = kernel / kernel.sum()
        channels = x.shape[1]
        x = F.conv2d(x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                      padding=(0, radius), groups=channels)
        x = F.conv2d(x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                      padding=(radius, 0), groups=channels)
    if noise_std > 0:
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        x = (x + noise * noise_std).clamp(0, 1)
    return x


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


@torch.no_grad()
def patch_scores(model, mem, x, device, neighbors=False, score_agg='max', blur_sigma=2.0,
                 neighbor_pad='zero'):
    feat = model(x.to(device))
    emb, (h, w) = embed(feat, neighbors, neighbor_pad)
    emb = emb.squeeze(0)                                    # N, C
    # 分块算最近邻距离，避免显存炸
    mins = []
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


def run_category(args, cat, device):
    root = os.path.join(args.data, cat)
    train_dir = os.path.join(root, 'train', 'good')
    test_dir = os.path.join(root, 'test')
    gt_dir = os.path.join(root, 'ground_truth')
    if not os.path.isdir(train_dir):
        print(f'[skip] {cat}: 找不到 {train_dir}')
        return None

    model = Features(pretrained=not args.no_pretrain, weights_path=args.weights,
                     imagenet_norm=getattr(args, 'imagenet_norm', False),
                     layers=args.layers).to(device).eval()
    train_imgs = list_images(train_dir)
    test_imgs = list_images(test_dir)
    if args.limit:  # 每个子目录各自截断，避免 --limit 只取到某一类缺陷
        def cap(lst, n):
            out = []
            for d in sorted({os.path.dirname(x) for x in lst}):
                out += [x for x in lst if os.path.dirname(x) == d][:n]
            return out
        train_imgs, test_imgs = cap(train_imgs, args.limit), cap(test_imgs, args.limit)

    # 0) 留出一部分 good 图专门标定阈值：用「建库图自己」打分会让距离趋近 0，阈值被系统性压低
    calib_imgs = []
    if args.calib_ratio > 0 and len(train_imgs) >= 5:
        idx = np.random.RandomState(args.split_seed).permutation(len(train_imgs))
        n_cal = max(2, int(len(train_imgs) * args.calib_ratio))
        calib_imgs = [train_imgs[i] for i in idx[:n_cal]]
        train_imgs = [train_imgs[i] for i in idx[n_cal:]]

    # 1) 少样本实验：在固定的 calibration split 后，按 seed 抽取建库图。
    # 不使用排序前 N 张，避免文件名顺序引入样本偏差。
    if args.train_limit > 0 and len(train_imgs) > args.train_limit:
        rng = np.random.RandomState(args.seed + 1)
        chosen = np.sort(rng.choice(len(train_imgs), size=args.train_limit, replace=False))
        train_imgs = [train_imgs[i] for i in chosen]

    # 固化每次运行的数据角色，便于审计划分是否发生变化。
    manifest_dir = os.path.join(args.out, 'manifests')
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = dict(
        category=cat, seed=args.seed, split_seed=args.split_seed, train_seed=args.seed + 1,
        calib_ratio=args.calib_ratio, train_limit=args.train_limit,
        train=[rel_path(p, args.data, cat) for p in train_imgs],
        calibration=[rel_path(p, args.data, cat) for p in calib_imgs],
        test=[rel_path(p, args.data, cat) for p in test_imgs],
    )
    with open(os.path.join(manifest_dir, f'{cat}.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 1) 建记忆库
    # 9 邻域有两种选点顺序（--neighbor-coreset）：
    #   base     : 先在 1536 维基础描述子上做 k-center，选完再对选中的 k 个位置补 9 邻域（省内存、名额不被平移副本挤占）
    #   expanded : 旧实现，先展开成 9x1536 再选点（候选池与选点成本都 ×9）
    t0 = time.time()
    use_base = bool(args.neighbors) and args.neighbor_coreset == 'base'
    bank, base_maps = [], []
    with torch.no_grad():
        for p in train_imgs:
            feat = model(load_img(p, args.size, args.preprocess, args.aspect_crop).to(device))
            if use_base:
                base_maps.append(base_desc(feat).squeeze(0).detach().cpu())
            else:
                emb, _ = embed(feat, args.neighbors, args.neighbor_pad)
                bank.append(emb.squeeze(0).detach().cpu())
    if use_base:
        if args.neighbor_pad == 'roll':
            print(f'[{cat}] 注意：base 选点不支持环形填充，按 zero 处理（--neighbor-pad roll 只在 expanded 模式下生效）')
        flat = torch.cat([m.reshape(m.shape[0], -1).t() for m in base_maps], 0)   # (N, C)
        if args.coreset_method == 'projected-greedy':
            _, sel = coreset_projected_greedy(flat, args.coreset_ratio, args.seed,
                                               args.projection_dim, args.starting_points, device)
        elif args.greedy:
            _, sel = coreset_greedy(flat, args.greedy, args.seed, return_idx=True)
        else:
            keep = max(1, int(flat.shape[0] * args.coreset_ratio))
            sel = torch.tensor(np.sort(np.random.RandomState(args.seed).choice(
                flat.shape[0], size=min(keep, flat.shape[0]), replace=False)))
        mem = expand_selected(base_maps, sel, args.neighbor_pad).to(device)
        bank_desc = f'{tuple(flat.shape)} (base) -> {len(sel)} 点 → 9 邻域展开'
    else:
        bank = torch.cat(bank, 0)
        if args.coreset_method == 'projected-greedy':
            mem, _ = coreset_projected_greedy(bank, args.coreset_ratio, args.seed,
                                              args.projection_dim, args.starting_points, device)
            mem = mem.to(device)
        elif args.greedy:
            mem = coreset_greedy(bank, args.greedy, args.seed).to(device)
        else:
            mem = torch.from_numpy(corset_random(bank.numpy(), args.coreset_ratio, args.seed)).to(device)
        bank_desc = f'{tuple(bank.shape)}'
    build_s = time.time() - t0
    print(f'[{cat}] 训练图 {len(train_imgs)} 张 -> patch {bank_desc} -> 记忆库 {tuple(mem.shape)}  ({build_s:.1f}s)')

    # 2) 阈值标定：优先用留出的 calib 图（未参与建库），否则退回建库图自评分
    score_src = calib_imgs if calib_imgs else train_imgs
    train_scores = []
    for i, p in enumerate(score_src):
        x_cal = load_img(p, args.size, args.preprocess, args.aspect_crop)
        if args.perturb_calibration:
            x_cal = perturb_img(x_cal, args.perturb_brightness, args.perturb_noise,
                                args.perturb_blur, args.seed + i)
        _, s = patch_scores(model, mem, x_cal, device,
                            args.neighbors, args.score_agg, args.blur_sigma, args.neighbor_pad)
        train_scores.append(s)
    thr = float(np.percentile(train_scores, args.thr_pct))

    # 3) 测试集评分
    y, score, px_true, px_score = [], [], [], []
    dump_px, dump_mask, dump_path = [], [], []
    lat, top = [], []
    for i, p in enumerate(test_imgs):
        label = 0 if os.path.basename(os.path.dirname(p)) == 'good' else 1
        x = perturb_img(load_img(p, args.size, args.preprocess, args.aspect_crop), args.perturb_brightness,
                        args.perturb_noise, args.perturb_blur, args.seed + 100000 + i)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()
        patch, s = patch_scores(model, mem, x, device, args.neighbors, args.score_agg,
                                args.blur_sigma, args.neighbor_pad)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
        y.append(label); score.append(s)
        top.append((s, p, patch.detach().cpu()))
        # 像素级标注：先把掩码统一取出来，缺掩码时用全零（既包含良品，也容忍个别缺文件的缺陷图），
        # 这样 px_true/px_score 与 --dump-pixel 的下标永远对得上。
        hw = patch.shape[-1]
        mask = None
        if label == 1:
            rel = os.path.relpath(p, test_dir)
            m = os.path.join(gt_dir, os.path.splitext(rel)[0] + '_mask.png')
            # VisA's official 1-class preparation stores masks as
            # ground_truth/bad/<image-stem>.png (without the MVTec suffix).
            if not os.path.exists(m):
                m = os.path.join(gt_dir, os.path.dirname(rel),
                                 os.path.splitext(os.path.basename(rel))[0] + '.png')
            if os.path.exists(m):
                if args.aspect_crop > 0 or args.preprocess != 'stretch':
                    # 几何变换必须和图像一致，否则 pixel AUROC 的绝对值不可用
                    mimg = load_mask(m, args.size, args.preprocess, args.aspect_crop)
                    mask = np.asarray(mimg.resize((hw, hw), Image.NEAREST)) > 128
                else:
                    mask = np.asarray(Image.open(m).convert('L').resize((hw, hw), Image.NEAREST)) > 128
        if mask is None:
            mask = np.zeros((hw, hw), bool)
        px_true.append(mask.reshape(-1))
        px_score.append(patch.detach().cpu().numpy().reshape(-1))
        if args.dump_pixel:
            dump_px.append(patch.detach().cpu().numpy().astype(np.float16))
            dump_mask.append(mask)
            dump_path.append(os.path.relpath(p, test_dir))

    img_auc = auc(y, score)
    px_auc = auc(np.concatenate(px_true), np.concatenate(px_score)) if px_true else float('nan')
    # 固化阈值下的招回/误报
    pred = [1 if s > thr else 0 for s in score]
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    rec = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + sum(1 for a, b in zip(pred, y) if a == 0 and b == 0))

    # 4) 存热力图（分数最高的 3 张）
    if args.dump_pixel:
        os.makedirs(args.dump_pixel, exist_ok=True)
        fp = os.path.join(args.dump_pixel, f'{cat}.npz')
        np.savez_compressed(fp, scores=np.stack(dump_px), masks=np.stack(dump_mask),
                            paths=np.array(dump_path), labels=np.array(y), thr=np.float32(thr),
                            size=np.int32(args.size))
        print(f'[dump] 像素分数图 {len(dump_px)} 张 → {fp}')

    os.makedirs(os.path.join(args.out, 'heatmaps'), exist_ok=True)
    for s, p, patch in sorted(top, key=lambda z: -z[0])[:3]:
        pm = patch.numpy()
        pm = (pm - pm.min()) / (pm.max() - pm.min() + 1e-8)
        Image.fromarray((pm * 255).astype(np.uint8)).resize((args.size, args.size)).save(
            os.path.join(args.out, 'heatmaps', f'{cat}_{os.path.basename(p)}_score{s:.2f}.png'))

    os.makedirs(os.path.join(args.out, 'scores'), exist_ok=True)
    with open(os.path.join(args.out, 'scores', f'{cat}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(category=cat, thr=thr, thr_pct=args.thr_pct, score_agg=args.score_agg,
                       preprocess=args.preprocess, aspect_crop=args.aspect_crop,
                       blur_sigma=args.blur_sigma, train_limit=args.train_limit,
                       train_seed=args.seed + 1,
                       perturb=dict(brightness=args.perturb_brightness,
                                    noise_std=args.perturb_noise,
                                    blur_sigma=args.perturb_blur,
                                    calibration_reperturbed=args.perturb_calibration),
                       calib=list(map(float, train_scores)),
                       calibration_source='held_out' if calib_imgs else 'train_self_score_fallback',
                       calibration_paths=[rel_path(p, args.data, cat) for p in calib_imgs],
                       test=[dict(path=rel_path(p, args.data, cat), label=l, score=s)
                             for p, l, s in zip(test_imgs, y, score)]),
                  f, ensure_ascii=False)

    res = dict(category=cat, n_train=len(train_imgs), n_calib=len(calib_imgs), n_test=len(test_imgs),
               # ---- 配置列：没有这些，几份 results*/ 目录事后无法区分 ----
               size=args.size, layers=args.layers, neighbors=int(bool(args.neighbors)),
               neighbor_coreset=(args.neighbor_coreset if args.neighbors else ''),
               neighbor_pad=(args.neighbor_pad if args.neighbors else ''),
               preprocess=args.preprocess, aspect_crop=args.aspect_crop,
               imagenet_norm=int(bool(getattr(args, 'imagenet_norm', False))),
               coreset=('projected_greedy' if args.coreset_method == 'projected-greedy'
                        else ('greedy' if args.greedy else 'random')),
               coreset_k=(args.greedy or None),
               coreset_ratio=args.coreset_ratio if args.coreset_method == 'projected-greedy' or not args.greedy else None,
               projection_dim=args.projection_dim if args.coreset_method == 'projected-greedy' else None,
               starting_points=args.starting_points if args.coreset_method == 'projected-greedy' else None,
               score_agg=args.score_agg, blur_sigma=args.blur_sigma,
               train_limit=args.train_limit, train_seed=args.seed + 1,
               perturb_brightness=args.perturb_brightness,
               perturb_noise=args.perturb_noise, perturb_blur=args.perturb_blur,
               perturb_calibration=int(args.perturb_calibration),
               calib_ratio=args.calib_ratio, thr_pct=args.thr_pct, seed=args.seed,
               split_seed=args.split_seed,
               weights=os.path.basename(args.weights) if args.weights else 'IMAGENET1K_V1',
               # ---- 指标列 ----
               feat_dim=int(mem.shape[1]), mem_patches=int(mem.shape[0]), img_auroc=round(img_auc, 4),
               pixel_auroc=round(px_auc, 4), threshold=round(thr, 4),
               recall_at_thr=round(rec, 4), false_positive_rate=round(fpr, 4),
               latency_ms=round(float(np.mean(lat)), 2),
               latency_p95_ms=round(float(np.percentile(lat, 95)), 2),
               build_seconds=round(build_s, 1))
    if device.type == 'cuda':
        res['peak_gpu_mb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    print('   ', json.dumps(res, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='MVTec AD 根目录，里面有 bottle/、cable/ 等')
    ap.add_argument('--category', default='bottle', help='类别名，或 all')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--layers', choices=['2+3', '1+2', '1+2+3'], default='2+3',
                    help='特征层组合：2+3（1536 维，/8 网格 40x40）；1+2（768）、1+2+3（1792）统一到 /4 网格 80x80')
    ap.add_argument('--preprocess', choices=['stretch', 'letterbox'], default='stretch',
                    help='resize 策略：stretch=直接拉伸；letterbox=等比缩放+灰边补齐（方图输入时两者等价）')
    ap.add_argument('--aspect-crop', type=float, default=0.0,
                    help='先中心裁剪成 h/w=该比例再预处理，用于模拟非方形相机（0=不裁）')
    ap.add_argument('--neighbors', action='store_true', help='论文版 9 邻域拼接（显存占用 x9）')
    ap.add_argument('--neighbor-coreset', choices=['base', 'expanded'], default='base',
                    help='9 邻域的选点顺序：base=先在基础描述子上选点再展开（默认，省内存）；'
                         'expanded=旧实现，先展开再选点（与历史 results_neigh 可比）')
    ap.add_argument('--neighbor-pad', choices=['zero', 'replicate', 'roll'], default='zero',
                    help='9 邻域边界填充：zero=与上游 Unfold(padding=1) 一致（默认）；'
                         'replicate=复制边界；roll=旧的环形平移（有环绕缺陷，仅用于复现旧结果）')
    ap.add_argument('--coreset-ratio', type=float, default=0.1,
                    help='random 模式保留比例；projected-greedy 模式为全候选池近似贪心保留比例')
    ap.add_argument('--coreset-method', choices=['legacy', 'projected-greedy'], default='legacy',
                    help='legacy 保留原随机/固定-k行为；projected-greedy 使用随机投影近似贪心+比例预算')
    ap.add_argument('--projection-dim', type=int, default=128)
    ap.add_argument('--starting-points', type=int, default=10)
    ap.add_argument('--greedy', type=int, default=0, help='k-center 贪心保留 patch 数，0=用随机抽样')
    ap.add_argument('--thr-pct', type=float, default=99.0, help='阈值取训练集分数的百分位，漏检优先就调高')
    ap.add_argument('--score-agg', choices=['max', 'top5'], default='max',
                    help='图像分数：max=最高 patch（默认），top5=最高 5%% patch 平均')
    ap.add_argument('--blur-sigma', type=float, default=2.0,
                    help='patch 分数图高斯平滑 sigma；0=关闭平滑')
    ap.add_argument('--train-limit', type=int, default=0,
                    help='建库最多使用 N 张 train/good（calibration 图另计）；0=全部')
    ap.add_argument('--perturb-brightness', type=float, default=1.0,
                    help='评测采集亮度乘数；默认 1.0')
    ap.add_argument('--perturb-noise', type=float, default=0.0,
                    help='评测图像高斯噪声标准差（像素范围 0..1）；默认 0')
    ap.add_argument('--perturb-blur', type=float, default=0.0,
                    help='评测图像高斯模糊 sigma（像素）；默认 0')
    ap.add_argument('--perturb-calibration', action='store_true',
                    help='对 calibration 良品使用同一扰动重新标定；默认冻结干净阈值')
    ap.add_argument('--limit', type=int, default=0, help='每类只用前 N 张图做冒烟测试')
    ap.add_argument('--calib-ratio', type=float, default=0.0,
                    help='从 train/good 里留出这一比例做阈值标定（0=旧行为，用建库图自评分，阈值会偏低）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split-seed', type=int, default=None,
                    help='固定 calibration/train 划分 seed；省略时沿用 --seed')
    ap.add_argument('--out', default='experiment_data/01_mvtec_gpu/runs/default')
    ap.add_argument('--no-pretrain', action='store_true', help='随机权重，仅用于离线冒烟测试（精度无意义）')
    ap.add_argument('--imagenet-norm', action='store_true',
                    help='对输入做 ImageNet 归一化（默认关；实测 screw 上反而降 3 点，见 README 消融）')
    ap.add_argument('--weights', default='', help='本地 wide_resnet50_2 权重文件路径，联网下载不了时用')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu', type=int, default=0, help='指定 GPU 序号，默认 0')
    ap.add_argument('--dump-pixel', default='',
                    help='把逐图像素分数图 + GT 掩码存成 <目录>/<类别>.npz，供 tools/pro_aupro.py 算 PRO/AUPRO')
    args = ap.parse_args()

    if args.split_seed is None:
        args.split_seed = args.seed

    if args.train_limit < 0:
        ap.error('--train-limit 必须为 0 或正整数')
    if args.blur_sigma < 0:
        ap.error('--blur-sigma 必须大于或等于 0')
    if args.perturb_brightness <= 0 or args.perturb_noise < 0 or args.perturb_blur < 0:
        ap.error('亮度必须大于 0，噪声和模糊强度必须大于或等于 0')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cpu' if args.cpu else 'cuda')
        if not args.cpu:
            torch.cuda.set_device(args.gpu)
    print('device =', device, '| torch', torch.__version__)
    os.makedirs(args.out, exist_ok=True)

    cats = ([c for c in sorted(os.listdir(args.data))
             if os.path.isdir(os.path.join(args.data, c))] if args.category == 'all' else [args.category])
    rows = []
    for c in cats:
        r = run_category(args, c, device)
        if r:
            rows.append(r)

    if rows:
        keys = list(rows[0].keys())
        with open(os.path.join(args.out, 'results.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(args.out, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        m = lambda k: float(np.nanmean([r[k] for r in rows]))
        print(f'\n{len(rows)} 类平均：image AUROC {m("img_auroc"):.4f} | pixel AUROC {m("pixel_auroc"):.4f} | '
              f'{m("latency_ms"):.1f} ms/张 | 记忆库 {int(m("mem_patches"))} patch')
        print('结果已写入', os.path.join(args.out, 'results.csv'))


if __name__ == '__main__':
    main()
