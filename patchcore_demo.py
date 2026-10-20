# -*- coding: utf-8 -*-
"""PatchCore 少样本无监督缺陷检测 demo（MVTec AD）

只依赖 torch / torchvision / numpy / pillow / matplotlib，不需要 anomalib。
思路与 PatchCore 论文一致：ImageNet 预训练特征 -> patch 级记忆库 -> 最近邻距离评分。

用法（先在 MVTec AD 的 bottle 类别上跑通，再上全类别）：
    python patchcore_demo.py --data D:/MVTec --category bottle
    python patchcore_demo.py --data D:/MVTec --category all
    python patchcore_demo.py --data D:/MVTec --category bottle --limit 20   # 冒烟测试

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


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_img(path, size):
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x


class Features(torch.nn.Module):
    """wide_resnet50_2 的 layer2 + layer3 特征拼接，输出 /8 分辨率 1536 通道。"""

    def __init__(self, pretrained=True, weights_path=None, imagenet_norm=False):
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
        # ImageNet 归一化做成开关：默认关（保持历史结果可复现）。
        # 实测在 screw 上打开反而更差（image AUROC 0.901→0.867），推测 BN 已吸收输入尺度差异，
        # 但只验过一个类，别当结论用。要开就加 --imagenet-norm。
        self.imagenet_norm = imagenet_norm
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)

    def forward(self, x):
        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        x = self.l1(self.stem(x))
        f2 = self.l2(x)                                     # B,512,H/8,W/8
        f3 = self.l3(f2)                                    # B,1024,H/16,W/16
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([f2, f3], dim=1)                   # B,1536,H/8,W/8


def embed(feat, neighbors=False):
    """把特征图转成 patch 描述子：(B, C, h, w) -> (B, h*w, C[*9])"""
    p = F.avg_pool2d(feat, 3, stride=1, padding=1)          # 3x3 邻域平均，压噪
    if neighbors:                                           # 论文版：再拼 9 个邻域位移
        outs = [torch.roll(p, shifts=(i, j), dims=(2, 3))
                for i in (-1, 0, 1) for j in (-1, 0, 1)]
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


def corset_random(feats, ratio, seed=0):
    n = feats.shape[0]
    keep = max(1, int(n * ratio))
    idx = np.random.RandomState(seed).choice(n, size=min(keep, n), replace=False)
    return feats[np.sort(idx)]


def coreset_greedy(feats, k, seed=0, ratio=0.1):
    """论文的 k-center 贪心选点（先随机降采样再贪心），k 为保留 patch 数。"""
    feats = corset_random(feats, ratio, seed)
    n = feats.shape[0]
    k = min(k, n)
    picked = [int(np.random.RandomState(seed).randint(n))]
    dist = torch.cdist(feats, feats[picked[0]].unsqueeze(0)).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(dist).item())
        picked.append(i)
        d = torch.cdist(feats, feats[i].unsqueeze(0)).squeeze(1)
        dist = torch.minimum(dist, d)
    return feats[torch.tensor(sorted(set(picked)))]


@torch.no_grad()
def patch_scores(model, mem, x, device, neighbors=False):
    feat = model(x.to(device))
    emb, (h, w) = embed(feat, neighbors)
    emb = emb.squeeze(0)                                    # N, C
    # 分块算最近邻距离，避免显存炸
    mins = []
    for i in range(0, emb.shape[0], 2048):
        d = torch.cdist(emb[i:i + 2048], mem)
        mins.append(d.min(dim=1).values)
    patch = torch.cat(mins).reshape(h, w)
    patch = gaussian_blur(patch)
    return patch, float(patch.max().item())


def run_category(args, cat, device):
    root = os.path.join(args.data, cat)
    train_dir = os.path.join(root, 'train', 'good')
    test_dir = os.path.join(root, 'test')
    gt_dir = os.path.join(root, 'ground_truth')
    if not os.path.isdir(train_dir):
        print(f'[skip] {cat}: 找不到 {train_dir}')
        return None

    model = Features(pretrained=not args.no_pretrain, weights_path=args.weights,
                     imagenet_norm=getattr(args, 'imagenet_norm', False)).to(device).eval()
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
        idx = np.random.RandomState(args.seed).permutation(len(train_imgs))
        n_cal = max(2, int(len(train_imgs) * args.calib_ratio))
        calib_imgs = [train_imgs[i] for i in idx[:n_cal]]
        train_imgs = [train_imgs[i] for i in idx[n_cal:]]

    # 1) 建记忆库
    t0 = time.time()
    bank = []
    with torch.no_grad():
        for p in train_imgs:
            feat = model(load_img(p, args.size).to(device))
            emb, _ = embed(feat, args.neighbors)
            bank.append(emb.squeeze(0).detach().cpu())
    bank = torch.cat(bank, 0)
    if args.greedy:
        mem = coreset_greedy(bank, args.greedy, args.seed).to(device)
    else:
        mem = torch.from_numpy(corset_random(bank.numpy(), args.coreset_ratio, args.seed)).to(device)
    build_s = time.time() - t0
    print(f'[{cat}] 训练图 {len(train_imgs)} 张 -> patch {tuple(bank.shape)} -> 记忆库 {tuple(mem.shape)}  ({build_s:.1f}s)')

    # 2) 阈值标定：优先用留出的 calib 图（未参与建库），否则退回建库图自评分
    score_src = calib_imgs if calib_imgs else train_imgs
    train_scores = []
    for p in score_src:
        _, s = patch_scores(model, mem, load_img(p, args.size), device, args.neighbors)
        train_scores.append(s)
    thr = float(np.percentile(train_scores, args.thr_pct))

    # 3) 测试集评分
    y, score, px_true, px_score = [], [], [], []
    lat, top = [], []
    for p in test_imgs:
        label = 0 if os.path.basename(os.path.dirname(p)) == 'good' else 1
        x = load_img(p, args.size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()
        patch, s = patch_scores(model, mem, x, device, args.neighbors)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
        y.append(label); score.append(s)
        top.append((s, p, patch.detach().cpu()))
        # 像素级标注
        if label == 1:
            rel = os.path.relpath(p, test_dir)
            m = os.path.join(gt_dir, os.path.splitext(rel)[0] + '_mask.png')
            if os.path.exists(m):
                hw = patch.shape[-1]
                mask = np.asarray(Image.open(m).convert('L').resize((hw, hw), Image.NEAREST)) > 128
                px_true.append(mask.reshape(-1))
                px_score.append(patch.detach().cpu().numpy().reshape(-1))
        else:
            px_true.append(np.zeros(args.size // 8 * (args.size // 8), bool))
            px_score.append(patch.detach().cpu().numpy().reshape(-1))

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
    os.makedirs(os.path.join(args.out, 'heatmaps'), exist_ok=True)
    for s, p, patch in sorted(top, key=lambda z: -z[0])[:3]:
        pm = patch.numpy()
        pm = (pm - pm.min()) / (pm.max() - pm.min() + 1e-8)
        Image.fromarray((pm * 255).astype(np.uint8)).resize((args.size, args.size)).save(
            os.path.join(args.out, 'heatmaps', f'{cat}_{os.path.basename(p)}_score{s:.2f}.png'))

    os.makedirs(os.path.join(args.out, 'scores'), exist_ok=True)
    with open(os.path.join(args.out, 'scores', f'{cat}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(category=cat, thr=thr, thr_pct=args.thr_pct,
                       calib=list(map(float, train_scores)),
                       test=[dict(path=p, label=l, score=s) for p, l, s in zip(test_imgs, y, score)]),
                  f, ensure_ascii=False)

    res = dict(category=cat, n_train=len(train_imgs), n_calib=len(calib_imgs), n_test=len(test_imgs),
               mem_patches=int(mem.shape[0]), img_auroc=round(img_auc, 4),
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
    ap.add_argument('--neighbors', action='store_true', help='论文版 9 邻域拼接（显存占用 x9）')
    ap.add_argument('--coreset-ratio', type=float, default=0.1, help='随机抽样的记忆库比例')
    ap.add_argument('--greedy', type=int, default=0, help='k-center 贪心保留 patch 数，0=用随机抽样')
    ap.add_argument('--thr-pct', type=float, default=99.0, help='阈值取训练集分数的百分位，漏检优先就调高')
    ap.add_argument('--limit', type=int, default=0, help='每类只用前 N 张图做冒烟测试')
    ap.add_argument('--calib-ratio', type=float, default=0.0,
                    help='从 train/good 里留出这一比例做阈值标定（0=旧行为，用建库图自评分，阈值会偏低）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-pretrain', action='store_true', help='随机权重，仅用于离线冒烟测试（精度无意义）')
    ap.add_argument('--imagenet-norm', action='store_true',
                    help='对输入做 ImageNet 归一化（默认关；实测 screw 上反而降 3 点，见 README 消融）')
    ap.add_argument('--weights', default='', help='本地 wide_resnet50_2 权重文件路径，联网下载不了时用')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu', type=int, default=0, help='指定 GPU 序号，默认 0')
    args = ap.parse_args()

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
