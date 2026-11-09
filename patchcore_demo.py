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


from patchcore import calibration as CAL
from patchcore import features as FE
from patchcore.calibration import SUITE_CONDITIONS, dump_suite, mask_for, peak_ws_mb
from patchcore.coreset import corset_random, coreset_greedy, coreset_projected_greedy, expand_selected
from patchcore.features import EMBED_STATE, PCA_STATE, Features, apply_pca, base_desc, embed, fit_pca
from patchcore.imaging import (IMG_EXTS, MEAN, STD, list_images, load_img, load_mask,
                               perturb_img, rel_path)
from patchcore.scoring import auc, gaussian_blur, patch_scores, patch_scores_batch, score_from_feat



# --profile 生效的覆盖项（结果列用；跨函数共享）
PROFILE_OVERRIDES = {}


def run_category(args, cat, device):
    root = os.path.join(args.data, cat)
    train_dir = os.path.join(root, 'train', 'good')
    test_dir = os.path.join(root, 'test')
    gt_dir = os.path.join(root, 'ground_truth')
    if not os.path.isdir(train_dir):
        print(f'[skip] {cat}: 找不到 {train_dir}')
        return None

    # 逐类显存峰值：不重置的话 --category all 的 peak_gpu_mb 是进程累计值（只增不减），
    # 后面的类别会被前面的类别污染。
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

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

    # 描述子构造方式：avgpool（历史默认）或 unfold_pool（对齐上游 MeanMapper）
    FE.EMBED_STATE['mode'] = getattr(args, 'embed_mode', 'avgpool')
    FE.EMBED_STATE['dim'] = int(getattr(args, 'embed_dim', 1024))
    if FE.EMBED_STATE['mode'] == 'unfold_pool':
        print(f"[{cat}] embed mode: unfold_pool -> {FE.EMBED_STATE['dim']} dims")
    # 0) 可选：在良品 patch 上拟合 PCA（只用训练集），把描述子降到低维再建库
    FE.PCA_STATE.clear()
    if getattr(args, 'pca_dim', 0) and not args.neighbors:
        sample = max(0, int(getattr(args, 'pca_sample', 400)))
        buf = []
        with torch.no_grad():
            for p in train_imgs:
                feat = model(load_img(p, args.size, args.preprocess, args.aspect_crop, args.crop_zoom).to(device))
                b = base_desc(feat).squeeze(0).reshape(feat.shape[1], -1).t().cpu()
                if sample and b.shape[0] > sample:
                    pick = np.random.RandomState(args.seed * 999 + 7).choice(
                        b.shape[0], sample, replace=False)
                    b = b[torch.from_numpy(np.sort(pick))]
                buf.append(b)
        xfit = torch.cat(buf, 0)
        mean, comp = fit_pca(xfit, args.pca_dim, getattr(args, 'pca_whiten', False))
        FE.PCA_STATE['mean'], FE.PCA_STATE['comp'] = mean, comp
        print(f'[{cat}] PCA {xfit.shape[1]} -> {comp.shape[1]} dims on {xfit.shape[0]} patches'
              f' (whiten={bool(getattr(args, "pca_whiten", False))})')
    elif getattr(args, 'pca_dim', 0):
        print(f'[{cat}] 注意：--pca-dim 与 --neighbors 不兼容，已忽略')

    # 1) 建记忆库
    # 9 邻域有两种选点顺序（--neighbor-coreset）：
    #   base     : 先在 1536 维基础描述子上做 k-center，选完再对选中的 k 个位置补 9 邻域（省内存、名额不被平移副本挤占）
    #   expanded : 旧实现，先展开成 9x1536 再选点（候选池与选点成本都 ×9）
    t0 = time.time()
    use_base = bool(args.neighbors) and args.neighbor_coreset == 'base'
    # --pool-cap：候选池上限（控制内存）。640 输入下 224 张 × 6400 patch × 1536
    # 的单精度池约 8.8 GB，cat 峰值翻倍；抽样到 15 万候选后约 0.9 GB，
    # 与 320 基线的贪心候选量同量级。抽样按 seed 固定，可复现。
    quota = 0
    if args.pool_cap > 0 and train_imgs:
        quota = max(1, int(args.pool_cap) // len(train_imgs))
    bank, base_maps = [], []
    grid_hw = None          # 特征网格 (h, w)，建库时记下，--save-bank 写 meta 用
    if quota and use_base:
        print(f'[{cat}] 注意：--pool-cap 在 base 邻域模式下不生效（需要完整空间网格做展开）')
    with torch.no_grad():
        for ii, p in enumerate(train_imgs):
            feat = model(load_img(p, args.size, args.preprocess, args.aspect_crop, args.crop_zoom).to(device))
            if grid_hw is None:
                grid_hw = (int(feat.shape[-2]), int(feat.shape[-1]))
            if use_base:
                base_maps.append(base_desc(feat).squeeze(0).detach().cpu())
            else:
                emb, _ = embed(feat, args.neighbors, args.neighbor_pad)
                e = emb.squeeze(0).detach().cpu()
                if FE.PCA_STATE:
                    e = apply_pca(e)
                if quota and e.shape[0] > quota:
                    rs = np.random.RandomState(args.seed * 10007 + ii)
                    pick = np.sort(rs.choice(e.shape[0], size=quota, replace=False))
                    e = e[torch.from_numpy(pick)]
                bank.append(e)
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
        if quota:
            full = int(len(train_imgs)) * int(args.size) * int(args.size) // 64
            print(f'[{cat}] pool cap: {bank.shape[0]} / ~{full} patches '
                  f'({100.0 * bank.shape[0] / max(1, full):.1f}%)')
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
    dump_dtype = np.float32 if getattr(args, 'dump_dtype', 'float16') == 'float32' else np.float16
    cal_px = []
    for i, p in enumerate(score_src):
        x_cal = load_img(p, args.size, args.preprocess, args.aspect_crop, args.crop_zoom)
        if args.perturb_calibration:
            x_cal = perturb_img(x_cal, args.perturb_brightness, args.perturb_noise,
                                args.perturb_blur, args.seed + i)
        patch_c, s = patch_scores(model, mem, x_cal, device,
                                  args.neighbors, args.score_agg, args.blur_sigma, args.neighbor_pad,
                                  fp16=args.fp16)
        train_scores.append(s)
        if args.dump_calib:
            cal_px.append(patch_c.detach().cpu().numpy().astype(dump_dtype))
    thr = float(np.percentile(train_scores, args.thr_pct))
    if args.dump_calib:
        os.makedirs(args.dump_calib, exist_ok=True)
        fp_cal = os.path.join(args.dump_calib, f'{cat}.npz')
        np.savez_compressed(fp_cal, scores=np.stack(cal_px),
                            paths=np.array([rel_path(p, args.data, cat) for p in score_src]),
                            labels=np.zeros(len(cal_px), np.int8), source=np.array('calibration'),
                            thr=np.float32(thr), thr_pct=np.float32(args.thr_pct),
                            size=np.int32(args.size),
                            dtype=np.array('float32' if dump_dtype is np.float32 else 'float16'))
        print(f'[calib-dump] 标定分数图 {len(cal_px)} 张 → {fp_cal}')

    # 3) 测试集评分
    y, score, px_true, px_score = [], [], [], []
    dump_px, dump_mask, dump_path = [], [], []
    lat, top = [], []
    batch = max(1, int(getattr(args, 'batch', 1)))
    for start in range(0, len(test_imgs), batch):
        chunk = test_imgs[start:start + batch]
        xs = [perturb_img(load_img(p, args.size, args.preprocess, args.aspect_crop, args.crop_zoom),
                          args.perturb_brightness, args.perturb_noise, args.perturb_blur,
                          args.seed + 100000 + start + j) for j, p in enumerate(chunk)]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()
        if len(xs) == 1:
            outs = [patch_scores(model, mem, xs[0], device, args.neighbors, args.score_agg,
                                 args.blur_sigma, args.neighbor_pad, fp16=args.fp16)]
        else:
            outs = patch_scores_batch(model, mem, xs, device, args.neighbors, args.score_agg,
                                      args.blur_sigma, args.neighbor_pad, fp16=args.fp16)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        per_img = (time.time() - t) * 1000 / len(xs)          # batch>1 时按张均摊
        for j, (p, (patch, s)) in enumerate(zip(chunk, outs)):
            i = start + j
            label = 0 if os.path.basename(os.path.dirname(p)) == 'good' else 1
            lat.append(per_img)
            y.append(label); score.append(s)
            top.append((s, p, patch.detach().cpu()))
            # 像素级标注：先把掩码统一取出来，缺掩码时用全零（既包含良品，也容忍个别缺文件的缺陷图），
            # 这样 px_true/px_score 与 --dump-pixel 的下标永远对得上。
            hw = patch.shape[-1]
            mask = mask_for(p, label, test_dir, gt_dir, hw, args)
            px_true.append(mask.reshape(-1))
            px_score.append(patch.detach().cpu().numpy().reshape(-1))
            if args.dump_pixel:
                dump_px.append(patch.detach().cpu().numpy().astype(dump_dtype))
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

    if args.dump_suite:
        dump_suite(args, model, mem, device, cat, score_src, test_imgs, test_dir, gt_dir, dump_dtype)

    # 4) 存热力图（分数最高的 3 张）
    if args.dump_pixel:
        os.makedirs(args.dump_pixel, exist_ok=True)
        fp = os.path.join(args.dump_pixel, f'{cat}.npz')
        np.savez_compressed(fp, scores=np.stack(dump_px), masks=np.stack(dump_mask),
                            paths=np.array(dump_path), labels=np.array(y), thr=np.float32(thr),
                            size=np.int32(args.size),
                            dtype=np.array('float32' if dump_dtype is np.float32 else 'float16'))
        print(f'[dump] 像素分数图 {len(dump_px)} 张 → {fp}')

    os.makedirs(os.path.join(args.out, 'heatmaps'), exist_ok=True)
    for s, p, patch in sorted(top, key=lambda z: -z[0])[:3]:
        pm = patch.numpy()
        pm = (pm - pm.min()) / (pm.max() - pm.min() + 1e-8)
        # 与项目里其它 resize 保持一致：低分辨率分数图用双线性放大（默认 BICUBIC 会让热力图边缘发虚）
        Image.fromarray((pm * 255).astype(np.uint8)).resize(
            (args.size, args.size), resample=Image.BILINEAR).save(
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

    if args.save_bank:
        os.makedirs(args.save_bank, exist_ok=True)
        bank_path = os.path.join(args.save_bank, f'{cat}.npz')
        np.savez_compressed(
            bank_path, mem=mem.detach().float().cpu().numpy(),
            layers=np.array(args.layers), feat_dim=np.int32(mem.shape[1]),
            size=np.int32(args.size), preprocess=np.array(args.preprocess),
            aspect_crop=np.float32(args.aspect_crop), neighbors=np.int8(int(bool(args.neighbors))),
            neighbor_pad=np.array(args.neighbor_pad), score_agg=np.array(args.score_agg),
            blur_sigma=np.float32(args.blur_sigma), thr=np.float32(thr),
            thr_pct=np.float32(args.thr_pct),
            weights=np.array(os.path.basename(args.weights) if args.weights else 'IMAGENET1K_V1'),
            grid=np.array(grid_hw if grid_hw is not None else (0, 0), dtype=np.int32), pool=np.array(bank_desc))
        print(f'[bank] 记忆库 {tuple(mem.shape)} → {bank_path}')

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
               score_agg=args.score_agg, blur_sigma=args.blur_sigma, fp16=int(bool(args.fp16)),
               batch=int(getattr(args, 'batch', 1)),
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
               build_seconds=round(build_s, 1),
               peak_cpu_mb=peak_ws_mb(),
               profile=(os.path.basename(args.profile) if getattr(args, 'profile', '') else ''),
               profile_overrides=','.join(sorted(PROFILE_OVERRIDES)),
               bank_file=(os.path.join(os.path.basename(args.save_bank), f'{cat}.npz')
                          if args.save_bank else ''))
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
    ap.add_argument('--preprocess', choices=['stretch', 'letterbox', 'crop'], default='stretch',
                    help='resize 策略：stretch=直接拉伸；letterbox=等比缩放+灰边补齐（方图输入时两者等价）')
    ap.add_argument('--crop-zoom', type=float, default=1.0,
                    help='\u4ec5 --preprocess crop \u751f\u6548\uff1a\u5148\u628a\u77ed\u8fb9\u7f29\u5230 size*zoom\uff0c\u518d\u4e2d\u5fc3\u88c1\u526a size\uff08\u4e0a\u6e38 366\u2192320 \u5373 1.144\uff09')
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
    ap.add_argument('--profile', default='',
                    help='按类覆盖的 JSON（configs/patchcore_profile.json）：只覆盖未显式给出的项')
    ap.add_argument('--embed-mode', choices=['avgpool', 'unfold_pool'], default='avgpool',
                    help='patch 描述子：avgpool=历史默认（3x3 均值，1536 维）；'
                         'unfold_pool=unfold 3x3 后池化到 --embed-dim（对齐上游 MeanMapper）')
    ap.add_argument('--embed-dim', type=int, default=1024,
                    help='unfold_pool 模式下的输出维度（默认 1024，与上游一致）')
    ap.add_argument('--pca-dim', type=int, default=0,
                    help='在良品 patch 上拟合 PCA 并降到该维度后再建库/打分（0=关闭）；与 --neighbors 不兼容')
    ap.add_argument('--pca-whiten', action='store_true',
                    help='PCA 白化（按特征值缩放），改变距离尺度')
    ap.add_argument('--pca-sample', type=int, default=400,
                    help='拟合 PCA 时每张训练图最多取多少 patch（默认 400）')
    ap.add_argument('--pool-cap', type=int, default=0,
                    help='候选池上限（patch 总数，0=不限）；640 等大输入建议 150000 以控制内存')
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
    ap.add_argument('--dump-calib', default=None, metavar='DIR',
                    help='把 calibration 良品的 patch 分数图存成 npz（离线比较聚合用）')
    ap.add_argument('--dump-dtype', choices=['float16', 'float32'], default='float16',
                    help='--dump-pixel/--dump-calib/--dump-suite 的存储精度；聚合对比建议 float32')
    ap.add_argument('--dump-suite', default=None, metavar='DIR',
                    help='同一记忆库对 5 组条件（干净/亮度/噪声 0.05/0.10/模糊）各存 calib+test 分数图')
    ap.add_argument('--perturb-calibration', action='store_true',
                    help='对 calibration 良品使用同一扰动重新标定；默认冻结干净阈值')
    ap.add_argument('--limit', type=int, default=0, help='每类只用前 N 张图做冒烟测试')
    ap.add_argument('--calib-ratio', type=float, default=0.0,
                    help='从 train/good 里留出这一比例做阈值标定（0=旧行为，用建库图自评分，阈值会偏低）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fp16', action='store_true',
                    help='前向 autocast(fp16) + 最近邻 fp16 计算；默认关（精度闸门见 docs/e2e-ceiling.md）')
    ap.add_argument('--batch', type=int, default=1,
                    help='测试集批推理大小（默认 1=逐张，与历史结果逐位一致；>1 只改前向并行度）')
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
    ap.add_argument('--save-bank', default='',
                    help='把记忆库与元数据存成 <dir>/<cat>.npz，供扰动/聚合实验复用')
    args = ap.parse_args()

    if getattr(args, 'pca_dim', 0) and getattr(args, 'embed_mode', 'avgpool') == 'unfold_pool':
        raise SystemExit('--pca-dim 与 --embed-mode unfold_pool 不能同时使用：'
                         'PCA 在 1536 维基础描述子上拟合，unfold_pool 输出 1024 维，投影矩阵不匹配')

    # --profile：defaults 对所有类生效，overrides 只对指定类生效（后者覆盖前者）。
    # 判定「未显式给出」= CLI 值与 argparse 默认值相同；命令行显式给了就以命令行为准。
    _prof_defaults, _prof_overrides = {}, {}
    if getattr(args, 'profile', ''):
        import json as _json
        with open(args.profile, encoding='utf-8') as _f:
            _prof = _json.load(_f)
        _prof_defaults = dict(_prof.get('defaults') or {})
        _prof_overrides = dict(_prof.get('overrides') or {})

    def _apply_profile(cat):
        """把 profile 里该类的覆盖项写进 args，返回实际生效的 {key: value}。"""
        eff = dict(_prof_defaults)
        eff.update(_prof_overrides.get(cat) or {})
        applied = {}
        for k, v in eff.items():
            if not hasattr(args, k):
                print(f'[profile] 警告：{k} 不是命令行参数，已忽略')
                continue
            try:
                is_default = getattr(args, k) == ap.get_default(k)
            except (KeyError, AttributeError):
                print(f'[profile] 警告：{k} 无法判定是否被显式指定，已忽略')
                continue
            if not is_default:
                continue                      # 命令行显式给了，profile 让位
            setattr(args, k, v)
            applied[k] = v
        globals()['PROFILE_OVERRIDES'] = applied
        for k, v in applied.items():
            print(f'[profile] {cat}.{k} = {v}')
        return applied

    def _profile_keys(cat):
        return {k for k in set(_prof_defaults) | set(_prof_overrides.get(cat) or {})
                if hasattr(args, k)}

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
        # 每一类都从同一份 CLI 参数出发再套 profile，避免上一个类的覆盖项泄漏到下一个类
        _saved = {k: getattr(args, k) for k in _profile_keys(c)}
        _apply_profile(c)
        try:
            r = run_category(args, c, device)
        finally:
            for k, v in _saved.items():
                setattr(args, k, v)
            globals()['PROFILE_OVERRIDES'] = {}
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
