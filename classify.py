# -*- coding: utf-8 -*-
"""classify.py —— PatchCore 三档判定（OK / 复检 / NG）与产线口径评测。

为什么单独写这个脚本：
    训练/评测脚本回答的是「AUROC 多少」，产线问的是「良品过杀多少、缺陷漏检多少、
    多少要人看」。单阈值只能二分类，工业现场必须给一个「灰区」让人工复核，
    这样才可能在「漏检 < 1%」的硬指标下把误杀控制在可接受范围。

三个子命令：
    build  : 用 train/good 建记忆库，并从留出的良品标定两条阈值 → experiment_data/04_memory_banks/banks/<类>.npz
    infer  : 单张图推理，输出分数、三档结论、热力图 PNG（面试现场演示用）
    eval   : 在整个 test 集上跑三档判定，输出混淆矩阵 / 漏检 / 过杀 / 复检率

用法（以 bottle 为例，与简历同配置：320 分辨率 + 2000 patch 贪心）：
    python classify.py build --data E:/MVTec --category bottle --weights torch_cache/wide_resnet50_2_timm.pth
    python classify.py infer --bank experiment_data/04_memory_banks/banks/bottle.npz --image test/broken_large/000.png --out heat.png
    python classify.py eval  --bank experiment_data/04_memory_banks/banks/bottle.npz --data E:/MVTec --category bottle

阈值口径（关键，面试会问）：
    thr_ok 默认 = 留出良品分数的 P(1-fpr) 分位 —— 经验分位，对应「名义」良品过杀率 fpr。
    加 --conformal 改用 split-conformal：k = ceil((n+1)(1-fpr))，阈值取第 k 小的标定分数，
      可给出「良品过杀 <= fpr」的统计保证（可交换性假设下）。代价是 n 张标定图最多只能
      保证到 1/(n+1) —— 要 0.5% 至少需要 199 张标定良品，本仓库各类只有 18~117 张，
      所以 0.5% 在现有 split 下不可达，报表会逐类打印真正可达的水平。
      详见 tools/conformal_threshold.py 与 docs/conformal-thresholds.md。
    thr_ng = 留出良品分数的 P(1-fpr/20) 分位，或用 thr_ok * --ng-ratio
    两者之间 = 灰区（复检）。灰区越宽，漏检越低、人工量越大 —— 这是一条可调的产线曲线。
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from patchcore.metrics import auc  # noqa: F401  （唯一实现见 patchcore/metrics.py）


# torch / PIL / patchcore_demo 只在 build / infer / eval 里用，report 不需要 —— 延迟到真正用到时再导入。
torch = _pcdemo = Image = None


def _need_torch():
    """build / infer / eval 的入口：这里才真正去 import 重依赖。"""
    global torch, _pcdemo, Image
    if torch is not None:
        return
    try:
        import torch as _t
        from PIL import Image as _I
        import patchcore_demo as _p
    except ImportError as e:
        raise SystemExit(
            f'当前子命令需要 torch / torchvision / Pillow（{e}）。'
            ' 只复算三档表请用：python classify.py report --scores experiment_data/00_baselines/results_best/scores')
    torch, Image, _pcdemo = _t, _I, _p
    torch.set_grad_enabled(False)


# ---------------------------------------------------------------- 打分的公共部分
# 注意：这里不能用 @torch.no_grad() 装饰器 —— torch 是延迟导入的，模块加载时还是 None。
# 改成在函数体内 with torch.no_grad()，效果一样。
def image_patch_scores(model, bank, path, size, device, neighbors=False, score_agg='max',
                       preprocess='stretch', aspect_crop=0.0, blur_sigma=2.0, neighbor_pad='zero'):
    """单张图 → (patch 分数图 hxw, 图像分数)。

    预处理/聚合必须跟建库时一致，否则分数不可比 —— 这些都从记忆库元数据里读出来传进来。
    """
    with torch.no_grad():
        x = _pcdemo.load_img(path, size, preprocess, aspect_crop)
        patch, score = _pcdemo.patch_scores(model, bank, x, device, neighbors, score_agg,
                                            blur_sigma, neighbor_pad)
        return patch.cpu().numpy(), score


def collect_patches(model, paths, size, device, neighbors=False, log_every=40):
    """把一批图的 patch 描述子堆成 (N, C) 张量。

    neighbors 必须与 build 时记录的 meta['neighbors'] 一致，否则建库是 C 维、
    查询是 9C 维，cdist 会因维度不匹配直接报错（早期版本就踩过这个坑）。
    """
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            x = _pcdemo.load_img(p, size).to(device)
            emb, _ = _pcdemo.embed(model(x), neighbors)
            # Keep the full training feature bank on CPU. VisA categories have
            # many more train images than MVTec, so retaining every patch on
            # the GPU can exceed the VRAM available on a desktop card.
            out.append(emb.squeeze(0).detach().cpu())
            if log_every and (i + 1) % log_every == 0:
                print(f'  特征 {i + 1}/{len(paths)}', flush=True)
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------- 热力图（不依赖 matplotlib）
def score_to_rgb(a):
    """0~1 的分数图 → 伪彩色 RGB（蓝→青→黄→红）。"""
    stops = np.array([[0.05, 0.15, 0.75], [0.0, 0.85, 0.85], [1.0, 0.95, 0.1], [0.85, 0.05, 0.05]])
    pos = np.array([0.0, 0.33, 0.66, 1.0])
    img = np.zeros(a.shape + (3,), dtype=np.float64)
    for c in range(3):
        img[..., c] = np.interp(a, pos, stops[:, c])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_heatmap(patch, path, image_path=None, size=256, gap=6):
    """拼一张三联图：原图 ｜ 热力图 ｜ 叠加图。面试/README 都用这一张。"""
    a = patch.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo + 1e-9)
    # 分数图是 /8 分辨率，先双线性放大再上色，边缘才不锯齿
    a_img = Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    a_up = np.asarray(a_img, dtype=np.float64) / 255.0
    heat = score_to_rgb(a_up)

    if image_path and os.path.isfile(image_path):
        orig = np.asarray(Image.open(image_path).convert('RGB').resize((size, size), Image.BILINEAR))
        overlay = (0.55 * orig + 0.45 * heat).astype(np.uint8)
        tiles = [orig, heat, overlay]
    else:
        tiles = [heat]
    h, w = tiles[0].shape[:2]
    canvas = np.full((h, w * len(tiles) + gap * (len(tiles) - 1), 3), 250, dtype=np.uint8)
    for i, t in enumerate(tiles):
        canvas[:, i * (w + gap):i * (w + gap) + w] = t
    Image.fromarray(canvas).save(path)
    return path


# ---------------------------------------------------------------- 三档判定
def verdict(score, thr_ok, thr_ng):
    if score < thr_ok:
        return 'OK'
    if score < thr_ng:
        return 'REVIEW'
    return 'NG'


def conformal_quantile(scores, alpha):
    """split-conformal 阈值：第 k 小的标定分数，k = ceil((n+1)(1-alpha))。

    返回 (thr, alpha_effective, reachable)。n 张标定图最多只能保证到 1/(n+1)，
    达不到目标 alpha 时 alpha_effective 会大于 alpha（reachable=False），
    此时阈值退化为 max(calib)——这是「数据只支持到这个水平」的诚实回答，
    而不是假装 0.5% 达标。
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError('calib 为空')
    alpha_eff = max(float(alpha), 1.0 / (n + 1))
    k = int(np.ceil((n + 1) * (1.0 - alpha_eff)))
    k = min(max(k, 1), n)
    return float(s[k - 1]), alpha_eff, bool(alpha_eff <= alpha + 1e-12)


def thr_ok_from(cal_scores, fpr, conformal):
    """按口径算 thr_ok；conformal 口径下打印可达水平（面试想听的就是这句）。"""
    if conformal:
        thr, alpha_eff, reach = conformal_quantile(cal_scores, fpr)
        print(f'  conformal 阈值 {thr:.2f}（n_calib={len(cal_scores)}，可达 {alpha_eff:.2%}'
              f'{"" if reach else "，未达目标 " + format(fpr, ".2%")}）')
        return thr
    return float(np.percentile(cal_scores, 100 * (1 - fpr)))


def _build_model(a, meta):
    """按记忆库记录的层组合建骨干 —— 否则 1+2 的库会喂给 2+3 的模型，维度直接对不上。"""
    return _pcdemo.Features(weights_path=_pick_weights(a, meta),
                            imagenet_norm=bool(meta.get('imagenet_norm', False)),
                            layers=meta.get('layers', '2+3')).to(torch.device(
                                'cuda' if torch.cuda.is_available() else 'cpu')).eval()


def _pick_weights(a, meta):
    """优先命令行给的权重，其次复用建库时记录的那份（不然会去联网下载 torchvision 权重）。"""
    for w in (a.weights, meta.get('weights')):
        if w and os.path.isfile(w):
            return w
    return a.weights


# 各层组合的特征维度（与 patchcore_demo.Features 一致）
LAYER_DIMS = {'2+3': 1536, '1+2': 768, '1+2+3': 1792}


def _npz_scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = z[key]
    if getattr(v, 'shape', None) == ():
        return v.item()
    return v


def _int_like(v, default=0):
    """`pool` 存的是 '(235200, 1536)' 这样的形状字符串，取第一个数字即可。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        digits = ''.join(ch if ch.isdigit() else ' ' for ch in str(v)).split()
        return int(digits[0]) if digits else default


def _meta_from_npz(z, path='<bank>'):
    """把两种记忆库统一成同一份 meta。

    - classify.py build 写的：`bank` + `meta`（JSON 字符串），可能没有 layers 字段 → 视为 2+3。
    - patchcore_demo.py --save-bank 写的：`mem` + 一堆标量（layers/feat_dim/grid/thr/...）。
    """
    if 'meta' in z.files:
        meta = json.loads(str(z['meta']))
        meta.setdefault('layers', '2+3')
        meta.setdefault('feat_dim', int(meta.get('dim') or LAYER_DIMS.get(meta['layers'], 1536)))
        meta.setdefault('preprocess', 'stretch')
        meta.setdefault('aspect_crop', 0.0)
        meta.setdefault('blur_sigma', 2.0)
        meta.setdefault('neighbor_pad', 'zero')
        meta.setdefault('imagenet_norm', False)
        meta.setdefault('score_agg', 'max')
        meta['bank_format'] = 'classify-build'
        meta.setdefault('category', os.path.splitext(os.path.basename(path))[0])
        return meta

    if 'mem' not in z.files:
        raise SystemExit(f'无法识别的记忆库：{path}（既没有 meta，也没有 mem）')

    layers = str(_npz_scalar(z, 'layers', '2+3'))
    feat_dim = int(_npz_scalar(z, 'feat_dim', LAYER_DIMS.get(layers, 1536)))
    thr = float(_npz_scalar(z, 'thr', float('nan')))
    meta = dict(
        category=os.path.splitext(os.path.basename(path))[0],
        layers=layers, feat_dim=feat_dim, dim=int(feat_dim),
        grid=int(_npz_scalar(z, 'grid', 0) or 0),
        size=int(_npz_scalar(z, 'size', 320) or 320),
        preprocess=str(_npz_scalar(z, 'preprocess', 'stretch')),
        aspect_crop=float(_npz_scalar(z, 'aspect_crop', 0.0) or 0.0),
        neighbors=bool(_npz_scalar(z, 'neighbors', False)),
        neighbor_pad=str(_npz_scalar(z, 'neighbor_pad', 'zero')),
        score_agg=str(_npz_scalar(z, 'score_agg', 'max')),
        blur_sigma=float(_npz_scalar(z, 'blur_sigma', 2.0) or 0.0),
        imagenet_norm=False,
        thr_ok=thr,
        thr_ng=thr * 1.2,
        thr_pct=float(_npz_scalar(z, 'thr_pct', 99.5)) if 'thr_pct' in z.files else None,
        weights=str(_npz_scalar(z, 'weights', '') or ''),
        n_used=_int_like(_npz_scalar(z, 'pool', 0), 0),
        greedy=int(z['mem'].shape[0]),
        bank_format='save-bank',
    )
    return meta


def expected_dim(meta):
    """按层组合与是否开邻域推出的记忆库维度。"""
    base = int(meta.get('feat_dim') or LAYER_DIMS.get(meta.get('layers', '2+3'), 1536))
    return base * (9 if meta.get('neighbors') else 1)


def load_bank(path, device):
    _need_torch()
    z = np.load(path, allow_pickle=True)
    meta = _meta_from_npz(z, path)
    key = 'bank' if 'bank' in z.files else 'mem'
    bank = torch.from_numpy(z[key]).float().to(device)
    check_bank_dims(bank, meta, path)
    return bank, meta


def describe_bank(meta, path=''):
    """一行说明这份库是怎么建的 —— 推理前先自己把口径读出来，别靠记忆。"""
    return (f'[bank] {path} 格式 {meta.get("bank_format")} 层组合 {meta.get("layers")} '
            f'{expected_dim(meta)} 维 网格 {meta.get("grid")} 预处理 {meta.get("preprocess")} '
            f'聚合 {meta.get("score_agg")} σ={meta.get("blur_sigma")} thr_ok={meta.get("thr_ok"):.2f}')


def check_bank_dims(bank, meta, path='<bank>'):
    """建库维度和 meta 里记的 neighbors 必须自洽，否则 cdist 会在推理时才炸。

    早期版本的 collect_patches 没把 --neighbors 传下去：建库恒为 1536 维，
    但 meta 里却写了 neighbors=True，于是查询侧算出 13824 维，
    报错信息是 "cdist 维度不匹配"，跟真正的原因（build 参数没生效）隔了很远。
    """
    want = expected_dim(meta)
    got = int(bank.shape[1])
    if got != want:
        raise SystemExit(
            f'记忆库维度不自洽：{path} 里是 {got} 维，但 meta 记录 layers={meta.get("layers")}、'
            f'neighbors={bool(meta.get("neighbors"))}，按此应为 {want} 维。\n'
            f'这通常意味着建库时的 --neighbors 与实际提取特征时不一致，请重新 build。'
        )


# ---------------------------------------------------------------- build
def cmd_build(a):
    _need_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(a.data, a.category)
    train_dir = os.path.join(root, 'train', 'good')
    imgs = _pcdemo.list_images(train_dir)
    if not imgs:
        raise SystemExit(f'找不到训练图：{train_dir}')

    # 与 demo 一致的留出逻辑：一部分良品只用于标定阈值，绝不进记忆库
    idx = np.random.RandomState(a.seed).permutation(len(imgs))
    n_cal = max(2, int(len(imgs) * a.calib_ratio))
    # Preserve the permutation order used by patchcore_demo.py. Sorting these
    # indices changes which feature rows the seeded coreset sampler selects.
    calib = [imgs[i] for i in idx[:n_cal]]
    used = [imgs[i] for i in idx[n_cal:]]
    print(f'[{a.category}] 训练图 {len(imgs)} → 建库 {len(used)} / 标定 {len(calib)}')

    model = _pcdemo.Features(weights_path=a.weights, imagenet_norm=a.imagenet_norm).to(device).eval()
    print('提取建库特征…', flush=True)
    feats = collect_patches(model, used, a.size, device, a.neighbors)
    print(f'  建库 patch {feats.shape[0]:,} × {feats.shape[1]}', flush=True)
    bank = _pcdemo.coreset_greedy(feats, a.greedy, seed=a.seed)
    print(f'  贪心 coreset → {bank.shape[0]} patch（{bank.shape[0] / feats.shape[0]:.2%}）', flush=True)
    del feats
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    bank = bank.to(device)

    print('标定阈值…', flush=True)
    cal_scores = np.array([image_patch_scores(model, bank, p, a.size, device,
                                              a.neighbors, a.score_agg)[1] for p in calib])
    thr_ok = thr_ok_from(cal_scores, a.fpr, getattr(a, 'conformal', False))
    if a.ng_ratio > 0:
        thr_ng = thr_ok * a.ng_ratio
    else:
        thr_ng = float(np.percentile(cal_scores, 100 * (1 - a.fpr / 20)))
    print(f'  良品分数 中位 {np.median(cal_scores):.2f} / P{a.fpr * 100:g}分位 {thr_ok:.2f}')
    print(f'  thr_ok(OK/REVIEW) {thr_ok:.2f} | thr_ng(REVIEW/NG) {thr_ng:.2f}')

    os.makedirs(a.out, exist_ok=True)
    meta = dict(category=a.category, size=a.size, neighbors=bool(a.neighbors), greedy=int(bank.shape[0]),
                score_agg=a.score_agg, layers=model.layers,
                feat_dim=int(LAYER_DIMS.get(model.layers, 1536)),
                preprocess='stretch', aspect_crop=0.0, blur_sigma=2.0, neighbor_pad='zero',
                calib_ratio=a.calib_ratio, fpr=a.fpr, ng_ratio=a.ng_ratio, seed=a.seed,
                imagenet_norm=bool(a.imagenet_norm), dim=int(bank.shape[1]),
                thr_ok=thr_ok, thr_ng=thr_ng, n_used=len(used), n_calib=len(calib),
                threshold_method='conformal' if getattr(a, 'conformal', False) else 'percentile',
                calib_median=float(np.median(cal_scores)), weights=a.weights or 'torchvision IMAGENET1K_V1')
    out = os.path.join(a.out, f'{a.category}.npz')
    np.savez_compressed(out, bank=bank.cpu().numpy(),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f'已保存 {out}（{os.path.getsize(out) / 1e6:.1f} MB）')


# ---------------------------------------------------------------- infer
def cmd_infer(a):
    _need_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bank, meta = load_bank(a.bank, device)
    print(describe_bank(meta, os.path.basename(a.bank)))
    model = _build_model(a, meta)
    device = next(model.parameters()).device

    paths = [a.image] if os.path.isfile(a.image) else _pcdemo.list_images(a.image)
    print(f'{"图":<44}{"分数":>9}  {"判定":<7}阈值 {meta["thr_ok"]:.2f} / {meta["thr_ng"]:.2f}')
    print('-' * 78)
    rows = []
    for p in paths:
        patch, score = image_patch_scores(model, bank, p, meta['size'], device,
                                          meta['neighbors'], meta.get('score_agg', 'max'),
                                          meta.get('preprocess', 'stretch'), meta.get('aspect_crop', 0.0),
                                          meta.get('blur_sigma', 2.0), meta.get('neighbor_pad', 'zero'))
        v = verdict(score, meta['thr_ok'], meta['thr_ng'])
        print(f'{os.path.relpath(p):<44}{score:9.2f}  {v}')
        rows.append(dict(image=os.path.relpath(p), score=round(score, 4), verdict=v))
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            base = f'{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}'
            save_heatmap(patch, os.path.join(a.out, f'{base}_{v}_{score:.1f}.png'),
                         image_path=p, size=meta['size'])
            rows[-1]['heatmap'] = f'{base}_{v}_{score:.1f}.png'
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    if a.out:
        with open(os.path.join(a.out, 'scores.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f'热力图与分数已写入 {a.out}')


# ---------------------------------------------------------------- eval
def cmd_eval(a):
    _need_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bank, meta = load_bank(a.bank, device)
    if meta.get('category') and meta['category'] != a.category:
        raise SystemExit(
            f"记忆库类别不匹配：bank 属于 {meta['category']}，评测类别是 {a.category}。"
            f"请使用对应类别的 bank，或将 --category 改为 {meta['category']}。"
        )
    print(describe_bank(meta, os.path.basename(a.bank)))
    model = _build_model(a, meta)
    device = next(model.parameters()).device
    test_dir = os.path.join(a.data, a.category, 'test')
    if not os.path.isdir(test_dir):
        raise SystemExit(f'找不到测试集：{test_dir}')

    # MVTec 的目录结构即标签：test/good 是良品，其余子目录名即缺陷类型
    items = []
    for sub in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, sub)
        if not os.path.isdir(d):
            continue
        for p in _pcdemo.list_images(d):
            items.append((p, 'good' if sub == 'good' else sub))
    if a.limit:
        rng = np.random.RandomState(a.seed)
        by = {}
        for p, t in items:
            by.setdefault(t, []).append((p, t))
        items = []
        for t, lst in by.items():
            # 0<limit<1 按比例抽样；limit>=1 按张数。原实现是 int 类型，'--limit 1' 会被
            # 算成 len*1=len（等于全跑），比例分支对 >=2 永远走不到。
            k = max(1, int(round(len(lst) * a.limit))) if 0 < a.limit < 1 else int(a.limit)
            sel = rng.choice(len(lst), size=min(k, len(lst)), replace=False)
            items += [lst[i] for i in sorted(sel)]

    print(f'[{a.category}] 测试图 {len(items)} 张，阈值 {meta["thr_ok"]:.2f} / {meta["thr_ng"]:.2f}')
    rows = []
    for i, (p, t) in enumerate(items):
        # 与 cmd_infer 完全同口径：预处理/裁剪/平滑/邻域填充都要从 meta 取，
        # 否则非默认记忆库（letterbox、aspect_crop、非 2.0 sigma）的评测定量会静默偏差。
        _, score = image_patch_scores(model, bank, p, meta['size'], device,
                                      meta['neighbors'], meta.get('score_agg', 'max'),
                                      meta.get('preprocess', 'stretch'), meta.get('aspect_crop', 0.0),
                                      meta.get('blur_sigma', 2.0), meta.get('neighbor_pad', 'zero'))
        rows.append(dict(path=os.path.relpath(p, test_dir), label=t, defect=int(t != 'good'),
                         score=score, verdict=verdict(score, meta['thr_ok'], meta['thr_ng'])))
        if (i + 1) % 40 == 0:
            print(f'  {i + 1}/{len(items)}', flush=True)

    # ---- 混淆矩阵（2 类真值 × 3 档判定）
    cnt = {}
    for r in rows:
        key = (r['defect'], r['verdict'])
        cnt[key] = cnt.get(key, 0) + 1
    ng, nrv, nok = (cnt.get((0, 'NG'), 0), cnt.get((0, 'REVIEW'), 0), cnt.get((0, 'OK'), 0))
    dg, drv, dok = (cnt.get((1, 'NG'), 0), cnt.get((1, 'REVIEW'), 0), cnt.get((1, 'OK'), 0))
    n_good, n_def = ng + nrv + nok, dg + drv + dok
    pct = lambda x, n: f'{x / n:.2%}' if n else '—'

    print('\n混淆矩阵（行=真值，列=判定）')
    print(f'{"":<8}{"NG":>8}{"REVIEW":>10}{"OK":>8}   合计')
    print(f'{"良品":<8}{ng:>8}{nrv:>10}{nok:>8}   {n_good}')
    print(f'{"缺陷":<8}{dg:>8}{drv:>10}{dok:>8}   {n_def}')

    print('\n产线口径')
    print(f'  过杀率（良品直接判 NG）      {pct(ng, n_good)}')
    print(f'  复检率（良品进灰区）          {pct(nrv, n_good)}')
    print(f'  漏检率（缺陷直接判 OK）      {pct(dok, n_def)}   ← 硬指标')
    print(f'  灰区兜住的缺陷              {pct(drv, n_def)}')
    print(f'  直接判出的缺陷              {pct(dg, n_def)}')
    print(f'  人工复核量（灰区占全量）      {pct(nrv + drv, len(rows))}')

    # 每个缺陷类型一行，便于发现哪类最差
    print('\n按缺陷类型')
    print(f'{"类型":<18}{"张数":>6}{"漏检":>8}{"复检":>8}{"判出":>8}')
    types = sorted({r['label'] for r in rows if r['defect']})
    for t in types:
        sub = [r for r in rows if r['label'] == t]
        n = len(sub)
        print(f'{t:<18}{n:>6}{pct(sum(r["verdict"] == "OK" for r in sub), n):>8}'
              f'{pct(sum(r["verdict"] == "REVIEW" for r in sub), n):>8}'
              f'{pct(sum(r["verdict"] == "NG" for r in sub), n):>8}')

    # ---- 灰区宽度扫描：产线真正要调的就是这一条曲线
    if a.sweep:
        print('\n灰区扫描（thr_ng = thr_ok × 倍数；倍数越大灰区越宽）')
        print(f'{"倍数":>6}{"thr_ng":>10}{"漏检":>9}{"复核量":>10}{"过杀(直接NG)":>13}')
        for r in (1.0, 1.2, 1.5, 2.0, 3.0):
            tn = meta['thr_ok'] * r
            miss = sum(1 for x in rows if x['defect'] and x['score'] < meta['thr_ok'])
            rv = sum(1 for x in rows if meta['thr_ok'] <= x['score'] < tn)
            ng2 = sum(1 for x in rows if not x['defect'] and x['score'] >= tn)
            print(f'{r:>6.1f}{tn:>10.2f}{pct(miss, n_def):>9}{pct(rv, len(rows)):>10}{pct(ng2, n_good):>13}')

    out = dict(category=a.category, n_good=n_good, n_defect=n_def,
               false_alarm=ng / n_good if n_good else None,
               review_good=nrv / n_good if n_good else None,
               miss_rate=dok / n_def if n_def else None,
               caught_in_review=drv / n_def if n_def else None,
               caught_direct=dg / n_def if n_def else None,
               review_load=(nrv + drv) / len(rows), rows=rows)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, f'{a.category}_triage.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'\n明细已写入 {p}')



# ---------------------------------------------------------------- report（离线：直接吃 demo 落盘的逐图分数）
def _cost_split(tests):
    """按路径排序后隔一张切一刀：一半做 validation（选工作点），另一半完全不参与选择。

    切分是确定性的（不依赖随机数），换机器结果一样；类别内交替取样，避免把同类缺陷全塞进一边。
    """
    ts = sorted(tests, key=lambda t: t['path'])
    return ts[1::2], ts[0::2]


def triage_grid(calib, tests, fpr_grid, ng_grid):
    """扫 (目标过杀, 灰区倍率) 二维网格，只数四个量：漏检、过杀、良品复核、缺陷复核。"""
    out = []
    d = [t for t in tests if int(t['label'])]
    g = [t for t in tests if not int(t['label'])]
    for fpr in fpr_grid:
        thr_ok = float(np.percentile(calib, 100 * (1 - fpr))) if len(calib) else 0.0
        for ng in ng_grid:
            thr_ng = thr_ok * ng
            out.append(dict(
                fpr=fpr, ng=ng, nd=len(d), ngood=len(g),
                miss=sum(1 for t in d if float(t['score']) < thr_ok),
                fa=sum(1 for t in g if float(t['score']) >= thr_ng),
                revd=sum(1 for t in d if thr_ok <= float(t['score']) < thr_ng),
                revg=sum(1 for t in g if thr_ok <= float(t['score']) < thr_ng)))
    return out


def cell_cost(acc, cost, prev):
    """把四类计数按缺陷先验 prev 折成「每件产品当量」成本。

    刻意不用 test 集的缺陷占比：MVTec test 集故意富集缺陷（本机 73%），
    直接乘会把「漏检更贵」放大 70 倍，选点必然滑向激进侧。
    """
    c_fn, c_fp, c_rev = cost
    miss_r = acc['miss'] / acc['nd'] if acc['nd'] else 0.0
    fa_r = acc['fa'] / acc['ngood'] if acc['ngood'] else 0.0
    revd_r = acc['revd'] / acc['nd'] if acc['nd'] else 0.0
    revg_r = acc['revg'] / acc['ngood'] if acc['ngood'] else 0.0
    return (prev * (c_fn * miss_r + c_rev * revd_r)
            + (1 - prev) * (c_fp * fa_r + c_rev * revg_r))


def cmd_report(a):
    """不重算特征，直接读 results_*/scores/*.json 里的逐图分数做三档判定。

    这一步是给「全 15 类」用的：重跑一遍特征要 40 分钟，而分数早就落盘了。
    """
    files = sorted(glob.glob(os.path.join(a.scores, '*.json')))
    if not files:
        raise SystemExit(f'{a.scores} 下没有 scores/*.json，先跑 patchcore_demo.py --out 同目录')

    rows_all, lines, skipped = [], [], []
    alpha_map, calib_map, tests_map = {}, {}, {}
    print(f'{"类别":<12}{"良品":>5}{"缺陷":>5}{"thr_ok":>8}{"过杀":>8}{"良品复核":>10}{"漏检":>8}{"复核量":>8}')
    print('-' * 74)
    for p in files:
        j = json.load(open(p, encoding='utf-8'))
        calib = np.asarray(j.get('calib') or [], dtype=np.float64)
        tests = j.get('test') or []
        # 早期/外部 scores 可能缺 category 字段：整个循环统一用 cat，避免 KeyError 让整份报告崩掉
        cat = j.get('category') or os.path.splitext(os.path.basename(p))[0]
        if len(calib) < 10 or not tests:
            # 早期版本这里是静默 continue，屏幕只留一屏「-」，看不出为什么少了一类。
            # 标定图少于 10 张时 P99.5 分位数不可靠，小样本冒烟（--limit）几乎必然触发。
            print(f'{j.get("category", os.path.basename(p)):<12}  跳过：'
                  f'标定 {len(calib)} 张（<10，分位数不可信）、测试 {len(tests)} 条')
            skipped.append(dict(category=j.get('category'), reason='calib<10', n_calib=len(calib), n_test=len(tests)))
            continue
        if a.conformal:
            thr_ok, alpha_eff, _ = conformal_quantile(calib, a.fpr)
            alpha_map[cat] = alpha_eff
        else:
            thr_ok = float(np.percentile(calib, 100 * (1 - a.fpr)))
        thr_ng = thr_ok * a.ng_ratio
        calib_map[cat], tests_map[cat] = calib, tests
        g = [t for t in tests if int(t['label']) == 0]
        d = [t for t in tests if int(t['label']) != 0]
        fa = sum(1 for t in g if t['score'] >= thr_ng)
        grv = sum(1 for t in g if thr_ok <= t['score'] < thr_ng)
        miss = sum(1 for t in d if t['score'] < thr_ok)
        drv = sum(1 for t in d if thr_ok <= t['score'] < thr_ng)
        rows_all += [(cat, int(t['label']), float(t['score']), thr_ok, thr_ng) for t in tests]
        pct = lambda x, n: f'{x / n:.1%}' if n else '-'
        print(f'{cat:<12}{len(g):>5}{len(d):>5}{thr_ok:>8.1f}{pct(fa, len(g)):>8}'
              f'{pct(grv, len(g)):>10}{pct(miss, len(d)):>8}{pct(grv + drv, len(g) + len(d)):>8}')
        lines.append(dict(category=cat, n_good=len(g), n_defect=len(d), thr_ok=round(thr_ok, 3),
                          thr_ng=round(thr_ng, 3), false_alarm=fa / len(g) if g else None,
                          review_good=grv / len(g) if g else None,
                          miss_rate=miss / len(d) if d else None,
                          review_load=(grv + drv) / (len(g) + len(d))))

    if not rows_all:
        # 全部类别都被跳过（典型场景：用 --limit 冒烟跑出来的 scores）
        # 此时继续往下会打一屏 0 与「-」，容易被当成程序坏了
        print(f'\n没有可评估的类别：{len(skipped)} 个文件全被跳过（原因见上）。')
        print('冒烟/小样本请用 classify.py eval（会重新建库标定），report 只吃完整跑的 scores。')
        return

    if a.conformal:
        bad = sorted((c, ae) for c, ae in alpha_map.items() if ae > a.fpr + 1e-12)
        if bad:
            need = int(np.ceil(1 / a.fpr - 1))
            print(f'\n⚠ conformal 口径：{len(bad)}/{len(alpha_map)} 类撑不起 {a.fpr:.2%}'
                  f'（至少需要 {need} 张标定良品），实际只能保证 1/(n+1)：')
            for c, ae in bad:
                print(f'    {c:<12} 可达 {ae:.2%}')
        else:
            print(f'\nconformal 口径：{len(alpha_map)}/{len(alpha_map)} 类均达到 {a.fpr:.2%}')

    # 汇总：所有类合并成一个混淆矩阵（每类用自己的阈值）
    G = [r for r in rows_all if r[1] == 0]
    D = [r for r in rows_all if r[1] == 1]
    fa = sum(1 for r in G if r[2] >= r[4])
    grv = sum(1 for r in G if r[3] <= r[2] < r[4])
    miss = sum(1 for r in D if r[2] < r[3])
    drv = sum(1 for r in D if r[3] <= r[2] < r[4])
    pct = lambda x, n: f'{x / n:.1%}' if n else '-'
    print('-' * 74)
    print(f'{"汇总":<12}{len(G):>5}{len(D):>5}{"":>8}{pct(fa, len(G)):>8}{pct(grv, len(G)):>10}'
          f'{pct(miss, len(D)):>8}{pct(grv + drv, len(rows_all)):>8}')
    print(f'\n目标过杀 {a.fpr:.1%}｜灰区 {a.ng_ratio:g}×｜合计 {len(rows_all)} 张')
    print(f'  漏检率（缺陷判 OK，硬指标）      {pct(miss, len(D))}')
    print(f'  灰区兜住的缺陷（人工复核可捞回）  {pct(drv, len(D))}')
    print(f'  复核后等效漏检（=漏检率：已判 OK 的件复核也看不到，灰区只能兜住 REVIEW 件）'
          f' {pct(miss, len(D))}')
    print(f'  人工复核量（占全量）            {pct(grv + drv, len(rows_all))}')

    # 真正的"产线曲线"是调 thr_ok 的分位（= 允许的良品过杀），漏检与过杀此消彼长。
    # 灰区只影响"人要复查多少"，不影响漏检（漏检由 thr_ok 决定）。
    print('\n产线曲线：目标过杀 → 实际漏检（每类各自阈值，灰区固定 ' + f'{a.ng_ratio:g}×）')
    print(f'{"目标过杀":>9}{"实际过杀":>10}{"漏检":>9}{"灰区捞回":>10}{"复核量":>9}')
    for tgt in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        ths = {}
        for cat_, calib_ in calib_map.items():
            if len(calib_) >= 10:
                ths[cat_] = float(np.percentile(calib_, 100 * (1 - tgt)))
        G2 = [r for r in rows_all if r[1] == 0]
        D2 = [r for r in rows_all if r[1] == 1]
        fa2 = sum(1 for r in G2 if r[2] >= ths.get(r[0], 0) * a.ng_ratio)
        rv2 = sum(1 for r in G2 if ths.get(r[0], 0) <= r[2] < ths.get(r[0], 0) * a.ng_ratio)
        ms2 = sum(1 for r in D2 if r[2] < ths.get(r[0], 0))
        dv2 = sum(1 for r in D2 if ths.get(r[0], 0) <= r[2] < ths.get(r[0], 0) * a.ng_ratio)
        print(f'{tgt:>9.1%}{pct(fa2, len(G2)):>10}{pct(ms2, len(D2)):>9}{pct(dv2, len(D2)):>10}'
              f'{pct(rv2 + dv2, len(rows_all)):>9}')

    # ---- AUC 口径：macro（每类算完再平均）与 pooled（所有图混在一起算）差别很大 ----
    # 各类缺陷的分数分布量级不同（screw 普遍高、leather 普遍低），混池后排序被打乱，
    # pooled 会比 macro 低一截。对外只能报一个数时必须说清是哪个口径。
    per_auc = []
    for cat_, ts in tests_map.items():
        if len(ts) < 2:
            continue
        a_ = auc([int(t['label']) for t in ts], [float(t['score']) for t in ts])
        if a_ == a_:          # 排除 nan
            per_auc.append((cat_, a_))
    pooled_auc = auc([r[1] for r in rows_all], [r[2] for r in rows_all]) if rows_all else float('nan')
    macro_auc = float(np.mean([v for _, v in per_auc])) if per_auc else float('nan')
    macro_std = float(np.std([v for _, v in per_auc])) if per_auc else float('nan')
    print('\nAUC 口径（同一批分数，两种算法）')
    print(f'  macro（{len(per_auc)} 类各自算 AUC 再平均） {macro_auc:.4f} ± {macro_std:.4f}')
    print(f'  pooled（全部 {len(rows_all)} 张混池后算）     {pooled_auc:.4f}')
    print(f'  差 {abs(macro_auc - pooled_auc) * 100:.1f} 个点 —— 对外只报一个数时必须标注口径')

    if skipped:
        print(f'\n跳过 {len(skipped)} 类（未计入汇总）：' + '、'.join(s['category'] for s in skipped))

    if a.cost:
        cost = tuple(float(x) for x in a.cost.replace('，', ',').split(','))
        if len(cost) != 3:
            raise SystemExit('--cost 需要三个数：漏检,过杀,复核（例：--cost 100,10,1）')
        prev = a.prevalence
        fpr_grid = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
        ng_grid = (1.0, 1.1, 1.2, 1.5, 2.0)
        per_class = []
        for p in files:
            j = json.load(open(p, encoding='utf-8'))
            calib = np.asarray(j.get('calib') or [], dtype=np.float64)
            ts = j.get('test') or []
            if len(calib) < 10 or not ts:
                continue
            val, _ = _cost_split(ts)
            per_class.append((calib, val, ts))

        def aggregate(idx):
            grid = {}
            for rec in per_class:
                for r in triage_grid(rec[0], rec[idx], fpr_grid, ng_grid):
                    key = (r['fpr'], r['ng'])
                    acc = grid.setdefault(key, dict(nd=0, ngood=0, miss=0, fa=0, revd=0, revg=0))
                    for k in acc:
                        acc[k] += r[k]
            return grid

        val_raw = aggregate(1)      # 只用来选点
        all_raw = aggregate(2)      # 全部 test 图，只报一次
        val_c = {k: cell_cost(v, cost, prev) for k, v in val_raw.items()}
        best = min(val_c, key=lambda k: val_c[k])
        acc = all_raw[best]
        pct = lambda x, n: f'{x / n:.1%}' if n else '-'
        n_all = acc['nd'] + acc['ngood']
        print(f'\n成本模型（假设：漏检 {cost[0]:g} / 过杀 {cost[1]:g} / 复核 {cost[2]:g}，缺陷先验 {prev:.2%}）')
        print('  单位「每件产品当量」；权重与先验都是业务假设，下面给出敏感度与先验扫描。')
        print(f'  validation 选点：目标过杀 {best[0]:.2%} × 灰区 {best[1]:g}× → {val_c[best]:.4f}/件')
        print(f'  锁定后 test（{n_all} 张）：{cell_cost(acc, cost, prev):.4f}/件'
              f'｜漏检 {pct(acc["miss"], acc["nd"])}｜过杀 {pct(acc["fa"], acc["ngood"])}'
              f'｜复核 {pct(acc["revd"] + acc["revg"], n_all)}')
        rl = (prev * (acc['revd'] / acc['nd'] if acc['nd'] else 0.0)
              + (1 - prev) * (acc['revg'] / acc['ngood'] if acc['ngood'] else 0.0))
        print(f'  产线口径复核负荷（按先验 {prev:.2%} 加权，不是 test 集构成）：{rl:.1%}'
              + (f'，约每 {1 / rl:.1f} 件 1 件进复检' if rl > 0 else ''))
        print('\n  validation 成本热力图（行=目标过杀，列=灰区倍率，单位 /件）')
        print('    ' + ''.join(f'{ng:>12g}×' for ng in ng_grid))
        for fpr in fpr_grid:
            cells = ''.join(f'{val_c[(fpr, ng)]:>13.4f}' for ng in ng_grid)
            star = '  *' if fpr == best[0] else ''
            print(f'  {fpr:>7.2%}' + cells + star)
        print('\n  漏检率 / 过杀率在网格上的走势（选完点再看代价）：')
        print(f'{"目标过杀":>9}{"灰区":>7}{"漏检":>9}{"过杀":>8}{"复核":>8}')
        for fpr in fpr_grid:
            r = all_raw[(fpr, best[1])]
            print(f'{fpr:>9.2%}{best[1]:>7g}{pct(r["miss"], r["nd"]):>9}'
                  f'{pct(r["fa"], r["ngood"]):>8}{pct(r["revd"] + r["revg"], n_all):>8}')
        print('\n  先验如何改变最优工作点（每种先验各自在 validation 上重选）：')
        for pv in (0.001, 0.005, 0.01, 0.05, 0.10, 0.30):
            vc = {k: cell_cost(v, cost, pv) for k, v in val_raw.items()}
            k = min(vc, key=lambda kk: vc[kk])
            st = all_raw[k]
            print(f'    先验 {pv:>6.2%} → 过杀 {k[0]:.2%} × 灰区 {k[1]:g}×'
                  f'｜test 漏检 {pct(st["miss"], st["nd"])}｜test 过杀 {pct(st["fa"], st["ngood"])}'
                  f'｜成本 {cell_cost(st, cost, pv):.4f}/件')
        print('\n  权重敏感度（换权重后 validation 重选，先验固定用上面的假设）：')
        for name, w in (('漏检 ×10', (cost[0] * 10, cost[1], cost[2])),
                        ('过杀 ×10', (cost[0], cost[1] * 10, cost[2])),
                        ('复核 ×10', (cost[0], cost[1], cost[2] * 10))):
            vc = {k: cell_cost(v, w, prev) for k, v in val_raw.items()}
            k = min(vc, key=lambda kk: vc[kk])
            st = all_raw[k]
            print(f'    {name:<10} → 过杀 {k[0]:.2%} × 灰区 {k[1]:g}×'
                  f'｜test 漏检 {pct(st["miss"], st["nd"])}｜test 过杀 {pct(st["fa"], st["ngood"])}'
                  f'｜成本 {cell_cost(st, w, prev):.4f}/件')
        print('  解读：漏检权重变大 → 选点左移（更容忍过杀）；过杀权重变大 → 选点右移。')

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, 'triage_all.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(dict(fpr=a.fpr, ng_ratio=a.ng_ratio, per_class=lines,
                           macro_auc=round(macro_auc, 4), macro_auc_std=round(macro_std, 4),
                           pooled_auc=round(pooled_auc, 4), skipped=skipped),
                      f, ensure_ascii=False, indent=2)
        print(f'\n明细已写入 {p}')


def main():
    ap = argparse.ArgumentParser(description='PatchCore 三档判定（OK / 复检 / NG）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='建记忆库并标定两条阈值')
    b.add_argument('--data', required=True, help='MVTec AD 根目录')
    b.add_argument('--category', required=True)
    b.add_argument('--out', default='experiment_data/04_memory_banks/banks')
    b.add_argument('--size', type=int, default=320, help='输入分辨率（简历配置 320）')
    b.add_argument('--greedy', type=int, default=2000, help='k-center 保留 patch 数')
    b.add_argument('--calib-ratio', type=float, default=0.3)
    b.add_argument('--fpr', type=float, default=0.005, help='允许的良品过杀率，决定 thr_ok 分位')
    b.add_argument('--conformal', action='store_true', help='用 split-conformal 分位（统计保证），替代经验分位')
    b.add_argument('--score-agg', choices=['max', 'top5'], default='max',
                   help='校准和推理共用的图像分数聚合方式')
    b.add_argument('--ng-ratio', type=float, default=0.0, help='thr_ng=thr_ok*该倍数；0=按 fpr/20 分位')
    b.add_argument('--neighbors', action='store_true')
    b.add_argument('--imagenet-norm', action='store_true')
    b.add_argument('--weights', default=None)
    b.add_argument('--seed', type=int, default=0)
    b.set_defaults(func=cmd_build)

    i = sub.add_parser('infer', help='单张/单个目录推理')
    i.add_argument('--bank', required=True)
    i.add_argument('--image', required=True)
    i.add_argument('--out', default=None, help='热力图输出目录')
    i.add_argument('--weights', default=None)
    i.add_argument('--json', action='store_true')
    i.add_argument('--imagenet-norm', action='store_true')
    i.set_defaults(func=cmd_infer)

    e = sub.add_parser('eval', help='在 test 集上评测三档判定')
    e.add_argument('--bank', required=True)
    e.add_argument('--data', required=True)
    e.add_argument('--category', required=True)
    e.add_argument('--limit', type=float, default=0.0,
                   help='每类缺陷抽样数：0<值<1 视为比例，>=1 视为张数（0=全跑）')
    e.add_argument('--out', default='experiment_data/03_triage')
    e.add_argument('--weights', default=None)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--imagenet-norm', action='store_true')
    e.add_argument('--sweep', action='store_true', help='扫描灰区宽度，输出漏检/复核量曲线')
    e.set_defaults(func=cmd_eval)

    rp = sub.add_parser('report', help='离线三档判定：直接读 scores/*.json，不重算特征')
    rp.add_argument('--scores', default='experiment_data/00_baselines/results_best/scores', help='含 <类别>.json 的目录')
    rp.add_argument('--fpr', type=float, default=0.005, help='目标良品过杀率（决定 thr_ok 分位）')
    rp.add_argument('--conformal', action='store_true', help='用 split-conformal 分位，并逐类报告真正可达的过杀水平')
    rp.add_argument('--ng-ratio', type=float, default=1.2, help='thr_ng = thr_ok × 该倍数（灰区宽度）')
    rp.add_argument('--cost', default=None, metavar='FN,FP,REV',
                   help='成本模式：漏检/过杀/复核的权重（如 100,10,1），扫阈值分位×灰区宽度')
    rp.add_argument('--prevalence', type=float, default=0.01,
                   help='成本模式下的缺陷先验（每件产品是缺陷的概率），默认 1%%')
    rp.add_argument('--out', default='experiment_data/03_triage')
    rp.set_defaults(func=cmd_report)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
