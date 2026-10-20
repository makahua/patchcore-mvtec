# -*- coding: utf-8 -*-
"""classify.py —— PatchCore 三档判定（OK / 复检 / NG）与产线口径评测。

为什么单独写这个脚本：
    训练/评测脚本回答的是「AUROC 多少」，产线问的是「良品过杀多少、缺陷漏检多少、
    多少要人看」。单阈值只能二分类，工业现场必须给一个「灰区」让人工复核，
    这样才可能在「漏检 < 1%」的硬指标下把误杀控制在可接受范围。

三个子命令：
    build  : 用 train/good 建记忆库，并从留出的良品标定两条阈值 → banks/<类>.npz
    infer  : 单张图推理，输出分数、三档结论、热力图 PNG（面试现场演示用）
    eval   : 在整个 test 集上跑三档判定，输出混淆矩阵 / 漏检 / 过杀 / 复检率

用法（以 bottle 为例，与简历同配置：320 分辨率 + 2000 patch 贪心）：
    python classify.py build --data D:/MVTec --category bottle --weights W.pth
    python classify.py infer --bank banks/bottle.npz --image test/broken_large/000.png --out heat.png
    python classify.py eval  --bank banks/bottle.npz --data D:/MVTec --category bottle

阈值口径（关键，面试会问）：
    thr_ok = 留出良品分数的 P(1-fpr) 分位 —— 直接对应「良品过杀率 fpr」
    thr_ng = 留出良品分数的 P(1-fpr/20) 分位，或用 thr_ok * --ng-ratio
    两者之间 = 灰区（复检）。灰区越宽，漏检越低、人工量越大 —— 这是一条可调的产线曲线。
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchcore_demo as pc          # 复用特征提取 / 记忆库 / 打分，避免两套实现漂移


# ---------------------------------------------------------------- 打分的公共部分
@torch.no_grad()
def image_patch_scores(model, bank, path, size, device, neighbors=False):
    """单张图 → (patch 分数图 hxw, 图像分数)。"""
    x = pc.load_img(path, size)
    patch, score = pc.patch_scores(model, bank, x, device, neighbors)
    return patch.cpu().numpy(), score


@torch.no_grad()
def collect_patches(model, paths, size, device, log_every=40):
    """把一批图的 patch 描述子堆成 (N, C) 张量。"""
    out = []
    for i, p in enumerate(paths):
        x = pc.load_img(p, size).to(device)
        emb, _ = pc.embed(model(x))
        out.append(emb.squeeze(0))
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


def _pick_weights(a, meta):
    """优先命令行给的权重，其次复用建库时记录的那份（不然会去联网下载 torchvision 权重）。"""
    for w in (a.weights, meta.get('weights')):
        if w and os.path.isfile(w):
            return w
    return a.weights


def load_bank(path, device):
    z = np.load(path, allow_pickle=True)
    bank = torch.from_numpy(z['bank']).float().to(device)
    meta = json.loads(str(z['meta']))
    return bank, meta


# ---------------------------------------------------------------- build
def cmd_build(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(a.data, a.category)
    train_dir = os.path.join(root, 'train', 'good')
    imgs = pc.list_images(train_dir)
    if not imgs:
        raise SystemExit(f'找不到训练图：{train_dir}')

    # 与 demo 一致的留出逻辑：一部分良品只用于标定阈值，绝不进记忆库
    idx = np.random.RandomState(a.seed).permutation(len(imgs))
    n_cal = max(2, int(len(imgs) * a.calib_ratio))
    calib = [imgs[i] for i in sorted(idx[:n_cal])]
    used = [imgs[i] for i in sorted(idx[n_cal:])]
    print(f'[{a.category}] 训练图 {len(imgs)} → 建库 {len(used)} / 标定 {len(calib)}')

    model = pc.Features(weights_path=a.weights, imagenet_norm=a.imagenet_norm).to(device).eval()
    print('提取建库特征…', flush=True)
    feats = collect_patches(model, used, a.size, device)
    print(f'  建库 patch {feats.shape[0]:,} × {feats.shape[1]}', flush=True)
    bank = pc.coreset_greedy(feats, a.greedy, seed=a.seed)
    print(f'  贪心 coreset → {bank.shape[0]} patch（{bank.shape[0] / feats.shape[0]:.2%}）', flush=True)
    del feats
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    print('标定阈值…', flush=True)
    cal_scores = np.array([image_patch_scores(model, bank, p, a.size, device)[1] for p in calib])
    thr_ok = float(np.percentile(cal_scores, 100 * (1 - a.fpr)))
    if a.ng_ratio > 0:
        thr_ng = thr_ok * a.ng_ratio
    else:
        thr_ng = float(np.percentile(cal_scores, 100 * (1 - a.fpr / 20)))
    print(f'  良品分数 中位 {np.median(cal_scores):.2f} / P{a.fpr * 100:g}分位 {thr_ok:.2f}')
    print(f'  thr_ok(OK/REVIEW) {thr_ok:.2f} | thr_ng(REVIEW/NG) {thr_ng:.2f}')

    os.makedirs(a.out, exist_ok=True)
    meta = dict(category=a.category, size=a.size, neighbors=bool(a.neighbors), greedy=int(bank.shape[0]),
                calib_ratio=a.calib_ratio, fpr=a.fpr, ng_ratio=a.ng_ratio, seed=a.seed,
                thr_ok=thr_ok, thr_ng=thr_ng, n_used=len(used), n_calib=len(calib),
                calib_median=float(np.median(cal_scores)), weights=a.weights or 'torchvision IMAGENET1K_V1')
    out = os.path.join(a.out, f'{a.category}.npz')
    np.savez_compressed(out, bank=bank.cpu().numpy(),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f'已保存 {out}（{os.path.getsize(out) / 1e6:.1f} MB）')


# ---------------------------------------------------------------- infer
def cmd_infer(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bank, meta = load_bank(a.bank, device)
    model = pc.Features(weights_path=_pick_weights(a, meta),
                        imagenet_norm=a.imagenet_norm).to(device).eval()

    paths = [a.image] if os.path.isfile(a.image) else pc.list_images(a.image)
    print(f'{"图":<44}{"分数":>9}  {"判定":<7}阈值 {meta["thr_ok"]:.2f} / {meta["thr_ng"]:.2f}')
    print('-' * 78)
    rows = []
    for p in paths:
        patch, score = image_patch_scores(model, bank, p, meta['size'], device, meta['neighbors'])
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bank, meta = load_bank(a.bank, device)
    model = pc.Features(weights_path=_pick_weights(a, meta),
                        imagenet_norm=a.imagenet_norm).to(device).eval()
    test_dir = os.path.join(a.data, a.category, 'test')
    if not os.path.isdir(test_dir):
        raise SystemExit(f'找不到测试集：{test_dir}')

    # MVTec 的目录结构即标签：test/good 是良品，其余子目录名即缺陷类型
    items = []
    for sub in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, sub)
        if not os.path.isdir(d):
            continue
        for p in pc.list_images(d):
            items.append((p, 'good' if sub == 'good' else sub))
    if a.limit:
        rng = np.random.RandomState(a.seed)
        by = {}
        for p, t in items:
            by.setdefault(t, []).append((p, t))
        items = []
        for t, lst in by.items():
            k = max(1, int(len(lst) * a.limit)) if a.limit <= 1 else a.limit
            sel = rng.choice(len(lst), size=min(k, len(lst)), replace=False)
            items += [lst[i] for i in sorted(sel)]

    print(f'[{a.category}] 测试图 {len(items)} 张，阈值 {meta["thr_ok"]:.2f} / {meta["thr_ng"]:.2f}')
    rows = []
    for i, (p, t) in enumerate(items):
        _, score = image_patch_scores(model, bank, p, meta['size'], device, meta['neighbors'])
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
def cmd_report(a):
    """不重算特征，直接读 results_*/scores/*.json 里的逐图分数做三档判定。

    这一步是给「全 15 类」用的：重跑一遍特征要 40 分钟，而分数早就落盘了。
    """
    files = sorted(glob.glob(os.path.join(a.scores, '*.json')))
    if not files:
        raise SystemExit(f'{a.scores} 下没有 scores/*.json，先跑 patchcore_demo.py --out 同目录')

    rows_all, lines = [], []
    print(f'{"类别":<12}{"良品":>5}{"缺陷":>5}{"thr_ok":>8}{"过杀":>8}{"良品复核":>10}{"漏检":>8}{"复核量":>8}')
    print('-' * 74)
    for p in files:
        j = json.load(open(p, encoding='utf-8'))
        calib = np.asarray(j.get('calib') or [], dtype=np.float64)
        tests = j.get('test') or []
        if len(calib) < 10 or not tests:
            continue
        thr_ok = float(np.percentile(calib, 100 * (1 - a.fpr)))
        thr_ng = thr_ok * a.ng_ratio
        g = [t for t in tests if int(t['label']) == 0]
        d = [t for t in tests if int(t['label']) != 0]
        fa = sum(1 for t in g if t['score'] >= thr_ng)
        grv = sum(1 for t in g if thr_ok <= t['score'] < thr_ng)
        miss = sum(1 for t in d if t['score'] < thr_ok)
        drv = sum(1 for t in d if thr_ok <= t['score'] < thr_ng)
        rows_all += [(j['category'], int(t['label']), float(t['score']), thr_ok, thr_ng) for t in tests]
        pc = lambda x, n: f'{x / n:.1%}' if n else '-'
        print(f'{j["category"]:<12}{len(g):>5}{len(d):>5}{thr_ok:>8.1f}{pc(fa, len(g)):>8}'
              f'{pc(grv, len(g)):>10}{pc(miss, len(d)):>8}{pc(grv + drv, len(g) + len(d)):>8}')
        lines.append(dict(category=j['category'], n_good=len(g), n_defect=len(d), thr_ok=round(thr_ok, 3),
                          thr_ng=round(thr_ng, 3), false_alarm=fa / len(g) if g else None,
                          review_good=grv / len(g) if g else None,
                          miss_rate=miss / len(d) if d else None,
                          review_load=(grv + drv) / (len(g) + len(d))))

    # 汇总：所有类合并成一个混淆矩阵（每类用自己的阈值）
    G = [r for r in rows_all if r[1] == 0]
    D = [r for r in rows_all if r[1] == 1]
    fa = sum(1 for r in G if r[2] >= r[4])
    grv = sum(1 for r in G if r[3] <= r[2] < r[4])
    miss = sum(1 for r in D if r[2] < r[3])
    drv = sum(1 for r in D if r[3] <= r[2] < r[4])
    pc = lambda x, n: f'{x / n:.1%}' if n else '-'
    print('-' * 74)
    print(f'{"汇总":<12}{len(G):>5}{len(D):>5}{"":>8}{pc(fa, len(G)):>8}{pc(grv, len(G)):>10}'
          f'{pc(miss, len(D)):>8}{pc(grv + drv, len(rows_all)):>8}')
    print(f'\n目标过杀 {a.fpr:.1%}｜灰区 {a.ng_ratio:g}×｜合计 {len(rows_all)} 张')
    print(f'  漏检率（缺陷判 OK，硬指标）      {pc(miss, len(D))}')
    print(f'  灰区兜住的缺陷（人工复核可捞回）  {pc(drv, len(D))}')
    print(f'  复核后等效漏检                  {pc(miss, len(D))}（灰区人工兜底）')
    print(f'  人工复核量（占全量）            {pc(grv + drv, len(rows_all))}')

    # 真正的"产线曲线"是调 thr_ok 的分位（= 允许的良品过杀），漏检与过杀此消彼长。
    # 灰区只影响"人要复查多少"，不影响漏检（漏检由 thr_ok 决定）。
    print('\n产线曲线：目标过杀 → 实际漏检（每类各自阈值，灰区固定 ' + f'{a.ng_ratio:g}×）')
    print(f'{"目标过杀":>9}{"实际过杀":>10}{"漏检":>9}{"灰区捞回":>10}{"复核量":>9}')
    for tgt in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        ths = {}
        for p in files:
            j = json.load(open(p, encoding='utf-8'))
            calib = np.asarray(j.get('calib') or [], dtype=np.float64)
            if len(calib) >= 10:
                ths[j['category']] = float(np.percentile(calib, 100 * (1 - tgt)))
        G2 = [r for r in rows_all if r[1] == 0]
        D2 = [r for r in rows_all if r[1] == 1]
        fa2 = sum(1 for r in G2 if r[2] >= ths.get(r[0], 0) * a.ng_ratio)
        rv2 = sum(1 for r in G2 if ths.get(r[0], 0) <= r[2] < ths.get(r[0], 0) * a.ng_ratio)
        ms2 = sum(1 for r in D2 if r[2] < ths.get(r[0], 0))
        dv2 = sum(1 for r in D2 if ths.get(r[0], 0) <= r[2] < ths.get(r[0], 0) * a.ng_ratio)
        print(f'{tgt:>9.1%}{pc(fa2, len(G2)):>10}{pc(ms2, len(D2)):>9}{pc(dv2, len(D2)):>10}'
              f'{pc(rv2 + dv2, len(rows_all)):>9}')

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, 'triage_all.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(dict(fpr=a.fpr, ng_ratio=a.ng_ratio, per_class=lines), f, ensure_ascii=False, indent=2)
        print(f'\n明细已写入 {p}')


def main():
    ap = argparse.ArgumentParser(description='PatchCore 三档判定（OK / 复检 / NG）')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='建记忆库并标定两条阈值')
    b.add_argument('--data', required=True, help='MVTec AD 根目录')
    b.add_argument('--category', required=True)
    b.add_argument('--out', default='banks')
    b.add_argument('--size', type=int, default=320, help='输入分辨率（简历配置 320）')
    b.add_argument('--greedy', type=int, default=2000, help='k-center 保留 patch 数')
    b.add_argument('--calib-ratio', type=float, default=0.3)
    b.add_argument('--fpr', type=float, default=0.005, help='允许的良品过杀率，决定 thr_ok 分位')
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
    e.add_argument('--limit', type=int, default=0, help='每类缺陷抽样张数（0=全跑）')
    e.add_argument('--out', default='triage')
    e.add_argument('--weights', default=None)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--imagenet-norm', action='store_true')
    e.add_argument('--sweep', action='store_true', help='扫描灰区宽度，输出漏检/复核量曲线')
    e.set_defaults(func=cmd_eval)

    rp = sub.add_parser('report', help='离线三档判定：直接读 scores/*.json，不重算特征')
    rp.add_argument('--scores', default='results_best/scores', help='含 <类别>.json 的目录')
    rp.add_argument('--fpr', type=float, default=0.005, help='目标良品过杀率（决定 thr_ok 分位）')
    rp.add_argument('--ng-ratio', type=float, default=1.2, help='thr_ng = thr_ok × 该倍数（灰区宽度）')
    rp.add_argument('--out', default='triage')
    rp.set_defaults(func=cmd_report)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
