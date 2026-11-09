# -*- coding: utf-8 -*-
"""Conformal 阈值统计：把「名义过杀率」换成「有统计保证的过杀率」。

背景
----
`patchcore_demo.py --thr-pct 99` 取 calib 分数的 P99，`classify.py --fpr 0.005` 取 P99.5。
两者都是**经验分位**：样本少时分位点极不稳定（18 张标定图估 P99.5，等于要估「第 0.09 张」）。
split-conformal 给出可辩护的替代：

    给定目标水平 alpha 与 n 张标定良品，
    k = ceil((n + 1) * (1 - alpha))
    若 k > n，则**该 alpha 在当前样本量下不可达**，最高只能到 alpha_min = 1 / (n + 1)；
    否则阈值 = 第 k 小的 calib 分数，可保证 P(良品分数 > 阈值) <= alpha（可交换性假设下）。

用法
----
    python tools/conformal_threshold.py --scores experiment_data/00_baselines/results_best/scores \
        --alpha 0.005 --ng-ratio 1.2

    # 同时输出 markdown 表格，可直接贴进 docs
    python tools/conformal_threshold.py --scores ... --markdown

只依赖标准库 + numpy，不需要 GPU / 数据集 / 权重。
"""
import argparse
import glob
import json
import math
import os


# ---------------------------------------------------------------- 工具
def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)，n 不大（<= 几千）时够用。"""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def cp_upper(k, n, conf=0.95):
    """Clopper-Pearson 单侧上界：观测到 k/n 次误报时，真实误报率的上界。"""
    if n == 0:
        return float('nan')
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1.0 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def conformal_quantile(scores, alpha):
    """返回 (threshold, alpha_effective, k) 或 None（不可达）。"""
    s = sorted(float(x) for x in scores)
    n = len(s)
    if n == 0:
        return None
    alpha_eff = max(alpha, 1.0 / (n + 1))
    k = int(math.ceil((n + 1) * (1.0 - alpha_eff)))
    k = min(max(k, 1), n)
    return s[k - 1], alpha_eff, k


def empirical(scores, labels, thr, strict=True):
    """在 (score, label) 上统计：良品误报、缺陷召回。label: 1=缺陷。"""
    fp = sum(1 for s, y in zip(scores, labels) if y == 0 and (s > thr if strict else s >= thr))
    n_good = sum(1 for y in labels if y == 0)
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and (s > thr if strict else s >= thr))
    n_def = sum(1 for y in labels if y == 1)
    fpr = fp / n_good if n_good else float('nan')
    rec = tp / n_def if n_def else float('nan')
    return dict(fp=fp, n_good=n_good, fpr=fpr, fpr_upper=cp_upper(fp, n_good), tp=tp, n_def=n_def, recall=rec)


def n_for_alpha(alpha):
    """达到 alpha 级别所需的最少标定良品数：k = ceil((n+1)(1-alpha)) <= n  <=>  n >= 1/alpha - 1。"""
    return int(math.ceil(1.0 / alpha - 1.0))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', default='experiment_data/00_baselines/results_best/scores')
    ap.add_argument('--alpha', type=float, default=0.005, help='目标名义过杀率（默认 0.5%%）')
    ap.add_argument('--conf', type=float, default=0.95, help='Clopper-Pearson 置信水平')
    ap.add_argument('--ng-ratio', type=float, default=1.2, help='灰区宽度 thr_ng = thr_ok × 倍数')
    ap.add_argument('--markdown', action='store_true', help='输出 markdown 表格')
    ap.add_argument('--out', default='', help='把结果 JSON 写到该路径（可选）')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.scores, '*.json')))
    if not files:
        raise SystemExit(f'{a.scores} 下没有 scores/*.json')

    print(f'目标 alpha = {a.alpha:.2%} | 所需最少 calib = {n_for_alpha(a.alpha)} 张 | conf = {a.conf:.0%}')
    print('备注：alpha 每降一档，所需 calib 约翻倍（0.5%→199，0.2%→499）。\n')

    rows = []
    skipped = []
    for p in files:
        d = json.load(open(p, encoding='utf-8'))
        cat = d.get('category') or os.path.basename(p)[:-5]
        calib = [float(x) for x in d['calib']]
        test = d['test']
        ts = [float(t['score']) for t in test]
        ty = [int(t['label']) for t in test]

        cq = conformal_quantile(calib, a.alpha)
        if cq is None:
            # 空 calib：conformal_quantile 返回 None，原来直接解包会 TypeError；
            # 这里显式跳过并记录，输出里能看出是哪一类缺标定。
            print(f'[skip] {cat}: calib 为空，无法给出 conformal 阈值')
            skipped.append(dict(category=cat, reason='empty_calib'))
            continue
        thr, alpha_eff, k = cq
        reachable = alpha_eff <= a.alpha + 1e-12
        ev = empirical(ts, ty, thr)
        # 对照：旧口径（经验分位 P(1-alpha)）
        naive = float(sorted(calib)[min(len(calib) - 1, int(round((len(calib) - 1) * (1 - a.alpha))))])
        ev_naive = empirical(ts, ty, naive)
        rows.append(dict(
            category=cat, n_cal=len(calib), n_good=ev['n_good'], n_def=ev['n_def'],
            alpha_target=a.alpha, alpha_effective=alpha_eff, reachable=bool(reachable), k=k,
            thr_conformal=round(thr, 4), thr_naive=round(naive, 4),
            fpr=round(ev['fpr'], 4), fpr_upper=round(ev['fpr_upper'], 4),
            recall=round(ev['recall'], 4),
            fp=ev['fp'], tp=ev['tp'],
            fpr_naive=round(ev_naive['fpr'], 4), recall_naive=round(ev_naive['recall'], 4),
        ))

    rows.sort(key=lambda r: r['category'])
    if a.markdown:
        print('| 类别 | n_calib | 可达 alpha | 阈值(conformal) | 阈值(P99) | 实测过杀 | 上界(CP95%) | 召回 | 漏检数 |')
        print('|---|---|---|---|---|---|---|---|---|')
        for r in rows:
            print(f"| {r['category']} | {r['n_cal']} | {r['alpha_effective']:.2%}{'' if r['reachable'] else ' (不可达)'} "
                  f"| {r['thr_conformal']:.2f} | {r['thr_naive']:.2f} | {r['fpr']:.2%} | {r['fpr_upper']:.2%} "
                  f"| {r['recall']:.1%} | {r['n_def'] - r['tp']}/{r['n_def']} |")
    else:
        hdr = f'{"类别":<12}{"n_cal":>6}{"可达alpha":>10}{"阈值":>9}{"实测过杀":>9}{"上界":>8}{"召回":>7}'
        print(hdr)
        print('-' * len(hdr))
        for r in rows:
            flag = '' if r['reachable'] else '*'
            print(f"{r['category']:<12}{r['n_cal']:>6}{r['alpha_effective']:>9.2%}{flag:<1}"
                  f"{r['thr_conformal']:>9.2f}{r['fpr']:>9.2%}{r['fpr_upper']:>8.2%}{r['recall']:>7.1%}")

    if not rows:
        print('\n没有任何可评估的类别（全部 calib 为空）。')
        if a.out:
            _write_out(a.out, a, rows, skipped)
        return

    reach = [r for r in rows if r['reachable']]
    print(f'\n可达 {a.alpha:.2%} 的类别：{len(reach)}/{len(rows)}'
          + (f'（{", ".join(r["category"] for r in reach)}）' if reach else '（一个都没有）'))
    # nan（该类没有良品/缺陷测试图）不能直接参与平均，否则整行输出变 nan
    def _mean(key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float('nan')

    mean_recall, mean_fpr = _mean('recall'), _mean('fpr')
    mean_recall_naive, mean_fpr_naive = _mean('recall_naive'), _mean('fpr_naive')
    print(f'macro 召回 {mean_recall:.1%} / 实测过杀 {mean_fpr:.2%}（conformal）')
    print(f'macro 召回 {mean_recall_naive:.1%} / 实测过杀 {mean_fpr_naive:.2%}（旧经验分位）')

    if a.out:
        _write_out(a.out, a, rows, skipped)


def _write_out(path, a, rows, skipped):
    """写结果 JSON。NaN 不是合法 JSON：缺良品/缺陷的类别写 null，别让下游 json.load 崩。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows_out = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
                for r in rows]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(alpha=a.alpha, conf=a.conf, n_required=n_for_alpha(a.alpha),
                       skipped=skipped, rows=rows_out),
                  f, ensure_ascii=False, indent=2, allow_nan=False)
    print('写入', path)


if __name__ == '__main__':
    main()
