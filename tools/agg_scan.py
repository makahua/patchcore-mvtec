# -*- coding: utf-8 -*-
"""Phase 0: offline aggregation scan from existing pixel dumps."""
import argparse
import glob
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR = os.path.join(ROOT, "experiment_data", "06_optimization", "10_pair")
PROFILE = {
    "screw": [os.path.join(ROOT, "experiment_data", "06_optimization", "16_screw12_p995",
                           "12_screw_seed_%03d" % s) for s in (0, 1, 2)],
    "capsule": [os.path.join(ROOT, "experiment_data", "06_optimization", "14_capsule12",
                             "12_capsule_seed_%03d" % s) for s in (0, 1, 2)],
}
MODES = ("max", "top1pct", "top5pct", "top10pct", "trimmed")


def cat_of(path):
    name = os.path.basename(path.rstrip("\\/"))
    body = name.split("_", 1)[1] if "_" in name else name
    return body.rsplit("_seed_", 1)[0]


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(pos.sum())
    n0 = int(neg.sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def agg(mode, patches):
    p = np.sort(np.asarray(patches, dtype=np.float64))[::-1]
    n = p.size
    if mode == "max":
        return float(p[0])
    if mode == "trimmed":
        k1 = max(1, int(np.ceil(n * 0.01)))
        k2 = max(1, int(np.ceil(n * 0.10)))
        tail = p[k1:k1 + k2]
        return float(tail.mean() if tail.size else p[-1])
    frac = {"top1pct": 0.01, "top5pct": 0.05, "top10pct": 0.10}[mode]
    k = max(1, int(np.ceil(n * frac)))
    return float(p[:k].mean())


def run_dump(run_dir):
    fs = glob.glob(os.path.join(run_dir, "pixel_dumps", "*.npz"))
    if not fs:
        return None
    z = np.load(fs[0])
    flat = z["scores"].reshape(z["scores"].shape[0], -1).astype(np.float32)
    labels = z["labels"].astype(np.int64)
    return {m: auc([agg(m, flat[i]) for i in range(flat.shape[0])], labels) for m in MODES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "aggregation-15class.json"))
    ap.add_argument("--md", default=os.path.join(ROOT, "docs", "aggregation-15class.md"))
    a = ap.parse_args()

    cats = sorted(cat_of(p) for p in os.listdir(PAIR) if p.startswith("23_"))
    res = {}
    for cat in cats:
        seeds = {}
        for s in (0, 1, 2):
            rd = (PROFILE[cat][s] if cat in PROFILE
                  else os.path.join(PAIR, "23_%s_seed_%03d" % (cat, s)))
            r = run_dump(rd)
            if r:
                seeds[s] = r
        if seeds:
            res[cat] = seeds
    missing = [c for c in cats if c not in res]

    table = {c: {m: (float(np.mean([v[m] for v in seeds.values()])),
                     float(np.std([v[m] for v in seeds.values()], ddof=1)) if len(seeds) > 1 else 0.0,
                     len(seeds)) for m in MODES} for c, seeds in res.items()}

    macro = {}
    for m in MODES:
        per_seed = []
        for s in (0, 1, 2):
            vals = [res[c][s][m] for c in res if s in res[c] and not np.isnan(res[c][s][m])]
            if vals:
                per_seed.append(float(np.mean(vals)))
        macro[m] = (float(np.mean(per_seed)),
                    float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0, len(per_seed))

    payload = {"categories": cats, "missing": missing,
               "per_class": {c: {m: {"mean": v[0], "std": v[1], "n_seeds": v[2]} for m, v in d.items()}
                             for c, d in table.items()},
               "macro": {m: {"mean": v[0], "std": v[1], "n_seeds": v[2]} for m, v in macro.items()},
               "note": "Phase 0 offline aggregation; image AUROC only; thresholds unchanged/not evaluated."}
    with io.open(a.out, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")

    lines = ["# 聚合方式对 image AUROC 的影响（15 类，离线重算）", "",
             "数据源：现有 `pixel_dumps`（当前协议：`2+3` 默认，screw/capsule 用 `1+2`）。",
             "只改图像级聚合，不重建库、不改阈值、不动模型。", "",
             "| 类别 | max | top1% | top5% | top10% | trimmed | Δ(top5%−max) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cat in sorted(table, key=lambda c: table[c]["max"][0]):
        d = table[cat]
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %+.4f |" % (
            cat, d["max"][0], d["top1pct"][0], d["top5pct"][0], d["top10pct"][0],
            d["trimmed"][0], d["top5pct"][0] - d["max"][0]))
    lines += ["", "## 15 类 macro", "", "| 聚合 | macro image AUROC |", "|---|---:|"]
    for m in MODES:
        lines.append("| %s | **%.4f** ± %.4f |" % (m, macro[m][0], macro[m][1]))
    if missing:
        lines += ["", "缺失（无 dump）：" + ", ".join(missing)]
    with io.open(a.md, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")

    print("macro:")
    for m in MODES:
        print("  %-9s %.4f ± %.4f" % (m, macro[m][0], macro[m][1]))
    print("missing:", missing)


if __name__ == "__main__":
    main()
