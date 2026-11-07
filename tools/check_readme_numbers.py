# -*- coding: utf-8 -*-
"""校验 README / docs/pair-baseline.json 与 10_pair 的 90 run 结果一致。

CI 用途：只依赖标准库，不装 torch、不读数据集——跑的是仓库里已提交的
metrics.json / per_image.csv / aupro.json。任何一侧被改动而另一侧没跟上，
这一步就会失败，避免 README 出现「跑不出来第二次」的数字。
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR = os.path.join("experiment_data", "06_optimization", "10_pair")
SLUG = "23"            # 默认配置：2+3
TOL = 5e-4             # 4 位小数 + 均值取整

# 换协议时最容易漏的地方：正文还在用旧数字。这里钉住「不许当成现役结论出现」的词，
# 允许出现在带「旧 / 历史 / 早期 / 已替换」标记的行里（保留可追溯性）。
STALE_TOKENS = ("0.9805", "0.8812", "19.7 ms", "pooled 0.8812")
STALE_MARKERS = ("旧", "历史", "早期", "已替换", "previous")
DOCS_WITH_NUMBERS = ("README.md", os.path.join("docs", "interview-notes.md"))


def pooled_auroc(dirs):
    byv = {}
    for d in dirs:
        with open(os.path.join(d, "per_image.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != "test":
                    continue
                bucket = byv.setdefault(float(row["score"]), [0, 0])
                bucket[int(row["label"])] += 1
    pos = sum(v[1] for v in byv.values())
    neg = sum(v[0] for v in byv.values())
    if not pos or not neg:
        return float("nan")
    cum_good = 0
    acc = 0.0
    for k in sorted(byv):
        good, bad = byv[k]
        acc += bad * cum_good + 0.5 * bad * good
        cum_good += good
    return acc / (pos * neg)


def recompute(root):
    per_seed = {}
    for seed in (0, 1, 2):
        dirs = sorted(glob.glob(os.path.join(root, PAIR, "%s_*_seed_%03d" % (SLUG, seed))))
        if not dirs:
            raise SystemExit("[error] 找不到 %s_*_seed_%03d 的运行目录" % (SLUG, seed))
        rows = []
        for d in dirs:
            m = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))["categories"][0]
            a = json.load(open(os.path.join(d, "aupro.json"), encoding="utf-8"))["per_class"][0]
            rows.append(dict(m, aupru=a["aupru"], aupru_01=a["aupru_01"]))
        per_seed[seed] = dict(
            img=st.mean([r["img_auroc"] for r in rows]),
            px=st.mean([r["pixel_auroc"] for r in rows]),
            au=st.mean([r["aupru"] for r in rows]),
            au10=st.mean([r["aupru_01"] for r in rows]),
            lat=st.mean([r["latency_ms"] for r in rows]),
            pooled=pooled_auroc(dirs),
        )
    agg = {}
    for key in ("img", "px", "au", "au10", "lat", "pooled"):
        vals = [per_seed[s][key] for s in per_seed]
        agg[key] = st.mean(vals)
    return agg


def main():
    ap = argparse.ArgumentParser(description="README 主表数字一致性检查")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    doc = json.load(open(os.path.join(a.root, "docs", "pair-baseline.json"), encoding="utf-8"))
    readme = open(os.path.join(a.root, "README.md"), encoding="utf-8").read()
    got = recompute(a.root)

    bad = []
    for key, label in (("img", "macro image AUROC"), ("px", "macro pixel AUROC"),
                       ("au", "macro AUPRO@30"), ("au10", "macro AUPRO@10"),
                       ("lat", "算法段 ms"), ("pooled", "pooled AUROC")):
        want = doc["aggregate"][key]["mean"]
        if abs(want - got[key]) > TOL:
            bad.append("%s: json %.4f vs 复算 %.4f" % (label, want, got[key]))

    # README 必须写的是同一批数字（防止只改 json 不改正文）
    for text in ("%.4f" % got["img"], "%.4f" % got["px"], "%.4f" % got["au"], "%.4f" % got["pooled"]):
        if text not in readme:
            bad.append("README 缺少数字 %s" % text)

    # 过期数字守卫：文档里不许把旧协议的数当现役结论
    for rel in DOCS_WITH_NUMBERS:
        path = os.path.join(a.root, rel)
        if not os.path.isfile(path):
            continue
        for lineno, line in enumerate(open(path, encoding="utf-8").read().split("\n"), 1):
            if any(tok in line for tok in STALE_TOKENS) and not any(m in line for m in STALE_MARKERS):
                bad.append("%s:%d 出现过期数字 %r（若为历史对照，请在同句标注「旧」）"
                           % (rel, lineno, line.strip()[:60]))

    print("| 指标 | pair-baseline.json | 复算 |")
    print("|---|---:|---:|")
    for key, label in (("img", "image AUROC"), ("px", "pixel AUROC"), ("au", "AUPRO@30"),
                       ("au10", "AUPRO@10"), ("lat", "算法段 ms"), ("pooled", "pooled")):
        print("| %s | %.4f | %.4f |" % (label, doc["aggregate"][key]["mean"], got[key]))

    if bad:
        print("\n[FAIL] " + "\n[FAIL] ".join(bad))
        return 1
    print("\n[ok] README / pair-baseline.json / 90 个 run 三者一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
