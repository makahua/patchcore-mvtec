# -*- coding: utf-8 -*-
"""校验 README / docs/pair-baseline.json 与运行产物一致（按类配置感知）。

数据源由 `tools/build_pair_baseline.py` 决定：未覆盖的类别取 10_pair 的 2+3 运行，
`configs/patchcore_profile.json` 里的覆盖类别取各自 evidence 目录。
CI 只依赖标准库与仓库内已提交的产物，不装 torch、不读数据集。
"""
import argparse
import json
import os

import build_pair_baseline as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 5e-4

# 换协议时最容易漏的地方：正文还在用旧数字。这里钉住「不许当成现役结论出现」的词，
# 允许出现在带「旧 / 历史 / 早期 / 已替换」标记的行里（保留可追溯性）。
STALE_TOKENS = ("0.9805", "0.8812", "19.7 ms", "pooled 0.8812")
STALE_MARKERS = ("旧", "历史", "早期", "已替换", "previous")
DOCS_WITH_NUMBERS = ("README.md", os.path.join("docs", "interview-notes.md"))


def main():
    ap = argparse.ArgumentParser(description="README 主表数字一致性检查")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    doc = json.load(open(os.path.join(a.root, "docs", "pair-baseline.json"), encoding="utf-8"))
    readme = open(os.path.join(a.root, "README.md"), encoding="utf-8").read()
    got = B.build(a.root)

    bad = []
    for key, label in (("img", "macro image AUROC"), ("px", "macro pixel AUROC"),
                       ("au", "macro AUPRO@30"), ("au10", "macro AUPRO@10"),
                       ("lat", "算法段 ms"), ("pooled", "pooled AUROC")):
        want = doc["aggregate"][key]["mean"]
        if abs(want - got["aggregate"][key]["mean"]) > TOL:
            bad.append("%s: json %.4f vs 复算 %.4f" % (label, want, got["aggregate"][key]["mean"]))

    # triage 是整数，直接比
    for seed in ("0", "1", "2"):
        if doc["triage"][seed] != got["triage"][seed]:
            bad.append("triage seed %s: json %s vs 复算 %s"
                       % (seed, doc["triage"][seed], got["triage"][seed]))

    # README 必须写的是同一批数字（防止只改 json 不改正文）
    for text in ("%.4f" % got["aggregate"]["img"]["mean"], "%.4f" % got["aggregate"]["px"]["mean"],
                 "%.4f" % got["aggregate"]["au"]["mean"], "%.4f" % got["aggregate"]["pooled"]["mean"]):
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
        print("| %s | %.4f | %.4f |" % (label, doc["aggregate"][key]["mean"],
                                        got["aggregate"][key]["mean"]))
    print("| 漏检件数 (3 seed) | %s | %s |"
          % ([doc["triage"][s]["miss"] for s in "012"], [got["triage"][s]["miss"] for s in "012"]))

    if bad:
        print("\n[FAIL] " + "\n[FAIL] ".join(bad))
        return 1
    print("\n[ok] README / pair-baseline.json / 逐类运行产物三者一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
