# -*- coding: utf-8 -*-
"""导出 / 校验逐类三档阈值（docs/thresholds.json）——按类配置感知。

数据源与 `tools/build_pair_baseline.py` 一致：未覆盖的类别取 10_pair 的 2+3 运行，
`configs/patchcore_profile.json` 里覆盖的类别（screw / capsule）取各自的 evidence 目录。
本文件解决两个常见误用：
1. 拿某一类的 `thr_ok` 去套别的类 —— 各类分数尺度不同，阈值不可跨类；
2. 拿旧 P99 / 旧权重的阈值直接用 —— 权重一换阈值量级就变（38.18 → 1.99）。

只依赖标准库，CI 可直接跑 `--check`。
"""
import argparse
import json
import os
import statistics as st

import build_pair_baseline as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (0, 1, 2)
NG_RATIO = 1.2
TOL = 1e-4


def collect(root):
    prof = B.load_profile(root)
    cats = {}
    for seed in SEEDS:
        for cat, d in B.cats_for_seed(root, prof, seed):
            c = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))["categories"][0]
            cats.setdefault(cat, []).append(c)
    out = []
    for cat in sorted(cats):
        rows = sorted(cats[cat], key=lambda r: r["seed"])
        thr = [r["threshold"] for r in rows]
        out.append(dict(
            category=cat,
            layers="1+2" if cat in (prof.get("overrides") or {}) else "2+3",
            n_calib=int(rows[0]["n_calib"]),
            n_test=int(rows[0]["n_test"]),
            seeds=sorted(int(r["seed"]) for r in rows),
            thr_ok=[round(t, 4) for t in thr],
            thr_ok_mean=round(st.mean(thr), 4),
            thr_ok_std=round(st.stdev(thr), 4) if len(thr) > 1 else 0.0,
            thr_ng_mean=round(st.mean(thr) * NG_RATIO, 4),
            recall_mean=round(st.mean([r["recall_at_thr"] for r in rows]), 4),
            fpr_mean=round(st.mean([r["false_positive_rate"] for r in rows]), 4),
            img_auroc_mean=round(st.mean([r["img_auroc"] for r in rows]), 4),
            pixel_auroc_mean=round(st.mean([r["pixel_auroc"] for r in rows]), 4),
        ))
    over = sorted((prof.get("overrides") or {}).keys())
    return dict(
        protocol=dict(layers=("2+3（默认）" + ("；%s=1+2" % "/".join(over) if over else "")),
                      size=320, thr_pct=99.5, ng_ratio=NG_RATIO,
                      calib_ratio=0.3, split_seed=42, seeds=list(SEEDS),
                      weights="wide_resnet50_2_timm.pth (timm racm_in1k)",
                      dataset_role="MVTec AD, development only (test seen many times)"),
        note=("thr_ok = 每类留出良品的 P99.5 分位；thr_ng = thr_ok x 1.2。"
              "阈值与类别绑定，不可跨类或跨权重迁移；0.5% 名义过杀在 18-117 张标定下不可达（conformal 15/15 类不达标）。"
              "layers 列说明该类用哪套特征层；服务时按 configs/patchcore_profile.json 取。"),
        classes=out,
    )


def main():
    ap = argparse.ArgumentParser(description="导出 / 校验 docs/thresholds.json")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--check", action="store_true", help="只校验仓库里的 json 是否与 run 一致")
    a = ap.parse_args()

    doc = collect(a.root)
    path = os.path.join(a.root, "docs", "thresholds.json")

    if a.check:
        if not os.path.isfile(path):
            print("[FAIL] 缺少 docs/thresholds.json")
            return 1
        old = json.load(open(path, encoding="utf-8"))
        bad = []
        if old.get("protocol") != doc["protocol"]:
            bad.append("protocol 不一致")
        old_c = {c["category"]: c for c in old.get("classes", [])}
        for c in doc["classes"]:
            o = old_c.get(c["category"])
            if o is None:
                bad.append("%s 缺失" % c["category"])
                continue
            if o.get("layers") != c["layers"]:
                bad.append("%s.layers: json %s vs 复算 %s" % (c["category"], o.get("layers"), c["layers"]))
            for k in ("thr_ok_mean", "thr_ng_mean", "recall_mean", "fpr_mean"):
                if abs(float(o.get(k, 0)) - c[k]) > TOL:
                    bad.append("%s.%s: json %.4f vs 复算 %.4f" % (c["category"], k, o.get(k, 0), c[k]))
        if bad:
            print("[FAIL] " + "\n[FAIL] ".join(bad))
            return 1
        print("[ok] thresholds.json 与逐类运行产物一致（%d 类）" % len(doc["classes"]))
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote %s（%d 类）" % (path, len(doc["classes"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
