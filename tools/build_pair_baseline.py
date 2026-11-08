# -*- coding: utf-8 -*-
"""生成 docs/pair-baseline.json（按类配置感知）。

读 configs/patchcore_profile.json：
  - overrides：按类覆盖的 CLI 项（screw / capsule → layers 1+2）
  - evidence ：被覆盖类别的运行目录（该类别用哪批 run 出数）
其余类别一律取 10_pair/23_<cat>_seed_NNN（冻结协议 2+3）。

只依赖标准库与仓库内已提交的产物，CI 可跑。
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR = os.path.join("experiment_data", "06_optimization", "10_pair")
SLUG = "23"
CATS = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
        "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")
SEEDS = (0, 1, 2)
NG_RATIO = 1.2


def load_profile(root):
    path = os.path.join(root, "configs", "patchcore_profile.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cats_for_seed(root, prof, seed):
    """返回 [(category, run_dir)]，按 CATS 顺序。"""
    ev = prof.get("evidence") or {}
    out = []
    for cat in CATS:
        if cat in ev:
            pat = os.path.join(ev[cat]["run_root"], ev[cat]["run_glob"] % seed)
            hits = sorted(glob.glob(os.path.join(root, pat)))
        else:
            hits = sorted(glob.glob(os.path.join(root, PAIR, "%s_%s_seed_%03d" % (SLUG, cat, seed))))
        if not hits:
            raise SystemExit("[error] 缺 %s / seed %d 的运行目录" % (cat, seed))
        out.append((cat, hits[0]))
    return out


def pooled_and_triage(pairs):
    byv, tri = {}, dict(miss=0, n_bad=0, over=0, n_good=0, review=0, n_test=0)
    for cat, d in pairs:
        m = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))["categories"][0]
        thr = float(m["threshold"])
        ng = thr * NG_RATIO
        with open(os.path.join(d, "per_image.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != "test":
                    continue
                score, label = float(row["score"]), int(row["label"])
                byv.setdefault(score, [0, 0])[label] += 1
                tri["n_test"] += 1
                if label == 1:
                    tri["n_bad"] += 1
                    if score < thr:
                        tri["miss"] += 1
                    elif score < ng:
                        tri["review"] += 1
                else:
                    tri["n_good"] += 1
                    if score >= ng:
                        tri["over"] += 1
                    elif score >= thr:
                        tri["review"] += 1
    pos = sum(v[1] for v in byv.values())
    neg = sum(v[0] for v in byv.values())
    cum_good, acc = 0, 0.0
    for k in sorted(byv):
        good, bad = byv[k]
        acc += bad * cum_good + 0.5 * bad * good
        cum_good += good
    return (acc / (pos * neg) if pos and neg else float("nan")), tri


def build(root):
    prof = load_profile(root)
    per_seed, triage, per_class_raw = {}, {}, {}
    for seed in SEEDS:
        pairs = cats_for_seed(root, prof, seed)
        rows = []
        for cat, d in pairs:
            m = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))["categories"][0]
            a = json.load(open(os.path.join(d, "aupro.json"), encoding="utf-8"))["per_class"][0]
            t = json.load(open(os.path.join(d, "timing.json"), encoding="utf-8"))["per_category"]
            lat = {r["category"]: r["latency_ms"] for r in t} if isinstance(t, list) else t
            peak = {r["category"]: r["peak_gpu_mb"] for r in t} if isinstance(t, list) else {}
            rows.append(dict(category=cat, n_train=m["n_train"], n_calib=m["n_calib"], n_test=m["n_test"],
                             img=m["img_auroc"], px=m["pixel_auroc"], au=a["aupru"], au10=a["aupru_01"],
                             thr=m["threshold"], rec=m["recall_at_thr"], fpr=m["false_positive_rate"],
                             lat=float(lat.get(cat, m["latency_ms"])), peak=float(peak.get(cat, 0) or 0)))
            per_class_raw.setdefault(cat, []).append(rows[-1])
        pooled, tri = pooled_and_triage(pairs)
        per_seed[seed] = dict(img=st.mean([r["img"] for r in rows]),
                              px=st.mean([r["px"] for r in rows]),
                              au=st.mean([r["au"] for r in rows]),
                              au10=st.mean([r["au10"] for r in rows]),
                              lat=st.mean([r["lat"] for r in rows]),
                              pooled=pooled)
        triage[str(seed)] = tri

    def agg(key, nd):
        vals = [per_seed[s][key] for s in SEEDS]
        return dict(mean=round(st.mean(vals), nd), std=round(st.stdev(vals), nd))

    per_class = []
    for cat in CATS:
        rs = per_class_raw[cat]
        per_class.append(dict(category=cat, n_train=rs[0]["n_train"], n_calib=rs[0]["n_calib"],
                              n_test=rs[0]["n_test"],
                              img=round(st.mean([r["img"] for r in rs]), 4),
                              px=round(st.mean([r["px"] for r in rs]), 4),
                              au=round(st.mean([r["au"] for r in rs]), 4),
                              thr=round(st.mean([r["thr"] for r in rs]), 2),
                              rec=round(st.mean([r["rec"] for r in rs]), 4),
                              fpr=round(st.mean([r["fpr"] for r in rs]), 4),
                              lat=round(st.mean([r["lat"] for r in rs]), 2),
                              peak=round(st.mean([r["peak"] for r in rs]), 1)))

    layers = "、".join("%s=1+2" % c for c in sorted((prof.get("overrides") or {})))
    return dict(
        protocol=("P99.5 / legacy greedy 2000 / 320 stretch / split-seed 42 / seeds 0-2 / "
                  "layers 2+3（默认）%s" % (("；" + layers) if layers else "")),
        data_role="MVTec test = development-only",
        aggregate=dict(pooled=agg("pooled", 4), img=agg("img", 4), px=agg("px", 4),
                       au=agg("au", 4), au10=agg("au10", 4), lat=agg("lat", 4)),
        triage=triage,
        per_class=per_class,
    )


def main():
    ap = argparse.ArgumentParser(description="生成 docs/pair-baseline.json（按类配置感知）")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true", help="只比对，不写")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "docs", "pair-baseline.json")
    data = build(a.root)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if a.check:
        old = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
        if json.loads(old) == json.loads(text):
            print("[ok] pair-baseline.json 与运行产物一致")
            return 0
        print("[FAIL] pair-baseline.json 过期，运行 python tools/build_pair_baseline.py 重建")
        return 1
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    ag = data["aggregate"]
    print("img %.4f  px %.4f  au %.4f  au10 %.4f  lat %.2f  pooled %.4f"
          % (ag["img"]["mean"], ag["px"]["mean"], ag["au"]["mean"], ag["au10"]["mean"],
             ag["lat"]["mean"], ag["pooled"]["mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
