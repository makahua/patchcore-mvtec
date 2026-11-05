# -*- coding: utf-8 -*-
"""Paired statistics + default-switch gates for a two-config protocol sweep.

Directory convention (produced by tools/run_experiment.py):
  <root>/<baseline_slug>_<category>_seed_<NNN>/
  <root>/<candidate_slug>_<category>_seed_<NNN>/

Reports per-seed macro metrics, seed mean/std (ddof=1), paired difference
(candidate - baseline) per seed, per-class degradation, triage rates with
numerators/denominators, and the pre-registered gate table.

Usage:
  python tools/pair_stats.py --root experiment_data/06_optimization/10_pair \
      --baseline 23 --candidate 12 --out experiment_data/06_optimization/10_pair/pair_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NG_RATIO = 1.2


def load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = metrics.get("per_category") or metrics.get("categories") or []
    if isinstance(rows, dict):
        rows = [dict(category=k, **v) for k, v in rows.items()]
    cat = None
    entry = {"category": None, "image": {}, "pixel": {}, "latency": {}}
    for row in rows:
        c = row.get("category") or cat
        entry["category"] = c
        entry["image"][c] = row.get("img_auroc")
        entry["pixel"][c] = row.get("pixel_auroc")
        entry["latency"][c] = row.get("latency_ms")
        entry["threshold"] = entry.get("threshold") or {}
        entry["threshold"][c] = row.get("threshold")
    aupro_path = run_dir / "aupro.json"
    entry["aupro"] = {}
    if aupro_path.is_file():
        data = json.loads(aupro_path.read_text(encoding="utf-8"))
        per = data.get("per_class") or data.get("per_category") or []
        if isinstance(per, dict):
            per = [dict(category=k, **v) for k, v in per.items()]
        for r in per:
            if isinstance(r, dict) and r.get("category"):
                entry["aupro"][r["category"]] = r.get("aupru", r.get("aupro"))
    entry["triage"] = triage(run_dir, entry["threshold"])
    return entry


def triage(run_dir: Path, thresholds: dict) -> dict:
    """Counts under the two-stage rule with thr_ok from the run and thr_ng = thr_ok * 1.2."""
    per_image = run_dir / "per_image.csv"
    out = {}
    if not per_image.is_file():
        return out
    rows = list(csv.DictReader(open(per_image, encoding="utf-8-sig")))
    by_cat = {}
    for r in rows:
        if r.get("split") == "test":
            by_cat.setdefault(r["category"], []).append(r)
    for cat, items in by_cat.items():
        thr = thresholds.get(cat)
        if thr is None:
            continue
        ng = thr * NG_RATIO
        good = [float(r["score"]) for r in items if r["label"] == "0"]
        bad = [float(r["score"]) for r in items if r["label"] == "1"]
        miss = sum(1 for s in bad if s < thr)
        over = sum(1 for s in good if s >= ng)
        rev_good = sum(1 for s in good if thr <= s < ng)
        rev_bad = sum(1 for s in bad if thr <= s < ng)
        out[cat] = dict(n_good=len(good), n_bad=len(bad), miss=miss, over=over,
                        review_good=rev_good, review_bad=rev_bad,
                        miss_rate=miss / len(bad) if bad else None,
                        over_rate=over / len(good) if good else None,
                        review_rate=(rev_good + rev_bad) / len(items) if items else None)
    return out


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default="23")
    ap.add_argument("--candidate", default="12")
    ap.add_argument("--out", default="")
    ap.add_argument("--latency-gate", type=float, default=25.0)
    ap.add_argument("--degrade-gate", type=float, default=0.03)
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_absolute():
        root = ROOT / root

    def collect(slug: str):
        runs = {}
        for d in sorted(root.glob(f"{slug}_*_seed_*")):
            prefix, _, seed_txt = d.name.rpartition("_seed_")
            if not prefix.startswith(slug + "_") or not seed_txt.isdigit():
                continue
            cat = prefix[len(slug) + 1:]
            seed = int(seed_txt)
            entry = load_run(d)
            if entry:
                runs.setdefault(seed, {})[cat] = entry
        return runs

    base, cand = collect(a.baseline), collect(a.candidate)
    if not base or not cand:
        print(f"[错误] 未找到配对运行：基线 {len(base)} 个 seed，候选 {len(cand)} 个 seed")
        return 2

    cats = sorted({c for runs in base.values() for c in runs})
    seeds = sorted(set(base) & set(cand))
    print(f"根目录：{root}")
    print(f"基线 {a.baseline}：seed {sorted(base)}；候选 {a.candidate}：seed {sorted(cand)}；共同 seed {seeds}")

    per_seed = {}
    for seed in seeds:
        row = {"image": [], "pixel": [], "aupro": [], "latency": []}
        for cat in cats:
            b = base[seed].get(cat)
            if not b:
                continue
            row["image"].append(b["image"].get(cat))
            row["pixel"].append(b["pixel"].get(cat))
            row["aupro"].append(b["aupro"].get(cat))
            row["latency"].append(b["latency"].get(cat))
        per_seed[seed] = {k: mean(v) for k, v in row.items()}

    print("\n## 每 seed 的 15 类 macro（基线）")
    print("| seed | image AUROC | pixel AUROC | AUPRO@30 | 算法段均值 ms |")
    print("|---:|---:|---:|---:|---:|")
    for seed in seeds:
        r = per_seed[seed]
        print(f"| {seed} | {fmt(r['image'])} | {fmt(r['pixel'])} | {fmt(r['aupro'])} | {fmt(r['latency'], 2)} |")

    print("\n## seed 汇总（均值 ± 样本标准差，ddof=1）")
    for key, label in (("image", "image AUROC"), ("pixel", "pixel AUROC"),
                       ("aupro", "AUPRO@30"), ("latency", "算法段 ms")):
        vals = [per_seed[s][key] for s in seeds]
        print(f"- {label}: {fmt(mean(vals))} ± {fmt(std(vals))}")

    print("\n## 配对差（候选 − 基线，逐 seed）")
    paired = {}
    for key, label in (("image", "image AUROC"), ("pixel", "pixel AUROC"), ("aupro", "AUPRO@30")):
        diffs = []
        for seed in seeds:
            bc = mean([base[seed][c][{"image": "image", "pixel": "pixel", "aupro": "aupro"}[key]].get(c)
                       for c in cats if c in base[seed]])
            cc = mean([cand[seed][c][{"image": "image", "pixel": "pixel", "aupro": "aupro"}[key]].get(c)
                       for c in cats if c in cand[seed]])
            if bc is None or cc is None:
                continue
            diffs.append(cc - bc)
        paired[key] = diffs
        if diffs:
            print(f"- {label}: " + ", ".join(f"{d:+.4f}" for d in diffs) +
                  f" → 均值 {mean(diffs):+.4f}，范围 [{min(diffs):+.4f}, {max(diffs):+.4f}]")

    print("\n## 单类退化（候选 − 基线，三 seed 平均）")
    worst = []
    for cat in cats:
        bs = [base[s][cat]["image"].get(cat) for s in seeds if cat in base[s]]
        cs = [cand[s][cat]["image"].get(cat) for s in seeds if cat in cand[s]]
        ba = [base[s][cat]["aupro"].get(cat) for s in seeds if cat in base[s]]
        ca = [cand[s][cat]["aupro"].get(cat) for s in seeds if cat in cand[s]]
        di = (mean(cs) - mean(bs)) if mean(bs) is not None and mean(cs) is not None else None
        da = (mean(ca) - mean(ba)) if mean(ba) is not None and mean(ca) is not None else None
        worst.append((cat, di, da))
    worst.sort(key=lambda t: (t[1] if t[1] is not None else 0))
    print("| 类别 | Δimage | ΔAUPRO |")
    print("|---|---:|---:|")
    for cat, di, da in worst:
        print(f"| {cat} | {fmt(di, 4)} | {fmt(da, 4)} |")

    print("\n## 三档判定（候选，逐 seed 汇总；thr_ng = thr_ok × 1.2）")
    for seed in seeds:
        miss_n = miss_d = over_n = over_d = rev_n = rev_d = 0
        for cat, e in cand[seed].items():
            t = e["triage"].get(cat)
            if not t:
                continue
            miss_n += t["miss"]; miss_d += t["n_bad"]
            over_n += t["over"]; over_d += t["n_good"]
            rev_n += t["review_good"] + t["review_bad"]; rev_d += t["n_good"] + t["n_bad"]
        print(f"- seed {seed}: 漏检 {miss_n}/{miss_d} = {fmt(miss_n / miss_d if miss_d else None, 4)}；"
              f"实际 NG 过杀 {over_n}/{over_d} = {fmt(over_n / over_d if over_d else None, 4)}；"
              f"复核 {rev_n}/{rev_d} = {fmt(rev_n / rev_d if rev_d else None, 4)}")

    print("\n## 默认切换门槛")
    gates = []
    n_seeds = len(seeds)
    gates.append(("完整性：两组各 15 类 × 3 seed",
                  len(cats) == 15 and n_seeds == 3,
                  f"类别 {len(cats)}，共同 seed {n_seeds}"))
    for key, label in (("image", "macro image AUROC"), ("aupro", "macro AUPRO@30")):
        d = mean(paired.get(key, []))
        gates.append((f"总体质量：{label} 配对均值差 ≥ 0", d is not None and d >= 0,
                      "—" if d is None else f"{d:+.4f}"))
    deg_i = [x for _, di, _ in worst if di is not None]
    deg_a = [x for _, _, da in worst if da is not None]
    ok_deg = (not deg_i or min(deg_i) >= -a.degrade_gate) and (not deg_a or min(deg_a) >= -a.degrade_gate)
    gates.append((f"单类退化：任一类三 seed 平均下降 ≤ {a.degrade_gate}", ok_deg,
                  f"最差 Δimage {min(deg_i):+.4f}" if deg_i else "—"))
    lat = mean([per_seed[s]["latency"] for s in seeds])
    gates.append((f"算法时延：15 类均值 ≤ {a.latency_gate} ms", lat is not None and lat <= a.latency_gate,
                  "—" if lat is None else f"{lat:.2f} ms"))
    print("| 门槛 | 结果 | 实测 |")
    print("|---|---|---|")
    for name, ok, detail in gates:
        print(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    if a.out:
        out = Path(a.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(
            root=str(root), baseline=a.baseline, candidate=a.candidate,
            seeds=seeds, categories=cats, per_seed={str(k): v for k, v in per_seed.items()},
            paired={k: v for k, v in paired.items()},
            per_class_degrade=[dict(category=c, d_image=di, d_aupro=da) for c, di, da in worst],
            gates=[dict(name=n, passed=bool(ok), detail=d) for n, ok, d in gates],
        ), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
