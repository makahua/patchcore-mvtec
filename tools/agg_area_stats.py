# -*- coding: utf-8 -*-
"""Phase 0b: defect-area statistic per category vs best aggregation mode.

Defect area fraction is computed from ground_truth masks only (no test scores),
so a rule based on it can be pre-registered before seeing AUROC.
"""
import glob
import io
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "MVTec", "mvtech_anomaly_detection")
SCAN = os.path.join(ROOT, "docs", "aggregation-15class.json")


def main():
    with io.open(SCAN, encoding="utf-8") as f:
        scan = json.load(f)
    rows = []
    for cat in sorted(scan["per_class"]):
        fracs = []
        for p in glob.glob(os.path.join(DATA, cat, "ground_truth", "*", "*.png")):
            m = np.asarray(Image.open(p).convert("L"))
            fracs.append(float((m > 127).mean()))
        med = float(np.median(fracs)) if fracs else float("nan")
        p90 = float(np.percentile(fracs, 90)) if fracs else float("nan")
        d = scan["per_class"][cat]
        best = max(("max", "top1pct", "top5pct", "top10pct", "trimmed"), key=lambda m: d[m]["mean"])
        rows.append({"category": cat, "n_masks": len(fracs), "median_area": med,
                     "p90_area": p90, "max": d["max"]["mean"], "top1pct": d["top1pct"]["mean"],
                     "top5pct": d["top5pct"]["mean"], "best_mode": best,
                     "best_gain": d[best]["mean"] - d["max"]["mean"]})
    rows.sort(key=lambda r: r["median_area"])
    out = os.path.join(ROOT, "docs", "aggregation-area-rule.json")
    with io.open(out, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("%-12s %6s %8s %8s %8s %8s  %s" % ("category", "masks", "med_area", "max", "top1%", "top5%", "best"))
    for r in rows:
        print("%-12s %6d %8.4f %8.4f %8.4f %8.4f  %s (%+.4f)" % (
            r["category"], r["n_masks"], r["median_area"], r["max"], r["top1pct"],
            r["top5pct"], r["best_mode"], r["best_gain"]))


if __name__ == "__main__":
    main()
