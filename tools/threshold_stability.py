"""Resample the saved calibration scores to show threshold uncertainty.

This is a diagnostic, not a new independent test: the saved test set is reused
for each calibration bootstrap replicate. No model inference is required.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# 本文件在 tools/ 下，默认路径一律锚到仓库根，避免在别的目录执行时把结果写到别处
ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path, fpr: float, ng_ratio: float, repeats: int, seed: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    calib = np.asarray(data.get("calib", []), dtype=float)
    test = data.get("test", [])
    labels = np.asarray([int(row["label"]) for row in test])
    scores = np.asarray([float(row["score"]) for row in test])
    if len(calib) < 2 or len(labels) == 0:
        raise ValueError(f"{path}: calibration/test scores missing")
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(repeats):
        sample = rng.choice(calib, size=len(calib), replace=True)
        thr_ok = float(np.percentile(sample, 100 * (1 - fpr)))
        thr_ng = thr_ok * ng_ratio
        normal = labels == 0
        anomalous = labels == 1
        rows.append({
            "replicate": rep,
            "thr_ok": thr_ok,
            "thr_ng": thr_ng,
            "good_fpr": float(np.mean(scores[normal] >= thr_ok)) if normal.any() else float("nan"),
            # 漏检 = 缺陷被判 OK（< thr_ok）；用 thr_ng 会把灰区件同时算进漏检与 review_rate
            "defect_miss": float(np.mean(scores[anomalous] < thr_ok)) if anomalous.any() else float("nan"),
            "review_rate": float(np.mean((scores >= thr_ok) & (scores < thr_ng))),
        })
    return data.get("category", path.stem), len(calib), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=str(ROOT / "experiment_data/00_baselines/results_best/scores"))
    ap.add_argument("--fpr", type=float, default=0.005)
    ap.add_argument("--ng-ratio", type=float, default=1.2)
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "experiment_data/01_mvtec_gpu/threshold_stability"))
    a = ap.parse_args()
    if not 0 < a.fpr < 1 or a.ng_ratio < 1 or a.repeats < 1:
        ap.error("fpr must be in (0,1), ng-ratio >= 1, and repeats >= 1")
    paths = sorted(Path(a.scores).glob("*.json"))
    if not paths:
        ap.error(f"no score JSON files found in {a.scores}")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in paths:
        category, n_calib, reps = summarize(path, a.fpr, a.ng_ratio, a.repeats, a.seed)
        for key in ("thr_ok", "thr_ng", "good_fpr", "defect_miss", "review_rate"):
            values = np.asarray([row[key] for row in reps], dtype=float)
            summary.append({"category": category, "calib_n": n_calib, "metric": key,
                            "mean": float(np.nanmean(values)), "std": float(np.nanstd(values)),
                            "p05": float(np.nanpercentile(values, 5)),
                            "p95": float(np.nanpercentile(values, 95))})
        (out_dir / f"{category}.json").write_text(
            json.dumps({"category": category, "calib_n": n_calib, "fpr_target": a.fpr,
                        "ng_ratio": a.ng_ratio, "repeats": a.repeats, "seed": a.seed,
                        "warning": "Bootstrap diagnostic reuses the fixed test set; not an independent confidence interval.",
                        "replicates": reps}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{category}: calib={n_calib}, thr_ok median={np.median([r['thr_ok'] for r in reps]):.4f}")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved bootstrap diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
