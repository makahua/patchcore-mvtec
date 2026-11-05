"""Threshold working points and bootstrap intervals (plan task F).

Two working-point modes are compared on a deterministic development split of the
saved test set:

* ``global``     -- one FPR target for every category (the current default);
* ``per-class``  -- per-category FPR chosen on the validation half only.

The split is path-sorted and alternating, both halves keep good and defect
images.  It is a *development* split: the underlying MVTec test set has already
been inspected many times, so no number here is an independent holdout.  The
test half is evaluated once, after the per-class FPR has been frozen on the
validation half.

Bootstrap replicates resample the calibration scores (to rebuild the threshold)
and the reported test half (to rebuild the rates).  This quantifies sampling
noise; it is not a conformal guarantee, and with 18-117 calibration images a
0.5% target is not reachable by conformal calibration at all (see
``docs/conformal-thresholds.md``).

No model inference is required.
"""
import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_COST = (100.0, 10.0, 1.0)  # FN, FP, review
DEFAULT_GRID = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)


def load_scores(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    calib = np.asarray(data.get("calib", []), dtype=float)
    test = data.get("test", [])
    if calib.size < 2 or not test:
        raise ValueError(f"{path}: calibration/test scores missing")
    paths = [row["path"] for row in test]
    labels = np.asarray([int(row["label"]) for row in test])
    scores = np.asarray([float(row["score"]) for row in test])
    return data, calib, paths, labels, scores


def dev_split(paths):
    order = np.argsort(np.asarray(paths))
    val = np.sort(order[0::2])
    test = np.sort(order[1::2])
    return val, test


def threshold_at(calib, fpr):
    return float(np.percentile(calib, 100.0 * (1.0 - fpr)))


def mean_or_nan(values):
    return float(np.mean(values)) if values.size else float("nan")


def rates(scores, labels, thr_ok, thr_ng, cost, prev):
    good = labels == 0
    defect = labels == 1
    fpr = mean_or_nan(scores[good] >= thr_ok)
    miss = mean_or_nan(scores[defect] < thr_ng)
    review_d = mean_or_nan((scores[defect] >= thr_ok) & (scores[defect] < thr_ng))
    review_g = mean_or_nan((scores[good] >= thr_ok) & (scores[good] < thr_ng))
    review = mean_or_nan((scores >= thr_ok) & (scores < thr_ng))
    c_fn, c_fp, c_rev = cost
    total = prev * (c_fn * miss + c_rev * review_d) + (1.0 - prev) * (c_fp * fpr + c_rev * review_g)
    return {"fpr": fpr, "miss": miss, "review": review, "cost": float(total)}


def select_fpr(calib, scores, labels, grid, cost, prev, ng_ratio):
    best = None
    for fpr in grid:
        thr_ok = threshold_at(calib, fpr)
        r = rates(scores, labels, thr_ok, thr_ok * ng_ratio, cost, prev)
        row = {"fpr_target": fpr, "thr_ok": thr_ok, **r}
        if best is None or row["cost"] < best["cost"]:
            best = row
    return best


def bootstrap_fixed(calib, scores, labels, fpr, ng_ratio, cost, prev, repeats, seed):
    rng = np.random.default_rng(seed)
    n_calib = calib.size
    n_test = scores.size
    out = {k: np.empty(repeats) for k in ("thr_ok", "fpr", "miss", "review", "cost")}
    for rep in range(repeats):
        c = calib[rng.integers(0, n_calib, n_calib)]
        idx = rng.integers(0, n_test, n_test)
        thr_ok = threshold_at(c, fpr)
        r = rates(scores[idx], labels[idx], thr_ok, thr_ok * ng_ratio, cost, prev)
        out["thr_ok"][rep] = thr_ok
        for key in ("fpr", "miss", "review", "cost"):
            out[key][rep] = r[key]
    return out


def bootstrap_perclass(calib, val_s, val_y, test_s, test_y, grid, ng_ratio, cost, prev, repeats, seed):
    """Resample calibration, validation and test; refit the FPR pick each replicate.

    This is the honest interval for the per-class procedure: it carries both the
    threshold noise and the instability of choosing a target on a small half.
    """
    rng = np.random.default_rng(seed)
    n_calib, n_val, n_test = calib.size, val_s.size, test_s.size
    out = {k: np.empty(repeats) for k in ("thr_ok", "fpr", "miss", "review", "cost", "fpr_target")}
    for rep in range(repeats):
        c = calib[rng.integers(0, n_calib, n_calib)]
        vi = rng.integers(0, n_val, n_val)
        pick = select_fpr(c, val_s[vi], val_y[vi], grid, cost, prev, ng_ratio)
        thr_ok = threshold_at(c, pick["fpr_target"])
        ti = rng.integers(0, n_test, n_test)
        r = rates(test_s[ti], test_y[ti], thr_ok, thr_ok * ng_ratio, cost, prev)
        out["thr_ok"][rep] = thr_ok
        out["fpr_target"][rep] = pick["fpr_target"]
        for key in ("fpr", "miss", "review", "cost"):
            out[key][rep] = r[key]
    return out


def ci(values):
    return {
        "p2_5": float(np.nanpercentile(values, 2.5)),
        "p50": float(np.nanpercentile(values, 50)),
        "p97_5": float(np.nanpercentile(values, 97.5)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="experiment_data/06_optimization/00_baseline/mvtec_seed_000/native/scores")
    ap.add_argument("--out", default="experiment_data/06_optimization/06_threshold/bootstrap_seed_000")
    ap.add_argument("--fpr", type=float, default=0.005, help="global FPR target")
    ap.add_argument("--grid", default=",".join(f"{g:g}" for g in DEFAULT_GRID),
                    help="candidate FPR targets for per-class selection")
    ap.add_argument("--cost", default=",".join(f"{c:g}" for c in DEFAULT_COST), metavar="FN,FP,REV")
    ap.add_argument("--prevalence", type=float, default=0.01)
    ap.add_argument("--ng-ratio", type=float, default=1.2)
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    grid = [float(x) for x in a.grid.split(",") if x.strip()]
    cost = tuple(float(x) for x in a.cost.split(","))
    if len(cost) != 3 or not 0 < a.fpr < 1 or a.repeats < 1 or not grid:
        ap.error("need --cost FN,FP,REV, 0<fpr<1, repeats>=1 and a non-empty --grid")

    score_paths = sorted(Path(a.scores).glob("*.json"))
    if not score_paths:
        ap.error(f"no score JSON files found in {a.scores}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    t0 = time.time()

    rows = []
    store = []
    for path in score_paths:
        data, calib, paths, labels, scores = load_scores(path)
        cat = data.get("category", path.stem)
        val_idx, test_idx = dev_split(paths)
        val_s, val_y = scores[val_idx], labels[val_idx]
        test_s, test_y = scores[test_idx], labels[test_idx]

        # global mode: fixed FPR target, evaluated on the test half
        thr_g = threshold_at(calib, a.fpr)
        g = rates(test_s, test_y, thr_g, thr_g * a.ng_ratio, cost, a.prevalence)
        gb = bootstrap_fixed(calib, test_s, test_y, a.fpr, a.ng_ratio, cost, a.prevalence, a.repeats, a.seed)

        # per-class mode: FPR frozen on the validation half, then one test pass
        pick = select_fpr(calib, val_s, val_y, grid, cost, a.prevalence, a.ng_ratio)
        fpr_star = pick["fpr_target"]
        thr_p = threshold_at(calib, fpr_star)
        p = rates(test_s, test_y, thr_p, thr_p * a.ng_ratio, cost, a.prevalence)
        pb = bootstrap_perclass(calib, val_s, val_y, test_s, test_y, grid, a.ng_ratio,
                                cost, a.prevalence, a.repeats, a.seed)

        n_calib = int(calib.size)
        alpha_min = 1.0 / (n_calib + 1.0)
        rows.append({
            "category": cat,
            "n_calib": n_calib,
            "n_val": int(val_idx.size),
            "n_test": int(test_idx.size),
            "val_good": int((val_y == 0).sum()),
            "val_defect": int((val_y == 1).sum()),
            "test_good": int((test_y == 0).sum()),
            "test_defect": int((test_y == 1).sum()),
            "conformal_alpha_min": alpha_min,
            "conformal_ok_0005": bool(alpha_min <= 0.005),
            "global_thr": thr_g, "global_fpr_measured": g["fpr"], "global_miss": g["miss"],
            "global_review": g["review"], "global_cost": g["cost"],
            "global_thr_ci": ci(gb["thr_ok"]), "global_miss_ci": ci(gb["miss"]),
            "global_fpr_ci": ci(gb["fpr"]), "global_review_ci": ci(gb["review"]),
            "global_cost_ci": ci(gb["cost"]),
            "perclass_fpr_target": fpr_star,
            "perclass_val_cost": pick["cost"],
            "perclass_thr": thr_p, "perclass_fpr_measured": p["fpr"], "perclass_miss": p["miss"],
            "perclass_review": p["review"], "perclass_cost": p["cost"],
            "perclass_thr_ci": ci(pb["thr_ok"]), "perclass_miss_ci": ci(pb["miss"]),
            "perclass_fpr_ci": ci(pb["fpr"]), "perclass_review_ci": ci(pb["review"]),
            "perclass_cost_ci": ci(pb["cost"]),
            "perclass_target_dist": {f"{v:g}": int(c) for v, c in
                                     zip(*np.unique(pb["fpr_target"], return_counts=True))},
        })
        log_lines.append(
            f"{cat}: calib={n_calib} alpha_min={alpha_min:.4f} | global fpr={a.fpr:g} "
            f"miss={g['miss']:.3f} review={g['review']:.3f} cost={g['cost']:.4f} | "
            f"per-class fpr*={fpr_star:g} miss={p['miss']:.3f} review={p['review']:.3f} cost={p['cost']:.4f}"
        )
        store.append((cat, calib, test_s, test_y))

    # Cost/precision frontier: same global target applied to every class, point estimates only.
    frontier = []
    for fpr in grid:
        vals = []
        for _cat, calib, test_s, test_y in store:
            thr = threshold_at(calib, fpr)
            vals.append(rates(test_s, test_y, thr, thr * a.ng_ratio, cost, a.prevalence))
        frontier.append({
            "fpr_target": fpr,
            "macro_fpr": float(np.nanmean([v["fpr"] for v in vals])),
            "macro_miss": float(np.nanmean([v["miss"] for v in vals])),
            "macro_review": float(np.nanmean([v["review"] for v in vals])),
            "macro_cost": float(np.nanmean([v["cost"] for v in vals])),
        })

    def macro(key):
        return float(np.mean([r[key] for r in rows]))

    modes = {}
    for mode in ("global", "perclass"):
        modes[mode] = {
            "macro_fpr": macro(f"{mode}_fpr_measured"),
            "macro_miss": macro(f"{mode}_miss"),
            "macro_review": macro(f"{mode}_review"),
            "macro_cost": macro(f"{mode}_cost"),
            "fpr_targets": sorted({r[f"{mode}_fpr_target"] if mode == "perclass" else a.fpr for r in rows}),
        }
    modes["perclass"]["fpr_target_by_class"] = {r["category"]: r["perclass_fpr_target"] for r in rows}

    elapsed = time.time() - t0
    fieldnames = ["category", "n_calib", "n_val", "n_test", "val_good", "val_defect", "test_good",
                  "test_defect", "conformal_alpha_min", "conformal_ok_0005",
                  "global_fpr_target", "global_thr", "global_fpr_measured", "global_miss", "global_review",
                  "global_cost", "perclass_fpr_target", "perclass_val_cost", "perclass_thr",
                  "perclass_fpr_measured", "perclass_miss", "perclass_review", "perclass_cost"]
    per_class = out_dir / "per_class.csv"
    with per_class.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["global_fpr_target"] = a.fpr
            w.writerow(r2)

    summary = out_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode", "macro_fpr", "macro_miss", "macro_review", "macro_cost"])
        for mode in ("global", "perclass"):
            w.writerow([mode, f"{modes[mode]['macro_fpr']:.4f}", f"{modes[mode]['macro_miss']:.4f}",
                        f"{modes[mode]['macro_review']:.4f}", f"{modes[mode]['macro_cost']:.4f}"])
        for row in frontier:
            w.writerow([f"global@{row['fpr_target']:g}", f"{row['macro_fpr']:.4f}", f"{row['macro_miss']:.4f}",
                        f"{row['macro_review']:.4f}", f"{row['macro_cost']:.4f}"])

    metrics = {
        "protocol": {
            "source_scores": str(a.scores),
            "split": "path-sorted alternating halves (val=even, test=odd); development split of an already "
                     "inspected test set, not an independent holdout",
            "global_fpr": a.fpr,
            "fpr_grid": grid,
            "cost_fn_fp_rev": list(cost),
            "prevalence": a.prevalence,
            "ng_ratio": a.ng_ratio,
            "repeats": a.repeats,
            "seed": a.seed,
            "warning": "Bootstrap quantifies sampling noise; it is not a conformal guarantee. "
                       "With these calibration sizes a 0.5% target is unreachable by conformal calibration.",
        },
        "modes": modes,
        "frontier": frontier,
        "per_class": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps({
        "tool": "tools/threshold_ci.py",
        "command": " ".join(["python", "tools/threshold_ci.py", "--scores", a.scores, "--out", a.out,
                             "--fpr", f"{a.fpr:g}", "--repeats", str(a.repeats), "--seed", str(a.seed)]),
        "scores_dir": a.scores,
        "repeats": a.repeats,
        "seed": a.seed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "timing.json").write_text(json.dumps({"duration_seconds": elapsed}, indent=2), encoding="utf-8")
    (out_dir / "run.log").write_text("\n".join(log_lines) + f"\n[elapsed] {elapsed:.2f} s\n", encoding="utf-8")

    print("\n".join(log_lines))
    print(f"global   macro fpr={modes['global']['macro_fpr']:.4f} miss={modes['global']['macro_miss']:.4f} "
          f"review={modes['global']['macro_review']:.4f} cost={modes['global']['macro_cost']:.4f}")
    print(f"perclass macro fpr={modes['perclass']['macro_fpr']:.4f} miss={modes['perclass']['macro_miss']:.4f} "
          f"review={modes['perclass']['macro_review']:.4f} cost={modes['perclass']['macro_cost']:.4f}")
    for row in frontier:
        print(f"frontier fpr={row['fpr_target']:g}: macro fpr={row['macro_fpr']:.4f} "
              f"miss={row['macro_miss']:.4f} cost={row['macro_cost']:.4f}")
    print(f"Saved to {out_dir} ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
