"""Summarize packaged experiment runs (read-only).

Scans <root>/**/metrics.json, derives a method label from the recorded command,
and prints per-seed rows plus cross-seed mean/std for the coreset study.

per_image.csv counts are per category and per split (held-out calibration +
test), not the whole batch: an all-class baseline run writes one CSV holding
every category, so a naive row count over-reports a single category (e.g. grid
would show 2810 rows instead of 79 + 78). n_calib / n_test come from
metrics.json's category row and are printed separately so nobody mistakes the
CSV row count for the test sample size.

Usage:
  .venv/Scripts/python.exe tools/summarize_runs.py --root experiment_data/06_optimization
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args_of(command: list[str]) -> dict[str, str]:
    return dict(zip(command, command[1:]))


def method_label(command: list[str], row: dict) -> str:
    argv = parse_args_of(command)
    mem = row.get("mem_patches")
    if argv.get("--coreset-method") == "projected-greedy":
        ratio = argv.get("--coreset-ratio", "?")
        try:
            pct = f"{float(ratio) * 100:.3f}%"
        except (TypeError, ValueError):
            pct = str(ratio)
        return f"projected {pct} (k={mem})"
    greedy = argv.get("--greedy", "0")
    return f"legacy k={greedy}" if greedy not in ("0", None) else f"legacy random (k={mem})"


def seed_of(command: list[str]) -> str:
    return parse_args_of(command).get("--seed", "?")


def per_image_counts(run: Path, category: str) -> tuple[int, int, int]:
    """Rows of per_image.csv for this category, split into calib / test."""
    path = run / "per_image.csv"
    if not path.is_file():
        return 0, 0, 0
    n_calib = n_test = 0
    with path.open(encoding="utf-8-sig") as f:
        for rec in csv.DictReader(f):
            if rec.get("category") != category:
                continue
            split = (rec.get("split") or "").strip()
            if split in ("held_out", "calib"):
                n_calib += 1
            elif split == "test":
                n_test += 1
    return n_calib, n_test, n_calib + n_test


def f4(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "NA"


def f2(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "NA"


def f1(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "NA"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiment_data/06_optimization")
    ap.add_argument("--category", default="", help="only this category")
    ap.add_argument("--include-smoke", action="store_true", help="also include --limit smoke runs")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root

    groups: dict[tuple[str, str], list[dict]] = {}
    for metrics_path in sorted(root.rglob("metrics.json")):
        run = metrics_path.parent
        config_path = run / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        command = config.get("command", [])
        if "--limit" in command and not args.include_smoke:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("return_code") not in (0, None):
            print(f"# skip {run.relative_to(root)}: return_code={metrics.get('return_code')}")
            continue
        for row in metrics.get("categories", []):
            category = row.get("category", "?")
            if args.category and category != args.category:
                continue
            n_calib, n_test, n_rows = per_image_counts(run, category)
            groups.setdefault((category, method_label(command, row)), []).append(
                {"seed": seed_of(command), "row": row, "dir": str(run.relative_to(root)),
                 "n_calib": n_calib, "n_test": n_test, "n_rows": n_rows})

    if not groups:
        print("no packaged runs found")
        return 1

    print("## per-seed")
    print("| category | method | seed | mem | img AUROC | pixel AUROC | miss | fpr | latency ms | build s | peak MB | n_calib | n_test | per_image rows | dir |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (category, method), items in sorted(groups.items()):
        for it in sorted(items, key=lambda x: x["seed"]):
            row = it["row"]
            rec, fp = row.get("recall_at_thr"), row.get("false_positive_rate")
            miss = f"{100 * (1 - rec):.1f}%" if isinstance(rec, (int, float)) else "NA"
            fpr = f"{100 * fp:.1f}%" if isinstance(fp, (int, float)) else "NA"
            print(f"| {category} | {method} | {it['seed']} | {row.get('mem_patches')} | "
                  f"{f4(row.get('img_auroc'))} | {f4(row.get('pixel_auroc'))} | {miss} | {fpr} | "
                  f"{f2(row.get('latency_ms'))} | {f1(row.get('build_seconds'))} | "
                  f"{f1(row.get('peak_gpu_mb'))} | {it['n_calib']} | {it['n_test']} | "
                  f"{it['n_rows']} | {it['dir']} |")

    print()
    print("## across seeds (mean ± sample std)")
    print("| category | method | seeds | mem | img AUROC | pixel AUROC | miss | fpr | latency ms | build s | peak MB | n_calib | n_test |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def fmt(values: list, digits: int = 4) -> str:
        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            return "NA"
        return f"{nums[0]:.{digits}f}" if len(nums) == 1 else (
            f"{st.mean(nums):.{digits}f} ± {st.stdev(nums):.{digits}f}")

    def fmt_pct(values: list) -> str:
        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            return "NA"
        return f"{nums[0]:.1f}%" if len(nums) == 1 else (
            f"{st.mean(nums):.1f}% ± {st.stdev(nums):.1f}%")

    for (category, method), items in sorted(groups.items()):
        items = sorted(items, key=lambda x: x["seed"])
        rows = [it["row"] for it in items]
        seeds = ",".join(it["seed"] for it in items)
        print(f"| {category} | {method} | {seeds} | {rows[0].get('mem_patches')} | "
              f"{fmt([r.get('img_auroc') for r in rows])} | {fmt([r.get('pixel_auroc') for r in rows])} | "
              f"{fmt_pct([100 * (1 - r['recall_at_thr']) for r in rows])} | "
              f"{fmt_pct([100 * r['false_positive_rate'] for r in rows])} | "
              f"{fmt([r.get('latency_ms') for r in rows], 2)} | "
              f"{fmt([r.get('build_seconds') for r in rows], 1)} | "
              f"{fmt([r.get('peak_gpu_mb') for r in rows], 1)} | "
              f"{items[0]['n_calib']} | {items[0]['n_test']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
