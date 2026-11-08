"""Run patchcore_demo.py and package a reproducible experiment record.

Usage:
  .venv/Scripts/python.exe tools/run_experiment.py --run-dir experiment_data/06_optimization/00_baseline/bottle_seed_000 -- --data MVTec/mvtech_anomaly_detection --category bottle --size 320 --greedy 2000 --calib-ratio 0.3 --seed 0 --split-seed 42 --weights torch_cache/wide_resnet50_2_timm.pth

The wrapper refuses to overwrite a non-empty run directory. Native output is
stored under <run-dir>/native; normalized artifacts are written at <run-dir>.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("patchcore_demo.py", "tools/run_experiment.py")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    """Dependency-free AUROC for the pooled per-image scores."""
    if not labels or len(set(labels)) < 2:
        return None
    ordered = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and scores[ordered[j]] == scores[ordered[i]]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        for pos in range(i, j):
            ranks[ordered[pos]] = avg_rank
        i = j
    positives = sum(labels)
    negatives = len(labels) - positives
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def package_artifacts(run_dir: Path, native: Path, command: list[str], started: float,
                      return_code: int) -> None:
    results_path = native / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    score_dir = native / "scores"

    per_image_path = run_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["category", "split", "path", "label", "score"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if score_dir.exists():
            for score_path in sorted(score_dir.glob("*.json")):
                data = json.loads(score_path.read_text(encoding="utf-8"))
                category = data.get("category", score_path.stem)
                calib = data.get("calib", [])
                calib_paths = data.get("calibration_paths", [])
                calib_source = data.get("calibration_source", "held_out" if calib_paths else "train_self_score_fallback")
                for i, score in enumerate(calib):
                    writer.writerow({"category": category, "split": calib_source,
                                     "path": calib_paths[i] if i < len(calib_paths) else "",
                                     "label": 0, "score": score})
                for item in data.get("test", []):
                    writer.writerow({"category": category, "split": "test",
                                     "path": item.get("path", ""),
                                     "label": item.get("label", ""),
                                     "score": item.get("score", "")})

    image_labels, image_scores = [], []
    for score_path in sorted(score_dir.glob("*.json")) if score_dir.exists() else []:
        data = json.loads(score_path.read_text(encoding="utf-8"))
        image_labels.extend(int(item["label"]) for item in data.get("test", []))
        image_scores.extend(float(item["score"]) for item in data.get("test", []))
    macro = {}
    for key in ("img_auroc", "pixel_auroc", "recall_at_thr", "false_positive_rate",
                "latency_ms", "latency_p95_ms", "build_seconds", "peak_gpu_mb"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        macro[key] = sum(values) / len(values) if values else None
    (run_dir / "metrics.json").write_text(
        json.dumps({"status": "completed" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "aggregate": {"macro": macro,
                                  "pooled_image_auroc": auc_rank(image_labels, image_scores),
                                  "pooled_test_images": len(image_labels),
                                  "dataset_role": "development; MVTec test has been repeatedly inspected"},
                    "categories": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = run_dir / "summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_path.write_text("category,status\n,failed\n", encoding="utf-8")

    weight_path = None
    for i, arg in enumerate(command[:-1]):
        if arg == "--weights":
            weight_path = Path(command[i + 1])
            if not weight_path.is_absolute():
                weight_path = ROOT / weight_path
            break

    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = None
    try:
        import torch
        import torchvision
        torch_version = torch.__version__
        torchvision_version = torchvision.__version__
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        torch_version = torchvision_version = cuda_version = gpu = None

    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_commit = None

    config = {
        "command": [sys.executable, "patchcore_demo.py", *command],
        "working_directory": str(ROOT),
        "started_epoch": started,
        "finished_epoch": time.time(),
        "duration_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy_version,
        "torch": torch_version,
        "torchvision": torchvision_version,
        "cuda": cuda_version,
        "gpu": gpu,
        "git_commit": git_commit,
        "weights": str(weight_path) if weight_path else None,
        "weights_sha256": sha256(weight_path) if weight_path else None,
        "native_results": str((native / "results.json").relative_to(ROOT)),
        "manifests": sorted(str(p.relative_to(ROOT)) for p in (native / "manifests").glob("*.json")),
        "note": "MVTec test scores are development results; this wrapper does not make them an independent holdout.",
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "banks": sorted(str(p.relative_to(ROOT)) for p in (run_dir / "banks").glob("*.npz")),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    timing = {
        "duration_seconds": config["duration_seconds"],
        "per_category": [{k: row.get(k) for k in
                          ("category", "latency_ms", "latency_p95_ms", "build_seconds", "peak_gpu_mb", "peak_cpu_mb", "bank_file")}
                         for row in rows],
        "timing_semantics": "PatchCore native latency fields; consult tools/latency_breakdown.py for end-to-end stage timing.",
    }
    (run_dir / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="unique destination; existing non-empty dirs are rejected")
    parser.add_argument("--package-existing", action="store_true",
                        help="rebuild normalized artifacts from an existing native run without rerunning inference")
    parser.add_argument("--return-code", type=int, default=None,
                        help="with --package-existing: original exit code; default reads metrics.json, "
                             "otherwise -1 (unknown, stays status=failed)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="patchcore_demo.py arguments after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if args.package_existing:
        native = run_dir / "native"
        config_path = run_dir / "config.json"
        if not config_path.is_file() or not (native / "results.json").is_file():
            parser.error("existing run must contain config.json and native/results.json")
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        old_command = old_config.get("command", [])
        original_args = old_command[2:] if len(old_command) >= 2 else []
        # 重打包不能假定原来成功：默认沿用 metrics.json 的 return_code，
        # 没有记录就留 -1（unknown），宁可标 failed 也不要把失败运行洗成 completed。
        old_code = args.return_code
        if old_code is None:
            old_code = -1
            old_metrics = run_dir / "metrics.json"
            if old_metrics.is_file():
                try:
                    old_code = int(json.loads(old_metrics.read_text(encoding="utf-8")).get("return_code", -1))
                except (TypeError, ValueError, json.JSONDecodeError):
                    old_code = -1
        if old_code != 0:
            print(f"warning: repackaged run keeps a non-zero/unknown return_code ({old_code}); "
                  f"pass --return-code 0 only if the original run really succeeded")
        package_artifacts(run_dir, native, original_args,
                          float(old_config.get("started_epoch", time.time())), old_code)
        print(f"Repackaged experiment artifacts: {run_dir}")
        return 0
    if not command:
        parser.error("provide patchcore_demo.py arguments after --")
    if "--save-bank" not in command:
        command = command + ["--save-bank", str(run_dir / "banks")]

    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    native = run_dir / "native"
    native.mkdir()
    if "--out" in command:
        parser.error("do not pass --out; wrapper sets an isolated native output directory")
    command += ["--out", str(native)]

    full_command = [sys.executable, "-u", str(ROOT / "patchcore_demo.py"), *command]
    started = time.time()
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(full_command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(full_command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            encoded = line.encode(sys.stdout.encoding or "utf-8", errors="replace")
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
            log.write(line)
        return_code = proc.wait()
    package_artifacts(run_dir, native, command, started, return_code)
    print(f"\nExperiment artifacts: {run_dir}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
