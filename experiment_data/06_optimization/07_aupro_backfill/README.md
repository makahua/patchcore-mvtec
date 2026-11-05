# 07_aupro_backfill：coreset 关键点的 AUPRO 补测

> 目的：`docs/coreset-budget.md` §6 当时写着「未计算 AUPRO」。这一批把 8 个关键 run 重跑一遍并落
> `--dump-pixel`，补上区域级指标，顺便验证重跑是否逐位复现。
> 数据角色：**development only**（MVTec test 已被多轮查看，不是 holdout）。只做 seed 0，不据此改默认配置。

## 为什么重跑而不是复用

`pixel_dumps/*.npz` 需要原始像素分数图，而 `01_coreset_fixed/`、`02_coreset_projected/`、`04_representative/`
的 run 当初没有开 `--dump-pixel`。这里用 `tools/run_experiment.py` 在**新目录**重跑（不覆盖旧结果），
随后 `tools/pro_aupro.py` 计算 AUPRO。

重跑结果的 image / pixel AUROC 与原始 run **逐位一致（差值 0.0）**，证明补测没有改变被测对象。

## 内容

| run 目录 | 类别 | 配置 | mem |
|---|---|---|---|
| `screw_legacy_k0500` | screw | legacy `--greedy 500` | 500 |
| `screw_legacy_k1000` | screw | legacy `--greedy 1000` | 1000 |
| `screw_legacy_k2000` | screw | legacy `--greedy 2000` | 2000 |
| `screw_legacy_k4000` | screw | legacy `--greedy 4000` | 4000 |
| `screw_projected_000558` | screw | projected-greedy `ratio=0.005580357142857143` | 2000 |
| `screw_projected_001` | screw | projected-greedy `ratio=0.01` | 3584 |
| `bottle_legacy_k2000` | bottle | legacy `--greedy 2000` | 2000 |
| `grid_legacy_k2000` | grid | legacy `--greedy 2000` | 2000 |

公共配置：`--size 320 --calib-ratio 0.3 --thr-pct 99 --blur-sigma 2 --seed 0 --split-seed 42`，
权重 `torch_cache/wide_resnet50_2_timm.pth`。

每个 run 目录含六件套（`config.json` / `per_image.csv` / `metrics.json` / `summary.csv` / `timing.json` / `run.log`）、
`native/`、`pixel_dumps/<category>.npz` 与 `aupro.json`。汇总表：`aupro_summary.json`。

## 结果（seed 0）

| run | image AUROC | pixel AUROC | AUPRO@30% | AUPRO@10% | PRO@5% |
|---|---|---|---|---|---|
| screw k=500 | 0.9053 | 0.9678 | 0.8587 | 0.7213 | 0.7600 |
| screw k=1000 | 0.9227 | 0.9789 | 0.8919 | 0.7867 | 0.8391 |
| screw k=2000 | 0.9240 | 0.9823 | 0.9270 | 0.8315 | 0.9007 |
| screw k=4000 | 0.9151 | 0.9841 | 0.9377 | 0.8422 | 0.9073 |
| screw projected 0.558%（k=2000） | 0.9650 | 0.9811 | 0.9094 | 0.8079 | 0.8635 |
| screw projected 1%（k=3584） | 0.9619 | 0.9856 | 0.9361 | 0.8438 | 0.9100 |
| bottle k=2000 | 1.0000 | 0.9856 | 0.9463 | 0.8416 | 0.9204 |
| grid k=2000 | 0.9641 | 0.9555 | 0.8567 | 0.7568 | 0.8387 |

结论写进了 `docs/coreset-budget.md` §8：AUPRO 到 k=4000 仍未饱和（与在 k=1000 就饱和的 image AUROC 不同）；
同预算下 projected 的定位反而更差（0.9094 vs 0.9270）。

## 复现

见 `docs/coreset-budget.md` §8 末尾的两条命令（`run_experiment.py ... --dump-pixel` 后接 `pro_aupro.py`）。
AUPRO 口径与限制见 `docs/pro-aupro.md`。
