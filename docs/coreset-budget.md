# Coreset 预算复验（legacy fixed-k vs projected greedy）

> 运行日期：2026-11-05　｜　数据角色：**development only**
> 本文所有 MVTec 数字都来自已被多轮查看的 test 划分，只用于开发期比较，**不作为 holdout 或量产指标**；也不据此修改默认配置。

## 1. 目的

`docs/OPTIMIZATION_PLAN.md` §3.1 要求回答两个问题：

1. 固定 k 从 500 到 4000，质量、漏检、时延、建库和显存怎么变？
2. 上游那种「按比例选点」的近似思路，在这份**独立实现**（projected-greedy 近似，不是官方复现）上是否可行、代价多大？

本轮只做这两问，不碰权重对照（§3.2）、部署优化（§4.1）和阈值调参（§5）。

## 2. 协议

| 项 | 值 |
|---|---|
| 数据 | `MVTec/mvtech_anomaly_detection` |
| 输入 | `--size 320`，`--preprocess stretch`，无邻域 |
| 标定 | `--calib-ratio 0.3`（从 train/good 留出），`--thr-pct 99` |
| 划分 | `--split-seed 42`（固定，跨 seed 不变） |
| 重复 | `--seed 0/1/2`（影响建库采样与 projected 投影矩阵/起点） |
| 权重 | `torch_cache/wide_resnet50_2_timm.pth`（config.json 内含 sha256） |
| 执行 | `tools/run_experiment.py`，每次运行独立目录，六件套齐全 |
| 结果目录 | `experiment_data/06_optimization/{01_coreset_fixed,02_coreset_projected,04_representative}/` |

两种方法的实际含义（`patchcore_demo.py:216-302`）：

- **legacy fixed-k**：先随机抽 10% 候选池，再在池内做精确 k-center 贪心，`--greedy K` 直接指定记忆库大小。
- **projected greedy（独立近似实现）**：`--coreset-method projected-greedy`，全候选池先投影到 128 维（`--projection-dim`），在投影空间做近似贪心，目标大小 `k = int(n * ratio)`，`--starting-points 10` 决定初始中心。
  这是本项目自己的近似：投影矩阵与起点都由 RNG 决定，投影维度、距离计算与上游实现不同，**不能与官方数字逐项对拍**，也不称「官方复现」。
  screw 的候选池是 224 张建库图 × 1600 patch = 358 400；`ratio=0.01` → **k=3584**，`ratio=0.005580357142857143` → **k=2000**（同预算）。

两种预算分开读：

- **1% projected（k=3584）** = 当前测试配置对照，**预算不同**，只能说明「各自默认预算下的表现」，不能说明纯算法优劣。
- **projected 0.558%（k=2000）** = 与 legacy k=2000 的同预算对照，见 §3 末两行与 §5。

## 3. 结果：screw（3 seed，均值 ± 样本标准差）

| 方法 | mem | image AUROC | pixel AUROC | 漏检率@P99 | 误报率 | 算法时延 | 建库 | GPU 峰值 |
|---|---|---|---|---|---|---|---|---|
| legacy k=500 | 500 | 0.8831 ± 0.0231 | 0.9638 ± 0.0042 | 37.5% ± 3.8% | 8.1% ± 2.8% | 18.06 ± 1.06 ms | 57.5 ± 4.1 s | 171.7 MB |
| legacy k=1000 | 1000 | 0.9249 ± 0.0132 | 0.9778 ± 0.0014 | 30.8% ± 6.2% | 7.3% ± 2.4% | 18.57 ± 0.65 ms | 104.1 ± 4.8 s | 175.2 MB |
| legacy k=2000 | 2000 | 0.9231 ± 0.0065 | 0.9840 ± 0.0017 | 33.3% ± 4.2% | 4.9% ± 2.4% | 20.19 ± 0.68 ms | 201.1 ± 9.4 s | 188.8 MB |
| legacy k=4000 | 4000 | 0.9226 ± 0.0078 | 0.9856 ± 0.0014 | 36.4% ± 5.1% | 4.9% ± 0.0% | 23.42 ± 0.71 ms | 392.1 ± 20.5 s | 223.9 MB |
| projected 0.558%（k=2000，同预算） | 2000 | **0.9534 ± 0.0101** | 0.9819 ± 0.0011 | **25.2% ± 8.0%** | 4.1% ± 2.8% | 20.54 ± 0.09 ms | **29.0 ± 0.1 s** | 661.6 MB |
| projected 1%（k=3584，预算不同） | 3584 | 0.9565 ± 0.0050 | 0.9861 ± 0.0008 | 24.6% ± 6.4% | 3.3% ± 1.4% | 22.84 ± 0.68 ms | 44.5 ± 0.8 s | 661.6 MB |

逐 seed 原始值见各 `metrics.json`；seed 0 的 k=2000 取自 `00_baseline/mvtec_seed_000`（同一协议、同一 split-seed，仅与 15 类同批跑）。同预算 projected 三 seed 位于 `02_coreset_projected/screw_ratio_000558_seed_00{0,1,2}/`，`metrics.json` 中 `mem_patches=2000`、`n_calib=96`、`n_test=160`，与 legacy 一致。

## 4. 结果：bottle / grid（代表类复验）

| 类别 | 方法 | mem | image AUROC | pixel AUROC | 漏检率 | 误报率 | 时延 | 建库 | 峰值 |
|---|---|---|---|---|---|---|---|---|---|
| bottle | legacy k=2000 | 2000 | 1.0000 ± 0.0000 | 0.9855 ± 0.0002 | 3.2% ± 0.0% | 0.0% | 20.48 ± 0.51 ms | 128.9 ± 5.0 s | 188.7 MB |
| bottle | projected 1%（预算不同） | 2352 | 1.0000 ± 0.0000 | 0.9853 ± 0.0001 | 1.6% ± 0.0% | 0.0% | 21.37 ± 0.23 ms | **22.6 ± 1.0 s** | 474.8 MB |
| grid | legacy k=2000 | 2000 | 0.9708 ± 0.0067 | 0.9586 ± 0.0044 | 8.8% ± 1.8% | 3.2% ± 5.5% | 20.81 ± 0.65 ms | 163.6 ± 8.5 s | 189.0 MB |
| grid | projected 1%（预算不同） | 2960 | **0.9766 ± 0.0017** | 0.9606 ± 0.0017 | 6.4% ± 1.0% | 3.2% ± 2.7% | 22.75 ± 0.20 ms | **34.2 ± 3.1 s** | 566.4 MB |

bottle 的 image AUROC 三个 seed 都是 1.0000（83 张 test，单一 easy 类），对方法**没有区分度**；grid 与 screw 才有信息量。

## 5. 判断（可作为面试口径）

1. **固定 k 在 screw 上 1000 之后基本饱和**：image AUROC 从 0.9249（k=1000）到 0.9226（k=4000）无提升，pixel AUROC 则单调小涨（0.9778→0.9856）。继续加 k 买到的是定位分辨率，不是整图判定——这是当时基于 pixel AUROC 的推断；AUPRO 补测（§8）直接证实：k=500→4000 的 AUPRO@30% 是 0.8587 → 0.8919 → 0.9270 → 0.9377，**定位指标到 k=4000 仍未饱和**，而 image AUROC 从 k=1000 起就平了。
2. **k 的代价是超线性的**：k 从 1000 到 4000，建库 104 s→392 s（3.8×），时延 18.6→23.4 ms（+26%），显存 175→224 MB（+28%）。
3. **同预算（k=2000）对照**：projected 的 image AUROC 0.9534 ± 0.0101，比 legacy 0.9231 ± 0.0065 高 **3.0 点**。表中 ± 是 n=3 的样本标准差，不是置信区间；本次三 seed development 运行的观察均值差为 3.0 个百分点，样本量不足以声称统计显著。漏检 25.2% ± 8.0% vs 33.3% ± 4.2%；pixel AUROC 略低（0.9819 vs 0.9840，−0.2 点）；算法时延持平（20.54 vs 20.19 ms）；建库快 **6.9×**（29.0 s vs 201.1 s）；峰值显存 661.6 MB vs 188.8 MB。
   读法：同预算下它把「整图判定」做上去了、定位略降、建库快一个量级，代价是显存峰值。
4. **1%（k=3584）与 legacy k=2000 不是同预算**：那一行观察到 +3.34 点，但由于预算和方法同时变化，无法分解归因；只能写成「当前测试配置对照，预算不同」。
5. **峰值显存的口径**：`peak_gpu_mb` 取自 `torch.cuda.max_memory_allocated`，是 PyTorch **已分配 CUDA 张量**的峰值，不等于进程总 VRAM，也不等于主机 RAM。同预算（k=2000）与 1%（k=3584）的峰值都是 661.6 MB，说明它不随最终记忆库大小变化，而主要由建库阶段随候选池增长的中间张量决定（本轮只有两次运行的峰值可对比，未逐项拆解到具体张量）；4 GB 卡上仍可跑，但更高分辨率或更大类要先算清候选池。
6. **seed 噪声不可忽略**：screw k=500 的 image AUROC 标准差 0.0231（0.8592–0.9053），k≥1000 才收到 0.006–0.013；同预算 projected 的标准差是 0.0101。任何「单 seed 提升」在这套样本量下都不可信。

## 6. 未完成 / 局限

- 同预算对照只覆盖 screw（3 seed）；bottle/grid 仍未做同预算 projected，仍只有 1% 一点。
- 只做了 screw 的完整 k 曲线；bottle/grid 只做 k=2000 一点。
- pixel AUROC 基于 40×40 分数图与原图 mask 的双线性上采样，绝对值不可与 320/224 的公开数字并列。
- AUPRO 只补了 seed 0 的 8 个关键点（`07_aupro_backfill/`，见 §8），没有跨 seed；bottle/grid 也只有 legacy k=2000 一点；阈值固定 P99，未做阈值—预算联合优化。
- 峰值显存只记 `torch.cuda.max_memory_allocated`（PyTorch 张量口径），不是进程总 VRAM/RAM 的实测值。
- `experiment_data/06_optimization/00_protocol_smoke/` 六件套不全（只有 `native/` 与 `run.log`），保留原样未重建。

## 7. 复算

```powershell
# 固定 k（以 screw、seed 1、k=1000 为例）
.venv\Scripts\python.exe tools\run_experiment.py `
  --run-dir experiment_data/06_optimization/01_coreset_fixed/screw_k1000_seed_001 -- `
  --data MVTec/mvtech_anomaly_detection --category screw --size 320 `
  --greedy 1000 --calib-ratio 0.3 --thr-pct 99 --split-seed 42 --seed 1 `
  --weights torch_cache/wide_resnet50_2_timm.pth

# projected greedy 1%（k=3584，预算与 legacy k=2000 不同）
.venv\Scripts\python.exe tools\run_experiment.py `
  --run-dir experiment_data/06_optimization/02_coreset_projected/screw_ratio_001_seed_001 -- `
  --data MVTec/mvtech_anomaly_detection --category screw --size 320 `
  --coreset-method projected-greedy --coreset-ratio 0.01 --projection-dim 128 `
  --starting-points 10 --calib-ratio 0.3 --thr-pct 99 --split-seed 42 --seed 1 `
  --weights torch_cache/wide_resnet50_2_timm.pth

# projected greedy 同预算（k=2000；ratio = 2000 / 358400 = 0.005580357142857143）
.venv\Scripts\python.exe tools\run_experiment.py `
  --run-dir experiment_data/06_optimization/02_coreset_projected/screw_ratio_000558_seed_001 -- `
  --data MVTec/mvtech_anomaly_detection --category screw --size 320 `
  --coreset-method projected-greedy --coreset-ratio 0.005580357142857143 `
  --projection-dim 128 --starting-points 10 --calib-ratio 0.3 --thr-pct 99 `
  --split-seed 42 --seed 1 --weights torch_cache/wide_resnet50_2_timm.pth

# 汇总（跨 seed 均值 ± 标准差；只读，跳过 --limit 冒烟运行）
.venv\Scripts\python.exe tools\summarize_runs.py --root experiment_data/06_optimization
```

## 8. AUPRO 补测（2026-11-05 追加，seed 0）

§6 原先写着「未计算 AUPRO」，这条已补上：对 8 个关键 run 重跑推理并落 `--dump-pixel`，再用
`tools/pro_aupro.py` 计算区域级指标。**重跑得到的 image / pixel AUROC 与原始 run 逐位一致（差值 0.0）**，
说明补测没有改变被测对象；AUPRO 是我们之前缺的那一维。

| run | mem | image AUROC | pixel AUROC | AUPRO@30% | AUPRO@10% | PRO@5%FPR | 区域数 |
|---|---|---|---|---|---|---|---|
| screw legacy k=500 | 500 | 0.9053 | 0.9678 | 0.8587 | 0.7213 | 0.7600 | 140 |
| screw legacy k=1000 | 1000 | 0.9227 | 0.9789 | 0.8919 | 0.7867 | 0.8391 | 140 |
| screw legacy k=2000 | 2000 | 0.9240 | 0.9823 | 0.9270 | 0.8315 | 0.9007 | 140 |
| screw legacy k=4000 | 4000 | 0.9151 | 0.9841 | 0.9377 | 0.8422 | 0.9073 | 140 |
| screw projected 0.558%（k=2000，同预算） | 2000 | 0.9650 | 0.9811 | 0.9094 | 0.8079 | 0.8635 | 140 |
| screw projected 1%（k=3584，预算不同） | 3584 | 0.9619 | 0.9856 | 0.9361 | 0.8438 | 0.9100 | 140 |
| bottle legacy k=2000 | 2000 | 1.0000 | 0.9856 | 0.9463 | 0.8416 | 0.9204 | 70 |
| grid legacy k=2000 | 2000 | 0.9641 | 0.9555 | 0.8567 | 0.7568 | 0.8387 | 174 |

产物：`experiment_data/06_optimization/07_aupro_backfill/<run>/{metrics.json,aupro.json,pixel_dumps/,summary.csv,...}`，
汇总表 `07_aupro_backfill/aupro_summary.json`。

三条读法：

1. **AUPRO 与 image AUROC 的饱和点不同**。image AUROC 在 k=1000 就到顶（0.9249）并在 k=4000 反而略降（0.9226）；
   AUPRO@30% 一路涨到 0.9377（k=4000 比 k=500 高 7.9 点）。**加预算买到的是定位质量，不是整图判定**，
   这条从推断变成了实测。
2. **同预算下 projected 输在定位**：k=2000 时 projected AUPRO 0.9094 vs legacy 0.9270（−1.76 点），
   而它的 image AUROC 高 3.0 点。两个指标指向相反——写结论时必须同时给，不能只挑赢的那个。
   1% projected（k=3584，AUPRO 0.9361）与 legacy k=4000（0.9377）在定位上接近，但预算仍不同。
3. **grid 的定位明显弱于它的 pixel AUROC 所暗示的水平**：pixel 0.9555 看着还行，AUPRO 只有 0.8567
   （174 个区域、缺陷细而密，40×40 分数图的分辨率不够）。这类「面积加权好看、区域级露馅」正是 PRO 要抓的东西，
   与 `docs/pro-aupro.md` 的 tile 现象一致。

复算：

```powershell
.venv\Scripts\python.exe tools/run_experiment.py `
  --run-dir experiment_data/06_optimization/07_aupro_backfill/screw_legacy_k0500 -- `
  --data MVTec/mvtech_anomaly_detection --category screw --size 320 --greedy 500 `
  --calib-ratio 0.3 --thr-pct 99 --blur-sigma 2 --seed 0 --split-seed 42 `
  --weights torch_cache/wide_resnet50_2_timm.pth `
  --dump-pixel experiment_data/06_optimization/07_aupro_backfill/screw_legacy_k0500/pixel_dumps

.venv\Scripts\python.exe tools/pro_aupro.py `
  --dumps experiment_data/06_optimization/07_aupro_backfill/screw_legacy_k0500/pixel_dumps `
  --data MVTec/mvtech_anomaly_detection --category screw `
  --out experiment_data/06_optimization/07_aupro_backfill/screw_legacy_k0500/aupro.json
```

AUPRO 的口径与限制沿用 `docs/pro-aupro.md`（40×40 分数图双线性上采样、原图 GT 连通域 4 邻接、
AUPRO@30% 截断）；本轮只做 seed 0，不能据此声称跨 seed 稳定。
