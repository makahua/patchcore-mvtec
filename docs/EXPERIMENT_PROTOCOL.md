# 实验协议与数据集角色

## 冻结协议（2026-11-06）

先固定配置与验收规则，再运行实验；后续实验不得逐项改协议。

| 项目 | 固定值 |
|---|---|
| 权重 | 现有 timm `racm_in1k`（`torch_cache/wide_resnet50_2_timm.pth`，SHA256 见 `experiment_data/05_indexes/weights_sha256.txt`） |
| 输入 | 320×320，stretch，不做 ImageNet 归一化 |
| 记忆库 | legacy greedy：先抽取 10% 候选，再选最多 2000 个中心 |
| 特征对照 | `2+3`（1536 维，/8 网格）对 `1+2`（768 维，/4 网格） |
| 分数处理 | 高斯平滑 σ=2，图像聚合 `max` |
| 良品划分 | 70% 建库、30% calibration，`split-seed=42` |
| 重复次数 | coreset seed 0、1、2 |
| 主判定口径 | 经验 P99.5，`fpr=0.005`，灰区倍数 1.2 |
| 次判定口径 | P99，仅用于连接历史结果与敏感性分析 |
| 精度与批量 | FP32、batch=1 |
| 数据角色 | MVTec development-only |

三个 seed 共用完全相同的建库、标定、测试清单，只改变 coreset 随机性。所有参数显式记录在 `config.json`；除 Git commit 外还记录 `source_sha256`（`patchcore_demo.py` 与 `tools/run_experiment.py` 的实际内容哈希），防止未提交代码让记录失真。

### 三档判定定义

- `score < thr_ok`：OK。
- `thr_ok ≤ score < thr_ng`：REVIEW。
- `score ≥ thr_ng`：NG。
- `thr_ok` 由该运行的 calibration 良品分数计算（经验分位）。
- `thr_ng = thr_ok × 1.2`。

名义 `fpr=0.005` 指**超过 OK 阈值的比例，即 REVIEW+NG**；而「过杀率」另定义为良品被判 NG 的比例。二者不能混称，报告时必须写清分子/分母。

P99.5 是可计算的经验分位，但现有标定样本量不足以提供 0.5% 的有限样本 conformal 保证。conformal 可达水平单独报告（见 `docs/conformal-thresholds.md`）；bootstrap 只用于描述敏感性，不替代保证。

## 统计口径

- 每个 seed 先算 15 类 macro，再对三个 macro 取均值与样本标准差（`ddof=1`）。
- 每个 seed 分别计算 pooled，再汇总；**不把三个 seed 的重复测试图片拼成更大的样本集**。
- 配对收益按 `1+2 − 2+3` 逐 seed 计算，再给差值均值与范围。
- 该标准差描述 coreset 随机性，不代表独立测试集不确定性。

## 验收矩阵

> **15 类 × 3 seed × 2 层组合 = 90 个类别运行。**

任何类别或 seed 缺失都不能宣布完成验收；已完成部分可以报告，但不据此切换默认。默认切换门槛见 `docs/OPTIMIZATION_PLAN.md` 第三节（完整性、配对均值、单类退化 ≤0.03、15 类算法段均值 ≤25 ms，外加漏检/过杀不得上升的业务保护）。

## 数据集角色

- **MVTec AD 当前划分**：已用于指标分析、平滑消融、少样本曲线和鲁棒性探索，正式标为开发集。后续可继续用于调试，但结果不得作为最终独立泛化成绩。
- **VisA PCB1**：已有探索结果，标为开发集，不用于最终外部验证。
- **VisA PCB2**：已按旧锁定配置完成一次外部检查，结果在 `experiment_data/02_visa/results_external_visa_pcb2/`，后续不可再把它当成未触碰的 holdout；也不作为新配置的外部验证成绩。
- **真正产线验证**：仍需采集独立批次/独立日期的真实工件；公开数据集验证不能替代。

## 数据清单与复现

`patchcore_demo.py` 每次运行在 `manifests/<category>.json` 写下建库、calibration 和 test 的相对路径清单。`--split-seed` 固定 calibration/train 划分；`--seed` 控制建库抽样与 coreset。

实验统一用 `tools/run_experiment.py --run-dir <唯一目录> -- <patchcore_demo.py 参数...>` 启动，产出六件套：`config.json`、`per_image.csv`、`metrics.json`、`summary.csv`、`timing.json`、`run.log`，并把原生输出放进 `native/`。`--save-bank <run>/banks` 保存记忆库与元数据（层组合、特征维度、网格、预处理、聚合、σ、阈值、权重），供扰动实验复用，不必每种聚合重新建库。

辅助工具：

- `tools/verify_protocol.py --root <目录>`：校验 train/calibration/test 两两互斥、跨 seed 清单一致、类别齐全。
- `tools/pair_stats.py --root <目录> --baseline 23 --candidate 12`：逐 seed macro、均值±std、配对差、单类退化、三档判定与门槛表。
- `tools/pro_aupro.py --dumps <run>/pixel_dumps ...`：区域级 PRO/AUPRO。
- `tools/summarize_features.py` / `tools/summarize_runs.py`：只读汇总。
- `tools/export_thresholds.py`：从 10_pair 汇总逐类 `thr_ok`/`thr_ng` 到 `docs/thresholds.json`，`--check` 供 CI 校验；阈值与类别绑定，不可跨类或跨权重迁移。

旧结果可用 `--package-existing` 补生成标准产物，不会重跑模型。`per_image.csv` 标记 held-out calibration、train self-score fallback 和 test，不能把 fallback 自评分误认为留出 calibration。

MVTec 没有独立 validation split；其官方 test 已反复参与探索，故全部 MVTec test 结果按 development analysis 解释。不可把同一 test 拆出一半改名为「独立 holdout」。

结果目录必须独立、不覆盖历史结果。宏平均与 pooled 分开报告。PR-AUC 需附测试集缺陷占比；当前测试集缺陷比例约 73%，不能直接外推到低缺陷率产线。

## 已知限制

MVTec 测试集已反复用于探索，存在基于该测试集选择实验方向的风险；多 seed 共用同一测试集时，seed 标准差只描述训练/校准抽样敏感度。标定样本每类 18–117 张，0.5% 的有限样本保证在本数据上普遍不可达。CPU 内存峰值随候选池线性增长（`1+2` 在最大类别约 8 GB），本机只能单进程串行跑。

## PCB2 一次性外部检查（旧配置）

旧锁定配置下，PCB2 共 631 张建库良品、270 张 calibration 良品、200 张测试图；image AUROC 0.8794、pixel AUROC 0.9766，P99 阈值下召回 41.0%、误报率 3.0%，推理约 18.43 ms/张。建库 706.9 秒。该结果说明固定 PatchCore 配置可迁移运行，但与 MVTec 各类指标不是同口径；PCB2 已揭盲，不再作为后续改动的独立测试集。
