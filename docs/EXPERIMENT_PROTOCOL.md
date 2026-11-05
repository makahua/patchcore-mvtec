# 实验协议与数据集角色

## 数据集角色

- **MVTec AD 当前划分**：已用于指标分析、平滑消融、少样本曲线和鲁棒性探索，正式标为开发集。后续可继续用于调试，但其结果不得作为最终独立泛化成绩。
- **VisA PCB1**：已有探索结果，标为开发集，不用于最终外部验证。
- **VisA PCB2**：已按锁定配置完成一次外部检查；结果保存在 `experiment_data/02_visa/results_external_visa_pcb2/`，后续不可再把它当成未触碰的 holdout。
- **真正产线验证**：仍需采集独立批次/独立日期的真实工件；公开数据集验证不能替代。

## 锁定配置

| 项 | 锁定值 |
|---|---|
| backbone/权重 | WideResNet50-2；`wide_resnet50_2_timm.pth` |
| 输入尺寸 | 320 |
| 记忆库 | k-center greedy，最多 2000 patch |
| patch 后处理 | score aggregation `max`；patch 分数 blur sigma=2 |
| 校准 | 仅正常训练图中留出，`calib_ratio=0.3`，P99 阈值 |
| 随机性 | 配置 seed 与 split seed 都显式记录 |
| 主要指标 | 逐类 image/pixel AUROC，固定阈值召回/误报、延迟、峰值显存 |

## 数据清单与复现

`patchcore_demo.py` 每次运行在 `manifests/<category>.json` 写下建库、calibration 和 test 的相对路径清单。`--split-seed` 固定 calibration/train 划分；`--seed` 控制建库抽样及 coreset。少样本对照应跨训练数量共用相同 split seed 与 test 清单。

新的优化实验使用 `tools/run_experiment.py --run-dir <唯一目录> -- <patchcore_demo.py 参数...>` 启动。它将原生输出写入 `native/`，并在 run 根目录整理 `config.json`、`per_image.csv`、`metrics.json`、`summary.csv`、`timing.json`、`run.log`。旧结果可用 `--package-existing` 补生成标准产物，不会重跑模型。`per_image.csv` 标记 held-out calibration、train self-score fallback 和 test，不能把 fallback 自评分误认为留出 calibration。

MVTec 当前流程没有独立 validation split；其官方 test 已反复参与项目探索，故全部 MVTec test 结果均按 development analysis 解释。不可把同一 test 拆分重命名为独立 holdout。需要真正外部泛化结论时，使用未参与选择的新批次或未触碰外部数据。

结果目录必须独立，不覆盖历史结果。宏平均与 pooled 指标分开报告。PR-AUC 需附测试集缺陷占比；MVTec 当前测试集缺陷比例约 73%，该 PR-AUC 不能直接外推到低缺陷率产线。

## 已知限制

MVTec 测试集已反复用于探索，存在基于该测试集选择实验方向的风险。多 seed 共用测试集时，seed 标准差只描述训练/校准抽样敏感度，不是独立测试集不确定性。PCB2 只做一次锁定配置的外部检查；更强的结论需要新采集样品和独立批次。

## PCB2 一次性外部检查

锁定配置下，PCB2 共 631 张建库良品、270 张 calibration 良品、200 张测试图；image AUROC 0.8794、pixel AUROC 0.9766，P99 阈值下召回 41.0%、误报率 3.0%，推理约 18.43 ms/张。建库耗时 706.9 秒，主要来自该类别更大的正常图集合与 greedy coreset。该结果说明固定 PatchCore 配置可迁移运行，但与 MVTec 各类指标不是同口径，也不代表 VisA 全类别表现。PCB2 已揭盲并用于本项目，不再作为后续改动的独立测试集。
