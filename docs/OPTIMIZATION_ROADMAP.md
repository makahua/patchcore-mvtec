# PatchCore 项目优化路线图

## 状态总览（2026-11-04 复核）

| 阶段 | 状态 | 已经有的 / 还缺什么 |
|---|---|---|
| 0 基线锁定与复现记录 | ⚠️ 部分 | 有：`results.csv` 逐行配置列、`05_indexes/all_results.csv`（36 列）、`file_inventory.csv`、PyQt 演示流程固定。缺：Python/torch/CUDA/权重 SHA256 未写进结果元数据，逐文件划分清单未落盘 |
| 1 实现差异审计 | ✅ 已完成 | `docs/vs-upstream.md`（10 项差异、2 项待量化、1 项实现缺陷）；数据角色与口径见 `docs/EXPERIMENT_PROTOCOL.md` |
| 2 官方细节消融 | ⚠️ 部分 | 已做：`--blur-sigma` 扫 σ=0/0.5/1/2（pixel 峰值在 σ=1）；预处理对照 stretch vs letterbox（`docs/preprocessing-ablation.md`，MVTec 全方图故两者等价，4:3 模拟下分形态有差）。coreset 顺序与边界填充已做（`docs/neighbor-ablation.md`：zero vs roll +1.93 点，base 顺序建库 8.7×）。未做：输入分辨率 320 vs 224 的直接对照 |
| 3 评测协议与阈值可靠性 | ⚠️ 部分 | 有：`tools/threshold_stability.py`（bootstrap 敏感性）、`--calib-ratio 0.3` 留出良品、名义/实际 FPR 分开报告、split-conformal 阈值（`classify.py --conformal`，0.5% 在 18–117 张标定下 0/15 可达）、PRO/AUPRO（`tools/pro_aupro.py`，AUPRO@FPR≤30% 0.9211 ± 0.0398）。缺：多种子划分 |
| 4 PyQt 演示 | ⚠️ 部分 | 单图检测、热力图、OK/REVIEW/NG、耗时显示已有；类别/bank 切换与样例按钮未确认 |
| 5 性能与鲁棒性 | ⚠️ 部分 | 扰动：亮度（×0.8）、噪声（σ=0.05/0.10）、模糊（σ=1px）各跑冻结/重标定两组，汇总 `docs/robustness.md`；旋转/JPEG 未实现；**耗时 8 段拆解已做**（`docs/latency-breakdown.md`，端到端 35.5 ms @900²，1024² 62.4 ms）；FAISS 对照未做 |
| 6 对外材料 | ✅ 已完成 | README 压到 125 行（指标表 + 六条结论 + 设计决策 + 边界）；长尾拆到 `docs/install.md`、`docs/interview-notes.md`（坑 / 简历 / 面试问答 / 30 分钟复现 / 自检与 CI）；CI 已建（`.github/workflows/ci.yml`，numpy-only 23 条测试 + 断链检查）；`experiments.md`、`interview-qa.md` 不再单列，内容并入上述两篇 |

> 与 `docs/OPTIMIZATION_PLAN.md` 的 §0.0 表互相对应；两边口径要一起改。

---

## 目标

把当前面向面试演示的 MVTec AD 独立实现，逐步变成**可解释、可复现、可公平对照 AWS 官方实现**的项目。优化的首要目标不是追一个更高 AUROC，而是先保证每个实验都能说明白“改了什么、结果是否同口径、代价是多少”。

当前基线记录：15 类 image AUROC macro 0.9805、pixel AUROC macro 0.9709、pooled image AUROC 0.8812、AUPRO@FPR≤30% macro 0.9211（@10% 0.8068）；greedy coreset 2,000 patch，输入 320。上述数字是当前实现已有结果，不是下列计划任务的验收承诺。每类已有独立分数文件，可离线复算。

## 约束与对照原则

- 目标机器为 RTX 3050 Laptop、4 GB 显存；优先保持 320 输入、2,000 patch 可运行。
- 一次只改变一个算法因素，固定权重文件、随机种子、训练/标定/test 文件清单和输入处理。
- `macro` 与 `pooled` 必须同时报告；不把不同权重、输入尺寸、后处理或评分公式的数字直接称为提升。
- 官方仓库公开的最高结果来自其特定配置；官方 README 同时提供 WR50 示例与训练/评测命令。与本项目比较时，应按实际配置匹配，不拿 ensemble 最高值当单模型等价基线。[AWS 官方 README](https://github.com/amazon-science/patchcore-inspection/blob/main/README.md)
- 每组性能测试预热后重复测量，报告 latency median/p95、吞吐、建库时间和峰值显存；推理时间不混入读图和建库时间。

## 阶段 0：锁定当前基线与复现记录（P0）

> **状态（2026-11-04）：⚠️ 部分完成。** `results_best/results.csv` 每行已带 size/neighbors/coreset/calib/thr/seed/weights；`05_indexes/all_results.csv` 统一了 36 列；`file_inventory.csv` 是文件校验清单。**还缺**：Python/torch/CUDA/权重 SHA256 未写进结果元数据（scores JSON 只有 category/thr/thr_pct/calib/test），逐文件划分清单未单独落盘。

### 工作项

1. 为当前 `experiment_data/00_baselines/results_best` 保存一份机器可读配置：Python、torch/torchvision、CUDA、GPU、权重 SHA256、尺寸、预处理、提取层、coreset 类型/大小/种子、calibration 划分、评分公式、平滑参数。
2. 保存每类实际使用的 train、calibration、test 文件清单及划分种子；保证同一图像不跨集合。
3. 统一汇总表字段：逐类指标、macro、pooled、样本数、标定样本数、延迟、建库时间、显存。
4. 把现有 PyQt 单图演示的启动/建库流程固定下来；把 MVTec、权重、bank 作为本地资源，不提交大型数据文件。

### 验收

- 从记录文件可以准确复现某个结果目录的配置和数据划分。
- 离线报告同时显示 macro 与 pooled，不再只在 README 手工抄指标。
- 权重 SHA256 与结果元数据绑定；同一配置、同一数据划分可稳定重跑，或明确记录随机波动。

## 阶段 1：实现差异审计，先对齐 PatchCore 关键链路（P0）

> **状态：✅ 完成（2026-11-04）。** 见 `docs/vs-upstream.md`：预处理、邻域聚合（`torch.roll` 环绕缺陷）、coreset 机制、权重、阈值口径、PRO 缺失等 10 项，逐项标了影响与是否已量化。

建立 `docs/vs-upstream.md` 差异表，逐项对照 AWS 官方实现和当前脚本，标注“已一致 / 不同 / 未确认 / 影响待测”：

1. **Backbone 与输入**：WR50 权重、RGB 范围、归一化、resize/crop、插值与 antialias。
2. **Patch 特征**：官方 patchify（patch size/stride）、多层空间对齐、特征预处理/维度映射；对照当前 3×3 平均池化和 layer2/layer3 直接拼接。
3. **Coreset**：候选池抽样、随机投影/降维空间、k-center greedy 具体实现、预算和随机种子。
4. **最近邻与图像分数**：精确/近似检索、距离定义、patch score、image score、官方 reweighting 是否启用及其实现。
5. **像素热图**：插值到输入大小的时机、平滑顺序、sigma、边界行为。
6. **评测口径**：mask 对齐、image/pixel AUROC、PR-AUC、PRO、macro/pooled 聚合。

### 验收

- 差异表每项有官方代码位置、当前代码位置、实测影响及对照实验计划。
- 所有“与官方差 X 个点”的表述，注明比较配置和指标口径；配置不一致时只称为参考值。

## 阶段 2：做官方关键细节的可控消融（P1）

> **状态：⚠️ 部分完成。** 只做了实验 D 的一部分：`--blur-sigma 0/0.5/1/2` 在 screw 上扫过（image AUROC 0.9158/0.9326/0.9543/0.9553，pixel 峰值在 σ=1，见 `docs/RESULTS_OPTIMIZATION_PLAN.md`）；实验 A/B/C 未做。

每项先在 bottle、screw、wood、tile 四类进行筛查，再对有效且显存可承受的配置跑 15 类。保留当前实现作为对照组。

### 实验 A：patch 特征与维度映射

- 对照：当前 3×3 avg pool + 1,536 维拼接；官方 patchify 与特征预处理/映射配置。
- 指标：逐类 image/pixel AUROC、PR-AUC、patch 数、推理耗时、峰值显存。
- 目的：判断差异来自局部描述子还是后续打分，不预设官方路径在当前不同权重下必然更好。

### 实验 B：coreset 选择

- 对照：当前 greedy 2,000、随机 10%、官方近似 greedy/对应采样方式。
- 固定最终预算，额外记录候选池规模、选点耗时、特征维数和检索时延。
- 4 GB 卡上优先避免在 GPU 同时保留全部候选特征与检索临时矩阵；需要时将候选特征放 CPU、分块计算。

### 实验 C：image-level scoring / reweighting

- 对照：当前 patch 最大值作为图像分数；官方 scorer 及其 reweighting。
- 先验证逐图分数方向、极端样本、良品排序，再计算 AUROC；保证 calibration 和 test 完全不参与建库或阈值调优。

### 实验 D：热图后处理

- 对照：当前 patch 网格平滑与官方先上采样后平滑的完整流程。
- 指标：pixel AUROC、PRO、缺陷区域召回；重点查看细小缺陷类别，不能只看整图 AUROC。

### 验收

- 每个实验都有一行配置、结果和结论；至少 3 个 seed 报告均值与标准差。
- 只把在固定划分、重复种子中稳定改善且资源预算满足的改动合入默认配置。

## 阶段 3：补齐评测协议和阈值可靠性（P1）

> **状态：⚠️ 部分完成（2026-11-04）。** 有 `tools/threshold_stability.py`（bootstrap 敏感性诊断）、`--calib-ratio 0.3` 留出良品标定、名义目标过杀与实际观测过杀分开报告（README 产线曲线）、split-conformal 阈值、**PRO/AUPRO**（`tools/pro_aupro.py` + `docs/pro-aupro.md`：AUPRO@30% 0.9211、@10% 0.8068，tile 的 image AUROC 满分而 AUPRO 仅 0.8181），以及 **成本模型**（`report --cost FN,FP,REV --prevalence P`：validation 选点 5% × 1.5×，test 0.1814/件，比默认 0.5% × 1.2× 省 22%，见 `docs/cost-model.md`）。**还缺**：多种子划分（每类目前 1 个 seed）。

1. ~~增加 PRO 指标并明确其积分范围和像素 mask 处理方式。~~ ✅ 已完成：`tools/pro_aupro.py`（FPR 截断 0.3 与 0.1 两档、GT 掩码在原图分辨率读取、4 邻接连通域、每块等权），见 `docs/pro-aupro.md`。
2. 当前阈值稳定性 bootstrap 仅作敏感性诊断。扩大独立良品 calibration 集，并单独保存验证集；test 集不用于选阈值、NG ratio 或成本参数。
3. 用多个图像级 seed 重复划分；报告阈值分布、良品过杀、缺陷漏检、复检比例的均值/区间和分母。
4. 区分“名义目标 FPR”和“有限 calibration 样本上的实际观测 FPR”。若 0.5% 目标的样本量不足以分辨，明确标记不可可靠估计。
5. 评估固定阈值与独立重标定两种场景；不得在扰动后的 test 上重挑阈值。

### 验收

- 报表同时给出 AUROC、PR-AUC、PRO 和固定阈值工作点。
- 工作点数字附样本量与不确定性，不写成量产保证。

## 阶段 4：完善 PyQt 演示（P2）

> **状态：⚠️ 部分完成。** 单图检测、热力图、OK/REVIEW/NG 三档、耗时显示已在 `pyqt_demo.py`（7 KB）；类别/bank 切换、样例按钮、DPI 验证未确认。

1. 保留已有单图检测、热力图、分数和 OK/REVIEW/NG。
2. 增加类别/bank 选择、阈值说明、设备和推理耗时显示，以及错误提示。
3. 加载模型和 bank 后复用实例，避免每次点击都重新加载权重。
4. 提供几张良品/缺陷样例按钮，便于面试现场演示；样例图仅从本地数据读取。
5. 对 UI 调用的推理线程、结果路径及 DPI/不同屏幕尺寸做验证。

### 验收

- 干净启动命令可打开窗口；良品、缺陷各完成一次端到端检测，窗口不冻结。
- 输出能对应到保存的模型配置和阈值，不把演示阈值解释成实际产线保证。

## 阶段 5：性能与鲁棒性（P2）

> **状态：⚠️ 部分完成。** 扰动只落地了两个：亮度 ×0.8（可重标定恢复）与模糊 σ=1px（排序崩到 0.4583，重标定也救不回），见 `docs/RESULTS_OPTIMIZATION_PLAN.md`；旋转/JPEG 未实现。耗时分解、FP16、FAISS 对照均未做。

### 性能

- 先量出 feature extraction、最近邻检索、热图后处理各自耗时，再决定优化点。
- 评估 batch、分块大小、CPU/GPU bank、torch autocast/FP16 对延迟、显存和指标的影响。
- 需要 FAISS 时作为可选 backend，与 `torch.cdist` 用相同查询和库做精度/速度对照；不因用了库就宣称算法精度变化。
- 报告 warmup、同步方式、重复次数、median/p95、吞吐和峰值显存。

### 鲁棒性

- 做受控亮度/对比度、轻微模糊、压缩、旋转和背景变化实验。
- 每个扰动分成阈值冻结和独立良品重新标定两组；报告分数漂移、误报、漏检。
- 先做轻量级扰动，再扩到全类别；记录可能失效的类别和原因。

## 阶段 6：收口与对外材料（P3）

> **状态：❌ 大部分未做。** README 已有「独立简化实现」定位；`experiments.md`、`interview-qa.md` 不存在。

- README 给出“独立简化实现”的定位、官方差异、复现环境和指标口径。
- `docs/experiments.md` 记录全部成功和失败消融，保留失败实验。
- `docs/interview-qa.md` 准备记忆库、coreset、阈值、macro/pooled、失败类别和官方差异问答。
- 一键脚本只负责调用已固定命令，不把下载大数据、安装依赖和跑实验混成不可追踪的一步。

## 建议执行顺序与时间

| 顺序 | 交付物 | 粗略投入 |
|---|---|---:|
| 0 | 基线配置、数据清单、可复算结果 | 0.5–1 天 |
| 1 | 上下游实现差异审计 | 0.5–1 天 |
| 2 | 4 项关键消融（先 4 类，再全量） | 1–3 天 GPU 时间 |
| 3 | PRO 与阈值/划分可靠性 | 1–2 天 |
| 4 | PyQt 演示完善 | 0.5–1 天 |
| 5 | 性能与扰动评估 | 1–2 天 |
| 6 | README、实验记录、面试材料 | 0.5–1 天 |

**P0 完成标准**：阶段 0 和阶段 1。完成前不把项目称为“官方 PatchCore 完整复现”。

**最小高收益版本**：先完成阶段 0、阶段 1、阶段 3 的阈值可靠性与 PRO，再做阶段 2 的评分和特征消融；这几项最能提升结果可信度和面试抗追问能力。
