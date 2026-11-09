# 与官方 PatchCore 的逐项差异（vs-upstream）

> 目的：回答「你这个实现和论文/官方仓库到底差在哪、差多少」。
> 维护约定：**每改一次实现或跑完一次对照实验，就更新对应行的「影响」列**；行内的行号指本仓库当前版本。
> 核对方式说明：`代码` = 已逐行读本仓库源码；`论文/README` = 依据论文与官方公开文档（本机无外网，未与上游逐行 diff）。

## 0. 一句话结论

本项目是**按论文思路的独立实现**，主干（WR50 layer2+3 特征、patch 级记忆库、最近邻距离、coreset、平滑后取 max）与论文一致；已知差异集中在**预处理、coreset 选点机制、边界处理、工作点定义**四处，其中两处会系统性影响与官方数字的可比性，一处（边界 wrap）是明确的实现缺陷。

## 1. 差异总表

| # | 维度 | 本项目 | 官方 / 论文 | 影响方向 | 是否已量化 | 核对方式 |
|---|---|---|---|---|---|---|
| 1 | 预处理 | 320 直接拉伸（`--preprocess stretch`，可切 letterbox / `--aspect-crop`） | 224 直接 resize；可用 `--preprocess` 对比；letterbox 与 stretch 在方图上等价 | ✅ 已量化（P0-3 v2：mask 与图像同步变换后，4:3 视野下 letterbox 的 pixel AUROC +2.43/+0.65/+6.19 点） | 论文 / 本地实测 |
| 2 | 特征层与拼接 | layer2(512) + layer3(1024)，layer3 双线性上采样到 H/8 后拼接 = 1536 维，`/8` 网格（`patchcore_demo.py:115-122`） | 同为 WR50 layer2+layer3 拼接 | 基本一致；`/8` 上 40×40=1600 patch（320 输入），官方 224 输入为 28×28=784 | ✅ 已实现一致 | 代码 |
| 3 | 邻域聚合 | 先 `avg_pool2d(3, stride=1, padding=1)`，可选再拼 9 个位移；`--neighbor-pad {zero,replicate,roll}`（zero 为默认，等价上游 `Unfold(padding=1)`；roll 仅用于复现旧结果）、`--neighbor-coreset {base,expanded}`（base 默认：先在 1536 维基础描述子上选点再展开；expanded：旧顺序，先展开 9×1536 再选点） | 3×3 邻域平均后再做邻域 patch 拼接（padding 展开，无环绕） | **已修**：zero 比 roll 同预算 image AUROC +1.93 点（0.8546→0.8739）；base 与 expanded 精度持平而建库 194→22 s、候选池内存 2.1 GB→236 MB；但 9 邻域整体仍比不开差 0.34 点、时延 2.9×，默认关 | ✅ 已量化（`docs/neighbor-ablation.md`） | 代码 + 实测 |
| 4 | coreset 选点 | 随机保留 10% → **精确贪心 k-center**，k 固定 2000（`patchcore_demo.py:165-184`） | 随机线性投影到低维（约 128）→ 贪心 k-center，按**比例**（约 1%）保留 | k 固定 2000 时：小类过采样（多于 1%）、大类欠采样（320 输入下 2,000/233k≈0.86%）；精确贪心比投影贪心慢但更准；另：旧实现「先展开 9 倍再选点」，现已可选 `--neighbor-coreset base`（先选点后展开），两者精度持平、后者建库快 8.7× | ✅ 部分：greedy vs 随机 10% = AUROC +0.75 点、28→16 ms、574→169 MB、std 0.039→0.027 | 代码 + 论文 |
| 5 | 图像分数聚合 | 平滑后取 `max`，可选 top5%（`patchcore_demo.py:199-206`） | 平滑后取 `max`（论文同义） | 概念一致；差异在平滑核（本项目 k=5, σ=2，`patchcore_demo.py:136-144`） | ✅ σ 扫描（screw）：σ=0/0.5/1/2 → image AUROC 0.9158/0.9326/0.9543/0.9553 | 代码 |
| 6 | 骨干权重 | `timm/wide_resnet50_2.racm_in1k`（RandAugment+Mixup 重训）转 state_dict，命中 318/320 层 | `torchvision IMAGENET1K_V1/V2`（官方默认 V2） | 特征分布不同 → 绝对 AUROC 与官方不可直比；**换官方 V1 的影响按类别而异**（screw image −2.12 点但 AUPRO +1.35；bottle image 打平、pixel/AUPRO 略降），且分数尺度差约 19 倍、阈值必须重标定 | ✅ 已量化（任务 C，2026-11-05：`experiment_data/06_optimization/02_weights/README.md` §9） | 源码（`torchvision/models/resnet.py:596-636`）+ 文件 SHA256 溯源 |
| 7 | 阈值 / 工作点 | 从 train/good 留出 `--calib-ratio 0.3`，取分位：demo 用 `--thr-pct 99`（=1%），`classify` 用 `--fpr 0.005`（=0.5%）（`patchcore_demo.py:285`、`classify.py:215-219`） | 官方**不定义阈值**，只报 AUROC/PRO | 「召回/过杀」是自定工作点，不能当论文指标；且本项目内部并存两套口径（见 P0-2） | ✅ 已有产线曲线：0.5% 目标 → 漏检 20.0%、复核 22.7%、实际过杀 0.2% | 代码 |
| 8 | 评测指标 | image/pixel AUROC、image PR-AUC、macro 与 pooled 分开报 | image AUROC + **PRO**（区域重叠），论文最高 0.996 为 ensemble | PRO/AUPRO 已补齐（`tools/pro_aupro.py` + `docs/pro-aupro.md`）：AUPRO@30% 0.9211、@10% 0.8068；**定位质量仍不可与论文并列**（320 vs 224、权重不同、单 seed） | ✅ 已实现（P1-3） | 论文/README + 实测 |
| 9 | 输入归一化 | ImageNet 归一化默认**关**（`--imagenet-norm` 才开） | 默认开启 | 本项目实测开反而更差：screw image AUROC 0.901→0.867（只验 1 类，别当结论） | ⚠️ 单类已量化 | 代码 |
| 10 | 记忆库存储/检索 | `.npz` float32 全量 + `torch.cdist` 分块暴力检索 | FAISS 索引（可 GPU），支持 half | 本项目 19.7 ms（2,000 patch）来自小库；库大时暴力检索会成瓶颈 | ✅ 时延/显存已入表 | 代码 + 论文 |

## 2. 已确认的实现缺陷（非「设计选择」）

1. **`torch.roll` 边界环绕**（第 3 行）：**已修（2026-11-04，P3）**。默认改为 `--neighbor-pad zero`（`F.pad` 后切片，与上游 `Unfold(padding=1)` 一致），保留 `replicate` 与 `roll`（后者只为复现旧结果）。同预算对照：image AUROC 0.8546 → 0.8739（+1.93 点），pixel 持平；详见 `docs/neighbor-ablation.md`。
2. **两套 calib/建库逻辑**：`patchcore_demo.py` 与 `classify.py` 各自实现一遍留出与贪心，`classify.py:196-199` 的注释就是「排序索引会改变贪心选点」这件事的补丁痕迹。长期必然漂移，应抽到公共模块（P2）。

## 3. 与官方数字的正确比较方式

- 不能直接说「我 0.9830 vs 论文 0.996」（旧 P99 口径是 0.9805）：口径是 **单模型 WR50 / 320 输入 / timm 权重 / 自定阈值**，论文 0.996 是 **ensemble + 官方权重 + PRO 体系**。
- 可以说的三句话：
  1. 「单模型 WR50、320 输入、2,000 patch 记忆库下，15 类 macro image AUROC 0.9830，pooled 0.8564，macro 与 pooled 差 12.7 点来自类间分数尺度（旧 P99 口径是 0.9805 / 0.8812、差 9.9 点）」（✅ 可复算）；
  2. 「与官方实现的已知差异共 10 项，其中权重差异已量化（任务 C：按类别方向不同，且分数尺度差约 19 倍），边界 wrap 已修复并量化（+1.9 点）」（✅ 本表 + docs/neighbor-ablation.md）；
  3. 「官方 0.996 是 ensemble 结果，本项目没有 ensemble，也不打算用它凑数」（✅ README 已写明）。

## 4. 待办（按本表派生）

> 更新：P0-1/P0-2/P0-3、P1-3、P3 与任务 C 已完成 —— 第 1 行预处理已量化（`docs/preprocessing-ablation.md`），第 3 行边界环绕已修并量化（`docs/neighbor-ablation.md`），第 7 行阈值口径已给出 conformal 统计（`docs/conformal-thresholds.md`），第 8 行 PRO/AUPRO 已实现（`docs/pro-aupro.md`）。

- [x] 第 1 行：letterbox / 保长宽比 vs 直接拉伸，在 screw、pill、transistor 上对照（P0-3）
- [x] 第 3 行：`roll` → `pad+unfold` 修复，在 grid 上做同预算对照（+1.93 点），并新增 `--neighbor-coreset base`（建库 8.7×、内存 8.9×）；见 `docs/neighbor-ablation.md`（P3）
- [x] 第 6 行：torchvision `IMAGENET1K_V1` 与 timm `racm_in1k` 同配置对照（任务 C，2026-11-05）：官方 V1 权重已取回并校验（SHA256 `95faca4d…`、320/320），成对跑 screw/bottle；结论按类别给（screw image −2.12 点 / AUPRO +1.35 点，bottle image 打平 / AUPRO −1.42 点），详见 `experiment_data/06_optimization/02_weights/README.md` §9
- [x] 第 8 行：实现 PRO/AUPRO（P1）
- [x] 第 7 行：统一为 `fpr=0.005` 单一口径 + conformal 区间（P0-2）
