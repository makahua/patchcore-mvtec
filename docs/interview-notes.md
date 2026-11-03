# 面试笔记：坑、简历、问答、复现路径、自检与 CI

> README 里只留结论，细节与话术都在这里。

## 六个必踩的坑

1. **mask 尺寸不一致** —— ground_truth 的 mask 是原图尺寸，而分数图是 /8 分辨率。脚本里已经 resize 到 patch 尺寸；自己改代码时别在这上面翻车（这是 pixel AUROC 算成 0.5 的最常见原因）。
2. **阈值别用测试集定，也别用「建库图自评分」定** —— 产线拿不到缺陷样本，阈值只能用 good 分布；
   但更隐蔽的是**自匹配偏差**：建库图自己的 patch 就在库里，最近邻距离≈0，拿它们算 P99 会把阈值压低（初版误报 28% 就是这么来的）。
   正确做法是留出一部分 good 图只做标定（`--calib-ratio 0.3`）。
3. **显存爆炸** —— `train/good` 有的类别 200+ 张，全量 patch 堆在 GPU 上会 OOM。脚本先在 CPU 上建库再抽样搬 GPU。
4. **图像拉成正方形** —— 脚本直接 resize 到 256×256，会轻微变形；要严谨就改成 keep-aspect + pad，并在 README 里说明。
5. **好图被判坏（误报）** —— good 的 test 图也可能有高响应（反光、边缘）。所以要看 `false_positive_rate`，不能只看 AUROC。
6. **权重下载失败** —— 报 `URLError` 就是墙。设代理或用 `TORCH_HOME` 预先放权重。

---

## 简历怎么写

**版式 A（3 行，推荐）**

> **【工业质检】** MVTec AD 少样本无监督缺陷检测（PatchCore）　｜　个人项目　｜　2026.10　｜　代码开源（见 GitHub）
> - **方法：** WideResNet50 的 layer2/layer3 特征融合出 /8 分辨率 patch 描述子，构建正常样本记忆库，测试时以最近邻距离作为异常分数；用高斯平滑抑制单点噪声，再做 image-level 判定。
> - **工程：** 用随机抽样 / k-center 贪心两种 coreset 策略控制记忆库规模，把每类约 27 万 patch 压到 2,000（<1%），单张 20 ms（RTX 3050 Laptop）、显存 0.19 GB；同分辨率下比随机抽样快 42%、省一半显存且精度反高 0.7 点。
> - **结果：** MVTec AD 15 类共 1,725 张测试图，image AUROC 0.980（15 类均值）、pixel AUROC 0.971、图像级 PR-AUC 0.993；自研 OK / 复检 / NG 三档判定（每类阈值由 18–117 张留出良品标定，名义目标过杀 0.5%）：当前测试划分观测到良品直接判废 0.2%、转复核 2.1%，缺陷漏检 20.0%、灰区再捞回 30%。这些低误报工作点来自小标定集，应作为探索性结果，不表述为产线保证。

（以上为**已跑出的真实版本**，简历里就是用这一版。注意第一句的 `0.980（15 类均值）` 是 macro 口径，
括号不能省——省了就和 pooled 0.8812 打架，见上一节 ⚠ 提示。）

**版式 B（2 行，版面紧张时）**

> **【工业质检 · 无监督】** 少样本异常检测（PatchCore）　｜　个人项目　｜　【2026.xx】
> - 基于 WideResNet 多尺度特征 + 正常样本记忆库的最近邻异常评分，MVTec AD 15 类平均 image AUROC【0.9xx】/ pixel AUROC【0.9xx】，单张【xx ms】；对比随机抽样与 k-center 贪心两种记忆库压缩策略对精度—显存的影响，阈值只由 good 样本分布确定（P【99】），漏检率【x%】、误报【x%】。

**照抄时改三处**：日期、AUROC 真值、耗时/显存真值。**没跑出真值就别把这行加上去。**

### 需要抄进简历的 4 个字段

`experiment_data/00_baselines/results_best/results.csv` 里这几列就是简历要的：

| CSV 列 | 简历里写什么 | 注意 |
|---|---|---|
| `img_auroc` | image AUROC 0.9xx | **必须标「15 类均值」**（macro），否则与 pooled 冲突 |
| `pixel_auroc` | pixel AUROC 0.9xx | 同样按类平均 |
| `latency_ms` | 单张 xx ms | 注明机器型号与分辨率 |
| `mem_patches` + `peak_gpu_mb` | 记忆库 xx patch、显存 x.x GB | 显存取 `peak_gpu_mb`，别用 nvidia-smi 的整卡占用 |
| `recall_at_thr` / `false_positive_rate` | 漏检 x% / 误报 x% | 这两列是**单阈值**结果；三档判定的数要跑 `classify.py report` |

**另外，`experiment_data/00_baselines/results_best/results.csv` 现在会自己记录配置**（`size` / `neighbors` / `imagenet_norm` /
`coreset` / `coreset_k` / `coreset_ratio` / `calib_ratio` / `thr_pct` / `seed` / `weights`）。

之前这几个字段是不写的，结果仓库里同时躺着 `experiment_data/00_baselines/results/`、`experiment_data/00_baselines/results_fix/`、`experiment_data/00_baselines/results_greedy/`、
`experiment_data/00_baselines/results_320/`、`experiment_data/00_baselines/results_best/`、`experiment_data/00_baselines/results_neigh/`、`experiment_data/00_baselines/results_normtest/` 七个目录，光看 CSV
根本分不清哪个是 256 哪个是 320、哪个开了邻域——事后只能靠显存和时延反推。
现在每行自带配置，**任何一个结果目录都能自证**。

---

## 面试会问什么

1. **“为什么不直接训分类网络？”** —— 缺陷类型不可穷举、样本极少；无监督只用 good 样本，未知缺陷也能给分数。
2. **“记忆库为什么不全存？”** —— patch 数量 = 图数 × H/8 × W/8，200 张图就 20 万 patch，检索和显存都撑不住；coreset 保留代表性 patch，精度损失很小。
3. **“为什么要 3×3 邻域平均 / 高斯平滑？”** —— patch 级特征对微小位移敏感，邻域平均提升局部一致性，减少单点噪声造成的误报。
4. **“阈值怎么定，为什么不用测试集？”** —— 产线拿不到缺陷样本；用 good 的分数分布取高分位（P99），再按漏检/过杀成本调整分位。
5. **“AUROC 高就能上线吗？”** —— 不能。要看固定阈值下的漏检率/过杀率、单张耗时是否匹配节拍、以及换批次/换光照后的稳定性。
6. **“怎么和 Halcon 的方案比？”** —— Halcon 强在标定好、光照稳、缺陷形态已知的快速落地（模板匹配 + Blob + 卡尺）；无监督异常检测强在缺陷未知、样本少。产线上常常是「传统算子做定位和尺寸 + 异常检测做外观」，两者组合。

---

## 30 分钟复现路径

数据 5.27 GB 和权重 267 MB 都不在仓库里（MVTec 许可证禁止再分发、GitHub 也有单文件上限），
所以仓库只放**代码 + 结果表 + 逐图分数**。想验证结论，最短路径是这两条：

| 想验证什么 | 要准备什么 | 命令 | 耗时 |
|---|---|---|---|
| 三档判定那 15 行 + 两个 AUC 口径 | **只要有 numpy** | `python classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2` | 1 秒 |
| 某类能跑到同样 AUROC | 数据集 + timm 权重 | `python patchcore_demo.py --data E:/MVTec --category bottle --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce` | bottle 约 2–3 分钟（受设备负载影响） |

> 第二条命令的预期值（bottle，RTX 3050 Laptop）：`img_auroc 1.0000 / pixel 0.9857 / threshold 37.9943 / mem 2000 / peak 188.7 MB`。
> 时延会在 17–24 ms 之间抖动（笔记本 GPU 动态频率），**别把时延当复现失败的判据，看 threshold 和 AUROC**。

## 进阶

- 加 **PaDiM** 对比（多元高斯 + 马氏距离）：改动约 50 行，能讲“为什么 PatchCore 比 PaDiM 强：不受高斯假设约束”。
- 把 `patchcore_demo.py` 包成 **FastAPI 服务**（本机毕设那套 backend 骨架可复用）：多一句「异常检测服务化」。
- 换**自己的工件**：拍 30 张正常螺栓/PCB，跑同一套流程——这一步能把“公开数据集”升级成“真实工件验证”，面试杀伤力翻倍。

---

## 自检与 CI

```powershell
python -m pytest tests/ -q        # 有 pytest（23 条）
python tests/test_metrics.py      # 没 pytest 也能直接跑，纯 numpy，15 条
python tests/test_pro.py          # PRO/AUPRO 自检，8 条，需 pillow
```

CI 在 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)，每次 push / PR 跑三件事：

| job | 做什么 | 为什么这么设计 |
|---|---|---|
| `metrics`（py3.10 + 3.12） | 23 条测试（15 指标/记忆库/成本 + 8 PRO/AUPRO）+ `classify.py report` 冒烟，钉住 macro `0.9805` / pooled `0.8812` | **只装 numpy + pytest**，不装 torch——能在十几秒内跑完，且证明「核心指标不依赖深度学习框架」 |
| `docs` | `tools/check_links.py` 检查所有 md 的相对链接 | README 曾链到不存在的 `experiment_data/README.md`，这种断链只有点进去才发现 |

本地跑不了 pytest 也不影响（开发机的 PyPI 可能被代理挡住）：`python tests/test_metrics.py` 自带运行器，同样 15/15；PRO 那组用 `python tests/test_pro.py`，8/8。

`tests/test_metrics.py` 只依赖 numpy，共 15 条，钉的是**面试官唯一会当场复算的那几个数**：

- `auc()` 是手写的 rank 法（为了不依赖 sklearn）。并列值如果按 argsort 名次硬算，
  所有指标会整体偏一点却**不报错**——所以专门有「全并列应为 0.5」「打乱顺序结果不变」
  「与两两比较定义式对拍」三条。
- `auprc()` 的单调性（排序更好的分数必须拿到更高 PR-AUC）。
- `check_bank_dims()`：记忆库维度必须和 meta 里记的 `neighbors` 自洽。
  这条是 **BUG-1 的回归测试**——早期 `classify.py` 建库时没把 `--neighbors` 传下去，
  建库恒为 1536 维、查询却是 13824 维，要到推理才抛一个跟根因无关的 cdist 维度错误。


