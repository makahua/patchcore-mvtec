# 特征层消融：2+3、1+2、1+2+3 三个组合的实测对照（任务 D）

> **数据角色：development-only。** MVTec AD 的 test 集在本项目里已被反复用于开发决策，本轮结果只能作为开发集对照，不能宣称独立 holdout 或生产泛化。
>
> 目录：`experiment_data/06_optimization/03_features/`；每次运行一个目录，六件套（`config.json` / `per_image.csv` / `metrics.json` + `summary.csv` / `timing.json` / `run.log`），另有 `native/`、`pixel_dumps/`、`aupro.json`。汇总表由只读工具 `tools/summarize_features.py` 生成。

## 1. 要回答的三个问题

1. 把 layer1（256 通道、/4 分辨率）拉进来，image / pixel / 区域级指标是涨还是跌？
2. 再加 layer3（1024 通道、/16）凑成 1792 维，是否还会继续涨？
3. 涨的部分值不值它的建库时间、显存与推理时延？

## 2. 固定协议

| 项 | 值 |
|---|---|
| 数据根 | `MVTec/mvtech_anomaly_detection` |
| 骨干 | timm `wide_resnet50_2.racm_in1k`（`torch_cache/wide_resnet50_2_timm.pth`，SHA256 `1810845932…7dc64`，命中 318/320 层） |
| 输入 | 320×320，直接拉伸（不用 letterbox），无 ImageNet 归一化 |
| 记忆库 | coreset 贪心，`k = 2000`（三个组合同一预算，可公平比较） |
| 标定 | `--calib-ratio 0.3` 留出良品，阈值 `--thr-pct 99` |
| 划分 | `--split-seed 42`；运行 `--seed 0/1/2` |
| 类别 | screw（224 张建库）、toothbrush（42）、pill（267） |
| 评测 | image AUROC、pixel AUROC（40×40 或 80×80 上采样回原图）、AUPRO@30% / @10% / PRO@5%FPR（`tools/pro_aupro.py`） |
| 计时 | 算法段（`patch_scores` 内部，不含读图解码），与 `results.csv` 口径一致 |

## 3. 实现与网格

`Features` 直接把骨干拆成 `stem / layer1 / layer2 / layer3`，`--layers` 控制融合：

| 组合 | 融合方式 | 维度 | 网格 | patch/图 | 候选池 @320 |
|---|---|---|---|---|---|
| `2+3`（默认） | layer3 双线性上采样到 layer2 的 /8 网格后拼接 | 1536 | 40×40 | 1600 | 1600 × 图数 |
| `1+2` | layer2 双线性上采样到 layer1 的 /4 网格后拼接 | 768 | 80×80 | 6400 | **4×** |
| `1+2+3` | layer2、layer3 都上采样到 /4 后与 layer1 拼接 | 1792 | 80×80 | 6400 | 4× 且更宽 |

**口径更正。** 第一版把 layer1 用 `avg_pool2d(2,2)` 降到 /8，等于只做通道拼接、把新增低层的空间分辨率丢掉；那两次试跑（`_invalid_layer1_downsampled/`）只作调试记录，**不计入任何结论**。现行实现保持 layer1 原网格，符合"加低层必须付出 4 倍 patch 代价"的对照本意。

## 4. 结果（均值，n 为 seed 数）

| 组合 | 类别 | n | image AUROC | pixel AUROC | AUPRO@30% | AUPRO@10% | PRO@5% | 召回 | 过杀 | 算法时延 | 建库 | GPU 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `2+3` | screw | 3 | 0.9231 | 0.9840 | 0.9293 | 0.8331 | 0.8969 | 0.6667 | 0.0488 | 19.72 ms | 196 s | 188.0 MB |
| `2+3` | toothbrush | 3 | 0.9426 | 0.9903 | 0.9076 | 0.7291 | 0.8301 | 0.9778 | 0.3055 | 25.29 ms | 36 s | 188.0 MB |
| `2+3` | pill | 3 | 0.9519 | 0.9764 | 0.9508 | 0.8767 | 0.9368 | 0.6903 | 0.0000 | 19.12 ms | 160 s | 188.7 MB |
| `1+2` | screw | 3 | **0.9737** | 0.9897 | **0.9523** | 0.9042 | 0.9366 | 0.8291 | 0.0488 | 25.06 ms | 622 s | 229.6 MB |
| `1+2` | toothbrush | 3 | **0.9750** | 0.9893 | **0.9399** | 0.8323 | 0.9341 | 1.0000 | 0.3333 | 25.02 ms | 70 s | 229.6 MB |
| `1+2` | pill | 3 | **0.9623** | 0.9608 | **0.9662** | 0.9260 | 0.9605 | 0.6643 | 0.0128 | 24.73 ms | 351 s | 229.6 MB |
| `1+2+3` | screw | 1 | 0.9705 | 0.9841 | 0.9276 | 0.8627 | 0.9014 | 0.9076 | 0.0732 | 37.11 ms | 1315 s | 354.5 MB |
| `1+2+3` | toothbrush | 3 | 0.8824 | 0.9912 | 0.9315 | 0.7979 | 0.9386 | 1.0000 | 0.3611 | 37.71 ms | 179 s | 354.5 MB |
| `1+2+3` | pill | 0 | — | — | — | — | — | — | — | — | — | — |

`1+2+3 / screw` 只有 1 个 seed、`1+2+3 / pill` 没有数据，原因见第 6 节；这两格**不参与"某组合更好"的结论**。

逐 seed（image AUROC / AUPRO@30%）：

- `2+3` screw 0.9240/0.9291/0.9162，AUPRO 0.9270/0.9356/0.9252
- `1+2` screw 0.9703/0.9764/0.9744，AUPRO 0.9562/0.9515/0.9493
- `1+2+3` screw 0.9705（seed0）
- `1+2+3` toothbrush 0.8806/0.8861/0.8806，AUPRO 0.9328/0.9313/0.9304 —— 三个 seed 几乎不动，**不是 seed 噪声**
- `2+3` toothbrush 0.9417/0.9361/0.9500；`1+2` toothbrush 0.9583/0.9833/0.9833

## 5. 判断

1. **加 layer1（`1+2`）在三个类上都是净收益，但幅度不同。** image AUROC +5.1 / +3.2 / +1.0 点（screw / toothbrush / pill），AUPRO +2.3 / +3.2 / +1.5 点。screw 的收益最大也最合理：螺纹是细密周期结构，/4 网格才分辨得出螺距，/8 网格把两个螺峰平均进同一个 patch。
2. **代价是 2–3 倍建库与 +41.6 MB 峰值显存。** 候选池 4 倍、维度减半，净内存约 2 倍；建库时间 622 s vs 196 s（screw）、351 vs 160（pill）。算法时延只 +5 ms（+25%），因为骨干 forward 不变，多出来的只是更大网格的 `cdist` 与滑窗。
3. **`1+2+3` 相对 `1+2` 没有收益，反而在 image 级退化。** screw AUPRO −2.5 点、image −0.3 点；toothbrush image −9.3 点（0.9750 → 0.8824）而 pixel AUROC 反而 +0.2 点。解释必须谨慎：在**同样 2000 个中心**下，1792 维空间里 k-center 覆盖更稀，加上 layer3 的高层语义通道会把局部差异压平，image 级的 max 聚合因此更容易被单个高响应 patch 带偏；pixel 级仍有定位能力，说明掉的是"整图排序"而不是"区域可分"。这是假设，不是已证机制。
4. **pixel AUROC 与 AUPRO 会给出相反的排序。** pill 上 `1+2` 的 pixel AUROC 比 `2+3` 低 1.6 点，但 AUPRO 高 1.5 点——AUPRO 按区域等权、只看 FPR≤30% 的那段，pixel AUROC 被大片良品背景主导。对外报指标时必须同时给两者，且带截断说明。
5. **toothbrush 的过杀率 30–36% 与层组合无关**，三种组合都在 0.25–0.5 之间；也就是说这里的阈值问题比特征问题更大，不能拿过杀率去论证"哪层更好"。

## 6. 资源与失败记录（如实）

- `1+2+3 / screw / seed0`：候选池 1,433,600 × 1792 ≈ 9.6 GB，可用内存 9.6 GB 时跑通，建库 1315 s、总墙钟 1348 s、GPU 峰值 354.5 MB。
- `1+2+3 / screw / seed1`：同配置启动后可用物理内存只剩 2.4 GB（页文件 35.9 GB），16 小时墙钟只累计 3503 s CPU，`run.log` 无任何进度行，人工终止。这是**内存换页 thrash**，不是数值或代码问题（同配置 seed0 已完成）。
- `1+2+3 / screw / seed2`、`1+2+3 / pill / seed0-2`：候选池 9.6 GB / 11.4 GB，在本机（物理 15.9 GB）没有安全余量，**未运行**，记 `not_run_resource`；不把它们当作"精度更差"的证据。
- `1+2+3 / toothbrush` 三个 seed 都跑通（池子仅 1.9 GB，176–186 s/run）。
- 失败的尝试与状态都留在 `sweep_status.json` 与 `_aborted/` 里（含被 argparse 拒绝的 7 个 `1+23` 目录，是驱动脚本拼参数的低级 bug，已修，不计入数据）。

## 7. 结论与建议

- **默认配置不改**：仍是 `2+3`、320、greedy 2000。理由是它在 2000 中心预算下给出最稳定的综合表现，建库 36–196 s，显存 188 MB，适合 4 GB 卡。
- 如果目标是把 screw 这类周期纹理做扎实，`1+2` 是**有数据支持**的选项：image +5.1 点、AUPRO +2.3 点，代价是建库 3 倍、显存 +42 MB。作为可选项写进文档即可，不做默认。
- `1+2+3` 不建议：没有精度收益，成本翻倍，且在 toothbrush 上显著退化。
- 面试口径：可以说"我做了低层特征的对照，`1+2` 在 screw 上带来 5 个点的 image AUROC，但建库时间翻三倍，所以默认仍守 2+3"；不能说"多层拼接一定更好"。

## 8. 局限

- 三个类别，不是 15 类横扫；screw/toothbrush/pill 是刻意选的"细结构/小目标/高反光"三个代表。
- test 集 development-only，且本轮部分类别的默认配置数字与 README 主表（`results_best`，早期快照）不完全一致，跨表比较只能看相对趋势。
- `1+2+3` 的 screw 只有 1 个 seed、pill 完全缺失，不能做跨 seed 统计。
- AUPRO 用 40×40 / 80×80 分数图双线性上采样回原图，分辨率差异会让细缺陷区域被平滑；跨组合比较只看同协议内的排序。
- 建库时间含首次权重加载之外的部分，但受磁盘与页缓存影响，同配置不同次可能差 ±10%。

## 9. 复现

```powershell
# 单个组合
.\.venv\Scripts\python.exe tools\run_experiment.py `
  --run-dir experiment_data\06_optimization\03_features\layer12_screw_seed_000 -- `
  --data MVTec/mvtech_anomaly_detection --category screw --layers 1+2 `
  --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --split-seed 42 --seed 0 `
  --weights torch_cache/wide_resnet50_2_timm.pth `
  --dump-pixel experiment_data\06_optimization\03_features\layer12_screw_seed_000\pixel_dumps

# 区域级指标
.\.venv\Scripts\python.exe tools\pro_aupro.py `
  --dumps experiment_data\06_optimization\03_features\layer12_screw_seed_000\pixel_dumps `
  --data MVTec/mvtech_anomaly_detection `
  --out experiment_data\06_optimization\03_features\layer12_screw_seed_000\aupro.json

# 汇总表
.\.venv\Scripts\python.exe tools\summarize_features.py
```

## 10. 一句话

在同一 2000 中心预算下，把 layer1 拉进来（`1+2`、80×80 网格）在三个代表类上都提升了 image AUROC 与 AUPRO，代价是建库 2–3 倍；再叠加 layer3（`1+2+3`）不仅没有额外收益，还在 toothbrush 上把 image AUROC 拉低 9 个点——所以默认保持 `2+3`，把 `1+2` 写成有条件的可选方案。
