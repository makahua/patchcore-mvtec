# 9 邻域消融：环形填充缺陷与 coreset 选点顺序

> 目标：把「9 邻域聚合」这一项从「跑通了但更差」变成「知道差在哪、哪个 bug 值得修、哪个代价值得付」。
> 全部实验在同一台机器上完成：RTX 3050 Laptop 4 GB、Python 3.13.9、torch 2.7.0+cu128。
> 数据：MVTec AD `grid`（185 张建库 / 79 张标定 / 78 张测试）。

---

## 1. 要回答的三个问题

`docs/vs-upstream.md` 在差异审计里记了三处与上游实现不一致的地方，本文逐条量化：

1. **环形填充**：旧实现用 `torch.roll` 平移特征图拼 9 邻域，左边缘会看到右边缘。这算 bug 还是可忽略？
2. **选点顺序**：旧实现先把每个 patch 展开成 9×1536 维再选点，候选池内存 ×9；改成「先在 1536 维基础描述子上选点、选完再展开」会不会掉精度？
3. **9 邻域值不值**：修完上面两条之后，默认开不开？

## 2. 协议

| 项 | 值 |
|---|---|
| 类别 | `grid`（基准最差的一类，patch 结构最规整） |
| 输入 | 320×320，`stretch` |
| 描述子 | layer2+layer3 拼接 1536 维，3×3 邻域平均 |
| 选点 | greedy coreset，`k=2000`（先随机 10% 再贪心） |
| 阈值 | `--calib-ratio 0.3 --thr-pct 99`，`--blur-sigma 2` |
| 权重 | `torch_cache/wide_resnet50_2_timm.pth` |
| 随机性 | `--seed 0`；小数据实验用 `--train-limit 24`（同 seed 随机抽样，非前 N 张） |

两组规模：**全量 185 张**（与主结果同口径）与**限流 24 张**（让「先展开再选点」在 4 GB 显存/本机内存内可行，用于顺序与填充的同预算对照）。

## 3. 结果

全量 185 张（银行 2000）：

| 配置 | image AUROC | pixel AUROC | 时延 | 峰值显存 | 建库 |
|---|---|---|---|---|---|
| 无邻域（基线，`results_best`） | **0.9641** | **0.9552** | **17.7 ms** | **189.5 MB** | 189.9 s |
| base 选点 + zero 填充 | 0.9607 | 0.9511 | 52.0 ms | 517.2 MB | 197.0 s |
| expanded 选点（旧顺序） | 不可行：候选池需 ≈16.4 GB 内存（见 §4） | — | — | — | — |

限流 24 张（银行同为 2000，四组同预算可比）：

| 配置 | image AUROC | pixel AUROC | 时延 | 峰值显存 | 建库 | 候选池内存 |
|---|---|---|---|---|---|---|
| 无邻域 | **0.8747** | 0.9431 | **18.9 ms** | 188.0 MB | 23.6 s | 236 MB |
| expanded + roll（旧实现） | 0.8546 | 0.9382 | 59.6 ms | 601.2 MB | 194.4 s | 2.1 GB |
| expanded + zero | 0.8739 | 0.9374 | 50.9 ms | 601.5 MB | 191.6 s | 2.1 GB |
| base + zero（新顺序） | 0.8722 | **0.9386** | 51.8 ms | 516.9 MB | **22.1 s** | 236 MB |
| base + replicate | 0.8622 | 0.9387 | 56.5 ms | 516.9 MB | 22.1 s | 236 MB |

## 4. 四条判断

**① `torch.roll` 的环绕填充是真实缺陷，修它不花钱。**
同一预算下 zero 比 roll 的 image AUROC 高 **1.93 点**（0.8546 → 0.8739），pixel 基本不变（0.9382 → 0.9374）。原因是环形平移让最左列 patch 的邻域里混进了最右列的特征，网格图案在边界处被伪造成「跨边界」结构；补 0 与上游 `Unfold(padding=1)` 完全一致。改动只有一行填充方式，零额外成本，已设为默认（`--neighbor-pad zero`）。

**② 选点顺序可以改，精度等价、建库快 8.7×。**
base 与 expanded 在 24 图上 image AUROC 差 0.17 点（0.8722 vs 0.8739），在单类单 seed 的噪声范围内；pixel 甚至略好（0.9386 vs 0.9374）。但建库时间从 194 s 降到 22 s，候选池内存从 2.1 GB 降到 236 MB，GPU 峰值从 601 MB 降到 517 MB。关键在于「2000 个名额」不再被同一块纹理的平移副本重复占用。
（过程中发现并修掉一个真 bug：`coreset_greedy(..., return_idx=True)` 原先返回的是「随机降采样后」的行号，映射回原池会把 2000 个点全取成池子最前面约 2.4 张图的 patch，image AUROC 直接崩到 0.655；现在返回的是原池行号，`flat[idx]` 与选中的特征逐位相等。）

**③ expanded 顺序在全量数据上根本跑不动。**
185 张建库图 = 296,000 个 patch；先展开成 9 邻域 = 296,000 × 13,824 × 4 B ≈ **16.4 GB**，本机 4 GB 显存 + 16 GB 内存无解（这也是历史 `results_neigh` 只做到 18,944 个 patch 的原因）。base 顺序把这一步省掉，全量 185 张跑通：197 s、517 MB。

**④ 但 9 邻域本身仍然不划算，默认继续关。**
全量口径下，加 9 邻域后 image 0.9641 → 0.9607（−0.34 点）、pixel 0.9552 → 0.9511（−0.41 点），时延 17.7 → 52.0 ms（2.9×），峰值显存 189.5 → 517.2 MB（2.7×）。**修 bug + 换顺序之后，9 邻域只是「不再更差」，并没有变成「更好」。** 所以默认仍是关；需要更强局部上下文时再开，并建议 `--neighbor-coreset base`。

## 5. 复现命令

```bash
# 全量：base 选点 + zero 填充
.venv\Scripts\python.exe patchcore_demo.py --data MVTec/mvtech_anomaly_detection --category grid \
  --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --blur-sigma 2 \
  --neighbors --neighbor-coreset base --neighbor-pad zero \
  --weights torch_cache/wide_resnet50_2_timm.pth --seed 0 \
  --out experiment_data/01_mvtec_gpu/results_p3_full_new_zero

# 同预算对照（四个配置只改 --neighbors / --neighbor-coreset / --neighbor-pad，都加 --train-limit 24）
--out experiment_data/01_mvtec_gpu/results_p3_lim24_none        # 无邻域基线
--neighbors --neighbor-coreset expanded --neighbor-pad roll     # results_p3_lim24_old_roll（旧实现）
--neighbors --neighbor-coreset expanded --neighbor-pad zero     # results_p3_lim24_old_zero
--neighbors --neighbor-coreset base     --neighbor-pad zero     # results_p3_lim24_new_zero
```

原始日志：`experiment_data/05_indexes/p3_neigh_run.log`；逐类结果在各 `results_p3_*/results.csv`（含 `neighbor_coreset` / `neighbor_pad` 两列，便于事后追溯配置）。

## 6. 局限

- 只在 `grid` 一类上做：它邻域结构最规整，对边界污染最敏感；换成 screw/leather 时 roll 的惩罚可能更小。
- 单 seed：0.17 点级别的差异不应解读为「base 更好」，只能解读为「base 没有变差」。
- 24 张限流组的绝对数字（0.87 一线）不能与主结果（0.96）并列，只用于同预算对照。
- `expand_selected` 是 Python 循环逐点取 3×3，2000 个点约 1–2 s；若银行放大到 2 万点需改成向量化实现。
- 未测 `--neighbors` 与更高分辨率的组合；理论上 640 时 patch 数 ×4，expanded 顺序只会更不可行。

## 7. 面试一句话

「9 邻域我跑过、也调过：先修掉了环形填充的边界污染（+1.9 点），再把选点从『展开后选』改成『选完再展开』（建库快 8.7×、内存省 8.9×，精度持平）；但修完仍然比不开 9 邻域差 0.3 点、时延贵 2.9×，所以默认关——这不是没做完，是量过之后的选择。」
