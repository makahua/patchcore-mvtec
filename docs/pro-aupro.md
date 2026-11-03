# 区域级指标：PRO / AUPRO（2026-11-04）

> 一句话结论：本机配置下 **AUPRO@FPR≤30% = 0.9211 ± 0.0398**（15 类 macro），
> 但把截断收到 10% 只剩 0.8068、收到 5% 只有 0.8841 的 PRO ——
> 而 tile 这类「图级全对、区域级崩」的类别说明：
> **image AUROC 1.0000 不代表定位能用**。
>
> 与 pixel AUROC 的相关系数只有 0.819，与 image AUROC 是 **−0.019**（完全不相关）。
> 三个指标不能互相替代，面试时任何「AUROC 0.98」都要说清是哪一层。

## 1. 为什么补这个指标

项目原有三组指标：image AUROC（判有没有）、pixel AUROC（逐像素排序对不对）、
以及产线曲线（过杀/漏检/复核量）。它们共同的缺口是：

- **pixel AUROC 按像素面积加权**。一块 200×200 的大缺陷能贡献 4 万个正样本，
  把 7×7 的小划痕的排序错误完全掩盖。工业上后者才是漏检事故。
- **image AUROC 只看 max 分数**。一张图里只要有一个像素分数超过阈值就算抓到，
  至于抓到的区域是不是缺陷本身、有没有把 80% 的缺陷区域盖住，指标完全不看。

PRO（Per-Region Overlap）就是补这个缺口：把 GT 按连通域切成一块块缺陷，
**每块等权**统计「被检出像素占比」，再对像素级 FPR 积分。MVTec 论文与 PatchCore /
PaDiM 都以 AUPRO 作为区域级主指标，也是本项目的竞品/论文对比口径。

## 2. 口径（可复现的定义）

| 项 | 取值 | 说明 |
|---|---|---|
| 分数图 | `patchcore_demo.py --dump-pixel` 的 (N, 40, 40) patch 分数 | size=320 → 8× 下采样 |
| 上采样 | 双线性 → 原图分辨率 | 与 PatchCore 把 anomaly map 放大回原图再算指标一致 |
| GT 来源 | `ground_truth/<缺陷类型>/<名字>_mask.png` 原图读取 | 不用 40×40 降采样掩码做判定，避免细缺陷被抹掉 |
| 连通域 | 4 邻接（行游程 + 并查集，无 scipy 依赖） | 对角相接算两块，比 8 邻接保守 |
| 区域权重 | 每块等权 | 官方口径；不是按面积加权 |
| 阈值 | 200 个等距阈值，覆盖全数据集分数范围 | 「分数 > t」判为异常 |
| FPR | 全图非 GT 像素中被判异常的比例 | 良品图全部像素 + 缺陷图的非缺陷区域 |
| AUPRO | PRO-FPR 曲线在 FPR ≤ 0.3 的梯形积分 / 0.3 | 另给 @0.1 与 PRO@5%FPR 两档 |

代码：`tools/pro_aupro.py`（纯 numpy；`--dumps/--data/--category/--limit/--out/--markdown`）。
测试：`tests/test_pro.py` **8/8**（连通域、完美/反向/半检出的 AUPRO、区域计数、像素 AUROC）。

**复算一致性**：同一批 dump 重算的 pixel AUROC 与 `results_pro_dump/results.csv`
最大差 **5×10⁻⁵**（15 类），说明 dump 的 (N, 40, 40) 分数图没有丢信息。

## 3. 结果（15 类，320 / greedy 2000 / calib 0.3 / P99 / timm WR50）

按 AUPRO@30% 升序：

| 类别 | image AUROC | pixel AUROC | AUPRO@30% | AUPRO@10% | PRO@5%FPR | 区域数 | 测试图 |
|---|---|---|---|---|---|---|---|
| tile | 1.0000 | 0.9198 | **0.8181** | 0.6002 | 0.6476 | 89 | 117 |
| grid | 0.9641 | 0.9553 | 0.8611 | 0.7655 | 0.8486 | 174 | 78 |
| wood | 0.9912 | 0.9081 | 0.8746 | 0.7210 | 0.7791 | 171 | 79 |
| toothbrush | 0.9472 | 0.9900 | 0.9035 | 0.7213 | 0.8343 | 79 | 42 |
| transistor | 0.9962 | 0.9490 | 0.9115 | 0.7912 | 0.8514 | 44 | 100 |
| cable | 0.9846 | 0.9801 | 0.9284 | 0.8068 | 0.8941 | 163 | 150 |
| zipper | 0.9677 | 0.9801 | 0.9331 | 0.8297 | 0.9166 | 183 | 151 |
| screw | 0.9557 | 0.9845 | 0.9354 | 0.8399 | 0.9040 | 140 | 160 |
| carpet | 0.9843 | 0.9858 | 0.9413 | 0.8481 | 0.9173 | 98 | 117 |
| metal_nut | 0.9990 | 0.9838 | 0.9424 | 0.8373 | 0.9307 | 136 | 115 |
| capsule | 0.9625 | 0.9889 | 0.9445 | 0.8360 | 0.9317 | 118 | 132 |
| bottle | 1.0000 | 0.9858 | 0.9463 | 0.8446 | 0.9228 | 70 | 83 |
| hazelnut | 1.0000 | 0.9872 | 0.9510 | 0.8605 | 0.9483 | 148 | 110 |
| pill | 0.9555 | 0.9781 | 0.9521 | 0.8807 | 0.9432 | 248 | 167 |
| leather | 1.0000 | 0.9876 | **0.9725** | 0.9189 | 0.9924 | 102 | 124 |
| **macro** | **0.9805 ± 0.0189** | **0.9709** | **0.9211 ± 0.0398** | **0.8068 ± 0.0760** | **0.8841 ± 0.0815** | 1963 | 1725 |

## 4. 四条判断

1. **区域级比像素级更悲观，而且类间差异被放大。**
   macro 从 0.9709（pixel）掉到 0.9211（AUPRO@30%），
   跨类标准差从 0.0252 涨到 0.0398（约 1.6×），
   截断收到 10% 时进一步涨到 0.0760。也就是说：**指标越贴近「盖住整块缺陷」，越暴露出弱类**。

2. **image AUROC 与区域可用性零相关（−0.019）。**
   tile 的 image AUROC 是满分 1.0000，AUPRO 却全场最低 0.8181；
   transistor image 0.9962 但 AUPRO 0.9115 倒数第五；
   反过来 toothbrush image AUROC 最低（0.9472），AUPRO 0.9035 反而中游。
   机制：image 分数是 max 聚合，只要有一个像素爆掉就判 NG；
   区域指标要求整块缺陷都被抬起来，纹理类（tile 的 glue/rough）恰恰是「局部爆点、整体不亮」。
   这条与 `docs/FAILURE_ANALYSIS.md` 里 screw/thread_side 全漏的结论互为印证：
   **图级召回好看不等于产线不退货**。

3. **AUPRO 对 FPR 截断极敏感，报数必须带 limit。**
   同一批结果：@30% → 0.9211，@10% → 0.8068（−11.4 点），PRO@5%FPR → 0.8841。
   低过杀区（≤10%）才是产线会用的工作点，所以「AUPRO 0.92」这个数不能单独引用，
   必须写成「AUPRO@FPR≤30%」并同时给出 @10% 一档。

4. **pooled 是陷阱。** 把 15 类的像素合并算一条曲线得 AUPRO 0.7808（@10% 仅 0.6099），
   因为各类分数尺度不同、合并后阈值语义混乱 —— 与 README 里
   「pooled image AUROC 0.8812 vs macro 0.9805 差 9.9 点」是同一类错误。
   **跨类结论只能看 macro。**

## 5. 复现

```bash
# 1) 落盘逐图像素分数图（15 类，本机约 45 分钟；结果与 results_best 完全一致）
.venv\Scripts\python.exe patchcore_demo.py --data MVTec\mvtech_anomaly_detection \
    --category all --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 \
    --weights torch_cache\wide_resnet50_2_timm.pth \
    --out experiment_data\01_mvtec_gpu\results_pro_dump \
    --dump-pixel experiment_data\05_indexes\pixel_dumps

# 2) 算 PRO/AUPRO（约 75 秒）
.venv\Scripts\python.exe tools\pro_aupro.py \
    --dumps experiment_data\05_indexes\pixel_dumps \
    --data MVTec\mvtech_anomaly_detection --category all \
    --out experiment_data\05_indexes\pro_aupro.json \
    --markdown experiment_data\05_indexes\pro_table.md

# 3) 自检
.venv\Scripts\python.exe tests\test_pro.py      # 8/8
```

`pixel_dumps/` 由 `--dump-pixel` 再生成（约 5 MB），已在 `.gitignore` 里；
`results_pro_dump/results.csv` 与 `05_indexes/pro_aupro.json` 入库。

## 6. 局限

1. **基础分辨率是 40×40**：上采样只能插值出平滑台阶，边界定位精度受 size=320 的分辨率上限约束。
   换成 size=320 + 更细的 patch 网格（或 9 邻域）才能验证分辨率对 AUPRO 的真实影响。
2. **没有官方数字对照**：本机无法访问 GitHub/arxiv（DNS 与代理均被禁止），
   官方 PatchCore 在 224 输入 + 1% coreset 下的 AUPRO 无法在这里逐行核对，
   所以本文只报自己的绝对值和类间排序，不做「比官方高/低」的结论。
3. **4 邻接 vs 官方实现**：官方用 skimage/scipy 的默认连边方式（2D 下通常为 8 邻接），
   我们取 4 邻接偏保守；同一张图里斜角相接的两块缺陷在我们这里算两块（等权后被放大权重）。
4. **单 seed**：与三档表一致，`--seed 0` 一次；AUPRO 的 seed 方差未测。
5. **mask 的灰度处理**：`>128` 二值化，与 `patchcore_demo.py` 的像素标签处理一致。

## 7. 后续（给 P1-4 成本模型与 P2 的输入）

- 成本模型里「区域级漏检」应使用 PRO@5%FPR 的口径，而不是 image 召回；
- 若做 9 邻域/更大 size 实验，把 AUPRO 作为主指标之一，观察「定位精度」是否随分辨率单调改善；
- tile/grid/wood 三类是本项目的区域级弱项（< 0.9），任何「全类可用」的说法都应先解释这三类。
