# 预处理对照：letterbox vs 直接拉伸（P0-3）

> 复核时间 2026-11-04（**v2：GT mask 已随几何变换同步**）· 代码 `patchcore_demo.py --preprocess {stretch,letterbox} --aspect-crop`
> 结果目录 `experiment_data/01_mvtec_gpu/results_preproc_{mode}_{类别}/`

## 0. 结论先说

1. **MVTec AD 15 类全部是方图**（900²–1024²），在方图上 `letterbox` 与 `stretch` **逐位等价**（实测张量 `maxdiff = 0.0`）。所以在这份数据上「保不保长宽比」不是一个可测变量——原计划的对照会得到两行完全相同的数字。
2. 为了把变量逼出来，加 `--aspect-crop 0.75`：先把图中心裁剪成 4:3（模拟非方形相机视野），再分别拉伸 / 留边到 320。**只改预处理，其余（权重、建库图、标定集、阈值、种子）全同。**
3. 结果**分缺陷形态**，不能一句「letterbox 更好」：
   - **screw**（细长划痕，各向异性）：letterbox image AUROC **+1.45 点**、pixel **+2.43 点**、召回 58.8%→65.5%、测试良品过杀 9.8%→4.9%。
   - **pill**（近对称主体）：image 基本无差别（−0.08 点），pixel **+0.65 点**。
   - **transistor**（细引脚且大量贴边）：image +0.33 点，pixel **+6.19 点**（0.8957 → 0.9576），阈值下召回 80%→60%。
4. 结论：**拉伸畸变伤害的是各向异性结构和定位质量**；在「mask 与图像对齐」的口径下，letterbox 的 pixel AUROC **三同类没有一类更差**（+2.43 / +0.65 / +6.19），此前的「灰边污染边界」是错误归因（详见 §5 口径更正）。代价是两条：灰边让建库/标定里的 patch 分布多出一块恒值区（对贴边缺陷类的**阈值行为**仍需单独看，transistor 召回从 80% 掉到 60%），以及非方视野下有效像素变少。产线仍应按缺陷形态 + 视野比例决定，但**「letterbox 会毁掉定位」这个说法不成立**。

## 1. 为什么原始对照是空的

| 类别 | bottle | cable | capsule | carpet | grid | hazelnut | leather | metal_nut |
|---|---|---|---|---|---|---|---|---|
| 原始尺寸 | 900² | 1024² | 1000² | 1024² | 1024² | 1024² | 1024² | 700² |

| 类别 | pill | screw | tile | toothbrush | transistor | wood | zipper |
|---|---|---|---|---|---|---|---|
| 原始尺寸 | 800² | 1024² | 840² | 1024² | 1024² | 1024² | 1024² |

方图上「等比缩放到长边 → 补边」与「直接拉伸到 S×S」得到同一个 S×S 张量，没有信息差异。这不是实现细节，是数据集性质——**MVTec 采图时就已经是正方形，长宽比在这份 benchmark 里根本不是风险点**，真正的风险在非方形的现场相机（或 VisA 的部分子集）。

## 2. 实验设置

| 项 | 值 |
|---|---|
| 类别 | screw / pill / transistor（各向异性、近对称、贴边细结构 三种形态各取一类） |
| 输入 | 320（`--size 320`），patch 网格 40×40 |
| 记忆库 | `--greedy 2000`，`--calib-ratio 0.3`，`--thr-pct 99` |
| 权重 | `torch_cache/wide_resnet50_2_timm.pth`（与主结果同） |
| 模拟非方形 | `--aspect-crop 0.75`：中心裁剪成 W×(0.75W)，即 4:3 视野 |
| 唯一变量 | `--preprocess stretch` vs `--preprocess letterbox` |
| 环境 | Python 3.13.9 / torch 2.7.0+cu128 / RTX 3050 Laptop |

## 3. 结果

### 3.1 image / pixel AUROC

| 类别 | 无裁剪（主结果，stretch） | 4:3 裁剪 + stretch | 4:3 裁剪 + letterbox | Δ(letterbox − stretch) |
|---|---|---|---|---|
| screw | 0.9553 / 0.9845 | 0.9154 / 0.9616 | **0.9299 / 0.9859** | image **+0.0145** / pixel **+0.0243** |
| pill | 0.9555 / 0.9781 | 0.9604 / 0.9714 | 0.9596 / 0.9779 | image −0.0008 / pixel **+0.0065** |
| transistor | 0.9958 / 0.9490 | 0.9917 / 0.8957 | **0.9950 / 0.9576** | image +0.0033 / pixel **+0.0619** |

三列的 mask 都与各自分支的几何变换一致（见 §5）。

### 3.2 固定阈值下的行为（`--thr-pct 99`，即名义 1% 过杀）

| 类别 | 预处理 | 召回 | 测试良品过杀 | 时延 ms |
|---|---|---|---|---|
| screw | stretch | 58.8% | 9.8% | 18.6 |
| screw | letterbox | **65.5%** | **4.9%** | 18.5 |
| pill | stretch | 78.0% | 3.8% | 25.5 |
| pill | letterbox | 80.9% | 0.0% | 18.5 |
| transistor | stretch | 80.0% | 0.0% | 33.1 |
| transistor | letterbox | 60.0% | 0.0% | 27.4 |

> 良品测试样本只有 20–41 张，过杀率 ±1 张就是 2–5 个点，这一列只能看量级，不宜当精确对比。

## 4. 机制分析

- **screw**：4:3 裁剪后 1024×768 被拉伸成 320×320，纵向压缩 1.33×。螺纹是沿单一方向延伸的高频结构，纵向压缩会让周期结构的 patch 描述发生各向异性形变，邻域 3×3 平均跨过的物理距离在横纵方向不再相等 → 异常分数被抹平。letterbox 保住比例（320×240 + 灰边），网格各向同性，所以分数更好、召回更高。
- **pill**：主体近似圆/椭圆，各向同性形变对圆目标几乎是保形的（圆 → 椭圆，但局部曲率统计变化小），所以两种预处理打平。
- **transistor**：引脚贴图边，letterbox 在两侧补 114 灰。image 级略好（+0.33 点）；pixel 级在 mask 对齐后反而**明显更好**（0.8957 → 0.9576）——因为拉伸把引脚间距压扁 1.33×，细引脚与背景的边界在 patch 分数上变得模糊；letterbox 保住比例后定位更准。代价出现在**阈值**一侧：灰边让标定分数分布整体左移/变窄，同一分位阈值下召回从 80% 掉到 60%，说明「更准的分数」不等于「更好的固定阈值」，产线要重新选阈值或改用分布无关的判定。

## 5. 局限（写在明面上）

1. **口径更正（2026-11-04 v2）**：v1 版把原始 GT mask 直接 resize 到 40×40，没有跟 `--aspect-crop` / `letterbox` 做同样的几何变换，导致 **pixel AUROC 系统性偏移**。v1 的结论「letterbox 在 transistor 上 pixel −1.57 点、灰边污染定位」就是这条 bug 的产物。现在 `load_mask(path, size, preprocess, aspect_crop)` 与 `load_img` 共用同一套裁剪与补边逻辑（mask 用 NEAREST、灰边填 0），六组实验已全部重跑。
   - 变化幅度：transistor letterbox pixel 0.9206 → **0.9576**、stretch 0.9363 → 0.8957；screw pixel 两臂各 +1 点上下；pill 基本不变。**image AUROC 完全不受影响**（mask 只进 pixel 指标），所以召回/过杀这些图像级结论不变。
2. 每类只跑 1 个 seed；过杀率基于 20–41 张良品。
3. 4:3 是模拟视野，真实产线要做的是「用现场相机比例与畸变参数重标定」，不是套 0.75。

## 6. 复现

```powershell
$py = '.venv\Scripts\python.exe'
foreach ($mode in 'stretch','letterbox') {
  foreach ($cat in 'screw','pill','transistor') {
    & $py patchcore_demo.py --data MVTec\mvtech_anomaly_detection --category $cat `
        --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 `
        --preprocess $mode --aspect-crop 0.75 `
        --weights torch_cache\wide_resnet50_2_timm.pth `
        --out experiment_data/01_mvtec_gpu/results_preproc_${mode}_${cat}
  }
}
```

等价性自检（方图，应当完全相同）：

```powershell
& $py -c "from patchcore_demo import load_img; import torch; p=r'MVTec\mvtech_anomaly_detection\screw\train\good\001.png'; print(torch.equal(load_img(p,320,'stretch'), load_img(p,320,'letterbox')))"
# True
```
