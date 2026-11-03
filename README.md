# PatchCore 少样本无监督缺陷检测 Demo（MVTec AD）

给机器视觉/工业质检岗准备的一个**能跑出真实数字**的项目。
不依赖 anomalib，只靠 torch + torchvision；核心就是 PatchCore：预训练特征 → patch 级记忆库 → 最近邻距离当异常分。

> **状态：已在真实 MVTec AD（15 类 / 1,725 张测试图）上跑完整流程，本文所有数字均为本机实测，可用 `analyze.py` 与 `classify.py report` 复现。**
> 环境：Python 3.12 + torch 2.7.0 (CUDA 12.8) + torchvision 0.22.0，单卡 RTX 3050 Laptop（4 GB 显存）。
> 主配置（`--greedy 2000 --size 320`，见「优化实验：五种配置对比」）：**image AUROC 0.9805 / pixel AUROC 0.9709 / image-level PR-AUC 0.993 / 推理与最近邻评分均值 19.7 ms / 显存峰值约 190 MB / 记忆库 2,000 patch**。延迟不含图像读取、建库和标定。
> 权重不是 torchvision 原版 `IMAGENET1K_V1`，而是 `timm/wide_resnet50_2.racm_in1k`（RandAugment+Mixup 重训）转出的 state_dict，见第 3 步，与论文口径可能有千分位差异。

> **复现来源与口径**：算法来自 PatchCore 论文 *Towards Total Recall in Industrial Anomaly Detection*（Roth et al., CVPR 2022；[arXiv:2106.08265](https://arxiv.org/abs/2106.08265)），参考 [AWS 官方实现](https://github.com/amazon-science/patchcore-inspection)。本项目代码按论文思路独立实现；并非逐行移植或完全对齐官方评测。本文 `image AUROC 0.9805` 指 15 类 macro 均值；同一批图像 pooled 结果是 0.8812。官方报告数字只有在 backbone、权重、输入和评测口径匹配时才可直接比较。

---

## 0. 为什么做这个（面试时的定位）

工业质检的现实是：**缺陷样本永远不够，缺陷类型永远有新的**。所以产线上更常见的是「只拿正常样本训练」的无监督/少样本异常检测，而不是凑够 7 类缺陷做分类。
这个 demo 正好证明三件事：
1. 你懂「只有 good 样本」这种数据条件该怎么建模；
2. 你会做记忆库 + 最近邻检索这一套工程实现（不是只会调 YOLO）；
3. 你能给出 image-level AUROC / pixel-level AUROC / 推理与最近邻评分耗时，并按产线策略调阈值。

---

## 1. 装环境（10 分钟）

```powershell
# 建议新建 conda 环境，别污染毕设那个
conda create -n ad python=3.12 -y   # 本项目实测 3.12；3.10 / 3.11 也能跑
conda activate ad

# 有 NVIDIA 显卡（按本机 CUDA/驱动选择 PyTorch 安装命令）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 没显卡就装 CPU 版：
# pip install torch torchvision

pip install numpy pillow matplotlib tqdm
```

验证：`python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`

> **只想复算三档表？上面这一整节都可以跳过。** `classify.py report` 只依赖 numpy，
> 装个 numpy 就能读 `experiment_data/00_baselines/results_best/scores/*.json` 复现 README 里的 15 行与两个 AUC 口径（1 秒）。
> 需要 torch 的只有 `build / infer / eval` 三个子命令和 `patchcore_demo.py` 本身。

> **权重不走 torchvision 的自动下载。** 本机 `download.pytorch.org` 不通（TLS 直接断），实际做法是从
> hf-mirror 下 timm 的 `wide_resnet50_2.racm_in1k`（275 MB safetensors）再转成 state_dict，见第 3 步。
> 只有在**省略** `--weights` 时，脚本才回退到 torchvision 自带的 `IMAGENET1K_V1` 自动下载。
> 想换缓存目录就设 `TORCH_HOME`（本机放在 `E:\torch_cache`；注意本机没有 D 盘）。

---

## 2. 拿数据集（MVTec AD）

- 官方：https://www.mvtec.com/company/research/datasets/mvtec-ad （注册后下 `mvtec_anomaly_detection.tar.xz`，约 4.9GB，15 个类别）

### 国内镜像（不用注册、不用翻墙）

**① hf-mirror 单文件直链（最省事，5.27GB zip，一个文件拉完）**

```powershell
# 断点续传，断了重跑同一条命令即可续上
New-Item -ItemType Directory -Force E:\MVTec_dl | Out-Null
curl.exe -L -C - -o E:\MVTec_dl\mvtec.zip "https://hf-mirror.com/datasets/hdtech/mvtech_anomaly_detection/resolve/main/mvtech_anomaly_detection.zip"

# 下完解压（约 5GB，Expand-Archive 慢但稳；有 7-Zip 就用 7z x）
Expand-Archive E:\MVTec_dl\mvtec.zip -DestinationPath E:\MVTec_dl\x
```

**② hf-mirror 原始目录版（文件多、可逐个续传，适合网络很抖）**

```powershell
pip install -U "huggingface_hub[cli]"
$env:HF_ENDPOINT = "https://hf-mirror.com"
hf download --repo-type dataset hdtech/mvtech_anomaly_detection --local-dir E:\MVTec_dl
# 或换成逐文件的版本：
# hf download --repo-type dataset TheoM55/mvtec_anomaly_detection --local-dir E:\MVTec_dl
```

**③ OpenDataLab（上海 AI Lab，国内官方副本，需登录拿 token）**

```powershell
pip install opendatalab
odl login            # 浏览器里授权
odl get OpenDataLab/MVTecAD
```

- **只要 bottle 一类就够跑通**：解压后保留 `bottle/` 目录即可，其余类别可以以后再加
- 磁盘要留 **11GB**（zip 5.27GB + 解压后约 5GB）
- 解压后若多出一层 `mvtec_anomaly_detection/`，`--data` 就指到那一层，判断标准是下面这个结构

解压后的目录结构必须是：

```
MVTec/
└── bottle/
    ├── train/good/*.png                (209 张)
    ├── test/good/*.png                 (20 张)
    ├── test/broken_large/*.png         (20 张)
    ├── test/broken_small/*.png
    ├── test/contamination/*.png
    └── ground_truth/broken_large/*_mask.png
```

---

## 3. 跑（3 条命令）

> **权重怎么来的（本机实测路径）**：DSH 沙箱里 `download.pytorch.org` 走不通（TLS 直接断），改从 hf-mirror 拿：
>
> ```powershell
> # 1) 下 timm 版权重（safetensors，275MB，断点续传）
> python tools\dl_mvtec.py "https://hf-mirror.com/timm/wide_resnet50_2.racm_in1k/resolve/main/model.safetensors" torch_cache\wr50.safetensors
> # 2) 转成 torchvision 能吃的 state_dict（无 safetensors 依赖，自解析头+裸 buffer）
> python tools\safetensors_to_pth.py torch_cache\wr50.safetensors torch_cache\wide_resnet50_2_timm.pth
> # 3) 指定权重跑
> python patchcore_demo.py --data ... --category all --weights torch_cache\wide_resnet50_2_timm.pth
> ```
>
> 加载时打印 `[weights] 命中 318/320 层`（fc 类被丢弃，属正常）——命中率不够它会报错退出，
> 不会悄悄用错权重跑出假数字。**坑**：`BobMcDear/wide_resnet50_2` 那个 `wide_resnet50_2.pickle`
> 是 **Flax/JAX** 权重（键只有 params/batch_stats），torchvision 只命中 53/320，别用。

```powershell
cd patchcore-demo

# ① 冒烟测试：每类 10 张，确认跑得通（1 分钟）
python patchcore_demo.py --data E:/MVTec --category bottle --limit 10 --weights torch_cache/wide_resnet50_2_timm.pth

# ② 正式跑 bottle
python patchcore_demo.py --data E:/MVTec --category bottle --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce

# ③ 全类别（15 类，GPU 上约 10-15 分钟）
python patchcore_demo.py --data E:/MVTec --category all --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce
```

产出：
- `experiment_data/00_baselines/results/results.csv` ← **简历上的数字全从这里抄**
- `experiment_data/00_baselines/results/results.json`
- `experiment_data/00_baselines/results_best/heatmaps/*.png` ← 异常热力图，README 里放 2 张，很有说服力

### 实测结果（本机 RTX 3050 Laptop，256×256，随机 coreset 10%，`--thr-pct 99`）

| 指标 | bottle | 全 15 类平均 |
|---|---|---|
| image-level AUROC | **1.0000** | **0.9703** |
| pixel-level AUROC | 0.9842 | **0.9701** |
| 推理与最近邻评分均值（不含读图/建库/标定） | 31.9 ms | 35.7 ms（各类均值） |
| 记忆库 patch 数 | 21401 | 24773 |
| 显存峰值 | 471 MB | 761 MB |
| 建库耗时 | 9.0 s | 3.3~24.5 s |

逐类（image / pixel）：

| 类别 | image | pixel | 类别 | image | pixel |
|---|---|---|---|---|---|
| bottle | 1.000 | 0.984 | metal_nut | 0.984 | 0.981 |
| cable | 0.985 | 0.980 | pill | 0.929 | 0.968 |
| capsule | 0.956 | 0.986 | screw | 0.892 | 0.980 |
| carpet | 0.978 | 0.986 | tile | 1.000 | 0.919 |
| grid | 0.947 | 0.962 | toothbrush | 0.967 | 0.989 |
| hazelnut | 0.999 | 0.985 | transistor | 0.988 | 0.966 |
| leather | 1.000 | 0.991 | wood | 0.982 | 0.905 |
| zipper | 0.948 | 0.970 | | | |

> ⚠️ **上面这张表是"初版基线"，它的阈值有 bug（见下一节），误报率 28% 不是模型差，是阈值被系统性压低了。**
> 真实工作点看「优化实验」那张表。记忆库最小的是 `toothbrush`（60 张训练图，6144 patch），所以它 24 ms 最快。

### 阈值标定的坑（这个必须修，面试也最爱问）

初版把阈值取成「**训练集自评分** P99」。问题是记忆库就是从这些训练图建的，**每张图自己的 patch 就在库里**，
最近邻距离趋近 0 → 训练分数整体偏低 → 阈值定得太低 → 测 good 图时大量误报（15 类平均 28%）。

修法：从 `train/good` 里留出一部分（`--calib-ratio 0.3`，本机每类留 18~117 张）**不参与建库**，只用它们算阈值。当前各类标定样本不多，尤其不足以稳定估计每类 0.5% 这类极低过杀率；文中测试集过杀/漏检数字是该固定划分上的观测值，不是产线保证。正式部署前需用更大的独立良品标定集验证。

```powershell
python patchcore_demo.py --data E:/MVTec --category all --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce
```

### 优化实验：五种配置对比（同机 RTX 3050 Laptop，都带 --calib-ratio 0.3）

| 配置 | image AUROC（macro） | pixel AUROC（macro） | image PR-AUC（macro） | 推理/检索均值 | 显存峰值 | 召回@0.5%名义过杀（macro） |
|---|---|---|---|---|---|---|
| 随机 10% / 256（初版，阈值未留出） | 0.9703 | 0.9701 | — | 35.7 ms | 761 MB | 阈值不可信 |
| 随机 10% / 256 | 0.9662 ± 0.039 | 0.9694 | 0.988 | 28.3 ms | 574 MB | 83.0% |
| **greedy 2000 / 256** | **0.9737 ± 0.027** | 0.9696 | **0.991** | **16.4 ms** | **169 MB** | **83.4%** |
| 随机 10% / 320 | 0.9763 ± 0.022 | 0.9706 | 0.992 | 58.2 ms | 935 MB | 81.4% |
| **greedy 2000 / 320（最终档）** | **0.9805 ± 0.019** | **0.9709** | **0.993** | 19.7 ms | 190 MB | 82.8% |
| 9 邻域 / 256（**仅跑了 grid**） | 0.9190 | 0.9537 | — | 360.3 ms | 2353 MB | — |

> **⚠ 口径：这一列的 image AUROC 是 macro**（每类各算一个 AUC，再对 15 类取平均）。
> 同一批分数如果按图片**汇总**（pooled，1725 张混在一起算一次）是 **0.8812**，比 macro 低 9.9 个点。
> 原因：各类分数分布量级不同（screw 整体偏高、leather 偏低），混池后排序被打乱。
> 两个数都对，但**必须说清是哪个**——面试官拿仓库里的 scores 复算，如果得到 0.88 而你写 0.98，会当场翻车。
> 复算命令见下面「三档判定」一节，classify.py report 会把两个口径一起打出来。

> 最终档（`experiment_data/00_baselines/results_best/`）就是简历上那组：留出 1085 张正常图做阈值标定，逐类最差三类
> toothbrush 0.947 / screw 0.955 / pill 0.956，都比 256 版好（256 版 screw 只有 0.901）。
> 
> **口径提醒（两个指标都各有一个口径，别混）**：
> - image AUROC：macro **0.9805**，pooled **0.8812**；
> - 召回@0.5%过杀：macro **82.8%**（漏检 17.2%），pooled **80.0%**（漏检 20.0%）——
>   screw / capsule 缺陷样本多且难，会把池化值拉低。
> 
> **对外怎么报**：报 macro 就必须写成「15 类均值 / 15 类平均」，不能只写「image AUROC 0.98」；
> 被问「全部混一起是多少」时直接答 pooled 0.8812，并解释原因（难类缺陷样本多、分数分布量级不同）。
> 简历上那版写的是 `image AUROC 0.980（15 类均值）`——带了这个限定，两个口径就不冲突；
> 千万不要在 A 处写 macro、B 处写 pooled 还都叫「AUROC」。

结论（可以直接当面试答案）：
1. **greedy coreset 是性价比最高的一步**：AUROC +0.75 点、时延 28→16 ms（-42%）、显存 574→169 MB、跨类标准差 0.039→0.027（更稳）。
   因为记忆库从 1.7 万压到 2000 patch，检索成本降 8 倍——这就是论文说"coreset 是效率关键"的实证。
2. **提分辨率的代价取决于记忆库大小**：随机抽样下 320 把时延 28→58 ms（记忆库随分辨率膨胀）；
   但 coreset 定长 2000 之后，320 只多 3 ms（16.4→19.7）就换来 AUROC +0.68 点（0.9737→0.9805）。
   **顺序很重要：先压 coreset，再提分辨率**；反过来就得多付一倍时延。
3. **9 邻域在 4GB 卡上跑通了，但不划算**：`experiment_data/00_baselines/results_neigh/` 在 grid 上完整跑完（18,944 patch、峰值 2,353 MB、360 ms），
   **没有 OOM**；可它是与 `experiment_data/00_baselines/results_fix/` 的 grid 同配置（256 / 随机 10%）的干净对照，
   精度反而 **0.9373 → 0.9190（−1.8 点）**，时延 30→360 ms（12×）、显存 449→2353 MB（5.2×）。
   原因大概率是维度从 1536 涨到 13824 后，coreset 在同等预算下覆盖变差，加上 `torch.roll` 的环绕填充污染了边界 patch。
   只跑了一个类，不推广到全部 15 类；但至少**没有证据支持"开邻域能提分"**，所以默认关。
4. **AUROC 高 ≠ 能上线**：0.5% 目标过杀下平均抓到 83% 缺陷（按类平均；汇总后漏检 20.0%）。工业线一般要漏检 <1~2%，
   所以真实方案得靠「双阈值 + 人工复核」或更高分辨率，这一点面试官一问就露。

### 三档判定：AUROC 之后，产线只问三件事（`classify.py`）

AUROC 是阈值无关的，产线不是。产线只问：**良品过杀多少 / 缺陷漏检多少 / 多少张要人看**。
单阈值只能二分类，工业现场必须留一个灰区让人复核，于是把阈值拆成两刀：

- `thr_ok` = 留出良品的 P(1−fpr) 分位 → 低于它判 **OK**
- `thr_ng` = `thr_ok` × 倍数（默认 1.2）→ 两者之间判 **REVIEW（复检）**，高于它判 **NG**

```powershell
python classify.py build  --data E:/MVTec --category bottle --weights torch_cache/wide_resnet50_2_timm.pth   # 建库 + 标定 → experiment_data/04_memory_banks/banks/bottle.npz
python classify.py infer  --bank experiment_data/04_memory_banks/banks/bottle.npz --image MVTec/mvtech_anomaly_detection/bottle/test/broken_large/000.png --out experiment_data/06_demo_outputs/demo_out
python classify.py eval   --bank experiment_data/04_memory_banks/banks/bottle.npz --data E:/MVTec --category bottle --sweep
python classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2   # 全 15 类，离线，秒出
```

> `report` **只需要 numpy**：不需要数据集、不需要 GPU、不需要权重，也不用装 torch / torchvision。
> 它只读仓库里已提交的 `experiment_data/00_baselines/results_best/scores/*.json`
> （15 个文件共 139 KB，`path` 已写成 `<类别>/test/...` 的相对形式，不含任何本机绝对路径）。
> 这是复算上表的最短路径：克隆仓库 → 一条命令 → 得到同样的 15 行与下面两个 AUC 口径。
> 
> 输出末尾会固定打印：
> 
> ```
> AUC 口径（同一批分数，两种算法）
>   macro（15 类各自算 AUC 再平均） 0.9805 ± 0.0189
>   pooled（全部 1725 张混池后算）    0.8812
>   差 9.9 个点 —— 对外只报一个数时必须标注口径
> ```

`infer` 会输出三联图（原图 ｜ 热力图 ｜ 叠加），归一化分数图、双线性放大后上色：

![三档判定热力图](experiment_data/06_demo_outputs/demo_out/broken_large_000_NG_57.1.png)

**全 15 类结果**（每类各自标定阈值，名义目标过杀 0.5%、灰区 1.2×，共 467 张良品 / 1,258 张缺陷；每类标定集为 18–117 张，低误报率估计有较大不确定性）：

| 类别 | 过杀 | 良品进灰区 | 漏检 | 复核量 |
|---|---|---|---|---|
| bottle | 0.0% | 0.0% | 0.0% | 8.4% |
| cable | 0.0% | 0.0% | 21.7% | 26.0% |
| capsule | 0.0% | 4.3% | 56.9% | 18.2% |
| carpet | 0.0% | 7.1% | 4.5% | 9.4% |
| grid | 0.0% | 9.5% | 7.0% | 24.4% |
| hazelnut | 0.0% | 0.0% | 0.0% | 17.3% |
| leather | 0.0% | 3.1% | 0.0% | 4.0% |
| metal_nut | 0.0% | 0.0% | 4.3% | 37.4% |
| pill | 0.0% | 0.0% | 22.0% | 52.1% |
| screw | 0.0% | 0.0% | **76.5%** | 17.5% |
| tile | 0.0% | 0.0% | 4.8% | 23.9% |
| toothbrush | 8.3% | 8.3% | 6.7% | 21.4% |
| transistor | 0.0% | 0.0% | 35.0% | 21.0% |
| wood | 0.0% | 0.0% | 10.0% | 13.9% |
| zipper | 0.0% | 9.4% | 8.4% | 26.5% |
| **汇总** | **0.2%** | **2.1%** | **20.0%** | **22.7%** |

**产线曲线**（调 `thr_ok` 的分位 = 允许的过杀，漏检与过杀此消彼长）：

| 目标过杀 | 实际过杀 | 漏检 | 灰区捞回 | 复核量 |
|---|---|---|---|---|
| 0.1% | 0.2% | 21.0% | 31.4% | 23.4% |
| 0.5% | 0.2% | 20.0% | 30.3% | 22.7% |
| 1.0% | 0.4% | 16.6% | 31.1% | 23.6% |
| 2.0% | 0.9% | 13.5% | 30.0% | 23.1% |
| 5.0% | 1.1% | 6.9% | 30.2% | 24.6% |
| 10.0% | 1.5% | **4.1%** | 27.2% | 23.9% |

三条结论，面试可以直接讲：

1. **AUROC 0.980 不等于能上线**：在当前固定数据划分和阈值方案下，名义目标过杀 0.5% 时观测漏检约 20%；小标定集使该工作点不确定性较大。论文报告的 99.6% 是其最高配置结果，不是“最优阈值下”的单模型 WR50 对照。
2. **灰区是便宜的那部分**：1.2× 的灰区只把复核量推到 23%，却兜回 30% 的缺陷；
   真正压漏检要放宽 thr_ok，代价是过杀变多（2% 过杀时漏检降到 13.5%）。
3. **screw 是最差的类（漏检 76.5%）**：螺纹上的细划痕在 /8 特征图上只占 1~2 个 patch，
   被 3×3 邻域平均抹平了 → 这类要么上更高分辨率，要么换 9 邻域（显存不够），要么改小视野重采图。
   **能指出自己方案在哪失效，比报一个漂亮均值有用。**

> 顺带一个失败消融：试过在输入处补 ImageNet 归一化（`--imagenet-norm`），screw 的 image AUROC
> 从 0.901 掉到 0.867，所以默认关掉。推测是 BN 已经吸收了输入尺度差异，但只验过一个类，
> 不下结论——写进 README 是因为**失败实验也是实验**，面试官问"你试过什么没成"时有话讲。

### 补齐评测（`analyze.py`）

AUROC 是阈值无关的，但产线只关心「固定过杀率下能抓多少」。所以把逐图分数落盘成 `scores/<类别>.json`，再用：

```powershell
python analyze.py experiment_data/00_baselines/results_greedy        # 也可以一次给多个目录对比
```

输出每类的 image/pixel AUROC、**图像级 PR-AUC**（本测试集缺陷图占比约 73%，此处不代表极低缺陷率场景；极低缺陷率下的 PR 指标需用符合产线比例的数据另测）、**召回@1.5%/0.5% 名义过杀率**、推理与最近邻评分耗时、记忆库大小，
最后一行固定给出 **macro vs pooled 的口径对照**（见上一节的 ⚠ 提示）——`analyze.py` 和 `classify.py report` 都会打，
这样「简历上的 0.98」和「别人复算出的 0.88」永远同时出现在你眼前，不会各说各话。

论文/官方仓库报告的最高 image AUROC 为 99.6%（那是 **WR50 + 多尺度/多邻域的 ensemble 配置**，不能当单模型基线），
WR50 单模型配置另有 99.16%（IM224）和 **99.30%（IM320）**；这些配置不能混为一个基线。本项目是 320 配置，为 0.980/0.971。
**对同分辨率 IM320 单模型基线差 1.25 个点**（对 99.6% 的 ensemble 差 1.55 个点，但那个对比没有意义）。差异候选包括：
1. ~~随机抽样代替贪心~~ —— 已用 `--greedy 2000` 补上（0.9662→0.9737），所以这条已经不算借口了；
2. **权重不是论文/官方那一套** —— 用的是 timm 的 RandAugment+Mixup 重训 ImageNet 权重；
3. **没开 9 邻域拼接**（`--neighbors`）；其精度/资源收益需在当前配置下实测，不能预设一定提分。

> **上游出处**：论文 [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
> （Roth et al., CVPR 2022），官方实现 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)。
> 本项目是**从零复现**（不 import anomalib，也不拷上游代码），与原版的差异见本节的差异候选清单与 `docs/OPTIMIZATION_PLAN.md` §7。

记住：**能解释差距比数字漂亮更重要**，面试官更想听你知不知道这 1.25 个点差在哪。

### 主要参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `--size` | 输入分辨率 | 默认 256；想更快用 224，想更准用 320 |
| `--coreset-ratio` | 记忆库随机抽样比例 | 默认 0.1；显存不够调到 0.03 |
| `--greedy N` | 改用论文的 k-center 贪心，保留 N 个 patch | `--greedy 1000`，慢但准确率高 |
| `--neighbors` | 9 邻域拼接（维度 x9） | 先做显存与精度实测；当前实验不支持固定的提分预期 |
| `--thr-pct` | 阈值取标定分数的百分位 | 默认 99；**产线漏检优先就调到 99.5** |
| `--calib-ratio` | 从 train/good 留出多少比例专门标定阈值 | **强烈建议 0.3**；不给就退回旧行为（阈值偏低、误报虚高） |
| `--limit` | 每类只取前 N 张 | 冒烟测试用 |

### PyQt 单图演示

桌面界面支持选择记忆库与待测图片，显示原图、异常热力图、异常分数和 OK / REVIEW / NG 三档结果。先按上文建立 bank，再启动：

```powershell
pip install PyQt5
python classify.py build --data E:/MVTec --category bottle --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/04_memory_banks/banks
python pyqt_demo.py
```

稳定性诊断可对留出的良品 calibration 分数做 bootstrap，观察阈值对有限标定样本的敏感度：

```powershell
python tools/threshold_stability.py --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2 --repeats 1000
```

这项诊断反复使用同一固定 test 集，输出不是独立置信区间或产线保证；使用方法与解释见 [复现说明](docs/REPRODUCE.md)。

### 新数据集：VisA

Amazon VisA 已下载并整理为 12 类 1-class split。PCB1 的可选实验记忆库在 `experiment_data/04_memory_banks/banks_visa_top5_fpr05/pcb1.npz`，使用最高 5% patch 均值和 5% 良品复检目标；当前官方测试集的 image AUROC 为 0.8491，异常直接漏判 43%。该测试集已用于探索，不作为独立验证，配置仍需用新样品验证。数据位置、PyQt 运行步骤和默认/实验阈值对比见 [VisA 接入指南](docs/VISA.md)。VisA 单独报告，不混入 MVTec AD 原有 15 类指标。

---

## 4. 六个必踩的坑

1. **mask 尺寸不一致** —— ground_truth 的 mask 是原图尺寸，而分数图是 /8 分辨率。脚本里已经 resize 到 patch 尺寸；自己改代码时别在这上面翻车（这是 pixel AUROC 算成 0.5 的最常见原因）。
2. **阈值别用测试集定，也别用「建库图自评分」定** —— 产线拿不到缺陷样本，阈值只能用 good 分布；
   但更隐蔽的是**自匹配偏差**：建库图自己的 patch 就在库里，最近邻距离≈0，拿它们算 P99 会把阈值压低（初版误报 28% 就是这么来的）。
   正确做法是留出一部分 good 图只做标定（`--calib-ratio 0.3`）。
3. **显存爆炸** —— `train/good` 有的类别 200+ 张，全量 patch 堆在 GPU 上会 OOM。脚本先在 CPU 上建库再抽样搬 GPU。
4. **图像拉成正方形** —— 脚本直接 resize 到 256×256，会轻微变形；要严谨就改成 keep-aspect + pad，并在 README 里说明。
5. **好图被判坏（误报）** —— good 的 test 图也可能有高响应（反光、边缘）。所以要看 `false_positive_rate`，不能只看 AUROC。
6. **权重下载失败** —— 报 `URLError` 就是墙。设代理或用 `TORCH_HOME` 预先放权重。

---

## 5. 简历怎么写（等你跑出真数字后替换【】）

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

## 6. 面试会问什么（现在就背下来）

1. **“为什么不直接训分类网络？”** —— 缺陷类型不可穷举、样本极少；无监督只用 good 样本，未知缺陷也能给分数。
2. **“记忆库为什么不全存？”** —— patch 数量 = 图数 × H/8 × W/8，200 张图就 20 万 patch，检索和显存都撑不住；coreset 保留代表性 patch，精度损失很小。
3. **“为什么要 3×3 邻域平均 / 高斯平滑？”** —— patch 级特征对微小位移敏感，邻域平均提升局部一致性，减少单点噪声造成的误报。
4. **“阈值怎么定，为什么不用测试集？”** —— 产线拿不到缺陷样本；用 good 的分数分布取高分位（P99），再按漏检/过杀成本调整分位。
5. **“AUROC 高就能上线吗？”** —— 不能。要看固定阈值下的漏检率/过杀率、单张耗时是否匹配节拍、以及换批次/换光照后的稳定性。
6. **“怎么和 Halcon 的方案比？”** —— Halcon 强在标定好、光照稳、缺陷形态已知的快速落地（模板匹配 + Blob + 卡尺）；无监督异常检测强在缺陷未知、样本少。产线上常常是「传统算子做定位和尺寸 + 异常检测做外观」，两者组合。

---

## 7. 30 分钟复现路径（给面试官 / 给未来的自己）

数据 5.27 GB 和权重 267 MB 都不在仓库里（MVTec 许可证禁止再分发、GitHub 也有单文件上限），
所以仓库只放**代码 + 结果表 + 逐图分数**。想验证结论，最短路径是这两条：

| 想验证什么 | 要准备什么 | 命令 | 耗时 |
|---|---|---|---|
| 三档判定那 15 行 + 两个 AUC 口径 | **只要有 numpy** | `python classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2` | 1 秒 |
| 某类能跑到同样 AUROC | 数据集 + timm 权重 | `python patchcore_demo.py --data E:/MVTec --category bottle --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce` | bottle 约 2–3 分钟（受设备负载影响） |

> 第二条命令的预期值（bottle，RTX 3050 Laptop）：`img_auroc 1.0000 / pixel 0.9857 / threshold 37.9943 / mem 2000 / peak 188.7 MB`。
> 时延会在 17–24 ms 之间抖动（笔记本 GPU 动态频率），**别把时延当复现失败的判据，看 threshold 和 AUROC**。

## 8. 进阶（做完再做，别一次贪多）

- 加 **PaDiM** 对比（多元高斯 + 马氏距离）：改动约 50 行，能讲“为什么 PatchCore 比 PaDiM 强：不受高斯假设约束”。
- 把 `patchcore_demo.py` 包成 **FastAPI 服务**（本机毕设那套 backend 骨架可复用）：多一句「异常检测服务化」。
- 换**自己的工件**：拍 30 张正常螺栓/PCB，跑同一套流程——这一步能把“公开数据集”升级成“真实工件验证”，面试杀伤力翻倍。

---

## 9. 自检（不需要 GPU、不需要数据集）

```powershell
python -m pytest tests/ -q        # 有 pytest
python tests/test_metrics.py      # 没 pytest 也能直接跑，纯 numpy
```

`tests/test_metrics.py` 只依赖 numpy，共 11 条，钉的是**面试官唯一会当场复算的那几个数**：

- `auc()` 是手写的 rank 法（为了不依赖 sklearn）。并列值如果按 argsort 名次硬算，
  所有指标会整体偏一点却**不报错**——所以专门有「全并列应为 0.5」「打乱顺序结果不变」
  「与两两比较定义式对拍」三条。
- `auprc()` 的单调性（排序更好的分数必须拿到更高 PR-AUC）。
- `check_bank_dims()`：记忆库维度必须和 meta 里记的 `neighbors` 自洽。
  这条是 **BUG-1 的回归测试**——早期 `classify.py` 建库时没把 `--neighbors` 传下去，
  建库恒为 1536 维、查询却是 13824 维，要到推理才抛一个跟根因无关的 cdist 维度错误。

---

有问题把 traceback 贴给我，我按报错改脚本。

---

## 面试材料与当前实验状态

- [集中实验数据总目录](experiment_data/README.md)：所有基线、GPU、VisA、triage、bank 和演示工件，以及逐类结果总表和文件校验清单。
- [实验协议与数据集角色](docs/EXPERIMENT_PROTOCOL.md)：MVTec 开发集、VisA 外部验证与当前口径。
- [难类失败分析](docs/FAILURE_ANALYSIS.md)：逐缺陷子类漏检与代表性标注图。
- [面试现场演示流程](docs/INTERVIEW_DEMO.md)：桌面/命令行演示步骤与预期输出。
- [结果优化计划与 GPU 实验记录](docs/RESULTS_OPTIMIZATION_PLAN.md)：平滑、少样本、校准和扰动实验。

**结果使用提醒：** MVTec 测试集已经用于多轮开发实验，应按开发集结果呈现；VisA PCB2 完成了一次锁定配置的外部检查，也不再是未触碰的 holdout。真实产线泛化仍需使用新的独立工件批次验证。

## 许可

MIT License, see [LICENSE](LICENSE). 第三方：PyTorch / torchvision（BSD-3-Clause）、MVTec AD 数据集（CC BY-NC-SA 4.0，仅限非商业研究）。
