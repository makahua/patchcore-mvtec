# 安装与数据准备

> README「快速开始」的展开版：环境、权重来源、MVTec AD 下载镜像与目录结构。

## 装环境

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
> hf-mirror 下 timm 的 `wide_resnet50_2.racm_in1k`（275 MB safetensors）再转成 state_dict（转换脚本 `tools/safetensors_to_pth.py`，纯 numpy，不依赖 safetensors 包），见第 3 步。
> 只有在**省略** `--weights` 时，脚本才回退到 torchvision 自带的 `IMAGENET1K_V1` 自动下载。
> 想换缓存目录就设 `TORCH_HOME`（本机放在 `E:\torch_cache`；注意本机没有 D 盘）。

---

## 拿数据集（MVTec AD）

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
