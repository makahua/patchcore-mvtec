# 接入 VisA 工业异常检测数据集

VisA 是 Amazon Research 发布的工业视觉异常检测数据集，包含 12 个类别、10,821 张图像及图像级/像素级标注；官方提供 1-class split 和整理脚本，可转成与本项目相同的 `train/good`、`test`、`ground_truth` 目录结构。数据集许可为 CC BY 4.0，使用或再分发时需遵守署名要求。[官方项目说明](https://github.com/amazon-science/spot-diff/blob/main/README.md) · [AWS Open Data 登记](https://registry.opendata.aws/visa/)

## 下载与解压

官方原始 tar 包约 1.93 GB。曾下载为 `VisA_20220922.tar` 并解压至 `VisA_raw`；**该 tar 已在 2026-11-09 的磁盘清理中删除（释放 1.8 GB）**，`VisA_raw` 与整理后的 `VisA_pytorch\1cls` 均保留。需要重建时重新下载：

```powershell
cd E:\面试项目文件夹\patchcore-mvtec
curl.exe -L -C - --fail -o VisA_20220922.tar `
  https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
New-Item -ItemType Directory -Force VisA_raw | Out-Null
tar.exe -xf VisA_20220922.tar -C VisA_raw
```

解压后应能找到 `candle`、`pcb1` 等 12 个类别目录，以及 `image_anno.csv`。

## 生成项目可读的 1-class split

仓库保留了 Amazon 官方的 `prepare_data.py` 和固定划分 `1cls.csv`：

```powershell
.\.venv\Scripts\python.exe tools\visa_prepare_data.py `
  --split-type 1cls `
  --data-folder VisA_raw `
  --save-folder VisA_pytorch `
  --split-file docs\samples\visa_1cls.csv
```

输出根目录为 `VisA_pytorch\1cls`，下面按类别提供 `train\good`、`test\good`、`test\bad` 和 `ground_truth\bad`。项目评测代码已兼容 VisA 的 `<stem>.png` mask 命名。

## 运行 PatchCore 评测

先用 PCB1 的少量图片确认环境和流程：

```powershell
.\.venv\Scripts\python.exe patchcore_demo.py `
  --data VisA_pytorch/1cls `
  --category pcb1 `
  --size 256 `
  --greedy 128 `
  --limit 20 `
  --calib-ratio 0.3 `
  --thr-pct 99 `
  --weights torch_cache/wide_resnet50_2_timm.pth `
  --out experiment_data/02_visa/results_visa
```

完整 PCB1 评测时删掉 `--limit 20`。数据集支持全部 12 类，把 `--category pcb1` 改成 `--category all` 即可；第一次建议保持 `--size 256`、`--greedy 128`，观察内存和耗时后再调整。

## 生成 PyQt 单图演示所需的记忆库

下面的命令使用 PCB1 官方训练集建库，输出 `experiment_data/04_memory_banks/banks\pcb1.npz`。建库时特征保存在系统内存中，减少显存占用：

```powershell
.\.venv\Scripts\python.exe classify.py build `
  --data VisA_pytorch/1cls `
  --category pcb1 `
  --size 192 `
  --greedy 256 `
  --calib-ratio 0.3 `
  --weights torch_cache/wide_resnet50_2_timm.pth `
  --out experiment_data/04_memory_banks/banks
```

建库完成后运行 `pyqt_demo.py`，在“记忆库”选择 `experiment_data/04_memory_banks/banks\pcb1.npz`，再选 PCB1 图片进行检测和热力图展示。

本机已用完整 PCB1 官方训练集建好 `experiment_data/04_memory_banks/banks\pcb1.npz`。启动界面：

```powershell
.\.venv\Scripts\python.exe pyqt_demo.py
```

在“记忆库”选择 `experiment_data/04_memory_banks/banks\pcb1.npz`，在“待检测图片”选择 `VisA_pytorch\1cls\pcb1\test\bad\041.JPG`。这张示例图当前判为 NG；界面也可直接选择其他 VisA JPEG 图片。`experiment_data/04_memory_banks/banks\pcb1.npz` 是较保守的阈值配置；若希望少漏检，可选择下方实验配置的记忆库。

## 当前 PCB1 实测结果

完整官方测试集共 100 张良品、100 张异常图。用 `size=192`、`greedy=256`、`calib-ratio=0.3` 的 PatchCore 评测，image AUROC 为 **0.8326**，pixel AUROC 为 **0.9818**。像素定位排序明显好于整图判别，说明主要瓶颈是整图分数与判定阈值。

把 `--fpr` 设为 `0.05`，会用留出的良品训练图计算更低的 OK/复检阈值。当前数据上，max 分数配置的异常直接漏判从 70% 降至 50%，但 5% 良品进入复检；另有 22% 缺陷进入复检。对应记忆库在 `experiment_data/04_memory_banks/banks_visa_fpr05\pcb1.npz`。

还可以试用 `--score-agg top5`，将最高 5% patch 分数取平均，减少单个噪声 patch 决定整图分数的影响。候选记忆库在 `experiment_data/04_memory_banks/banks_visa_top5_fpr05\pcb1.npz`。当前测试集上的 image AUROC 为 **0.8491**；在 5% 良品复检配置下，异常直接漏判 43%、进入复检 31%、直接判 NG 26%，良品复检 4%。PyQt 可以通过更换记忆库文件使用该配置。

需要重新生成该实验配置时：

```powershell
.\.venv\Scripts\python.exe classify.py build `
  --data VisA_pytorch/1cls `
  --category pcb1 `
  --size 192 `
  --greedy 256 `
  --calib-ratio 0.3 `
  --fpr 0.05 `
  --score-agg top5 `
  --weights torch_cache/wide_resnet50_2_timm.pth `
  --out experiment_data/04_memory_banks/banks_visa_top5_fpr05
```

这些结果是在已经用于探索的同一批官方测试图上得到的，只能作为实验候选，不能视作独立验证或生产保证。没有改动默认 `max` 配置。后续建议提高输入分辨率和核心集大小，并用全新留出的异常样品或产线数据验证；不要继续依据这 200 张测试图调参数。

VisA 官方划分保留了训练/测试分离。不要用官方 test 标签调 coreset、阈值或后处理。VisA 的结果应与 MVTec AD 分开汇报，说明数据集、划分及评估口径，不能将其混入原 MVTec 的 15 类 macro 指标。
