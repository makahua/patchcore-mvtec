# 复现说明

## 环境与依赖

推荐 Python 3.12；建库与推理需要 PyTorch、torchvision、NumPy、Pillow。仅运行离线报告和阈值稳定性分析需要 NumPy。PyQt6 仅用于桌面演示。

```powershell
pip install -r requirements.txt
pip install PyQt5
```

权重文件和 MVTec AD 数据集不随仓库分发。先按 README 的数据/权重步骤准备文件。数据根目录必须直接包含类别目录（例如 `E:/MVTec/bottle/train/good`）。

## 复算已提交结果

```powershell
python classify.py report --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2
python analyze.py experiment_data/00_baselines/results_best
```

前一条只读逐图分数 JSON，不需要 GPU、数据集或权重。后一条重新计算 image AUROC、pixel AUROC、PR-AUC 和 pooled/macro 口径。

## 重跑单类

```powershell
python patchcore_demo.py --data E:/MVTec --category bottle --size 320 --greedy 2000 --calib-ratio 0.3 --thr-pct 99 --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/01_mvtec_gpu/runs/reproduce
```

需比较时固定随机种子、权重文件、输入尺寸、coreset、标定比例和数据清单。建议将不同实验写入不同 `--out` 目录，避免覆盖逐图分数。

## 阈值波动诊断

```powershell
python tools/threshold_stability.py --scores experiment_data/00_baselines/results_best/scores --fpr 0.005 --ng-ratio 1.2 --repeats 1000 --seed 2026 --out experiment_data/01_mvtec_gpu/runs/reproduce/threshold_stability
```

脚本对已有 calibration 分数做有放回重采样，观察阈值和固定 test 集上的过杀/漏检/复检率如何变化。由于每次重复都复用同一 test 集，这只是标定敏感性诊断，不是独立置信区间，也不能替代更大的独立良品集。

## PyQt 单图演示

```powershell
pip install PyQt5
python classify.py build --data E:/MVTec --category bottle --weights torch_cache/wide_resnet50_2_timm.pth --out experiment_data/04_memory_banks/banks
python pyqt_demo.py
```

界面选择 `experiment_data/04_memory_banks/banks/bottle.npz`、图片和（可选）权重文件后运行。界面会显示原图、热力图、异常分数、OK/REVIEW/NG 结果及阈值。检测过程放在工作线程，不会冻结窗口。
