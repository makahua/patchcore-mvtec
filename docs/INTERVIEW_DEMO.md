# 面试现场演示流程

## 演示目标

用一张 MVTec bottle 异常图完整展示“输入图 → PatchCore 特征/记忆库检索 → OK/REVIEW/NG → 热力图”，并能解释分数阈值来自独立良品 calibration。该流程是桌面原型，不应表述为产线部署。

## 准备

当前工作区已有 `experiment_data/04_memory_banks/banks/bottle.npz`、权重文件和 MVTec 图像。桌面演示需要 `.venv` 中的 PyTorch、torchvision、Pillow 和 PyQt5。

启动界面：

```powershell
cd E:\面试项目文件夹\patchcore-mvtec
.\.venv\Scripts\python.exe pyqt_demo.py
```

界面中选择：

- 记忆库：`experiment_data/04_memory_banks/banks/bottle.npz`
- 权重：`torch_cache/wide_resnet50_2_timm.pth`
- 图片：`MVTec/mvtech_anomaly_detection/bottle/test/broken_large/000.png`

预期显示 NG、异常分数约 59.66、阈值约 OK≤39.94 / NG≥40.95，以及热力图。CLI 输出保存在 `experiment_data/06_demo_outputs/demo_out_interview/`。分数会因设备、依赖版本和计时环境有轻微差异。

## 命令行备用演示

如果 GUI 环境不可用，可运行同一张图的推理并保存三联图：

```powershell
.\.venv\Scripts\python.exe classify.py infer `
  --bank experiment_data/04_memory_banks/banks/bottle.npz `
  --image MVTec/mvtech_anomaly_detection/bottle/test/broken_large/000.png `
  --weights torch_cache/wide_resnet50_2_timm.pth `
  --out experiment_data/06_demo_outputs/demo_out_interview
```

本轮已验证该命令输出 NG，结果图为 `experiment_data/06_demo_outputs/demo_out_interview/broken_large_000_NG_59.7.png`。也已用 Qt offscreen 驱动界面中的异步推理路径，确认能完成并显示热力图。

## 面试讲解顺序

1. 说明建库只使用良品，异常图只用于推理展示。
2. 指出热力图是异常相似度响应，分数不是缺陷概率。
3. 解释 OK / REVIEW / NG 的双阈值如何从校准良品确定。
4. 展示一个失败样例，并说明 MVTec 当前是开发集，真实工件仍需独立批次验证。

避免声称此 GUI 已有 PLC、相机触发、节拍联锁或产线报警集成。
