# 推到 GitHub 的三步（10 分钟）

> 目标仓库名建议：`patchcore-mvtec`（比 patchcore-demo 更像"项目"而不是"练习"）。
> 仓库描述直接抄简历那句话：
> **MVTec AD 15 类无监督缺陷检测（PatchCore 复现 + coreset/分辨率/邻域消融 + 三档判定），image AUROC 0.980**

## 1. 先把本地仓库初始化（在 patchcore-demo 目录里执行）

```powershell
cd "E:\APP\DeepSeek Harness\DeepSeek\patchcore-demo"
git init -b main
git add .
git status            # 检查一遍：不该出现 MVTec_dl、*.pth、heatmaps（.gitignore 已挡）
git commit -m "feat: PatchCore on MVTec AD 15 classes (image AUROC 0.980) + three-tier triage"
```

## 2. 在 GitHub 上建空仓库

- 打开 https://github.com/new
- Repository name: `patchcore-mvtec`
- **不要**勾选 Add README / .gitignore / license（本地已有）
- Public（简历要放链接，必须公开）

## 3. 关联并推送

```powershell
git remote add origin https://github.com/makahua/patchcore-mvtec.git
git push -u origin main
```

> 首次推送会要求认证。用 **Personal Access Token**（GitHub → Settings → Developer settings →
> Personal access tokens → Tokens (classic) → Generate new token，勾 `repo`），
> 用户名填 `makahua`，密码处粘贴 token。

## 推完之后回来告诉我两件事

1. 仓库地址（我把它写进简历抬头，替换现在没有的 GitHub 行）；
2. 是否要顺手把 **毕设（YOLO11 多目标跟踪）** 也推一个仓库——那个含 5236 张数据集训练的完整
   pipeline + FastAPI + Vue3，是"工程能力"的另一半证据，比 PatchCore 更能证明你能交付系统。

## 注意

- **不要**把 `MVTec_dl` 或 `*.pth` 提交上去（仓库会变成 5 GB，且 MVTec 许可证不允许转发数据）；
- 你本机的 `wide_resnet50_2_timm.pth` 也别提，README 里已写清怎么下载（hf-mirror + safetensors 转换脚本）。
