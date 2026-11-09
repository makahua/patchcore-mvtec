# 一次性脚本（保留作证据）

这里的脚本都**不会被 CI 或任何流程调用**，只用于复现当时的下载/准备工作；
留着是为了让「数据与权重怎么来的」可追溯，而不是让人日常运行。

| 文件 | 用途 | 为什么留 |
|---|---|---|
| `dl_mvtec.py` | 断点续传下载器：`python tools/legacy/dl_mvtec.py <url> <out> [retries]`，走系统代理环境变量 | 文档里的镜像下载命令就是它跑出来的；数据集已在本地（`.gitignore` 排除），重下时仍需它 |

日常会用的脚本仍在上级目录（`run_experiment.py`、`agg_*.py`、`pro_aupro.py`、`check_*.py` 等）。
