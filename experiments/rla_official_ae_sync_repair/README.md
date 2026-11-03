# 有界同步修复

独立分支 `experiments/rla-official-ae-sync-repair`，基于 `835cd3116f12d2b6a5f0c003e2be5ece6a68a918`。
所有模型推理/训练与策略闭环禁止。代码只依赖历史数据和固定官方仿真环境，使用 CPU OSMesa。

运行次序（已执行的目录存在即拒绝覆盖，不能照抄重跑源动作）：

1. `common.py` 冻结协议。
2. `offline.py` 检查旧741帧。
3. `capture.py --suite ...` 每条至多一次运行时捕获；Object发生一次等待后记录故障，未重试。
4. `model_snapshot.py` 仅为缺少精确MJB的Object/Spatial重建相同模型，禁止任何step。
5. `color_audit.py --suite ...` 完整离线状态渲染。最终使用MJB副本、背景专用texture字节修改；旧XML实现保留为失败证据，不用于正式准入。
6. `index_pairs.py`、`deliver.py` 只做索引和CPU汇总。

CPU解释器：`/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python`。
实验目录：`/data/lyc/projects/rla-wm/experiments/rla_official_ae_sync_repair/run_20261103_01/`。

关键限制：A自恢复通过不蕴含B作者原视频对应通过。不得把这里的诊断成对RGB接入正式AE训练。完整控制器/观测缓存不属于旧候选状态，不能用渲染状态快照冒称可继续执行的控制器checkpoint。所有工程失败、实际计算与未完成证据见交付报告。
