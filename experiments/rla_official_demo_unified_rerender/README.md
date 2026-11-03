# 官方示范统一重渲染派生版

本分支不再追求作者旧视频的精确复现。父实验 `b581d02` 的 PARTIAL 与原视频对应失败结论保持不变。
作者 1653 train + 40 val 清单仅是身份与划分追溯基准，不自动认证原始 demo、状态或新图像。

## 数据身份与准入

`prepare.py` 按现有 RLA→RLDS 映射查找指定任务 HDF5。来源确认要求完整过滤动作逐元素相同、任务语义一致、全长关节观测路径独立最近邻且有明确间隔，并保存全部候选和差异。路径归一化仅用于身份诊断，不用于重建或插值输出状态。该联合证据不等于获得了未发布中间 HDF5 的字节证明。歧义条目不渲染；同一个原始文件哈希+demo 跨划分时隔离，不重划。

状态直接来自该 demo 的 `states[原始过滤索引]`。已固定官方 `scripts/create_dataset.py` 在动作后记录 obs，但输出动作前 states。本协议从动作前 states 同步重建 RGB、FK 与逐帧相机，不沿用源 RGB/proprio 的行相位，也不用旧重放状态。

源 XML 仅对资产的完整逻辑相对路径作本地官方资产映射，不重建 BDDL 场景替代源模型。编译后保存 MJB，副本只改已证明无前景别名的背景纹理。共享前景纹理当前失败关闭，不擅自整图或物体改色。数据世界状态不推进。

### 画面相关的状态字段

仅 `mj_forward` 不足以恢复所有官方视觉状态。FlatStove 的炉圈显示由按钮 qpos≥0.5 决定，并由 `set_visualization` 切换 site alpha。该固定官方规则逐帧一致作用于 Clean、增强与重载副本；它不是额外光照干预。LIBERO-10 第一遍漏掉此规则的完整失败产物保留，修复重渲染独立计数。其他画面所需未记录字段若无法确定，停止该条目。

官方 loader 的 `root_poses` 恒等占位值仅为兼容字段，不作为已测机器人基座位姿；相机和 FK 不使用该占位值。`target_qpos` 尾行保留 loader 的 achieved-qpos 兼容规则，不提供或伪造未来 RGB；合法 AE 窗口仍要求 `t+h<N`。

## 时间与公平对照接口

- 训练 h∈[2,64]，验证 h∈[10,64]，当前/未来索引 `[t,t+h-1]`；它是保留帧的数量，不等于固定物理秒数。
- `frame_mapping.csv`、`camera_time.json` 保存每帧原始索引与真实时间；`PairedInterface` 另返真实时间差。未对缺失时间插值或复制末帧。
- AE-Base 以后读取 `(Clean,Clean)`，AE-Aug/Aug-Cons 读取 `(Clean,Background)`；来源、划分、帧、相机与 Clean 处理完全共用。
- 配对 handle 的来源根目录按外观显式切换，其他身份不变。调用固定官方 loader，不构建 AE 或 DINO。
- 本轮不生成模型初始化、特征缓存或训练任务。

## 运行

```bash
# 在本实验 code worktree 中；所有路径由 common.py 固定。
/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python experiments/rla_official_demo_unified_rerender/prepare.py --freeze
/data/lyc/projects/rla-wm/.venv/bin/python experiments/rla_official_demo_unified_rerender/download.py --download
/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python experiments/rla_official_demo_unified_rerender/prepare.py
/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python experiments/rla_official_demo_unified_rerender/render.py --suite libero_spatial
/data/lyc/projects/rla-wm/.venv/bin/python experiments/rla_official_demo_unified_rerender/loader_check.py --pilots
# 四小样视觉与数值验收显式登记后才可执行：
/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python experiments/rla_official_demo_unified_rerender/batch.py
/data/lyc/projects/rla-wm/.venv/bin/python experiments/rla_official_demo_unified_rerender/loader_check.py
```

`--freeze` 仅首次执行，不覆盖原协议。批量支持轨迹级断点：完整结果校验后复用；不完整产物不自动覆盖或重跑。任一来源文件失败后停止继续派发该来源，已运行的条目保留结果。CPU OSMesa 8 个工作进程、各 2 线程，不使用 GPU，不操作其他任务。

## 统计与限制

完整数值检查不等于人工逐帧视觉审阅；固定 7 时刻×双相机是 AI 抽查，完整对比视频留本地。旧视频仅展示描述性差异。发布的源 XML 未附历史资产字节哈希，本轮记录固定官方本地资产哈希，不宣称完全复现历史渲染器。

身份不全、失败或跨划分冲突均阻止宣称完整 1693 条准入；不缩小集合偷偷启动训练。公开交付仅源码、协议和脱敏汇总，不含原始状态、模型 MJB、HDF5、PNG 或视频。
