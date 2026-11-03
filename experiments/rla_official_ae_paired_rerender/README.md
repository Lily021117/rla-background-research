# 官方 AE 数据配对重渲染（未放行全量训练）

报告见 [delivery/report.md](delivery/report.md)。范围固定为作者转换包的1653训练＋40验证，不使用历史单任务派生数据补数。

本轮：1693条转换轨迹→RLDS核验；四条固定源动作重放；Object一条143帧候选配对及loader跑通，但最终恢复后Clean直对原视频失败，另三条在前一层已失败。**最终配对准入0条**。重构状态不冒充作者原始生成状态。SAM、模型推理、训练及策略闭环均未执行。

## 路径和环境

`prepare.py` 中显式记录 ROOT/RUN/OFFICIAL，历史目录不覆盖。
本地 RUN：`/data/lyc/projects/rla-wm/experiments/rla_official_ae_paired_rerender/run_20261103_01`。

- 文件下载、报告、官方loader：项目 `.venv/bin/python`。
- TFDS身份匹配：项目 `.venv-sidecar/bin/python`（CUDA禁用）。
- MuJoCo/OSMesa重放、编译和渲染：`/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python`。
- LIBERO checkout 位于 RUN/libero_official，固定8f1084e…；无LIBERO-Plus、无环境升级。

## 可复现入口（不要对已存在产物再次运行生成器）

```bash
# 首次冻结；当前已经执行，不能覆盖本轮冻结协议
python experiments/rla_official_ae_paired_rerender/prepare.py

# 官方固定revision下载，支持相同来源的校验/断点续作，不用镜像
python experiments/rla_official_ae_paired_rerender/download_rlds.py
python experiments/rla_official_ae_paired_rerender/download_pilot_hdf5.py

# 四suite分别执行；枚举不是原转换ID，按完整数值签名+双RGB匹配
python experiments/rla_official_ae_paired_rerender/map_rlds.py --suite libero_object

# 每suite只固定转换ID000010的唯一动作候选；不自行换候选
python experiments/rla_official_ae_paired_rerender/replay_pilot.py --suite libero_object
python experiments/rla_official_ae_paired_rerender/post_audit.py --suite libero_object
python experiments/rla_official_ae_paired_rerender/review_contact.py --suite libero_object

# 资产隔离可单独验收；--render须numeric、pixel和真实视觉记录全部通过
python experiments/rla_official_ae_paired_rerender/paired_renderer.py --suite libero_object
python experiments/rla_official_ae_paired_rerender/paired_renderer.py --suite libero_object --render
python experiments/rla_official_ae_paired_rerender/export_pair.py --suite libero_object
python experiments/rla_official_ae_paired_rerender/check_pair_loader.py --suite libero_object
python experiments/rla_official_ae_paired_rerender/final_check.py
python experiments/rla_official_ae_paired_rerender/deliver.py
```

各命令使用对应环境；不得把示例解释为批量训练命令。渲染/导出目录若存在即拒绝覆盖。CPU映射恢复必须重新校验已保存前缀；图像匹配使用实际TFDS解码，不用PIL近似替代。

`paired_renderer.py`同状态恢复两个物理一致的引擎，只对具名背景geom复制材质/纹理引用。不会调用step；不关闭反射/阴影；保留弱增强。Object场景没有table_visual，本轮只改六面墙，没有擅自改floor。

`export_pair.py`额外143次状态设置/FK用于同步几何元数据，计入资源记录；不是环境积分或模型前向。官方流采用无损PNG目录，读取仍为原官方dataset。比较视频不用于训练。复制的全白foreground是官方占位，绝非真实mask。

## 全量停止门控

四小样最终均未通过，全量渲染/AE训练禁止。连续重放与再恢复render分开检查，容差不具备传递性；Object最终12个腕帧超门槛，具体证据保留。需要补齐再生成时状态/XML/资产/观测相位证据后，另行处理失败，不在本轮修改门槛或搜索环境版本。

Git仅包含代码、CSV、JSON、Markdown。下载源数据、完整XML/状态、PNG、MP4、NPZ留在RUN，不提交。
