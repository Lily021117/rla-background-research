# 原始 RGB 视频背景分割小规模验证

## 结论：PARTIAL，不能直接扩展全量或启动正式训练

已完成固定 **8条物理轨迹、16段完整相机视频、1288个物理帧（2576个相机帧）** 的 SAM 2.1 分割、一次有界提示修正、HSV配对增强、完整叠加视频及固定抽查。未改变训练/验证划分、源RGB或运动数据。
完整双视角轨迹准入：**0条完整通过，5条需修正，3条腕部跟踪失败**。这不是8条全失败：8段前视固定抽查均有可用的保守桌面区域；4条轨迹的指定双视角时间对通过实际官方loader接口检查。但这些局部结果不能替代全序列语义验收。

这条路径可以作为**人工辅助掩码制作原型**，目前不具备无人检查批量扩至1693条的条件。下一步只需对具体失败帧补语义提示、检验传播并逐段人工复核；本轮不继续推理、不扩量、不训练。

## 数据与身份

每套件按训练manifest的数字轨迹ID顺序，取首条及首个不同任务，均为000010/000011。四套件为libero_spatial、libero_object、libero_goal、libero_10，共8个不同任务。完整来源、视频SHA和固定抽查帧在selection_and_protocol.json，选择落盘先于图像检查。原官方套件级train/val划分未动。
原始HDF5映射及完整仿真状态未知仍如实保留，但本轮直接对原始RGB逐帧分割，不要求重渲染、不借用近似XML。官方loader的224 RGB与分割记录的224 RGB哈希实际相等；没有额外翻转。

## 分割模型、提示与边界

官方SAM 2.1：[facebookresearch/sam2](https://github.com/facebookresearch/sam2)，代码`2b90b9f5ceec907a1c18123530e92e794ad901a4`；配置`configs/sam2.1/sam2.1_hiera_b+.yaml`。
权重`sam2.1_hiera_base_plus.pt`，SHA256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`，来自官方dl.fbaipublicfiles.com链接，严格加载。隔离环境不修改RLA环境；BF16、无compile、冻结所有参数。未安装自定义CUDA扩展，官方额外mask后处理关闭，使用下面显式的保守规则；这项实现选择需要在复现时保持，不是已证明的最佳设置。

本原型**只标注明确桌面**；墙面、柜体等不确定区域保持原色，不能把全部非机器人区域叫背景。每段视频均有首帧桌面正点、机器人/任务物体负点；v2增加桌面框、8段腕部增加中间关键帧。实际坐标、帧号、依据见prompts_v1/v2.json和correction_log_v2.json。提示由AI根据原图制定，不是人工真值。
logit>0，原256分辨率7×7腐蚀；到224时nearest并3×3腐蚀。腐蚀不能保证消除错误的物体内部区域。两版本及失败均保留。

## 全部视频结果

| 视频 | 平均背景覆盖 | 近空帧/总帧 | 状态 |
|---|---:|---:|---|
| libero_spatial_000010_agentview_camera | 35.5% | 0/101 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_spatial_000010_wrist_camera | 52.4% | 1/101 | NEEDS_CORRECTION |
| libero_spatial_000011_agentview_camera | 32.7% | 0/84 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_spatial_000011_wrist_camera | 21.0% | 50/84 | NEEDS_CORRECTION |
| libero_object_000010_agentview_camera | 65.2% | 0/143 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_object_000010_wrist_camera | 23.0% | 72/143 | NEEDS_CORRECTION |
| libero_object_000011_agentview_camera | 64.9% | 0/164 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_object_000011_wrist_camera | 0.3% | 161/164 | FAILED_TRACKING |
| libero_goal_000010_agentview_camera | 33.1% | 0/114 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_goal_000010_wrist_camera | 0.0% | 114/114 | FAILED_TRACKING |
| libero_goal_000011_agentview_camera | 32.9% | 0/81 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_goal_000011_wrist_camera | 2.2% | 64/81 | FAILED_TRACKING |
| libero_10_000010_agentview_camera | 48.1% | 0/383 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_10_000010_wrist_camera | 58.3% | 0/383 | NEEDS_CORRECTION |
| libero_10_000011_agentview_camera | 49.9% | 0/218 | PARTIAL_USABLE_FIXED_FRAMES |
| libero_10_000011_wrist_camera | 51.2% | 0/218 | NEEDS_CORRECTION |


平均相邻mask IoU仅是图像平面稳定性指标，不是分割准确率；腕部运动/遮挡会影响它。见coverage_by_video.csv与all_frame_metrics.csv。
明确问题包括：spatial000011腕部t27把盘子内部纳入桌面；spatial000010腕部t66有盘沿疑似误涂；goal000011腕部t40疑似炉灶边界纳入。object000011、goal000010/11存在大段可见桌面漏分。不存在依据置信度自动把这些帧签成语义正确的处理。
所有前视首提示帧及若干腕部中间提示帧仍有碎片化。libero10两条腕部大部分固定帧较好，但同样不签完整通过。

## 检查范围，不混淆像素对齐和语义正确

每视频首/中/末及均匀固定帧在推理前冻结；16段原图、v1/v2叠加固定图均作AI视觉检查。另查看4段故障腕部首/末自动异常帧；异常帧不替代固定抽查。人工标注/人工逐帧验收数量=0，完整视频已保存但未声称人工全片看过。没有像素真值，不能给出前景误涂率、precision/recall或语义IoU。
全部帧自动检查源哈希、长度、二值mask、掩码外字节、覆盖、颜色变化和时间面积跳变。最终版近空 **462/2576 (17.9%)**，近整图0；近空与语义错误均未剔除，未用它们修改数据划分。

颜色弱定义：masked RGB平均绝对变化<1/255。最终版 **747/2576 (29.0%)**；其中非近空帧285。非近空且弱、同时超过一半masked像素S<0.05的帧为0。这不意味着不存在低饱和度影响；逐帧S比例已保存，不能把所有弱变化都归因于灰色。spatial000011的Δh≈−0.00305，object000011的Δh≈0.0000868，接近零以及uint8量化是本次弱变化的重要可见因素，未重抽参数。

## 配对HSV接入与门控

保持Δh~Uniform(-0.25,0.25)，S/V不变（uint8量化误差单测），背景外RGB逐字节拷贝。完整预览按轨迹共用一个参数；实际训练reader仍按既有record/occurrence独立PCG64规则，两时刻两相机共享，重复曝光允许新参数。没有改变动作、语言、proprio、内外参或时间。
实际官方loader对8条做关闭增强等价和门控检查；spatial000010[16,33]、object000010[94,118]、libero10_000010[191,254]、libero10_000011[144,180]在**显式prototype opt-in**下通过真实RGB哈希、mask、重复读取及metadata不变检查。跨度遵循官方span=gap+1，最大64，不套用策略h1…8。其余4条拒绝，无静默整图变色。
来源记为`reviewed_model_segmentation`，并注明AI review/human_ground_truth=false；不伪装sim_geom_whitelist。默认不开启模型估计mask，未审帧/失败帧无有效registry，正式训练仍阻塞。保留旧来源与所有身份门控。单测6项通过。
本轮未重新提取DINO；以后训练须从x/x′分别提取冻结DINO特征，不能改旧缓存冒充RGB增强。

## 资源、计数与异常

两次有意分割遍历：SAM image_encoder **5168**、mask_decoder **5152**、memory_encoder **5152** 次组件前向。不能把三者相加称独立视频样本；额外16次image encoder来自8个腕部中间提示。
v2第15段导出时外部SIGTERM/exit143，完整218帧raw mask已保存。只做CPU重导出，最后未处理腕部才继续推理；无分割重放。该段218次/组件根据完整输出和执行路径推算，未持久化hook计数，明确区别。该段精确推理时间缺失，不补造。初次权重下载也被exit143中断，原官方URL断点续传完成。
v1推理90.27秒，含导出完整运行134.10秒；v2已记录15段推理84.51秒，另218帧推理计时未知。峰值torch显存约1.03GiB；nvidia-smi采样含运行库约1.2GiB，不把它称高频监测峰值。运行GPU0 UUID及前后ECC在资源记录中，未新增ECC；任务推理进程已退出。
当前原型产物约106.6MiB（含两轮、重复审计及恢复文件，源视频/权重/隔离环境另计）。全量546930相机帧线性估算单轮推理5.32–5.45小时，v1同口径推理+导出约7.91小时；不含下载、提示、人工修正，也不保证长视频同吞吐。两轮产物线性约22.1GiB，具体见cost_projection.json。
3386段全量视频若每段提示1–3分钟，约56–169人小时；每段8帧、每帧5–15秒另约38–113人小时，均为规划假设而非已发生人工工时；异常修正尚无法可靠预算。当前失败比例下，人工成本比GPU推理更需要验证。
**AE/DINO/策略前向=0；训练更新=0；backward=0；策略闭环=0；仿真重放=0。** 未创建正式训练任务。旧观测退化任务保持暂停。

## 产物与修正方式

本地根目录：`$RLA_ROOT/experiments/rla_official_background_hue/mask_pilot_20261102_01`。
- `selection_and_protocol.json`、`prompts_v1.json`、`prompts_v2.json`、`correction_log_v2.json`：选择/提示/修订。
- `v1/`保留首次结果；最终使用`v2/`前14段，最后前视在`v2_recovered_front/`、腕部在`v2_tail/`。每段`overlay_full.mp4`左上原图、右上绿mask、左下增强、右下diff×4；10fps仅是检查播放速度，不是物理控制频率。
- 各段`fixed_review.png`是原图/mask/增强/差分固定帧；`overlay_review.png`快速检查；`raw_model_masks.npz`、`masks_224.npz`与`frame_identity.json`用于复现。
- `audit_final/semantic_review.json`、`trajectory_status.json`、`abnormal_review_index.json`列明确帧号；`paired_loader_validation.json`记录实际接口验收；`prototype_mask_registry.json`只放局部抽查通过帧。
- `audit/`保留初次CPU汇总，其中全空视频的相邻IoU写为nan；`audit_final/`已改为缺失/null，未改模型输出或重跑分割。

修正人员应先打开指定完整视频和对应原图，在原256×256坐标编辑prompts_v2的副本：桌面正点、盘子/碗/炉灶/机器人负点，`frame`是原视频帧号而非展示视频时间。每次记录原因与新增关键帧，使用新revision输出；不要覆盖v1/v2。具体操作命令见prompt_editing.md。重点不是继续侵蚀以掩盖盘内部误分，而是纠正语义提示和遮挡后的传播。
本轮结束，不自动执行下一轮修正或扩展。数据路径的局部可行性不等于颜色增强或一致性方法改善控制。
