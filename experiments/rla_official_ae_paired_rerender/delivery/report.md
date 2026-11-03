# 官方 LIBERO AE 配对背景重渲染：来源恢复与准入报告

## 结论：PARTIAL；全量准入未通过，不启动训练

保留原1653训练＋40验证、40任务、273465帧。**1693/1693转换轨迹已认证到源RLDS episode；4条固定小样取得唯一原HDF5动作候选并完整重放；1条完成候选配对渲染和官方loader读取，但最终恢复状态后的Clean直对原视频失败。最终准入0/1693。** 没有把动作相同、同名任务或同帧数单独当作示范认证。没有将候选重放状态冒充作者保存的完整状态。

Object/000010（milk→basket）143帧为本轮唯一已生成的候选配对原型，**不是合格原样本增强**：连续重放阶段通过，但最终重新恢复state、FK后渲染的腕图有12帧超过预设p99≤40门槛，最大56；最大MAE仍仅4.205/255。不能因为“接近”或A≈B、B≈C就宣称A≈C通过。Spatial图像门槛未过；Goal、10有数值/图像实质偏离。四条未全部通过，因此全量批处理为**NOT_STARTED_BLOCKED_BY_PILOT_ADMISSION**，不是因磁盘或GPU不足。

## 固定代码与真实数据链

- RLA `0fc6e97212442ebccffa06f1d42f5481e5f22950`；公开包 `xyzhang368/RLA-WM@5fc55b89ba14e85c1d8ac064fd8a6eb11b5329f8`。复用既有转换包与哈希证据，不重新下载10GB归档。
- RLDS `openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`：Spatial复用本地，其他三套单独下载并核验。TFDS4.9.3 / TensorFlow2.15.1 CPU解码。
- 原HDF5来自 `yifengzhu-hf/LIBERO-datasets`：Spatial/Object/Goal固定`97773100c1474cd0d686ebd173cc0e4fd5442466`；该版本没有LIBERO10，事前查元数据后将唯一所需10任务文件固定到`f13aa24a3da8c43c7225569f28c562979fa0e35a`。只下载四个任务文件；未用Cosmos/旧单任务派生样本替代。
- 运行环境：官方LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`，robosuite1.4.1、MuJoCo3.3.2、OSMesa CPU。无LIBERO-Plus、无环境版本搜索。发布数据未附确切再生成环境/资产字节哈希，因此这是有验证门槛的候选复现，不是声称完全相同的作者环境。
- RLA README/16x64配置/转换器/下载器/DATA.md已核验，哈希见evidence_manifest。DATA.md仍有“未上传”旧说明，以固定HF实际文件为准。当前下载只用官方HF，没有启用旧脚本的镜像回退。

链路：原始公开HDF5 → OpenVLA regenerated--no_noops（该中间文件/状态未取得）→ RLDS → 作者转换视频。RLDS file_path实际给出任务HDF5路径，**不含demo编号**；RLDS observation.state为8维机器人EEF/夹爪，不是全场景MuJoCo state。

## 全量映射与划分

| suite | split | trajectories | frames | rlds_sources_matched | hdf5_demo_candidates | candidate_state_sequences | original_state_sequences | clean_correspondence_pass | paired_sequences |
|---|---|---|---|---|---|---|---|---|---|
| libero_spatial | train | 422 | 51732 | 422 | 1 | 1 | 0 | 0 | 0 |
| libero_spatial | val | 10 | 1238 | 10 | 0 | 0 | 0 | 0 | 0 |
| libero_object | train | 444 | 65552 | 444 | 1 | 1 | 0 | 0 | 0 |
| libero_object | val | 10 | 1432 | 10 | 0 | 0 | 0 | 0 | 0 |
| libero_goal | train | 418 | 50545 | 418 | 1 | 1 | 0 | 0 | 0 |
| libero_goal | val | 10 | 1497 | 10 | 0 | 0 | 0 | 0 | 0 |
| libero_10 | train | 369 | 98711 | 369 | 1 | 1 | 0 | 0 | 0 |
| libero_10 | val | 10 | 2758 | 10 | 0 | 0 | 0 | 0 | 0 |

逐条清单：`source_mapping.csv`。验证ID仍是每套件0…9，不是每任务10条。原盘点的有效起点/时间对统计原样核对；没有更换为约2000条原示范、libero90或历史60条子集。

全1693条以完整action、joint+gripper、EEF位置数组签名找唯一候选，再比对语言及双视角像素。普通轨迹每相机固定5帧，四pilot每帧比对。所有最终映射通过冻结像素/数值门槛；这不是普通轨迹全帧几何认证。直接TFDS枚举与转换编号**不总相同**，例如Spatial枚举32对应转换54；转换器分worker读取TFDS切片，不能把直接枚举号当转换ID。使用真实TFDS图像解码器后验证，不能拿PIL解码差异误判身份。

每条映射保留RLDS枚举索引、源任务文件、数值/图像覆盖、原视频hash、metadata hash。公开表将上游私人根路径脱敏，只保留逻辑任务路径及原字符串hash。4条以外尚未连接到具体demo_N，不把知道任务文件称为已找到完整源示范。

## 时间、XML和状态：不是简单逐行贴状态

按公开OpenVLA再生成脚本：env.reset → 设置原states[0] → 10个零运动/开夹爪等待 → 按“运动范数<1e-4且夹爪未变化”过滤noop → 保存动作前obs → 执行动作。只保留成功重放的源生成规则已记录；**本轮不再按重放结果筛选或替换四条样本**。

该脚本首个保存state直接复制等待前原states[0]，但首个RGB/机器人obs来自等待后；所以不能拿再生成states首行直接配首图。本次四个原initial时间均0.25s，重放候选首时刻0.75s；Goal第一个保留动作原索引为1，其余为0。没有删首帧，没有整体平移序列，没有插值。候选时间步0.05s，真正来源身份仍靠上述多模态检查。

原HDF5 XML逐条保留，源/运行joint名称与顺序核对；原资产的完整逻辑相对路径映射到已固定本地资源，保存实际字节哈希，源历史字节是否同一仍未知。公开生成器按BDDL建环境并设置初态，不加载原demo XML；因此本次原XML与运行XML同时保存并比较，而不是随意把一个XML贴到另一份数据。body初始位置等存在差异，不能仅据state维度宣称模型相同。详见每条`asset_and_model_comparison.json`，不要把未使用相机差异直接说成失败原因。

## 四条完整小样结果

| suite | demo_candidate | frames | numeric_gate | pixel_gate | max_joint_error | max_eef_error_m | front_max_mae | wrist_max_mae | clean_admitted |
|---|---|---|---|---|---|---|---|---|---|
| libero_spatial | demo_6 | 101 | True | False | 0.0016329812568134372 | 0.0004279535177034166 | 5.046549479166667 | 6.5258534749348955 | False |
| libero_object | demo_11 | 143 | True | True | 0.0016231414723364423 | 0.0004825989749909221 | 3.1478118896484375 | 3.2350565592447915 | False |
| libero_goal | demo_19 | 114 | False | False | 0.046493052289852455 | 0.011610547786955916 | 8.310104370117188 | 19.014002482096355 | False |
| libero_10 | demo_24 | 383 | False | False | 0.04371761040529698 | 0.011527393959957416 | 7.995849609375 | 16.86986796061198 | False |

门槛在小样前冻结：逐帧最大关节绝对误差≤0.002、EEF位置分量误差≤0.002m；逐帧每视角RGB MAE≤8、p99≤40（0…255）；还需视觉检查事件、遮挡、物体与相机对应。阈值不是所有渲染环境的通用定律；Spatial失败可能包含渲染/纹理敏感性，但本轮没有看结果后放宽它。Goal及10的机器人/腕相机误差还提供了像素以外的阻断证据，不能只归为视频编码。

自动检查覆盖四条全部741帧。AI视觉：Object全部143帧六张逐帧缩略联系表＋7个固定高分辨率双视角对照；Spatial/Goal/10分别固定7/8/7帧高分辨率抽查。不是人工观看了全部视频，也不是分割/物体位姿真值。Object接近、夹持、搬运、篮中放置及干扰物罐子倾倒时序未见明显错位；Goal柜体/腕相机可见差异，10在中段操作后姿态分化。原视频与重放候选的完整对比视频四套件均保存在`video_index.csv`指向位置；失败样本没有伪造增强视频。

## 背景颜色实现与唯一已完成原型

背景按具名geom白名单选择，不用SAM；`table_visual`与六个wall视觉geom。只深复制所引用material和texture并改背景geom引用；其余前景共用引用不变。HSV只改H，S/V保持（uint8量化另有误差），纹理坐标/空间图案、反射、阴影、灯光不关闭不重写。所有原模型物理数组、网格、相机、灯光与改色编译模型逐元素比较通过。每条固定一个由独立SHA256/PCG64产生的delta_h，训练seed20261103、验证20261104；同轨迹所有时刻/两相机共享，完整参数见清单。

Object实际没有table_visual，只命中六面墙。没有把floor/机器人/商品/篮子自动加入白名单。墙纹理低饱和比例约86.7%；实际前视平均RGB变化0.146091/255，腕相机0.000000/255（143/143帧零变化，墙不可见）。这是**可追溯但较弱的墙面增强**，不声称对象任务有明显全视角改色，也没有自动加饱和度、改光照或扩白名单以追求效果。

对同一恢复state分别在物理相同的两个编译模型中渲染：286次状态设置、572次单相机RGB render、0次积分。每次render前后time/qpos/qvel完全不变，两路cam/body世界几何完全相同。Clean重渲染与候选视频比较通过，**但最终直接与作者原视频比较失败**（12个腕帧，见paired_final_clean_original_pixel_check.csv），不能用两级容差相加豁免。提示需要核查缓存obs与保存state/重新FK的相位，尚未确认它就是差异的唯一来源。实际c2w逐帧计算，wrist不静态；图像沿用OpenGL旋转180°、负fx约定。后台接口独立使用同步FK生成qpos/EEF，来源动作和语言原样保留，原metadata不覆盖；所有派生字段差值见export_audit。

已用固定官方TrajectoryDataset读取这条配对的合法h=2/10/64窗口，两时刻×两相机×3×224×224，时间索引、qpos、target_qpos、K与官方名为w2c的c2w字段逐元素一致。正式采样窗口保留官方尾部规则t+h<N，不跨episode、不用末帧拼未来；旧target_qpos末行重复仅为官方既有机器人字段，非新增有效AE未来目标。原视频只读；导出PNG流无损，比较MP4仅用于查看。保留的全白foreground流明确是官方占位，未用作背景真值。

AE-Base/Aug/Aug-Cons以后必须共享同一套**验收通过的派生Clean**、源ID、划分、时间采样和同步几何，不让只有增强组承担重渲染变化。接口见`fair_comparison_interface.json`。目前候选生成1条，最终训练准入0/1653、验证0/40，**不具备正式训练条件**。

## 具体阻塞与最小补齐方案

1. **episode身份**：全RLDS已找回，但其metadata无demo_N；仅四候选做过HDF5全动作join，Object连续重放阶段接近，但最终恢复后原视频对应仍未过关。
2. **生成时完整状态**：公开RLDS只有机器人信息；作者使用的regenerated--no_noops中间HDF5及其正确首图时刻state尚未取得。最小补齐是源再生成文件的demo-ID映射、逐帧完整场景state、生成XML/资产标识和obs采样时刻（首帧需另核实）。
3. **环境/资产和重放**：若拿不到上述状态，需要生成RLDS时的确切LIBERO/robosuite/MuJoCo/资产与reset初始化历史，才能在原四小样上复现；本轮仅测试一个固定官方环境，无无限版本搜索。当前不能断言漂移统一由哪一版本/物理因素造成。
4. **时间映射**：已建立公开代码的候选生成规则及原动作过滤索引，不意味着未通过轨迹取得真实图像时刻。不能用重新切帧或别的示范补数。

这不再是“缺少mask”。原视频、原HDF5、所有失败图表及候选状态均保留。暂停SAM，未创建任何AE训练任务；也没有因单条通过而偷偷缩小训练范围。

## 下载、资源、失败和批量预算

新增完整校验的数据文件合计 **12,226,065,269 bytes（11.386 GiB）**：三套RLDS 8,316,052,983 B，四HDF5 3,910,012,286 B。这是唯一文件净下载量；网络中断/断点重试的线缆传输总量未计量，不能冒称精确。另有官方git checkout，磁盘量与证据见resource_estimate，未把本地复用Spatial算新增下载。

全量两外观×两相机256²原始RGB约200.3 GiB，另加PNG/缓存、状态、原视频及检查视频。按本轮CPU候选重放/配对速度外推，串行约43.3h重放＋53.9h配对渲染；仅条件性容量规划，不是已放行ETA。磁盘充足，但来源/对应门槛阻断批量。支持RLDS已验记录断点续作；已验配对目录拒绝覆盖，批量执行尚未启动/验收。

工程故障完整列于incidents：官方网络中断，Object CPU映射exit143后按持久化前缀核验继续，最初错误JPEG解码/枚举假设修复；一次loader字段名和一次边界检查脚本修正。未重放物理轨迹来刷通过，没有删除失败。

## 实际计数与交付

| metric | value |
|---|---|
| render_calls | 3656 |
| physics_substeps | 19525 |
| state_restores | 433 |
| action_control_steps | 741 |
| settling_control_steps | 40 |
| replay_trajectories | 4 |
| candidate_frames | 741 |
| pair_and_export_integration_steps | 0 |
| kinematic_forwards_for_export | 143 |
| paired_physical_frames | 143 |
| paired_rgb_frames | 572 |
| paired_sequences_generated_candidate | 1 |
| paired_sequences_admitted | 0 |
| AE_forwards | 0 |
| DINO_forwards | 0 |
| policy_forwards | 0 |
| AE_updates | 0 |
| DINO_updates | 0 |
| policy_updates | 0 |
| backward | 0 |
| policy_closed_loops | 0 |
| SAM_forwards | 0 |
| rolling_replay_reproduction_failed_sequences | 3 |
| restored_clean_final_gate_failed_sequences | 1 |
| gpu_jobs | 0 |

4条源动作开环重放，共741个原动作＋40个稳定等待控制步，不是策略闭环，也不报告策略成功率。恢复741个候选frame state，不是741个已认证原始state；最终完整准入0。候选配对143物理帧/572张双外观双视角RGB，全部保留为未准入。FK是MuJoCo几何计算，不是AE/DINO/策略模型前向。状态设置包括导出同步FK；render_calls包括env隐式观测和显式相机渲染，不能仅按保存视频帧数算。

本地：`/data/lyc/projects/rla-wm/experiments/rla_official_ae_paired_rerender/run_20261103_01`。`delivery/source_mapping.csv`、`pilot_validation.csv`、`coverage_by_suite_split.csv`、`pairing_manifest.csv`、`video_index.csv`、`evidence_manifest.json`及各pilot证据可直接检查。完整视频/状态/下载数据只留本地。全部模型前向、更新、backward和策略闭环=0，GPU任务=0，CPU进程结束后释放本任务资源。代码/脱敏表及报告提交新分支，远端核验另见sync记录。本轮完成即停止，不自动训练或扩环境搜索。
