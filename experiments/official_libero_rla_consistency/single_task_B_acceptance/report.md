# 单任务普通 VLA-Adapter B：固定预算 pilot

状态：训练流程 PASS；闭环完成 2/2，成功 0/2。不是方法有效性 PASS。

## 数据身份与覆盖

`sealed_163_demo_12` 在历史 TZ manifest 的角色是 independent，后来明确划入 Step2 train；它已被用于开发，不再是独立测试。未打开其他封存 episode 的数据集内容。
本轮仅接纳当前清单明确许可的 cookie-box 任务 4 条 train（demo12/13/14/18）和 1 条 development（demo34）。没有把 HDF5 文件存在当作其他 episode 已获开发许可；20+5 目标未实现。这是已确认池的规模，不是整个 HDF5 的可用上限。
完整状态：train463/development89，共552帧；完整h8窗口431/81，共512。两种外观、两相机，共1104相机配对、2208张224px重渲染RGB。派生图像不是原始128px采集RGB，也没有冒用RLDS身份。
全部逐帧物理状态、相机、segmentation、depth配对检查通过；纹理干预在所有1104相机对生效。两种外观整条episode固定，reflection=1/MSAA=4，其余材质/灯光/物理设置不变。最终区域指标见 regional_summary_v2.json；阴影/接触无独立可靠mask，未声称逐像素机制全部归因。
区域分类修正：静态底座实际为 XML 的 robot0_link0，不按名称含 base 判断。v2 保留原始图像与全部像素，总误差/像素数量守恒，v1 原文件保留。底座内部276030个像素观测中差异为0；机器人/夹爪内部7506908个像素观测中330个变化，最大通道差31/255；其他不透明实体内部差异为0。炉灶内部133343个、边界453710个变化像素观测。机器人少量内部差异没有逐像素反事实归因，不能声称严格前景RGB不变；所有物理/几何检查仍通过。详见 regional_correction.json。
同状态FK按7个机械臂关节和2个手指qpos计算，552帧与完整场景FK核验通过（1e-12绝对数值检查）；部署采用同一机器人关节FK，不使用物体真值。raw proprio采样相位仍未知，无独立外部末端真值。

## 实际训练与官方偏离

官方RLA代码0fc6e97212442ebccffa06f1d42f5481e5f22950，VLA上游23fa0c9c159e2aa04341cdd3e924f44061311060。使用官方Prismatic LLM/projector与规定的预训练视觉骨干；action head/proprio/query/LoRA按官方模块初始化，不用Spatial-Pro或历史退化权重。
网络、BC action L1、AdamW lr2e-4、rank64/alpha128/dropout0、双视角/proprio沿用官方实现。使用官方normalize_action_and_proprio、chunk_act_obs、RLDSBatchTransform和collator；外层是派生NPZ episode adapter与受限单进程训练循环，不是未经修改的官方TFDS RLDSDataset/launcher。无RLA target、未来图像监督或auxiliary head。
明确偏离：仅4条示范/单任务；train-only BOUNDS_Q99统计；显式STOP/mask修复；Math SDPA；只用冻结背景配对，不额外启用官方随机image_aug；固定预算替代官方Spatial20000步与checkpoint sweep。
正式270成功更新，microbatch8×accum2=16外观观测/8物理窗口每更新；2160物理曝光、4320外观观测、8640相机图像，约5.0116窗口epoch。耗时411.7s。独立smoke共2更新（batch1、batch8各1），从未作为正式初始化；总实际计算更新272，不将smoke隐藏在正式270中。
峰值GPU allocated 27.04 GiB；reserved 32.18 GiB。GPU0经用户授权，UUID核验，ECC未新增。无OOM、非有限值、跳步、恢复或重放。
step0及45/90/135/180/225/270保存参数delta、optimizer、scheduler和RNG；最终重载逐参数相等。不能单独把delta当完整Prismatic checkpoint。正式源码commit见initialization_provenance.json。

## 固定 development 离线结果

|step|外观|动作L1|平移L1（归一化）|旋转L1（归一化）|夹爪L1|
|---|---|---|---|---|---|
|0|clean|0.539063|0.568332|0.538657|0.452477|
|0|background|0.530331|0.509807|0.576146|0.454457|
|270|clean|0.211225|0.169875|0.280805|0.126533|
|270|background|0.209711|0.168925|0.278294|0.126320|

全部81个development窗口用于离线诊断；统计拟合仅使用训练集。中间验证未用于延长预算、选择checkpoint或调参。

## 固定 development 闭环

仅使用demo34的state0，seed0，最多220控制步，每8步重规划；无settling动作，初始图像与对应派生帧逐元素核验。此为历史已查看development，不是标准LIBERO完整基准或独立测试。
- clean: COMPLETE; success=False; steps=220; {"translation_saturation_fraction": 0.0, "gripper_values": [-1.0, -0.0], "min_eef_to_target_center_m": 0.06955778801865506, "target_max_lift_m": 0.02119605196761032, "target_final_displacement_m": 0.11337410175771931}
- background: COMPLETE; success=False; steps=220; {"translation_saturation_fraction": 0.0, "gripper_values": [-1.0, 1.0], "min_eef_to_target_center_m": 0.06348670711791213, "target_max_lift_m": 0.031137908051031227, "target_final_displacement_m": 0.042850324752256884}

未观察到 development 任务完成；离线拟合改善不能替代基础任务能力。
没有冻结初始化的配对闭环参照，因此即使成功也不声称证明了训练的因果收益。末端接近/碗抬升仅是轨迹诊断，不等于确认稳定抓取。
固定视频抽帧的视觉观察：两种条件均有接近饼干盒上目标后偏离的运动，未见稳定抓持并搬运到盘子的证据。这是视觉观察，不据此确定接触力学根因。
已保存轨迹的夹爪统计：Clean为218次open、2次neutral、0次close；Background为210次open、10次close。训练原始帧标签为255次open、208次close，并非缺少close标签。官方符号变换的开/闭端点往返正确；这些现象不能单独证明夹爪符号错误或唯一失败根因。详情见development/offline_analysis/gripper_diagnosis.json。

## 本轮决策

数据与机器人FK接口已得到当前5条示范范围的证据；没有证明20+5或其他任务可无条件接入。
不自动启动AE-C/AE-D或策略C/D。最优先问题是建立有足够独立训练/development覆盖的单任务基础控制证据：先核清更多候选的开发许可与身份，再预先固定新的B验证预算，而不是添加辅助loss来解释当前现象。
任何后续C/D从与B相同正式初始化开始，不能续训B。正式方法结论仍需更多任务、未见背景、独立状态和多seed。

## 保护与复现

历史数据/XML/来源manifest哈希复核保持不变，历史checkpoint未被保存/覆盖。当前研究分支新增并版本化适配脚本；既有历史结果目录不变。未执行C/D、AE训练或额外rollout重试。
运行顺序：single_task_preflight.py → render_admitted.py → accept_rendered_pool.py → baseline_update_smoke.py（隔离batch1、8）→ baseline_update_smoke.py --mode formal --batch-size 8 → --mode development。所有缓存/视频/模型在数据盘；公开可分享材料不包含权重、图像、轨迹。
