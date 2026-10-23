# 训练数据覆盖与夹爪失败诊断

结论：数据不足来自旧脚本选取范围；已实际扩充并冻结20条train+5条development。没有发现夹爪标签/符号/部署命令错误。当前动作闭合已学会，但未来chunk闭合时机仍未学准，闭环接近位姿也存在偏差。此为定位证据，不是唯一根因。

## 1. 数据覆盖的真实原因

源任务cookie-box共有50条示范。按旧seal和释放记录，仅2条仍封存；另5条属于Step1冻结test，均只核对身份与schema元数据，不读取其数据内容。43条允许开发，其中38条此前只因旧split/cache入口没遍历而未使用；没有以缺RLDS身份哈希排除。上一轮把4+1当作导出范围过于保守，现已修复。

43条均有完整states/actions、双相机字段与原始model_file XML；逐条state/action有限性、长度、20Hz时序及资产模型预检通过。时间/字段/资产/长度预检排除0。选定25条完整配对回放的几何、segmentation、depth、FK检查均通过；其余18条是固定20+5预算后的候选，不是被判坏数据，未对它们宣称完整渲染验收。保护身份内容未读取；仅metadata不等于其数据已验收。

新manifest入口从原始HDF5/XML生成派生ID，资产仅做路径重定位，背景仅修改tex-table/tex-wall文件引用；不依赖旧geometry缓存。原4train/1dev角色不变，新增加16train+4dev，按照预先固定的SHA256(seed|source_demo)顺序选择，不用成功率或模型得分。源rewards/dones作为来源记录保留，不用它筛选。

冻结数据：train2050帧/1890窗口，development506帧/466窗口；共2556物理帧、2356完整h8窗口、5112相机配对、10224张RGB。实际新增2004帧/8016张RGB，其余复用既有5条。20条train有22个open→close事件，5条dev有6个。每条示范及所有外观/窗口保持同一split。

数据保留原反射=1、MSAA=4，允许纹理引起的间接RGB响应；不是严格前景RGB固定。每帧K/逐帧wrist外参、state/proprio/action/camera映射与原始来源均保存。raw末端派生传感器刷新相位仍未知，训练输入继续使用同一保存状态的机器人关节FK，不声称恢复原RLDS像素或raw proprio。

## 2. 夹爪链路与监督覆盖

原始-1=open/+1=close → 官方clip+invert得到1=open/0=close → Q99最后维mask=False → 无sigmoid连续回归 → 不反归一化夹爪维 → -sign(2*p-1) → 仿真+1=close/-1=open，等于0.5时neutral。与官方transform逐元素一致；真实batch动作标签等于源actions的正确chunk，没有把两指qpos当动作。全部有效8×7动作参与BC L1。源码位置及数值范围见gripper_pipeline_audit.md。

旧4条train共463帧，open255/close208；5个闭合事件全部包含在8个合法chunk位置，每事件实际得到80次动作槽监督（含两外观），当前观测曝光10次。没有闭合事件被padding或mask删除；每条最后一个被省略的动作均为open。独立示范仍只有4条、闭合事件只有5个，重复80次不能增加独立事件。末尾release当前观测窗口偏少，详细阶段代理覆盖见phase_window_coverage.csv，未把command/lift代理当确认抓取真值。

## 3. 固定观测模型诊断

4份checkpoint：step0/45/135/270，按预先固定的全512物理窗口×2外观推理；原来的train-only normalization不变。eval/inference_mode，全部参数冻结，Math SDPA。没有optimizer、反向传播或额外策略rollout。

最终checkpoint当前动作位置0的结果（窗口等权、窗口相关，不能当独立统计样本）：

|集合|外观|平移L1（归一化）|旋转L1（归一化）|夹爪L1|闭合召回|开口召回|
|---|---|---|---|---|---|---|
|train|clean|0.1969|0.2164|0.1320|91.35%|97.31%|
|train|background|0.1933|0.2150|0.1297|90.87%|96.86%|
|validation|clean|0.1898|0.2940|0.1123|100.00%|97.67%|
|validation|background|0.1885|0.2955|0.1127|97.37%|97.67%|

step270在所有5个train闭合事件与1个dev闭合事件的当前位置上均输出close，两种外观均正确（10/10和2/2事件-外观对）；阈值切换一般提前1–2步。因此“完全不会闭合”不受示范观测证据支持。训练集总体close召回约91%，不代表所有时刻无误。

关键局部欠拟合：最后一个chunk位置7在必须提前7步预测首次关闭的样本上为train 0/10、dev 0/2正确。平均预测边界比理想未来边界晚4.7/5.0步（0.235/0.250秒）。这来自固定示范离线轨迹，不是直接测得的闭环触发延迟。稳定闭合长区间的低L1会掩盖切换边界误差。图见plots/train_gripper_timeseries.png、validation_gripper_timeseries.png与close_event_chunk_timing.png。

16个部署入口对照中解码动作差0、夹爪阈值分歧0；同一输入重复推理差0。旧2条rollout共440条命令全量与预测后处理一致。没有D类证据：close不是被接口翻成open。

## 4. 闭环证据的边界

沿用原视频与保存状态，未重跑。Clean无关闭；Background仅10个关闭，分6个短段。两者执行chunk与220步限额符合冻结规则；没有持续动作饱和或长时间末端停滞证据。记录的关闭步手指确实朝关闭方向变化，最后4步缩小约19.8mm。

两者接近/接触正确的目标碗，但最近专家闭合相对位置分别仍差3.97cm、2.44cm，对应姿态差0.171rad、1.230rad。不能宣称在完全正确抓取位姿下只缺夹爪关闭，也不能把没有闭合单独当head故障。完整controller/actuator内部状态未保存，不能复原每个控制子步。详见rollout_failure_analysis.md。

综合定位：当前观测支持C；未来chunk边界支持A的局部版本；当前唯一dev示范不支持B；没有发现D。数据覆盖不足、未见状态下的接近/姿态偏差与未来切换欠拟合是有证据的候选，不是已经分离的唯一因果原因。

## 5. 下一轮只改变什么

只检验独立训练示范覆盖4→20，建议新的B4/B20共同step0受控对照，各270更新、同LR/seed/loss/阈值/chunk/曝光预算，保留原BC目标。两组共享从20条train拟合的同一统计并记录额外统计曝光；这样不把历史旧统计造成的额外裁剪混成数据量收益。旧B270只作历史参照。具体预注册草案见next_B_experiment.md与next_training_config_B4/B20.json。

现有合法数据足够20+5，不需要新采集。本轮未执行下一轮训练，也未启动AE-C/AE-D或策略C/D。

## 6. 保护、资源和交付

本轮optimizer updates=0，backward=0，新增策略闭环=0。GPU0仅冻结推理，峰值allocated 7.76GiB，reserved 11.03GiB，ECC无新增。数据生成是CPU保存状态重渲染，不执行action replay，不推进物理时间。全部输出/缓存在数据盘，旧checkpoint、数据与历史报告未覆盖。

metadata封存排除表、逐episode可用性、来源/帧身份与内容hash、分区完整性、25条真实loader的50个配对batch验收均保存。未使用新dev的策略表现选择数据或超参数。新dev仍是development，不能称全新独立测试。
本轮代码/脱敏汇总推送同一私有研究分支；实际GitHub commit和远端核验见github_sync_receipt.json。权重、完整图像、原始轨迹、逐样本预测和凭证不上传。
