# 夹爪完整链路审计

结论：没有发现开/闭符号、动作字段、normalization或部署阈值接入错误。发现的数据范围限制是旧split/cache入口只覆盖4+1，已通过显式新manifest与原始XML导出修复。

|阶段|实际定义与范围|执行证据|
|---|---|---|
|原始动作|HDF5 `actions[:,6]`，检查的旧5条为-1=open、+1=close；与frame/action缓存逐元素相等|`gripper_label_coverage.csv`；不是两指qpos|
|dataset transform|`1-clip(raw,0,1)`；因此1=open、0=close；连续区间有损clip但本数据二值端点可往返|本地`single_task_training.py:22`；官方`libero_dataset_transform` `upstream@23fa0c9c/prismatic/vla/datasets/rlds/oxe/transforms.py:827`；全5条数值相等|
|normalization|BOUNDS_Q99，action.mask=[T,T,T,T,T,T,F]；夹爪不做Q99缩放，min!=max故不触发unused-dimension归零|`upstream@23fa0c9c/prismatic/vla/datasets/rlds/utils/data_utils.py:52`；`batch_pipeline_audit.json`|
|batch与loss|8×7连续动作，完整有效chunk；所有夹爪维度进入mean L1，维度占比1/7，无额外gripper权重|`upstream@23fa0c9c/prismatic/vla/datasets/datasets.py:30`；`upstream@23fa0c9c/prismatic/util/data_utils.py:96`；`upstream@23fa0c9c/vla-scripts/finetune.py:418`|
|policy output|连续回归数值，无sigmoid；允许略超[0,1]，不能把数值0.7当已校准概率|所有checkpoint分布见`checkpoint_gripper_metrics.csv`|
|unnormalization|同一mask最后维False，夹爪保持原值；仅前6维按旧4条train统计解码|`upstream@23fa0c9c/prismatic/extern/hf/modeling_prismatic.py:786`|
|后处理|`command=-sign(2*p-1)`：p<0.5→+1 close；p>0.5→-1 open；p==0.5→0 neutral|`upstream@23fa0c9c/experiments/robot/libero/run_libero_eval.py:269`；`upstream@23fa0c9c/experiments/robot/robot_utils.py:151`|
|仿真入口|ControlEnv.step→SingleArm.control→Manipulator.grip_action→PandaGripper.format_action→actuator ctrlrange映射|robosuite1.4.1 `robots/manipulator.py:16`、`models/grippers/panda_gripper.py:43`|

PandaGripper按sign(command)更新内部current_action，speed=0.01；+1为关闭、-1为打开。两个手指qpos是proprio的最后两维，来源和形状都与动作第7维不同，未作为动作标签。

16个固定观测检查（step0/270、train/dev、起点/闭合事件、Clean/Background）：官方loss路径与实际predict_action入口解码后的动作最大差0，重复推理差0；没有未来图像或示范动作作为部署条件。历史两条rollout的440条执行命令与记录预测经官方后处理后逐元素相等。

Background的10条关闭命令均未被后处理翻成打开；记录的手指分离量在close命令步平均减少2.61mm。完整actuator ctrl和控制器current_action没有保存，不能声称重建了控制器每个内部子步。没有发现接口错误需要立刻重训；0.5 tie是官方中性行为，保持原阈值。

动作chunk位置0..7指a_t..a_{t+7}，不等同于RLA累计未来状态h1..8。当前B仍沿用完整未来状态筛选以保持既定协议，因此每条最后一个动作标签未进入任何有效chunk；旧5条被遗漏的均是open，闭合事件和close动作未被遗漏。尾部release的当前观测窗口较少，详见phase_window_coverage.csv，未悄悄改变mask。
