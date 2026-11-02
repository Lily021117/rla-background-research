# RLA配对数据V2原型

## 结论与范围

本轮完成两任务各5条来源明确的现存示范，全部标记为training-prototype。根据既有用途登记，10条均为新纳入训练池的独立源示范，复用旧训练示范0条；没有新采集。50个渲染版本不是50条独立示范。数据原型通过当前物理、几何及可见性验收，可以制定正式数据/训练协议，但未验证任何方法效果，也未证明一定产生策略性能差距。

任务是炉灶取黑碗放盘及打开柜子中层抽屉，不是从打开的抽屉取碗。历史负结果、文件及权重不变。只使用CPU/OSMesa，未分配GPU。

## 来源与时间语义

炉灶来自既有libero_spatial原始HDF5，选demo11/22/23/24/25；抽屉来自固定官方公开HDF5，选demo25/26/27/28/29。每任务按数字ID顺序检查并接收5条，没有拒绝源示范。抽屉文件来源为yifengzhu-hf/LIBERO-datasets、revision 97773100c1474cd0d686ebd173cc0e4fd5442466，SHA256 20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737。

按HDF5来源＋源demo身份与既有20训练/5development、受保护登记去重；新源动作序列哈希不与既有训练/development登记重复。不把Cosmos同名或重渲染文件当作新轨迹。来源内容可能曾被历史审计读取，不等于未曾曝光的测试数据；本轮仅新增训练原型。不宣称穷尽所有未知跨来源祖先关系。

原生成代码先执行action再记录obs，而states/actions使用同一原始索引。实际检查states[t+1]机器人关节/夹爪与obs[t]完全一致，结合生成代码确认源观测为动作后、保存状态为动作前。新RGB与FK均在保存的动作前状态生成，不复用相位不一致的源RGB。所有相邻保存状态间隔0.05秒，无断点；没有额外稳定等待、插值或重复尾帧。保留完整episode及原动作。

累计future索引沿用已验收extract_sequence.py的t+h,h=1..8。本轮只记录映射和有效mask，不计算特征或sidecar；末尾缺未来的索引标false，不重复末帧伪造有效标签。状态时间直接来自保存数组，不能将线性近似时间作为源证据。

来源文件整容器SHA校验读取文件字节，但未解码受保护episode的状态、图像或动作；只解码10条选中源示范。剩余潜在来源：炉灶5条、抽屉20条，均尚未内容验收，不能宣称已补齐。

## 干预与技术修正

仅操作对象表面或静态光照，四种类型均完成。对象1为较哑光/较暗表面，对象2为不同纹理调制和反光参数；均只修改目标视觉geom的隔离材质引用，复制共享材质和纹理，不误改其他抽屉。实际字段specular/shininess/reflectance，不使用不存在的roughness字段。

黑碗保留外部黑色及中性内纹；抽屉保留木/金属色调及中层身份。强度条件仅将既有灯的diffuse/specular/ambient乘0.65，方向条件仅绕世界Y旋转方向30度；不改光源位置、相机、几何、动力学、成功判据或语言。一整条轨迹内配置固定。确切XML字段和纹理哈希见appearance_configs.json。

初版碗纹理硬限亮度55/75使目标过暗，未通过预定可见性门槛；初版资料保留。revision2仅取消硬限幅，采用原灰度纹理×0.85，或×0.95×固定条纹调制。没有改变验收阈值，抽屉及光照未改，也没有模型筛选。

## 验收与局限

先固定每任务第一条源示范、5个均匀时间点，进行双相机RGB/深度/实例分割检查。分割以真实GEOM与目标body子树映射，不用SITE。各变体深度误差≤1e-6米，分割逐像素一致；编译物理参数相同。小样RGB及目标可见性阈值事先写入protocol：全图均值(3,252)、标准差>3；Clean目标≥4像素且均值>3时，变体目标均值≥max(2,0.35×Clean)且<250。

完整序列逐帧逐变体检查FK、目标位姿、相机矩阵完全一致，渲染前后time/qpos/qvel未改变；源动作与状态只有一份，全部版本共用。检查并保存末端、目标位置/朝向、夹爪、同步proprio、盘位置或抽屉关节，以及逐帧wrist外参。图像224×224，MSAA4，反射启用（MuJoCo默认flag实际核验为1），沿用水平翻转及负fx的K约定。没有改变相机几何。

深度/分割只在固定小样检查，不宣称每一完整帧都做了深度验收。完整帧的视觉统计见visual_summary.csv和本地逐帧文件。对象效果集中在目标表面，允许反射等邻域响应；光照作用全画面。不再使用前景RGB必须相同的旧背景规则。固定首/中/尾人工查看支持目标/夹爪/操作区仍可解释，但原Clean已有遮挡和wrist出画保留，不能保证每时刻两相机均完整看见目标。

源动作开环回放0条，仿真积分0步：本轮以保存状态直接渲染，XML白名单＋编译物理参数＋同状态FK验收支持配对，不声称验证过独立动作回放的终点复现。新外观的下游学习与控制价值未知。

## 数据格式与查看

revision2为正式原型，根目录revision1为失败小样与修正记录。source_manifest登记身份/来源/哈希；sources保存原状态、动作与派生XML；rendered/<episode>/<appearance>.npz含uint8 RGB[N,2,224,224,3]及实际timesteps。physical.npz保存状态、动作、物理量与逐帧K/c2w。pairing_manifest.jsonl每物理帧/外观一行，关联同一源身份、时间、horizon和mask；不冒用旧RLDS内容哈希。可读取RGB后转置为旧模型loader所需CHW，本轮未运行模型loader或特征缓存。

固定展示规则是每任务首条示范的时间中点PNG及全程均匀40帧短视频，5列条件×2行相机，5fps。视频为均匀抽样概览，不是20Hz实时回放，不用于判定任务成功。完整轨迹图像全部保留，没有截掉后半段。

## 计数

|项|实际数量|
|---|---:|
|independent_source_demonstrations|10|
|new_to_training_pool|10|
|reused_old_training|0|
|physical_frames|1468|
|complete_render_versions|50|
|full_rgb_images|14680|
|state_restorations|7440|
|render_calls|15280|
|sample_rgb_images_including_rejected_revision|200|
|source_action_open_loop_replays|0|
|model_forwards|0|
|backward|0|
|optimizer_updates|0|
|AE_updates|0|
|DINO_updates|0|
|policy_rollouts|0|
|simulation_steps|0|

固定小样两次尝试均计入状态恢复和render calls，完整原型RGB与验收RGB分开。render calls包含RGB、depth、seg，不是策略前向。所有环境操作仅状态设置与mj_forward（物理运动学，不是模型前向）。

|任务|物理帧|完整有效8未来窗口|独立源|渲染版本|
|---|---:|---:|---:|---:|
|stove|691|651|5|25|
|drawer|777|737|5|25|

## 下一步

见next_experiment_design.md。主要对照是相同新增增强下普通RLA与增加AE一致性的B/C，不将增强带来的增量混为一致性作用。现在只具备制定正式协议的条件，须先确定祖先划分、独立评价与资产隔离、有限训练预算；不默认1800步足够，不自动训练。
