# 原始 baseline 溯源

核实时间：2026-10-22。论文为 **Learning Visual Feature-Based World Models via Residual Latent Action**，arXiv **2605.07079v2**（2026-10-05）。官方仓库：[mlzxy/rla-wm](https://github.com/mlzxy/rla-wm)，本次固定 commit `0fc6e97212442ebccffa06f1d42f5481e5f22950`。这不是声称当前远端 main 未再更新。

论文定位：[§4.2、Fig.4、Table2](https://arxiv.org/html/2605.07079v2#S4.SS2)。这里的无 RLA-z 基线是 **BC-ResNet**。Table2 是 ManiSkill 五任务的少样本比较，使用50个评估种子42–91，并平均最后五个 checkpoint。§4.2 后段另有 LIBERO VLA-Adapter 扩展；Table3 的 VLA-Adapter 不是本次选择的 BC-ResNet。

## 实际代码证据

以下路径均相对于上述官方 commit；函数和行号可通过固定版本复核。

|问题|源码事实与位置|
|---|---|
|入口|`docs/applications.md:9`：`policies/train.sh fs_bc setting=1 n_shots=50`；`policies/config/fs_bc.yaml` 将 policy 指向 `VLABCPolicyRLAUnified` 并设置 `use_latent_head:false`。类名中的 Unified/RLA 不代表无 z 分支会构造 RLA。|
|视觉骨干|`policies/policy/utils.py:22 _make_resnet18_encoder`：`resnet18(weights=None)`，BN 替为 GroupNorm；空间特征512维。没有视觉预训练或现成机器人策略初始化。|
|策略主体|`policies/policy/vla_bc_policy_rla_unified.py:88,323`：state MLP256、state projection256、图像投影256、6个8-head attention blocks、16个learnable output queries；flatten后2048→2048 MLP，两处dropout0.1。|
|动作头|同文件 `__init__` 的 `action_head=Linear(2048,horizon*action_dim)`、`_decode_action`。确定性连续输出（train时dropout随机）；不是扩散或动作分类。|
|输入|`_forward_trunk` 只消费 image/state。配置单front_lower_camera、单观测时刻；没有语言、没有历史观测序列。state为机器人qpos，维数随机器人改变；不是当前LIBERO FK8。|
|初始化|`utils.py weights=None`；`train_vla_bc_workspace.py:50 __init__` seed42后实例化policy。新运行随机初始化；`resume:true`是恢复机制，不是预训练声明。新路线必须禁止自动恢复历史训练。|
|动作语义|`maniskill_sequence_dataset.py:196`读取 `target_qpos/qpos`；`fs_bc.yaml control_mode:pd_joint_pos`。机器人动作维数5/8/7、state6/9/12。不是LIBERO OSC xyz/轴角/夹爪7D。|
|normalizer|`ManiSkillSequenceDataset.get_normalizer:288`，LinearNormalizer limits逐维映射到[-1,1]，image range normalizer将[0,1]映射[-1,1]；没有Spatial-Pro quantile statistics。|
|BC loss|`VLABCPolicyRLAUnified.compute_loss:434`，默认全tensor均值MSE；`predict_action:410`反归一化。horizon12，执行前8（n_obs_steps=1）。|
|padding|dataset `_pad_sequence:316`边界复制；原 `compute_loss` 没有有效时间mask排除复制值。新数据仅验收8步，不能冒充12个有效动作。|
|图像处理|配置源512→policy128，crop_shape=null（random_crop=true此时不启用crop）、imagenet_norm=false、`use_foreground_mask:true`。不能称为整图背景训练。|
|数据预算|`fs_bc.yaml` 每任务50示范、PushT150（文档示例）；1000候选成功轨迹；BC仅使用有标签子集。batch64，epochs=40*(1000//n_shots)，因此50→800epochs，150→240epochs（整数除法）。这不是简单40epochs，也不是100updates。|
|优化器|AdamW lr3e-4，betas(.9,.999)，eps1e-8，wd1e-4；warmup200+cosine，累积1。workspace:131–143 给出实际scheduler；第一步lr=0。原配方含模型EMA用于评估，非一致性teacher。|
|评估|50episodes、max100steps、seeds42–91。配置保留top5/last、workspace按验证/rollout周期调用；论文“最后五个平均”与配置“top-k success”不能互换，新路线不执行任何筛选。|
|加RLA的变化|`fs_bc_rla.yaml`和`compute_loss:480`：冻结DINO与RLA encoder，以未来-当前特征产生target并除10；shared MLP增加linear latent head，0.05 L1+0.05 MSE；pixel-only数据缺动作时mask动作损失，以learnable proprio token代替state。无z分支全部不构造。|

## 实现与论文精简描述的差别

论文称ResNet BC，实际公开fs_bc含attention/query/shared MLP，不能用普通ResNet+单MLP替代。源码保留背景mask、12步预测及边界复制等细节，必须单列。官方few-shot配置 `n_shots` 默认10，但论文配套文档运行示例是50/150；本次不把默认值当论文设置。

官方LIBERO扩展另见 `rebuttal/exp2_vla_adapter/README.md` 及 `tools/train_eval.sh`：上游VLA-Adapter commit `23fa0c9c`、Prismatic DINO/SigLIP+Qwen2.5初始化、Pro版动作网络，额外RLA sidecar/head/loss。它是另一条官方扩展，公开Spatial-Pro已训练策略也不等于其从Prismatic开始的baseline。

## 本轮明确的跨域适配

实际调用固定官方类，`use_latent_head=False`；未使用历史自定义VLA unified trainer。保留随机初始化、ResNet/attention/MLP结构和12步输出。为现有数据，action_dim改7、state_dim改8（同状态FK）、单agentview、已渲染128px RGB；保留背景，不做原前景遮罩；只监督已验证前8步，后4步mask=false。训练只允许按任务单独建模，因为没有语言分支。不得直接混训需要语言区分目标的四任务。

**严格 P0-paper 配方已核实；本轮运行的是 BC-ResNet 的 LIBERO 适配接口，不能称为原论文数值复现。** P0-adapted/P1/P2采用相同跨域适配，才可解释其对照。若要求严格原论文P0，应另用ManiSkill qpos/target_qpos及官方前景mask数据；当前LIBERO配对集不能满足这一身份。

上游 `pyproject.toml` 声明MIT，但该commit没有独立LICENSE文本。保留来源声明，不自行制造作者版权/许可正文；已有仓库VLA-Adapter许可证仍保留。复现通过独立checkout获取官方源码，不重新发布未经核验的许可证。
