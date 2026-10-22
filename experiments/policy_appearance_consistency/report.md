# 原论文无RLA-z策略 + 跨外观动作一致性：首轮验收

状态：**PASS_LIMITED_INTERFACE；原论文数值复现未执行，正式训练未放行。** 当前Spatial-Pro/VLA-Adapter修复路线暂停。本轮仅CPU固定batch前向/梯度检查，共14次autograd gradient调用；optimizer创建/updates=0，闭环=0，保留集读取/评分=0。

## Baseline真实身份

选择论文§4.2/Table2的 **BC-ResNet**，官方 `fs_bc` 配置、commit `0fc6e97212442ebccffa06f1d42f5481e5f22950`。使用官方 `VLABCPolicyRLAUnified(use_latent_head=False)` 原类；它不构造DINO、RLA encoder或latent head。随机ResNet18+GroupNorm、6层空间attention、16queries、2048共享MLP及连续线性动作头，共28,938,132个可训练参数（LIBERO适配维度）。没有预训练策略、语言条件或历史图像输入。

原论文baseline不是当前Spatial-Pro、不是旧自定义unified trainer；官方LIBERO VLA-Adapter是论文另一个扩展。详细论文/代码定位、训练预算和配置差异见 `baseline_provenance.md`。

严格P0-paper使用ManiSkill joint-position target、qpos、前景mask、12步动作。现有LIBERO配对数据只有已核验8步OSC动作/FK8，故本轮保留骨干/随机初始化，明确适配action7/state8、agentview、保留背景、12输出中仅前8监督。**P0-adapted才与P1/P2公平对比；不能宣称它是原论文完整复现。** 没有语言输入，后续按任务独立训练。

## 实测公平性

固定清单在评分前写入：冻结顺序中的前2条不同训练episode各1窗口、首条验证episode1窗口，共3个clean/background pair。三者均属cookie-box任务。无结果驱动选样。全体190训练/30验证转移另外完成当前state/action索引与array哈希检查，不评分其他任务。

|检查|实测|
|---|---|
|P1与P2(λ=0) action输出最大差|0，3/3 pairs|
|λ=0 loss/全部参数gradient最大差|0，3/3 pairs|
|同图像+匹配dropout一致性输出最大差|0，3/3 pairs|
|不同背景一致性梯度|两路非零、有限、方向相反；可到达策略参数|
|最后4个无效位置梯度|0|
|参数/normalizer/buffer前后hash|相同|
|独立实例step0与真实entry|两实例同seed hash相同，loss/gradient精确一致|

广覆盖normalizer审计（只fit190训练窗口）：

|split/sample|归一化BC-MSE|一致性MSE|BC gradient norm|一致性gradient norm|λ=.1加权梯度比|
|---|---:|---:|---:|---:|---:|
|train #1|0.287662|0.000104092|2.983922|0.021894|0.000734|
|train #2|0.276058|0.000067775|2.924044|0.018154|0.000621|
|validation #1|0.310982|0.000092340|3.085584|0.019110|0.000619|

独立的公开per-task `PairedDataset` + `step_loss`入口另行验证：cookie-box训练4episodes/40windows、验证1episode/10windows，normalizer只fit该任务40训练窗口；首batch BC-MSE0.3269965、一致性MSE0.000121718，λ=0梯度仍精确一致。两类normalizer作用域已分别记录，不混同其数值。

λ=.1在初始样本上很弱（加权gradient约BC的0.06%–0.07%）。这是尺度证据，不是有效性证据；未搜索权重，也未依据验证成功率选择配置。常数输出也能取得小一致性，因此下一轮必须看BC与动作多样性。

## 数据与运行限制

- 沿用19训练/3验证episode划分，当前状态、动作区间[s:s+8]、FK proprio、相机和背景协议继承冻结验收；没有近邻帧配对。train/validation无episode交集。
- 主路径只读当前agentview RGB、当前FK和动作标签；未读取RLA targets、DINO特征、未来RGB或旧策略权重。背景材质反射/MSAA沿用冻结数据，未重新渲染。
- 全数据覆盖核验不等于全部batch梯度验收，本轮仅3个固定pairs及1个entry batch。没有证明训练收敛或闭环收益。
- 原始RGB是128px重渲染，不能追溯为512px采样同分布；与论文相机、控制/输入语义不同。炉灶验证缺失继续存在。
- CPU torch2.8.0，float32、deterministic algorithms，GPU未初始化，无GPU占用。广覆盖audit约5.14秒、峰值RSS约1.41GiB（不含独立entry进程）；环境导入耗时另计。
- 187个固定官方源码/配置及数据清单hash检查前后一致，原仓库和官方worktree Git状态保持原样。旧checkpoint从未打开写入，无新模型checkpoint。

## 后续条件

唯一建议：先做 **cookie-box单任务P1/P2各100次成功optimizer调用** 的训练管线pilot；共同随机初始化、batch4、原fs_bc AdamW/scheduler，固定λ=.1，最终100作为主checkpoint。原warmup200意味着此预算全在warmup内，首步lr0须分开记录；不等同充分训练。完整方案和曝光核算见 `next_training_plan.md`。

本轮交付可执行的dataset、两路forward、masked loss和dry-run入口，未实现或执行正式optimizer循环。开启小预算训练前，还需对新循环的更新计数、scheduler/EMA、保存恢复和资源做验收，并接受上述跨域/单任务限制。不能将此接口PASS当成方法有效性或四任务训练放行。

GitHub分支：`experiments/policy-appearance-consistency`。上传本轮脚本、配置、来源核验、脱敏报告和小型指标；不上传原始数据/图像/权重/凭证。确切commit与远端核验记录在本地 `github_sync_report.md`。
