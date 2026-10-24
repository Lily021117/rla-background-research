# 炉灶 D 全失败：冻结权重诊断

结论：已定位到接近/抓取阶段的行为差异，尚未确认唯一训练因果根因。未发现错任务训练、cookie 成品模型复用、错误 sidecar 或错误评估 checkpoint。D 在48/60次已有闭环中从未发出关闭命令，60/60未达到3 cm抬升；C在60/60发出关闭命令，59/60成功。不能把这个现象单独归咎于夹爪head。

## 1. 训练身份与保护

B/C/D各自使用同一炉灶20条训练示范（2,555个完整h=8窗口），同一4800次物理窗口采样、9600次外观曝光；不是20个窗口。5条离线development示范有703个完整窗口。闭环另有20个冻结development初始状态，每个三种外观，各模型60次；三种外观不是三个独立状态。

AE-C/AE-D使用完全相同的炉灶25条来源清单，仅20条train用于更新，5条用于固定诊断。策略sampling_order所有身份均在20条train中，实际预检模型tokens包含炉灶指令，正式代码每个batch也断言任务tokens。旧bootstrap cookie batch记录不参与正式forward。

策略B从原始未训练B step0开始，C/D从共同原始未训练RLA step0开始；参数与Python/NumPy/CPU/CUDA RNG逐项匹配。AE step0与官方LIBERO step90000经官方配置的mixed-FP16加载后逐参数一致；源文件FP32与实际FP16躯干有精度转换，不是换模型。AE不是论文step40000。

证据：checkpoint_state_audit.json、checkpoint_inventory.csv、protected_input_hashes.json、training_episode_identity.csv、target_join_verification.json。100个sidecar文件的SHA-256与冻结provenance一致，所有25×2模型×2外观的proprio身份键逐项通过；AE Encoder哈希与sidecar记录匹配，180次原评估记录的checkpoint哈希与实际最终文件匹配。

|阶段/组|实际冻结文件|SHA-256|
|---|---|---|
|AE/C/encoder|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/ae_run/AE_C_recovery1/encoder_step0600.pt|0530263fdd951ba578fbd0eabdd3de8ba8b31b10ce91e69826703aae60d10aab|
|AE/C/decoder|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/ae_run/AE_C_recovery1/decoder_step0600.pt|87b157a3aa505a90bd0fb50befd267ed44b7649e7e1c9603353d4852455c4562|
|AE/D/encoder|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/ae_run/AE_D_recovery1/encoder_step0600.pt|c7159652142dfaeac42cdd051bf7f4d48ed2a81b41530fb683fae5fd455c0a08|
|AE/D/decoder|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/ae_run/AE_D_recovery1/decoder_step0600.pt|891e077e0d2cb2d0e920b9e5502c9068ff95e543632d1cde6210f4876b35f89d|
|policy/B/trainables+optimizer+scheduler+RNG|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/policy_run/B/isolated_smoke/step0600.pt|87917b58a974818dee92fc294c10c22718fcaeb309c383b6fb90c09b42a17577|
|policy/C/trainables+optimizer+scheduler+RNG|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/policy_run/C/isolated_smoke/step0600.pt|87470f1a97daa913f377a5dc100d435b78ecc2f5d1a1521583630bd9a3d56325|
|policy/D/trainables+optimizer+scheduler+RNG|${PROJECT_ROOT}/experiments/official_libero_rla_consistency/run_20261023_10/stove/policy_run/D/isolated_smoke/step0600.pt|52734be38371ce1dca3dfe016c7830b6b7af60d18bb89ab6384fc362995f38db|

## 2. 唯一预期差异及恢复

AE-C：两路各自未来特征L1+MSE重建平均；AE-D另加latent MSE，lambda_cons=0.009127798272072366。两AE共同初始化、数据/horizon顺序、AdamW1e-4、600有效更新及解冻范围一致。C/D策略同结构/初始化/样本/动作/语言/两路图像/FK/STOP mask，AdamW2e-4、WD.01、Math SDPA BF16、appearance micro8×accum2、600有效更新、activation checkpointing、末步600 checkpoint选择相同；target分别来自AE-C/AE-D。辅助系数共同固定0.22408536585365854。

正式调用：`ae_consistency/policy_training.py:execute -> DerivedEpisodeDataset.sample -> RlaStore.frame -> official rla.loss.make_run_forward_pass -> finetune.run_forward_pass`。动作归一化与proprio来自同一炉灶训练统计；target统一采用 `(z−mu_C)/sigma_C_global`，sigma_C=2.68701489155。STOP修复/完整8步mask来自coverage_gripper/dataset.py，不用tail重复帧充当真实target。未来图像只用于AE/sidecar，实际策略输入仍是当前双视角、语言和proprio。

B/C/D评估调用同一policy_evaluation.py，使用各自最终600步、同一状态/seed/XML/动作后处理/220控制步上限。所有180条已有执行动作与模型chunk按官方夹爪公式−sign(2p−1)后的结果逐元素一致。

炉灶AE两组旧故障均从完整step0恢复，各重放44个已记录更新；最终各600有效、644已记录计算，旧故障可能各有1个未确认在途update。FP32 master、optimizer、RNG、采样位置恢复；AE无scheduler/动态AMP scaler，固定256 loss scale。44步标量前缀精确复现，不声称未持久化step44参数逐位已验证。策略B/C/D没有本轮中断恢复，final optimizer计数均600、scheduler600、采样位置4800。没有D独有的恢复配置变化证据。

## 3. 全部已有配对闭环的失败阶段

主阶段分类沿用评分前冻结的目标抬升>=3cm、EE到碗中心<=10cm代理。D：55/60到过附近但未抬升，5/60接近失败；0/60进入可靠抬升后的搬运阶段。C：59成功，1附近未抬升失败。失败都不是单纯完成搬运后漏掉末端判定。

|模型/外观|成功|曾发关闭命令|先于10cm附近关闭|重算手指接触|抬升>=3cm|
|---|---:|---:|---:|---:|---:|
|C/clean|20/20|20/20|0/20|20/20|20/20|
|C/seen_background|19/20|20/20|0/20|20/20|19/20|
|C/unseen_background|20/20|20/20|0/20|20/20|20/20|
|D/clean|0/20|1/20|0/20|18/20|0/20|
|D/seen_background|0/20|4/20|0/20|20/20|0/20|
|D/unseen_background|0/20|7/20|1/20|20/20|0/20|

C/D在同一初态的第一动作chunk已不同，全部60对在记录的第3个零基控制步后末端分离超过2cm。D实际发出关闭的12次，其首关时水平偏差平均约8–9cm；C约3–4cm。D通常未形成可成功抓取的位置与关闭组合。接触计数来自已记录state的mj_forward重算，只能说明几何接触候选，不能证明稳定夹持或接触力。

固定选取每外观第一个seed2026102500、沿C执行时长五个均匀视频帧，已人工查看：C完成抓取/抬升向盘移动，D保持在炉灶侧附近，未形成相同搬运轨迹。这是视觉观察，与全部轨迹量化分开；完整面板仅保存在本地video_panels_local_only。

发现记录精度限制：原日志在env.step后保存cached xpos，与记录qpos/qvel经mj_forward的末端/碗位置不逐位相同，抽查最大0.980mm。没有平移时间索引或修改原日志；阶段原统计保留，新增接触/关闭位置明确是同state重算值。这一共同记录相位限制不证明D特异故障，也不能用小差值断言严格同步已通过。训练派生FK与同state重算在25条示范上通过1e-7检查，不能混淆这两项。

## 4. 相同固定batch：动作与梯度

预先冻结每条示范首帧/首次关闭anchor/最后有效anchor，20train+5development，共150个外观观测。读取step0/1/100/300/600，全程eval+no-grad动作诊断；梯度在eval模式开启autograd，每split固定4条示范的关闭anchor×2外观，共8样本，micro2累积。梯度分别来自官方action L1和加权aux L1，作用于全部可训练VLM/backbone参数；不包括head内部共享块。它是确定性局部导数，不是重演训练噪声、优化器动量或完整训练分布。

action_by_split_appearance_horizon.csv / action_by_dimension_horizon.csv保留8个chunk位置和7个维度。全部chunk真实有效、mean reduction，无padding进入loss。归一化动作误差与Q99反变换后的控制命令误差分别记录，命令坐标不能解释为米或弧度，夹爪阈值另算。

|组/step|split|Clean action L1|Background action L1|Clean gripper L1|Background gripper L1|
|---|---|---:|---:|---:|---:|
|C/0|train|0.62195|0.57699|0.57571|0.55692|
|C/0|validation|0.60679|0.56701|0.52977|0.49829|
|C/1|train|1.69681|1.66440|1.46947|1.46072|
|C/1|validation|1.70361|1.67712|1.42295|1.41673|
|C/100|train|0.26232|0.26342|0.36688|0.36993|
|C/100|validation|0.25612|0.25626|0.27955|0.28636|
|C/300|train|0.21980|0.21922|0.35437|0.34908|
|C/300|validation|0.22058|0.21933|0.29243|0.28894|
|C/600|train|0.20633|0.20647|0.30021|0.29881|
|C/600|validation|0.20837|0.20917|0.27135|0.27019|
|D/0|train|0.62195|0.57699|0.57571|0.55692|
|D/0|validation|0.60679|0.56701|0.52977|0.49829|
|D/1|train|1.66794|1.63050|1.47310|1.43433|
|D/1|validation|1.67495|1.64142|1.42673|1.39281|
|D/100|train|0.28010|0.27472|0.39548|0.39715|
|D/100|validation|0.25969|0.25574|0.31839|0.32042|
|D/300|train|0.21957|0.21794|0.37313|0.37357|
|D/300|validation|0.22139|0.21831|0.31444|0.31412|
|D/600|train|0.20866|0.20775|0.35918|0.35599|
|D/600|validation|0.20065|0.19816|0.29864|0.29502|

C/D step0输出精确一致；首步两者都暂时变差，随后恢复。D的夹爪误差在100、300、600步均高于C，但最终总体development L1低于C。没有中间checkpoint闭环，不能确认哪一步开始闭环全失败，也不能把偏离step0直接定义为退化。

|组/step/split|action loss|raw aux|weighted aux|backbone action grad norm|weighted aux grad norm|cosine|
|---|---:|---:|---:|---:|---:|---:|
|C/0/train|0.56445|0.64746|0.14526|8.95596|0.43768|-0.1325|
|C/0/validation|0.59277|0.62988|0.14111|8.75545|0.46247|-0.1063|
|C/600/train|0.25342|0.41016|0.09192|0.33388|0.01595|+0.0145|
|C/600/validation|0.19946|0.35986|0.08057|0.26949|0.00853|+0.0646|
|D/0/train|0.56445|0.62109|0.13916|8.95596|0.50368|+0.0698|
|D/0/validation|0.59277|0.61133|0.13721|8.75545|0.51896|+0.1221|
|D/600/train|0.27295|0.14368|0.03220|0.56238|0.02436|+0.1138|
|D/600/validation|0.19812|0.12500|0.02802|0.47504|0.03497|+0.0367|

这批样本不支持D存在持续更强的反向梯度冲突：step0 C为负cosine而D为正；step1两者为负；后期D为正。D final raw aux更低，却在development的weighted auxiliary梯度norm约0.03497，C约0.00853。共同系数不等于相同有效梯度；只由aux loss大小判断学习强弱不成立。

示范首次关闭anchor的当前步：D在固定5条development上Clean正确关闭5/5、Background4/5，后续第7步两外观均5/5；训练示范当前步为14/20、15/20。这与闭环48/60不关闭并存，支持检查接近轨迹/闭环观测偏移及动作联合时序，不能说D完全不会输出关闭。具体结果见closure_anchor_predictions.csv。

## 5. AE表示、重建与尺度

|组/clean|latent norm|逐维方差均值|entropy有效秩|unit跨外观距|unit跨episode同h距|within/between|
|---|---:|---:|---:|---:|---:|---:|
|C|85.893|2.37999|72.408|0.43462|0.79292|0.54813|
|D|36.136|0.28193|43.955|0.24060|0.63735|0.37750|

同一400个development转移比较；同时给raw、每样本unit L2、官方共同C中心归一化三种结果。D尺度/方差/有效秩降低，归一化within/between也下降，说明并非只有等比例缩小，但不能据此称语义改善或坍塌。

未来DINO L1：C clean0.04751/background0.05223，D0.04859/0.05326；copy-current为0.0810/0.0822。D有小幅重建代价，且两者明显优于复制当前。AE_reconstruction_by_camera_horizon_phase.csv进一步按双相机、h1..8、approach/closure/transport代理阶段拆分，使用原来的完全相同样本；不是动作条件未来模型。

target_scale_and_offset.csv使用全部训练/验证有效sidecar rows：共同C居中后，D固定均值分量占平方能量83.6–84.3%，C约0.6–1.2%；D动态逐维方差约为C的12%。这不是aux L1或梯度的84%，不能作这样的换算。共同归一化符合冻结方案，尚未定位代码错误；但它使C/D监督的固定偏移、尺度与不变性一起变化，是明确的表示统计混杂。

## 6. 同容量冻结latent物理变化读出

从已验收的同state XML/qpos/qvel恢复碗world xyz和两指qpos；25条全部状态的FK proprio逐维核验通过。不是从图像猜标签。无独立真实世界位姿真值；本任务是炉灶上取碗，不是操作炉灶开关，因此未用无关旋钮变化替代任务后果。

每episode10个固定h8窗口：20train共200 clean样本拟合闭式ridge，5development共50转移分别评估clean/background。相同1024维输入、固定alpha=1（平均平方loss约定）、同样训练均值/逐维std标准化、同一输出容量；不调参，不对AE/策略新增监督或反向。标准化有意消除简单尺度差异，衡量线性可读信息，而不是策略实际aux学习难度。

|组/外观|碗位移MAE(mm)|碗位移R²|两指qpos变化MAE(mm)|
|---|---:|---:|---:|
|C/clean|6.954|0.8898|3.064|
|C/background|7.689|0.8694|3.715|
|D/clean|7.089|0.8904|2.839|
|D/background|7.294|0.8866|3.195|

D−C碗位移MAE：clean+0.135mm，episode bootstrap95%CI[−0.088,+0.372]mm；background−0.395mm，[−0.715,−0.075]mm。只5条development，小样本相关窗口不能当独立轨迹；结果不支持“D碗运动信息普遍不可读”，不排除更复杂语义、长时后果或闭环分布下的信息损失。未以EEF位移冒充物体后果。

## 7. 事实、排除与假设

- 已确认：炉灶各自正确数据/初始化/sidecar/最终checkpoint；配对部署规则一致；D在接近/抓取阶段偏离、48次从不关闭、无抬升；夹爪离线误差较C高；D表示尺度与有效秩降低；共同C中心引入显著D均值分量。
- 在现有证据范围内排除：错用cookie已训练模型、跨任务target串用、错评checkpoint、后处理吞掉关闭命令、D独有恢复预算变化、仅因总体离线action L1更差、仅因线性物体运动标签不可读。
- 仍待验证：目标中心/尺度变化如何影响共享head与策略优化；局部夹爪/接近时序误差怎样通过闭环状态偏移放大；单seed优化路径偶然性；head共享块/优化器动量层面的干扰。当前backbone局部cosine不能排除所有梯度干扰。

## 8. 唯一下一步：只改变D target中心偏移

建议一个冻结AE的两臂对照：D-control维持(z_D−mu_C)/sigma_C；D-recenter仅改为(z_D−mu_D_train)/sigma_C。mu_D只用已有20条训练的两外观sidecar计算；保留相同sigma_C、lambda_aux、AE-D权重、结构、初始化、所有数据、seed、LR/optimizer和600步预算。不能同时改变scale、辅助系数、夹爪权重或阈值。两臂均从同一原始step0开始，不续训退化策略；固定step0/100/300/600，final600为唯一比较。

放行前离线验证：target差异必须是逐坐标常数平移；所有转移间差分完全不变；共享初始参数/RNG/action输出一致；数据及batch逐项一致。后续若授权训练，统一评估冻结development，不挑checkpoint或状态；关注关闭行为、抓取位姿、实际SR。若recenter无改善，不追加调参。本轮没有执行该干预。

理由：这是隔离已观测到的固定均值混杂的最小改变；无须预设AE丢失物理信息，也无须增加物理loss。闭式读出目前未支持加入物理后果loss作为修复结论。

## 9. 范围、复现和技术记录

optimizer updates=0；AE/policy权重变化=0；新增闭环=0；未读取封存样本内容。诊断backward总320次（两次相同固定流程；第二次仅修正raw辅助loss的日志取值为乘系数前值，两次动作和梯度范数完全一致，float64 cosine因参数名集合求和顺序相差最多6.7e-16）。不把闭式ridge拟合称为策略训练。

GPU1 UUID与ECC在启动/结束核验，历史correctable2、uncorrectable parity1/SM1保持原值，无新增；GPU已释放，其他进程未干预。protected_input_hashes列出的历史输入全部复算一致。详细源代码和运行方式见README.md。

诊断过程保留了失败尝试：首次prepare误查仅preflight保存的batch证据，改为真实preflight路径后在新run02重做；AE Python环境没有MuJoCo，切回既有VLA环境；初次AE逐位比较未考虑官方FP16转换，随后验证确切cast等价；cached rollout姿态不逐位匹配被保留为记录限制，未放宽阈值宣称通过。
