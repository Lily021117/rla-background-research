# AE-C/AE-D表示对照与策略C/D实验准备

最终状态：**管线验收通过，表示存在明确取舍；方法有效性未证明。** 当前状态以`final_delivery_verification.json`为准；原`delivery_verification.json`保留CUDA故障现场的中间BLOCKED状态，不是最终状态。

## 1. 实际完成范围

- 使用冻结20条训练、5条development示范，全部外观/窗口按episode隔离；未读取封存内容。
- 10224张双视角Clean/Background RGB完成冻结DINOv3特征提取，FP32缓存，无DINO更新。原224px朝向、K与逐帧wrist几何沿用已验收版本。
- AE-C/AE-D各完成600次有效优化更新；C实际计算600次，D因中断共计算680次，其中80次为回滚重放。最终有效参数轨迹仍各600步。
- 生成100个sidecar episode数组，分C/D×Clean/Background四个根目录；每文件`(N,8,16,64)`，累计10224个anchor行、9424个完整有效窗口（这些包含模型/外观副本，不是独立物理轨迹）。12个从源RGB重新提取的检查及全部文件重载/有限性通过。
- 官方VLA-RLA结构的C/D共同step0、固定真实batch、action-only和auxiliary backward、target隔离、保存重载通过；**策略optimizer updates=0，新增策略闭环=0**。
- 新增20个合法reset seed，三种外观完成静态渲染及几何检查，未采集训练示范、未运行策略。历史5状态仍是回归development，不是独立测试。

## 2. 官方来源与唯一AE算法差别

固定RLA官方commit `0fc6e97212442ebccffa06f1d42f5481e5f22950`；双视角LIBERO的官方step90000初始化，**不是论文step40000**。确切Encoder/Decoder文件、SHA-256和配置见`official_initialization.json`。

复用官方`RlaAutoencoderMultiViewTrainer._compose_inverse_input/_forward_views/_plucker_rays`与extractor的特征/累计索引实现。双相机patch沿输入序列拼接，由同一组16个查询产生联合`16×64` latent；不是每相机16个token，也未跨相机强行按patch匹配。

AE-C重建为两种外观各自真实未来的`L1+MSE`之平均，mean覆盖batch/相机/patch/channel。AE-D相同重建，加`lambda_cons*mean((z_clean-z_background)^2)`；两外观使用相同物理转移和anchor相机几何。没有要求两外观未来特征相等，没有动作一致性、物理loss、对比负样本或新模型。

`training_plan.json`在development评分前冻结：每组600更新，microbatch物理1、累积8、有效物理batch8/外观batch16，4800物理转移曝光；约2.54个anchor窗口等效epoch，但仅0.317个完整anchor×horizon池epoch，不能声称收敛。h1–8均匀打乱采样，与原官方h2–64范围不同，是本pilot的明确偏离。C/D采样清单逐项相同、初始Encoder/Decoder逐参数相同。

采用官方AdamW LR1e-4、weight_decay0、全部Encoder/匹配Decoder解冻、DINO冻结。使用官方FP32 master/inflat_all工具；共同固定loss scale=256，替代官方初始2^20动态增长，依据固定训练batch有限梯度校准，不依据development。保存0/300/600，最终600预先指定。

一次训练集4转移校准得到`lambda_cons=0.009127798272072366`，规则`min(0.01,0.1*rec/cons)`。校准rec=0.101335、raw cons=1.110183；加权cons=0.010134。rec梯度norm0.28079，raw cons梯度7.68096，加权0.07011（约25% rec梯度norm）。这不是development权重搜索。

## 3. 表示结果：稳定性改善与退化风险同时存在

固定5条development，每条10个等距anchor×h1–8，共400个转移；400行相关，独立示范仍仅5条。原step90000是未追加训练的冻结参考。下面重建为双相机`L1+MSE`，不是RGB或动作误差。

|模型|Clean重建|Background重建|跨外观latent MSE|归一化单位向量L2|Cosine|
|---|---:|---:|---:|---:|---:|
|step90000|0.064145|0.133829|1.129617|0.403901|0.916868|
|AE-C|0.050797|0.057673|1.658375|0.477908|0.883816|
|AE-D|0.051714|0.058719|0.083087|0.255319|0.966963|

D相对C原始MSE下降约95%，单位向量距离也下降约47%，所以改善**不只是整体尺度缩小**。但它伴随明显尺度/有效维度收缩：

|Clean统计|step90000|C|D|
|---|---:|---:|---:|
|latent norm均值|83.84|85.76|35.96|
|逐维方差均值|1.8903|2.5054|0.2946|
|participation ratio|14.19|21.38|12.39|
|entropy effective rank|50.58|68.62|37.08|
|不同episode转移单位向量距离均值|0.7316|0.8290|0.6684|

因此不能把一致性项下降解释为表示全面更好。D不是全转移常数（非零跨转移差异与有效秩），但方差和有效秩明显下降，跨转移区分性存在风险。不同episode距离并不证明保留了所有物体运动信息；末端位移只作为分层元数据，不冒充目标物体位移。原始逐维方差数组、逐示范/逐horizon结果均已保存。

D相对C平均重建代价约+1.8%。复制当前tokens的Clean/Background基线分别0.104651/0.105316，C/D均显著优于该复制参照。固定h8的首/中/尾anchor消融中，C true/zero/其他episode z重建约0.05665/0.15013/0.12838，D约0.05804/0.13700/0.12888；Decoder确实使用latent，不能说只是复制当前输入。但这是已见未来的逆编码重建，**不是动作条件未来预测能力**。

Decoder的token_proj.weight norm从C11.3622到D11.5150，仅约+1.35%，未观察到简单的2.4倍反向放大；仍不能排除其他层/归一化的内部补偿。匹配Encoder/Decoder分别评估，没有将D Encoder拼到C或原始Decoder。

## 4. sidecar与归一化

最终模型选择由`active_models.json`显式固定：C为AE_C，D为AE_D_recovery_01；完整权重哈希见`final_checkpoint_inventory.csv`。官方累计索引是以同一t为参照的`min(t+h,N-1), h=1..8`，不是相邻帧转移。尾部clamp仅用于文件形状兼容，真实mask单独保存，策略样本只用完整h8窗口，不把重复末帧当有效target。

四个根目录使用独立appearance选择，state/FK身份hash只在对应根目录内join，避免Clean/Background覆盖。`sidecar_provenance.json`保留episode、source_demo、split、appearance、encoder hash、数组hash及mask路径。cache重算与12次独立源图重算通过；容差沿用官方fp16/batch形状数值误差规则，实际误差逐项记录。

为避免C/D各自归一化再引入不同尺度，本次四个store共用**仅由AE-C训练集、两种外观、完整窗口**拟合的center统计；D原始统计另存。这是相对官方默认每sidecar统计的明确偏离。action/proprio统计仍为B20的20train-only共同统计，development未参与拟合。

## 5. 策略C/D公平性验收与剩余资源条件

从B20正式训练的原始step0恢复共同VLM、proprio、动作参数，不从B20 step600续训。采用官方RLA head；与B baseline结构不同，536个同形head键精确复制，24个官方vision gate由原标量按相同初值扩为query级，新增参数在独立fork_rng中以同一seed初始化。没有把结构变化说成B/C完全同构。

C/D共同新结构、参数、初始化和RNG一致；action/aux输出相等，Math SDPA action-only梯度相对差0；aux梯度均进入共享backbone；改变真实z与数值动作标签不改变动作输出；前后参数hash一致，严格保存/重载通过。最初一次验收在`input_ids`不一致时停止，原因是官方随机padding token构造的Python RNG未同步；只修正验收两组构造前的RNG恢复，原失败日志保留，策略从未更新。

官方auto:0.25逐步计算`0.25*detach(actionL1)/clamp(detach(auxL1),1e-6)`，不同target会产生不同系数。当前只在固定C训练batch校准一次，C/D共用固定`lambda_aux=0.22408536585365854`。相对官方动态auto是有意偏离；未来训练中加权aux占比可能随action loss变化，不会依据development再调整。

|固定训练batch|C|D|
|---|---:|---:|
|归一化target norm|68.8285|61.0880|
|归一化target方差|0.28842|0.22704|
|原始aux L1|0.640625|0.617188|
|加权aux loss|0.143555|0.138303|
|加权aux backbone梯度norm|0.40161|0.56169|

共同系数不等于共同有效梯度；D约为C的1.40倍。后续D−C只能归因于这两套AE target整体差异，不能全部归因于纯外观不变性。C/D配置保持B20的数据、配对规则、原action/proprio统计、STOP修复、LR2e-4、AdamW、batch8/累积2、Math SDPA和600预算；从共同step0做action+aux联合阶段，不另插RLA-only stage1，明确不是完整官方多阶段预算复现。

本轮只验收一个物理转移的双外观batch2，峰值allocated约14.27GiB；**不证明正式batch8在15GiB可运行**。下一轮仍须资源和完整batch预检，不可因当前最小batch通过而盲目启动大batch。

## 6. 扩展development冻结协议

官方任务reset/placement sampler，`initialization_noise=None`；固定seed2026102300–2026102319，按合法性一次保留全部20，不基于模型表现筛选。保存state、XML和静态双相机预览。机器人初始关节/夹爪姿态相同，无新增相机/灯光/机器人姿态扰动；Clean/已见/未见三组state、body、camera、segmentation、depth匹配，RGB干预生效。

未见背景预先固定为同一tex-table/tex-wall机制的纯色纹理：桌面RGB(178,132,82)，墙面(84,127,132)，尺寸及纹理类型沿用原资源；不改变specular/shininess、灯光、反射1或MSAA4。允许固有反射/边界RGB响应，不要求前景逐像素不变。

20状态未与允许来源状态发生已检查的完全匹配；这不证明统计独立，也没有检查封存内容的状态重合。历史5状态单独作为回归development。后续新状态统一评估B20/C/D×20×3=180次，历史回归5状态×3模型×Clean/已见背景=30次，分开汇报，不能当独立最终测试。当前新增策略rollout为0。

## 7. 故障、保护和结论

GPU0最初26GiB门槛拦截后仅有24.6GiB，随后约3.8GiB，未强行启动。GPU1在D的380次成功更新后发生unspecified launch failure，ECC未见新增；依规则停止。用户随后明确允许GPU1/GPU3在15GiB以上使用，重新核验后在GPU1恢复；GPU3/GPU2均未使用，未重置驱动、未终止其他进程。

从step300恢复FP32 master、optimizer、RNG和采样位置。301–380的80步loss及Encoder/Decoder梯度norm与原日志完全一致。D最终有效600，实际计算680；不隐瞒额外计算，也不把中间step300冒充final。故障原目录、原日志及中间BLOCKED验收保留。C/D训练峰值allocated约7.78GiB；训练后策略验收峰值约14.27GiB。GPU任务已退出，所有新缓存/数据位于数据盘。

结论：D改善跨外观稳定性，但伴随明显尺度与有效维度收缩，不能宣称表示全面改善或物理动作信息更好；未完全坍缩，重建仍优于复制且依赖latent。两套sidecar可靠，策略C/D零更新公平性验收通过，扩展development已冻结。可以准备下一轮固定策略C/D受控检验，但必须保留这些表示取舍、梯度混杂和正式batch资源限制，不自动重选lambda或启动训练。

主要交付：`training_plan.json`、`consistency_scale_calibration.json`、`representation_summary.csv`、`reconstruction_by_demo_horizon.csv`、`latent_distribution_metrics.csv`、`decoder_ablation_metrics.csv`、`final_checkpoint_inventory.csv`、`sidecar_provenance.json`、`raw_source_recompute_audit.json`、`policy_readiness_retry1/isolated_smoke/`、`development_protocol/evaluation_protocol.json`及`final_delivery_verification.json`。GitHub仅同步源码、配置和脱敏汇总，不上传权重、图像、状态、sidecar或凭证。
