# 官方 LIBERO VLA-RLA C/D训练及统一development闭环

状态：训练与评估流程完成；不等于方法有效性通过。研究定义：基于LIBERO的自定义背景材质干预协议，不是官方LIBERO-Plus。

## 固定条件与训练

20条训练/5条development，来源与episode隔离不变；同一B20采样清单，每个物理窗口Clean/Background各一次，microbatch8、累积2、有效16外观，BF16、Math SDPA、AdamW LR2e-4、WD.01，600成功更新；600之后scheduler降LR，没有额外低LR更新。C/D均从原始共同step0开始，不从B20 final开始。官方RLA结构相同，唯一算法差别是冻结AE-C/AE-D target；两组共同lambda_aux=0.22408536585365854及C训练集RLA中心化统计，无按development调权。

B20与C/D使用相同观测外观及训练曝光规则。B20没有RLA查询/辅助头，C−B20代表完整RLA接入增量，不是等架构比较。C/D加入非reentrant activation checkpointing（Qwen/视觉/官方RLA头），保留RNG；另用expandable allocator和更新间释放空缓存，未改microbatch。B20未使用这些内存优化，浮点重算及额外RLA随机查询是可比性限制。

C/D参数及CPU/CUDA/Python RNG初始状态核验；两组各600有效更新。C在GPU0第2步前向OOM，已计算1步未持久化；在授权GPU1从共同step0重放，实际601步。D实际600步。C首步重放日志最大差：0。原失败目录/日志保留。

初次完整batch预检只在backward后预留optimizer内存，未覆盖下一步常驻状态，故并不充分；随后增加两个连续有效batch、前向期间常驻Adam moment等量存储的隔离预检，0更新，通过后才继续。GPU0上C一次OOM后未改LR/loss/数据；新进程GPU1运行。GPU1已有SRAM parity=1/SM=1风险记录，重新授权不是维修证明。

## AE-D恢复复核

恢复了匹配Encoder/Decoder、FP32 master、AdamW moments/step、CPU/CUDA/Python/NumPy RNG、采样位置和有效计数。AE固定LR，无scheduler；固定loss scale256，无动态AMP scaler。80步重放loss及梯度日志逐项一致，未见路径变化证据；AE-D有效600、实际680，AE-C有效/实际600。详见AE_D_recovery_audit.json；计时不包括全部排队/初始化/故障退出时间。

## 统一闭环：20个development物理状态

|模型|Clean|已见背景|未见背景|
|---|---:|---:|---:|
|B20|19/20 (95%)|17/20 (85%)|17/20 (85%)|
|C|20/20 (100%)|19/20 (95%)|20/20 (100%)|
|D|20/20 (100%)|20/20 (100%)|20/20 (100%)|

180有效闭环，0无效；同一状态三种外观只算一个物理状态。固定state/condition/model顺序，220控制步、chunk8、完整reset到1e-8；tex-table/tex-wall、反射1、MSAA4、固定已见/未见纹理、相同语言/FK proprio/action后处理。未读取封存测试集，没有运行旧5状态的额外回归闭环。

评估首次启动在trial前将状态数组内容SHA与NPY文件SHA误比较，已依据冻结生成代码修正为sha(state.tobytes())，未修改状态或协议；此时0次rollout，原日志和空目录保留，修正后在新目录执行。详见evaluation_pretrial_incident.json。

## 配对比较与不确定性

|比较|条件|成功率差|状态聚类bootstrap95%CI|胜/负翻转|
|---|---|---:|---|---|
|C-B20|clean|+5.0pp|[+0.0,+15.0]pp|1/0|
|C-B20|seen_background|+10.0pp|[-10.0,+30.0]pp|3/1|
|C-B20|unseen_warm_teal|+15.0pp|[+0.0,+30.0]pp|3/0|
|D-C|clean|+0.0pp|[+0.0,+0.0]pp|0/0|
|D-C|seen_background|+5.0pp|[+0.0,+15.0]pp|1/0|
|D-C|unseen_warm_teal|+0.0pp|[+0.0,+0.0]pp|0/0|
|D-B20|clean|+5.0pp|[+0.0,+15.0]pp|1/0|
|D-B20|seen_background|+15.0pp|[+0.0,+30.0]pp|3/0|
|D-B20|unseen_warm_teal|+15.0pp|[+0.0,+30.0]pp|3/0|

跨三外观等权汇总（仍为20个物理状态聚类，而非60个独立样本）：

- C-B20: +10.0pp，95%CI [+0.0,+21.7]pp。
- D-C: +1.7pp，95%CI [+0.0,+5.0]pp。
- D-B20: +11.7pp，95%CI [+3.3,+21.7]pp。

背景交互（背景−Clean之差的差）：

- seen_background: +5.0pp，95%CI [+0.0,+15.0]pp
- unseen_warm_teal: +0.0pp，95%CI [+0.0,+0.0]pp

区间以20个物理状态为重采样单位，保留三模型/三外观的相关性；绝对SR同时提供Wilson区间。若全成功，经验bootstrap会退化，不能解释为真实成功率确定为100%。例如20/20的Wilson95%为约83.9%–100%。区间/多项比较均为探索性，未作多重比较校正。单任务、单训练seed、20个development状态，无独立最终测试，均限制外推。

## 离线验证与失败解释

固定5条development示范的最终动作L1（各示范等权、归一化空间）：

|模型|Clean|Background|
|---|---:|---:|
|B20|0.201413|0.201866|
|C|0.191048|0.191322|
|D|0.198256|0.198865|

C离线L1略优于D，但闭环D只多1个成功；不能据离线L1选checkpoint或将其替代成功率。平移/旋转/夹爪和按示范数据见validation_summary.csv及原逐窗口日志。

前轮AE-D归一化跨外观距离改善，但latent norm/方差/有效秩明显下降，重建约增加1.8%；不是单一尺度改善，也不能据此证明运动信息保留。共同aux权重下，前轮D的辅助backbone梯度约为C的1.40倍；target整体差异/尺度与梯度强度是D−C解释的限制。不能仅根据策略成功/失败推断表示坍塌。

failure_stage_summary.csv按统一、事前固定规则使用目标物体最大抬升和末端到目标距离作描述性阶段代理；不是接触/抓取物理真值。全部8个任务失败（B20为7、C为1）均运行至220步；均达到目标中心10cm以内，但未出现高于初始目标位置3cm的抬升，属于接近后抓取/接触阶段代理，而不是已确认的夹爪head错误。

按每个模型/外观的最小失败seed统一选取4组视频面板（failure_evidence/manifest.json）：可见机械臂接近后对位/接触、目标物体在桌面附近移动，未完成稳定搬运到目标。这里只是视觉观察，不是独立接触真值，无法唯一判定夹爪、朝向或路径哪项是根因。不得将初始高度参照下未超过3cm直接等同于从未发生任何小幅抬升。D无失败，不人为挑选成功解释机制。

## 成本、保护与决策

- C: 600有效更新，记录训练/验收计时2088.9s，峰值allocated 19.00GiB。
- D: 600有效更新，记录训练/验收计时2085.9s，峰值allocated 19.00GiB。

C/D在同一GPU并行运行，不能将两个进程计时相加当作墙钟时间。180次trial记录耗时合计2434.0s（约40.6分钟），不含最初模型加载及检查；闭环峰值allocated约3.72GiB。本任务GPU进程均已退出，ECC无新增，其他任务未被终止；详见本地resource_release_receipt.json。

结论：D相对C只修复已见背景1个失败；已见背景交互+5pp、CI[0,+15]pp，未见背景交互0且两组均20/20。未观察到Clean代价，但不能证明跨外观一致性的稳定额外收益。C/D明显天花板，尤其D为60/60，当前协议对D−C的区分能力不足。

D−B20的跨外观聚类汇总区间在本次数据上高于0，是整体RLA方法相对action-only的探索性正向证据；这不等于一致性增量已获证明，更不能替代多训练seed/多任务/独立状态检验。C−B20跨外观汇总区间含0；D−C也含0。下一步优先预先扩大任务和独立合法状态的区分度，再做匹配多seed验证；不在本轮搜索让B失败而D成功的新背景或追加训练。

本轮未更新AE；策略有效更新C600/D600，C额外重放1；另有一次未完成更新的OOM前向，不计成功预算。新增有效闭环180；没有读取封存集。历史保护哈希检查通过。全部新产物位于数据盘。既有bootstrap目录名isolated_smoke不代表只做smoke：最终checkpoint scope为OFFICIAL_PAIRED_RLA_POLICY、global_step=600，完整optimizer/scheduler/RNG均核验。源码/配置/脱敏汇总同步私有研究分支，不上传权重、图像、sidecar、视频、原始轨迹或凭证。
