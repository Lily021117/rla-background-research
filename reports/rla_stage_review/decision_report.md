# RLA外观一致性研究：阶段复盘与投入决策

日期：2026-11-02。研究对象是基于LIBERO示范构建的配对数据与自定义背景材质干预，不是官方LIBERO-Plus成绩。本报告只读已有材料；证据编号对应末尾文件定位及 `evidence_manifest.json`，路径均相对项目根目录。

## 1. 决策先行

**暂停当前“AE latent一致性＋RLA辅助监督”实现作为论文主线的继续投入，保留并归档完整证据。** 归档只是研究状态，不删除、移动或改写历史权重、数据和日志。当前没有依据继续搜索系数、增加类似评测条件或直接启动大型WM/WAM实验。

理由不是某个失败seed，而是：表示一致性确有改善，控制收益却高度依赖训练条件；充分补齐当前低LR阶段后，development正差没有在新物理状态上复现，后续新背景与光照也没有总体收益。现有机制诊断不足以支持一个明确、可修复的统一根因。

这是一项针对**当前实现和证据范围**的投入决策，不证明所有外观不变性方法无效。“物理转移相同、观测外观变化”的研究假设仍合理，但尚未形成可靠控制收益证据。

## 2. 我们到底改变了什么？

### 两阶段数据流

```text
保存的同一物理转移 s(t)→s(t+h)
          ├─Clean双相机图像 ─冻结DINO─ 当前/未来特征 ─ AE-C或AE-D ─ z
          └─Background双相机图像 ─冻结DINO─ 当前/未来特征 ──────── z
AE训练：两路分别重建各自真实未来特征；D额外约束两路z接近
                         ↓ 冻结AE，离线提取并用训练集统计归一化
当前图像＋语言＋proprio → 策略共享网络 → 动作预测 → 示范动作L1
                                     └→ latent预测 → AE标签L1
部署：只有当前可观测输入；不需要未来图像、AE标签或Clean参考图像
```

设外观为 `a∈{clean,bg}`，当前与未来冻结DINO特征为 `f_t^a,f_(t+h)^a`，AE encoder输出 `z_h^a`，feature decoder预测未来特征 `fhat_(t+h)^a`。当前实现利用当前/未来特征差编码转移，并让decoder使用当前特征和z重建未来；它不是动作条件世界模型。

`L_rec^a = mean|fhat−f| + mean(fhat−f)^2`；`L_AE,C=(L_rec^clean+L_rec^bg)/2`；`L_AE,D=L_AE,C+lambda_cons·mean(z_clean−z_bg)^2`。两路重建各自外观，不要求未来RGB或DINO tokens完全相同。[A1,A2]

策略标签是 `y_h=(z_h−mu)/s`。从零AE主线的mu为每个latent位置16×64的训练均值，s为全部有效训练标签的全局标量尺度；C/D分别估计，不使用development拟合。策略损失为 `L_action+lambda_aux(k)·L_RLA`，两项分别是有效动作和latent元素的L1均值。标签是监督目标，**不是“最优latent”，也不是在线动作输入中的真实未来信息**。共同lambda不保证共同梯度强度；原始辅助loss也不能跨不同target分布直接排名。[P1,P2]

### 方法身份表

|阶段/标识|AE来源与训练|策略监督与初始化|必须保留的区别|
|---|---|---|---|
|早期微调C/D|官方LIBERO step90000，共同起点，任务内各600步；D加一致性|原始预训练VLA起点＋随机模块；动作＋RLA|每任务独立训练；不是作者原始step0，也不是论文精确复现|
|早期D-center|同一早期AE-D，不重训|D标签中心由mu_C改mu_D，仍用s_C|不是从零AE-D；中心化不能与一致性混称一个变量|
|从零AE-C/D|按固定官方实现构造encoder/feature decoder，共同未训练初始化；各2000步|后续冻结用于标签|DINO仍冻结预训练；不是“作者发布的原始step0”；仅一对AE训练结果|
|新C-fixed/D-fixed|上行新AE-C/D，各自训练统计|lambda0=0.22408536585365854，策略600步|fixed指策略辅助系数固定，不是AE未训练|
|新C-low/D-low|同一新AE-C/D|lambda0/2=0.11204268292682927|不要与早期0.1倍的C-low混淆|
|新D-decay|同一新AE-D|lambda0×(1−k/599)，k=0…599|均值等于半权重，但累计梯度作用不相等|
|1800步C-low/D-low|仍为同一新AE对|从600完整恢复，新增1200步LR2e-5|不是新AE复现，也不是纯步数消融|
|抽屉C-low|迁移上行AE-C，未在开抽屉训练AE|开抽屉策略从原始起点训练1800步|没有训练D；BC-standard架构不同，不能纯粹归因辅助loss|

AE一致性系数与策略辅助系数作用在不同阶段，不能互换解释。早期lambda_cons约0.0091278；从零主线为0.0000436240792631；策略lambda0另行固定。[H1,A2,P1]

`20261101/20261102`主要是共同初始化下的**采样顺序**，不是完整训练seed。`31001/31002/31003`重建原规则随机模块并改变训练RNG和采样，预训练权重不随机化；同seed内方法配对。但AE始终固定，不能称三个完整AE＋策略方法复现。[P3,S1]

“抽屉取碗放盘”仍是取放primitive；`libero_goal/open_the_middle_drawer_of_the_cabinet`是打开中层抽屉。二者数据、goal、动作需求不同，不能汇总为同一任务。

## 3. 实验演进：每一步回答了什么

|研究问题|主要设计与观测|可支持的结论／不能支持的结论|证据|
|预训练AE上加一致性是否有增量？|cookie C59/60、D60/60；新增抽屉取碗C1/60、D0/60；炉灶C59/60、D0/60|微小正差未跨任务重复；天花板/地板限制；59/60是策略成功次数，不是AE准确率|H1,H2|
|是否只是中心化或辅助强度？|D-old与D-center均0/60，但关闭/抬升代理显著改变；原lambda×0.1后C5/60、D-center9/60|标签处理和权重影响行为；没有恢复到原C，不支持“辅助权重越小越好”|H3,H4|
|更弱AE一致性、行为保持是否解决？|D-inv03 0/60；E-C59/60、E-D31/60、E-C-keep27/60、E-D-keep0/60|这些具体改法无稳定整体收益；E从C成品后训练，不能混进原step0对照|H5,H6|
|均衡采样/共享Clean标签是否稳定？|新四背景中两个顺序D-center−C为−2.5/−60pp，shared−C为+3.75/−53.75pp|局部正差与强顺序依赖并存，停止两个实现调参|H7|
|是否有梯度/行为统一根因？|梯度诊断PARTIAL；轨迹200对，局部接触未居中模式只在一个顺序突出|没有持续梯度冲突的充分证据；模式不等于原因，不能泛化|H8,H9|
|任务相关几何监督是否更好？|Geometry在两顺序优于匹配action-only，但均低于C|额外仿真监督可能有用，未产生超越普通RLA的价值；不是一致性消融|H10|
|一致性从第一次AE更新参与是否不同？|共同未训练AE，C/D各2000步；D重建略低、单位范数距离更小|建立有限重建基线；不是控制收益或物理因子识别证明|A1–A5|
|策略权重与随机条件是否重要？|fixed/half/decay及C-low补全出现顺序反转；新完整seed仍波动大|不能用单顺序或总loss讲稳定方法优势|P1–P5,S1–S3|
|补齐低LR阶段是否恢复能力？|六模型600→1800；C和D任务能力大幅恢复，D development背景+9.58pp|组合续训有帮助；不能分离步数与LR，也不能证明收敛|L1–L3|
|优势能否迁移新物理状态？|同六模型，50新生成状态，1500闭环；背景D−C−0.67pp|development背景优势未复现；不是全新纹理泛化|I1–I3|
|真正不同操作的基础能力/缺口？|开抽屉BC和C六模型均Clean≥16/20；背景效应seed依赖|没有达到预设推进门槛；不是D在抽屉失败|R1–R3|
|新纹理与光照呢？|同六炉灶模型，25旧验证状态，新900闭环；背景−2.67pp，光照−1.33pp|无总体收益；标签/动作更一致不能替代任务成功|N1–N5|

早期低LR、其他seed及采样恢复日志并非本轮逐一重新审计。本账本不是全仓库实验穷举；未查到的细节明确标为未核验，不用聊天补数。

## 4. 从零AE：有表示证据，但没有独立控制结论

按官方commit `0fc6e97212442ebccffa06f1d42f5481e5f22950`构造，初始化seed20261029；encoder和feature decoder不加载训练AE，冻结DINO。三个取碗放盘场景各20训练＋5development，共60/15示范；7231完整窗口、57848个anchor/horizon组合。每组2000更新、16000物理转移抽样、32000外观曝光，不是32000独立示范。[A1,A2,A6]

有效物理batch8、外观batch16、AdamW LR1e-4、WD0、无scheduler；Math SDPA，mixed FP16 torso＋FP32 masters，固定loss scale256。相对官方大规模recipe，数据、batch及h1…8范围均缩小，不称充分预训练。[A2]

独立重建预实验60步用于训练集尺度校准，正式两组重载共同未训练step0。初始decoder会使encoder重建梯度很小，因此没有只用step0比值。按10/30/60步预定规则得lambda_cons=4.362407926306656e-05，正式首步启用。校准节点加权encoder梯度比分别约2.015/0.1/0.0267，**不是每一步都控制在10%**，也不是动态梯度平衡。[A2,A6]

最终炉灶development重建L1+MSE：Clean C0.078634、D0.078220；背景C0.078959、D0.078754，差异较小。真实z优于打乱z/零z，支持decoder使用z，但不能证明z含控制充分信息或识别了物理因素。[A3]

同一最终炉灶验证样本：C/D latent范数207.23/148.03，逐坐标方差7.780/3.174，有效秩4.222/3.851；单位范数配对距离0.18619/0.13568。距离按单位向量L2，而非策略标签训练统计标准化；不能把不同口径混成同一曲线。有效秩受45样本、上限44限制，下降不是自动坍缩判据。三个任务方向见附表，不能只挑炉灶。[A4,A5]

## 5. 策略学习：权重收益与基础能力必须分开

所有新AE下策略实验仍从规定预训练VLA与随机动作/RLA模块开始，不从AE初始化策略。AE只负责离线标签；策略不计算未来重建或新增一致性loss。20炉灶训练示范、5探索性development；双相机、语言、FK proprio、chunk8、STOP/mask沿用验收版。通常每update8物理窗口、16外观样本；BF16/Math、AdamW，600步更新实际LR2e-4。[P1,S1,L1]

### 600步权重与顺序（每格Clean成功/20；背景为四条件等权百分比）

|方法|顺序20261101 Clean / 背景|顺序20261102 Clean / 背景|
|---|---|---|
|C-fixed|19 / 87.50%|1 / 22.50%|
|D-fixed|7 / 36.25%|15 / 61.25%|
|D-low|16 / 61.25%|17 / 67.50%|
|D-decay|0 / 0%|4 / 0%|
|C-low补全|0 / 0%|16 / 88.75%|

D-low较D-fixed改善，但C减半后一个顺序崩落、另一个恢复；因此不能总结为“减半普遍有益”或“D稳定胜出”。D-decay与D-low具有相同算术平均系数，结果却不同，不能把平均系数当作同一优化过程。C-low补全只新增200闭环，C-fixed/D-low的400历史结果未重跑。[P2,P4,P5]

### 三个完整策略seed：600→1800（Clean成功/20；背景百分比）

|seed|C-low 600|D-low 600|C-low 1800|D-low 1800|
|---|---|---|---|---|
|31001|6 / 23.75%|17 / 86.25%|19 / 85.00%|17 / 90.00%|
|31002|5 / 15.00%|1 / 13.75%|20 / 91.25%|20 / 96.25%|
|31003|0 / 2.50%|9 / 80.00%|19 / 80.00%|20 / 98.75%|

600步D背景均值60.00%、C13.75%，但D也有1/20的Clean失败seed。1800步C Clean96.67%、D95.00%；背景C85.42%、D95.00%，D−C+9.58pp，逐seed为+5/+5/+18.75pp；Clean D−C−1.67pp。基础能力恢复后仍有development正差，应保留，不因后续负结果抹去。[S2,L2]

续训新增7200次更新，恢复AdamW、scheduler、RNG与采样进度；第601–1800步2e-5。**更多更新与进入低LR阶段是组合干预**，不能分离贡献，不能证明1800收敛。600步阶段seed31002 D有49步历史重放；续训新增重放0，两者不得混记。[S3,L3]

## 6. 泛化证据：正差为何不足以继续当前主线

### 新物理状态（旧四背景）

50个按冻结初始化分布生成的状态，未参与本项目训练/调参；并非官方封存示范测试。来源审计曾读取候选封存内容，必须登记，不能写整段研究“封存内容从未读取”。旧候选未进入评分。无法保证所有上游预训练数据独立。[I1]

|seed|C Clean/50|D Clean/50|C背景均值|D背景均值|背景D−C|
|---|---:|---:|---:|---:|---:|
|31001|44|43|89.50%|82.50%|−7pp|
|31002|46|46|93.00%|98.00%|+5pp|
|31003|45|48|94.00%|94.00%|0pp|

背景等权均值C92.17%、D91.50%，D−C−0.67pp，状态聚类95%CI[−4.00,+2.67]pp；Clean D−C+1.33pp。**development背景优势没有复现**，但区间跨零不证明等效。这里使用同一对AE和同六个策略，是新状态检验，不是额外独立训练复现。[I2,I3]

### 打开中层抽屉：不能算成D失败

复用已准入公开示范20/5，六策略从各自规定初始状态各1800更新，共10800；0 AE/DINO更新。C-low迁移三取碗场景训练的AE-C，BC-standard无RLA结构；两者架构差异限制辅助loss的纯因果解释。[R1,R4]

|方法|三seed Clean成功/20|三seed背景均值|三seed背景−Clean|
|---|---|---|---|
|BC-standard|18,20,18|95.00%,77.50%,78.75%|+5,−22.5,−11.25pp|
|C-low|18,20,16|70.00%,98.75%,97.50%|−20,−1.25,+17.5pp|

两种方法均过Clean准入，所以完整执行600闭环。但均未满足“三seed都下降且均值至少下降10pp”的预设推进规则。BC平均下降9.58pp且状态CI不跨零，仍不能事后把推进门槛改掉；同样，不达门槛不证明背景影响不存在。局部背景2和seed反转明确保留。**没有训练抽屉D。** [R2,R3]

### 新纹理／静态光照（已有状态的新外观探索）

以SHA256固定规则从50状态取25，3新背景和3静态光照评分前冻结。900新有效闭环；150 Clean结果复用，不是本轮重跑。光照未用于AE一致性训练，属于跨干预类型探索；旧四背景不并入主要指标。[N1,N2]

|seed|背景C / D|背景D−C|光照C / D|光照D−C|
|---|---|---|---|---|
|31001|72/75 / 64/75|−10.67pp|66/75 / 73/75|+9.33pp|
|31002|70/75 / 75/75|+6.67pp|73/75 / 70/75|−4.00pp|
|31003|70/75 / 67/75|−4.00pp|72/75 / 65/75|−9.33pp|

主要背景指标C94.22%、D91.56%，差−2.67pp，CI[−7.56,+2.67]pp；次要光照C93.78%、D92.44%，差−1.33pp，CI[−6.22,+3.57]pp。Clean复用子集C67/75、D71/75（+5.33pp），不可替代完整50状态Clean结论。子集、seed和条件必须完整保留，不能择优。[N3]

同状态100anchor（5示范各20）显示：新背景D标签标准化MSE由C的0.05794降为0.03988，预测latent由0.001965降为0.001739；平移/旋转/夹爪输出漂移略小。但绝对示范L1中平移0.17271→0.18013、夹爪0.06085→0.06669更差，旋转0.21282→0.20804略好。光照也呈标签与输出漂移更小、平移/夹爪绝对误差更大。距离使用各AE已有训练统计标准化，动作使用共同评价尺度，不能混合不同物理单位。[N4,N5]

这支持“改善部分传到输出一致性”，不支持“控制改善”。示范误差不是实际闭环因果判定，不能据此断言AE丢失物理信息。

## 7. 证据边界与计数纪律

1. 本报告附账本按实验记录新增更新、有效闭环、技术无效、重放与复用来源；不把总展示行数当新增实验，也不提供跨全部历史的未经去重总数。
2. 固定一对AE、三个策略seed和同一批模型的多轮评测，仍只有这一对AE及这批策略。两个采样顺序不是两个完整seed，三个seed不足以精确估计训练随机性。
3. 状态bootstrap保留同一物理状态的全部条件/模型配对；只描述条件于当前模型的状态不确定性，不包含完整训练seed、AE初始化、任务或背景抽样不确定性。全0/全1的退化区间不证明真实确定性。
4. development反复使用；50新状态曾在当轮评分前冻结，最新25是其子集，因此不能再次称未使用封存测试。公开来源不自动等于未见任务泛化。
5. latent距离下降、重建下降、动作一致和闭环成功是不同命题。没有确认梯度冲突、表示坍缩、物理信息丢失或动作头连接方式中的任何一项为统一根因。
6. 梯度历史诊断为PARTIAL，100/300节点严格共享head覆盖缺失；不补实验。轨迹局部模式也见于不同成功路径，不能从最终参数倒推训练因果。[H8,H9]
7. 早期一些汇总只核验现存最终报告与小型统计，未重新遍历全部原轨迹或巨大checkpoint。相关提交未有直接收据时标为未核验。证据hash仅覆盖本次引用文件，未重新扫描权重或封存集。

### 直接观察／候选解释／尚未检验

|证据等级|内容|
|---|---|
|直接观察|D表示距离更小；部分输出漂移更小；权重/顺序/seed敏感；低LR续训恢复能力；新状态与新外观总体收益未复现|
|候选解释|辅助目标易拟合的外观稳定部分不一定与控制所需误差方向一致；辅助监督的优化作用可能随训练阶段、参数路径改变|
|尚未检验|哪种任务相关信息被保留/丢失；改变监督使用位置的因果收益；独立AE初始化的稳定性；跨广泛操作任务的外推|

## 8. 若决定重设计，最多两个候选（不执行）

### 候选一：重新限定表示监督必须保留的任务相关信息

- **依据**：一致性/重建改善没有转化为总体成功收益；早期几何target较匹配action-only好，但未超过C。[H10,N4]
- **尚缺证据**：具体缺失信息的可靠同状态读出或反事实证据；几何方案有额外仿真监督且只在早期模型阶段测试，不能搬作当前AE原因。
- **可检验预测**：若一个预先指定、可可靠标注的信息缺口确实关键，仅改变该监督内容应在匹配基线和独立状态上改善相关控制，而不是只降低latent距离。
- **潜在反例**：信息本已可读；读出提高但动作/闭环不改善；收益只来自增加监督预算。任一项都削弱该候选，不应继续加更多标签兜底。

### 候选二：重新限定策略如何使用已有辅助监督

- **依据**：策略系数与时间安排明显影响行为，且输出更一致仍可绝对动作误差更大。[P2,P4,N4]
- **尚缺证据**：不存在一个已确认的共享参数干扰位置或长期优化机制；历史梯度范围不完整，局部余弦不能替代AdamW作用。
- **可检验预测**：若能先指定一个有证据的使用环节，只改变该环节、保持target/初始化/有效预算不变，应同时改善动作绝对误差与匹配闭环，而非仅重现更好训练loss。
- **潜在反例**：局部梯度变好但闭环不变；改法同时改变容量或监督预算；收益只在一个seed。此时不能称机制被验证。

以上均不是已确认解决方案。本轮不生成训练任务，也不建议先上大型WM/WAM。下一版必须先写清要保留的信息、当前不足的证据、唯一修改环节及能否定修改的结果。

### 与师兄讨论的三个决定

1. 当前证据是否值得继续投入这一研究问题，而非仅继续救当前实现？
2. 若继续，应优先改变表示监督内容，还是策略使用监督的方式？哪项直接证据足以支持选择？
3. 哪一项最小实验能区分候选解释，并预先容许失败结论，而不是继续搜索正结果？

## 9. 本轮验收

本轮只读报告、配置、CSV/JSON，CPU重新计算主要比率并生成来源hash。策略更新=0、AE更新=0、DINO更新=0、模型前向=0、backward=0、新闭环=0、仿真重放=0；未读取新的封存内容。所有历史产物保持原位。详见 `arithmetic_checks.json`、`experiment_ledger.csv`、`evidence_manifest.json`。

## CPU核验附表

从A4原统计筛选最终2000步、validation、Clean行；配对距离本身涉及两外观。各任务45个固定anchor/horizon样本，非45条独立示范。

|任务|AE|latent范数|逐坐标方差|有效秩|单位范数配对L2|
|---|---|---:|---:|---:|---:|
|cookie|C|208.736|7.4845|4.0705|0.20826|
|drawer_bowl|C|205.583|8.8296|3.9986|0.16003|
|stove|C|207.234|7.7795|4.2219|0.18619|
|cookie|D|148.517|3.2265|3.8233|0.15453|
|drawer_bowl|D|148.095|3.6127|3.8561|0.12915|
|stove|D|148.027|3.1740|3.8514|0.13568|

## 引用文件定位

编号精确定位本次读取文件；内容hash、大小、相关提交和未核验项见manifest。下列路径相对 `<PROJECT_ROOT>`。

- **A1**：`experiments/rla_ae_from_scratch/run_20261029_01/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **S1**：`experiments/rla_scratch_low_policy_seeds/run_20261030_01/protocol.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **L1**：`experiments/rla_low_lr_continuation/run_20261030_01/protocol.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **I1**：`experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **R1**：`experiments/rla_drawer_baseline_gap/run_20261101_01/protocol.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N1**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/protocol.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **A2**：`experiments/rla_ae_from_scratch/run_20261029_01/frozen_plan.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **A3**：`experiments/rla_ae_from_scratch/run_20261029_01/delivery/reconstruction_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **A4**：`experiments/rla_ae_from_scratch/run_20261029_01/delivery/representation_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **A5**：`experiments/rla_ae_from_scratch/run_20261029_01/code/experiments/rla_ae_from_scratch/run.py`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **A6**：`experiments/rla_ae_from_scratch/run_20261029_01/initialization_audit.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **P1**：`experiments/rla_scratch_policy_aux/run_20261029_01/protocol.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **P2**：`experiments/rla_scratch_policy_aux/run_20261029_01/delivery/success_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **P3**：`experiments/rla_scratch_low_policy_seeds/run_20261030_01/protocol.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **P4**：`experiments/rla_scratch_policy_aux/run_20261030_01_c_low/delivery/success_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **P5**：`experiments/rla_scratch_policy_aux/run_20261030_01_c_low/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **S2**：`experiments/rla_scratch_low_policy_seeds/run_20261030_01/delivery/success_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **S3**：`experiments/rla_scratch_low_policy_seeds/run_20261030_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **L2**：`experiments/rla_low_lr_continuation/run_20261030_01/delivery/success_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **L3**：`experiments/rla_low_lr_continuation/run_20261030_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **I2**：`experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/success_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **I3**：`experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **I4**：`experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/paired_statistics.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **R2**：`experiments/rla_drawer_baseline_gap/run_20261101_01/delivery/success_by_seed_appearance.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **R3**：`experiments/rla_drawer_baseline_gap/run_20261101_01/delivery/method_summary.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **R4**：`experiments/rla_drawer_baseline_gap/run_20261101_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N2**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/clean_reuse.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N3**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/main_statistics.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N4**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/offline_stage_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N5**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/offline_by_demo.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N6**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/success_by_seed_condition.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **N7**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H1**：`experiments/official_libero_rla_consistency/run_20261023_10/recovery_20261024/completion/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H2**：`experiments/official_libero_rla_consistency/run_20261023_09/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H3**：`experiments/official_libero_rla_consistency/stove_centering/run_20261024_01/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H4**：`experiments/official_libero_rla_consistency/stove_aux_weight/run_20261025_01/completion_report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H5**：`experiments/rla_ae_consistency_strength/run_20261026_01/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H6**：`experiments/control_preserving_rla/run_20261025_01/results/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H7**：`experiments/rla_appearance_gap_eval/run_20261028_01/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H8**：`experiments/rla_aux_interference_diagnostic/run_20261028_01/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H9**：`experiments/rla_trajectory_divergence_analysis/run_20261028_01/analysis/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H10**：`experiments/rla_task_geometry_target/run_20261028_01/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Preport**：`experiments/rla_scratch_policy_aux/run_20261029_01/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Pverification**：`experiments/rla_scratch_policy_aux/run_20261029_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Qreport**：`experiments/rla_scratch_policy_aux/run_20261030_01_c_low/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Qverification**：`experiments/rla_scratch_policy_aux/run_20261030_01_c_low/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Sreport**：`experiments/rla_scratch_low_policy_seeds/run_20261030_01/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Sverification**：`experiments/rla_scratch_low_policy_seeds/run_20261030_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Lreport**：`experiments/rla_low_lr_continuation/run_20261030_01/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Lverification**：`experiments/rla_low_lr_continuation/run_20261030_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Ireport**：`experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Iverification**：`experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Rreport**：`experiments/rla_drawer_baseline_gap/run_20261101_01/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Rverification**：`experiments/rla_drawer_baseline_gap/run_20261101_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Nreport**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/report.md`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Nverification**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Averification**：`experiments/rla_ae_from_scratch/run_20261029_01/delivery/delivery_verification.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Adata**：`experiments/rla_ae_from_scratch/run_20261029_01/data_inventory.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **PnormC**：`experiments/rla_scratch_policy_aux/run_20261029_01/targets/C/statistics.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **PnormD**：`experiments/rla_scratch_policy_aux/run_20261029_01/targets/D/statistics.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H10stats**：`experiments/rla_task_geometry_target/run_20261028_01/success_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H7stats**：`experiments/rla_appearance_gap_eval/run_20261028_01/complete_comparison_table.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H8stats**：`experiments/rla_aux_interference_diagnostic/run_20261028_01/target_switch_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H9stats**：`experiments/rla_trajectory_divergence_analysis/run_20261028_01/analysis/paired_summary.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **H1stats**：`experiments/official_libero_rla_consistency/run_20261023_10/recovery_20261024/completion/per_trial_results_combined.csv`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Nrender**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/offline_render_acceptance.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Nae**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/offline_ae_acceptance.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
- **Npolicy**：`experiments/rla_frozen_nuisance_pilot/run_20261101_01/offline_policy_acceptance.json`（已读取；引用材料hash核验，不代表重审原始轨迹）。
