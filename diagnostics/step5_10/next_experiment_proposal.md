# 下一步：仅降低native action-head LR的10更新对照（未执行）

唯一主要组间改变：native Pro action head LR从1e-4降至1e-5。理由：仅加载head-step1就将固定验证动作L1从0.15268升至0.25213，三条验证示范均变差；单独VLM影响较小，proprio影响极小。没有定位新的实现错误；此方案验证优化敏感性，不预设降低LR可恢复闭环。

对照组：BC-only unified架构，各组LR均1e-4。实验组：同样BC-only，只有native head LR=1e-5。VLM/LoRA/action queries、proprio和保留的auxiliary head均1e-4；辅助权重均0，不新增物理、蒸馏或其他loss。可训练范围不变，不冻结backbone。两组都显式使用相同参数组划分，避免组结构不同。

从已核验共同step0参数开始，创建同样空AdamW状态，不能载入step1 optimizer。betas(.9,.999)、eps1e-8、weight_decay0、constant scheduler、BF16 native/FP32 LoRA、Math SDPA均不变；不加gradient clipping。microbatch2、累积2、有效batch4，原相同sampling_order前10行，seed20261021与原逐microbatch RNG规则相同，初始化与新增head相同。clean/background数据、语言/FK、归一化和背景协议不变。

两组训练loader共同补末尾STOP2、标签STOP与attention mask。共同实现该协议并通过eval模式训练forward/deploy动作max_abs=0后，才开始计数。历史Step5.7没有这一STOP，因此不能拿历史10步或100步直接当严格对照；必须重跑同STOP对照组。Training阶段原生0.02扰动保留，不能同时调整。

预算各10个成功optimizer updates；任何预检更新计入。保存step0/1/5/10并严格重载，主结果固定step10，不选最佳，不延长预算。记录每组动作/辅助0损失、参数与grad norm、dtype与moment状态、各任务曝光、峰值内存/ECC、跳过和恢复。特别记录BF16 head的实际变化坐标比例；降低LR可能使部分更新小于表示精度，这应如实分析，不能同时修改dtype。若需技术smoke，避免额外未计数更新。

观察：同一固定诊断样本的误差/漂移和chunk/分量；原固定30验证窗口按3条示范等权、clean/background分别报告。夹爪诊断必须包括已有训练/验证中的真实开合窗口，先固定清单，不能依结果更换；当前14个代表窗口全为open，不能据它们判断抓取时序。不得用保留集调参。

进入后续闭环验证的条件：两组共同协议、初始化/数据顺序/RNG及10次成功预算匹配；STOP/eval等价、有限值/重载/保护/ECC均通过；离线分项与逐示范结果无未解释严重恶化。若降低head LR仍恶化，应交付并停止。只有在完整审查后另行冻结配对任务/状态和成功判据，才建议开展小规模闭环，不能由离线L1宣布控制收益。两组最终checkpoint均固定step10，验证不用来挑checkpoint或追加步数。

本轮仅方案，不执行任何训练或闭环。
