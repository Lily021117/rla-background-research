# 后续唯一小预算方案（本轮未执行）

先限于cookie-box单任务，在固定split中使用全部合格训练窗口及该任务验证示范。没有语言输入，不能把多任务指令放在metadata里就宣称语言条件策略。炉灶验证缺失仍保留，暂不据此启动四任务泛化结论。

P1/P2从同一随机step0开始；seed42、同参数、同paired batch4、同采样序列、同dropout配对策略、同masked MSE。P1权重0，P2 **lambda=0.1**，不搜索。这一保守数值在看开发成功率前已写入脚本，归一化动作空间中与BC同为MSE。初始化的加权梯度比只有约0.0006–0.0007，可能非常弱；不能因短预算无收益就判断思想无效，也不因此自动扩大λ。

两组各 **100次成功optimizer调用**，预检update计入，不自动追加。沿用原fs_bc AdamW lr3e-4、wd1e-4、betas(.9,.999)、eps1e-8、warmup200及cosine定义，不采用旧Spatial-Pro的head1e-4修复路线。100步全部在warmup区间，原scheduler首步lr0必须记录（optimizer调用计数与非零参数更新次数分别报告）。这是可训练性/尺度pilot，不能当充分训练或与论文800epochs比较。保留step0/1/10/50/100，主结果固定最终100，不选最佳checkpoint。

有效batch为4个物理window，无累积；若资源要求改变batch，需要训练前共同冻结，不单独改一组。100updates=400物理window曝光、800当前RGB曝光/组。P0-adapted若以后加入同更新数/物理window预算，则只有400当前RGB曝光；额外图像算力要报告。独立轨迹数量另报，重复采样不增加episode数。

正式loop尚未在本轮实现或执行optimizer；提供的是已验收的`step_loss`与真实`PairedDataset`。下一次启动前需要审查循环、成功计数、恢复状态、scheduler/EMA和checkpoint重载，不应把本轮接口验收视为完成这些运行验收。训练资产只从新随机step0开始，无旧策略、RLA或未来帧目标。

放行：确认接受文档列出的LIBERO跨域适配；同任务train/validation及训练only normalizer冻结；GPU UUID/ECC/26GiB门槛通过或明确选择CPU；首batch完整loss/gradient/参数更新及保存恢复验收。P1/P2必须同时满足且预检更新计入预算。若不能满足，停止，不能替换Spatial-Pro初始化。

观察：训练/验证分开，BC及xyz/rotation/gripper误差、输出方差、常数baseline、预测动作与示范动作差异、一致性loss及其梯度比、资源/有效更新数。需要基础BC正常拟合且动作不坍缩；后续分别验证P2是否优于配对P1、是否损害Clean，以及独立新示范/新初始状态/未见背景能否保留收益。仅在后续另行授权并冻结控制和评估协议后开展闭环。本轮不评分任何保留集。
