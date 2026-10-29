# 新从零AE的下游策略验证：训练前验收

状态：标签生成、共同初始化、完整有效batch及零更新重载验收通过；正式训练和闭环结果尚未产生。

## 冻结资产

- 新AE-C/D均严格加载run_20261029_01正式step2000完整checkpoint；配置为官方双视角16x64。未加载校准pilot、未训练step0或历史step90000微调权重。
- AE-C checkpoint SHA256：8bfa58c22b906e200f511a8bb3876e7b5bb98f5a15ad424e59ec93508c1e8278。
- AE-D checkpoint SHA256：f2d82d593582cc8c6073a6d70e4a1569fb4d6201aa953099e59a7450bf8ebe0c。
- 本轮仅炉灶取碗放盘20训练、5探索性development示范，保持原划分、图像、FK proprio、动作统计。
- 训练有效物理窗口2555，development有效窗口703；每个窗口8个累计horizon，分别读取本外观标签，无共享/交换。
- 统计分别从各AE的20训练示范生成：两外观、所有有效窗口和horizon，每个16x64位置40880个观测。全局原始尺度C=6.517981275636939，D=4.598459492616209，不使用development。

## 完整batch预检（顺序20261101）

有效batch16外观=8物理窗口，microbatch8、累积2；BF16、Math SDPA和历史activation checkpointing。
关闭辅助系数后，仅替换C/D标签，动作输出、动作loss、全部参数动作梯度哈希精确一致。输入哈希不变。两种标签辅助梯度均可到达共享模块，数值有限，target不带梯度。
step0参数、buffers和RNG保护通过；零更新checkpoint严格保存/重载通过。预检optimizer updates=0。

固定训练batch在lambda0=0.22408536585365854下：

|target|action L1|辅助L1|加权辅助/动作梯度范数比|梯度余弦|
|---|---:|---:|---:|---:|
|新C|0.521484375|0.599609375|0.088254|0.084548|
|新D|0.521484375|0.57421875|0.089748|0.076868|

梯度范围为共同可训练VLA、proprio和head共享trunk；排除专用读出、动作专用fc1及位置/桥接gate，再按实际图依赖取交集。这是原始未缩放梯度，不是AdamW更新比例，不据此改变lambda0。

## 已冻结执行预算

两个原均衡采样顺序20261101/20261102，各C-fixed、D-fixed、D-low、D-decay四组，共4800成功更新。全部从对应C原始完整step0开始，不重新初始化辅助头。
lambda0固定历史值；D-low为一半；D-decay按已成功更新k=0..599线性降至0。各组保存step0/1/100/300/600，固定development评估0/100/300/600，共享梯度诊断0/300/600，诊断恢复RNG且不更新参数。

先160次Clean闭环。仅当至少一个方法在两个顺序均达到12/20，才对全部八组执行四套已冻结背景的640次闭环；否则停止。没有独立测试集结论。
GPU0串行执行；禁用故障GPU，不干扰其他进程。任何CUDA/ECC增量、OOM或非有限值停止并保留记录，不自动降配置或换卡。

完整证据：protocol.json、target_manifest.json、target_acceptance.json、各顺序run_plan.json、preflight/formal内的梯度与重载记录。AE/DINO更新0；正式策略训练不计算重建或额外一致性loss。
