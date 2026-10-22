# 跨外观动作一致性策略训练

研究比较是P1−P0-adapted（配对外观数据）与P2−P1（额外一致性），不是物理动作后果保真或WAM改进。

|组|结构和初始化|输入曝光|目标|
|---|---|---|---|
|P0-paper reference|官方fs_bc原配方|原ManiSkill单视角masked输入|官方全12步BC-MSE；只作溯源，不在本轮执行|
|P0-adapted|同一随机BC-ResNet，LIBERO接口适配|每物理window一个clean当前图像|masked BC-MSE|
|P1|与P0-adapted相同|clean/background各一次forward|两路BC平均；一致性计算记录、权重严格0|
|P2|与P1完全相同|同一对两路forward|相同BC加λ·masked动作MSE一致性|

训练只传当前图像和同状态FK proprio；没有语言网络，任务语言只作身份审计元数据。两路动作标签共用同一8×7张量，不读取未来图像、DINO、RLA sidecar/encoder或策略旧checkpoint。h=8只定义动作标签的区间，绝不是未来图像辅助监督。

所有动作维度使用原normalizer的limits规则；新任务stats仅从允许训练episodes计算。平移/旋转/夹爪分别归一化到无量纲空间，再按有效scalar数做MSE。夹爪保留HDF5控制符号，不使用旧VLA的翻转或binary后处理。MSE权重按每坐标一致，并不声称各坐标具有相同物理代价；后续应单列xyz、轴角、夹爪误差。禁止把混合无量纲误差称为物理距离。

模型输出12×7，只有前8步有合法label。mask最后4步false；未补造动作。共享BC与一致性使用完全同一mask与reduction，mask分母为有效时刻×7。原论文的复制padding没有有效时间mask，这是明确的数据适配差异。

train模式保留原dropout0.1。每物理pair保存Python、NumPy、torch CPU/CUDA RNG，clean后恢复pair起始RNG计算background，最后恢复clean结束RNG。两路匹配随机mask，推进一次随机流。P1/P2相同。当前CPU验证不使用CUDA；未来GPU必须重新验收确定性和资源。

不增加EMA teacher/stop-gradient，两个输出都收到梯度。原BC配方的evaluation EMA若后续启用，只用于固定checkpoint诊断，不为一致性提供target。

固定数据19训练示范190转移、3验证示范30转移，train/validation episode无交集。覆盖核验使用当前状态agentview，保留原tex-table/tex-wall、reflection=1、MSAA=4。允许固有反射/边界RGB响应；不另行渲染。未读取历史20条保留评估示范。

两个normalizer检查明确区分：广覆盖接口审计用190训练窗口拟合；可复现的per-task entry按所选任务训练窗口拟合，验证只复用训练stats。均不读取验证标签拟合。参数/模型结构/P1P2随机条件均一致。

动作一致性为零可由常数输出实现。因此接口PASS不代表有效性；后续同时报告BC、动作方差/不同动作可区分性、clean/background及真实闭环成功率。
