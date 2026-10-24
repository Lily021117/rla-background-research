# 实际执行与预设协议的差异/限制

1. **硬件与恢复**：初次GPU3出现correctable ECC增量，停止其上的抽屉缓存及炉灶AE进程；未再使用该卡。用户授权后使用GPU0/1，GPU2始终排除。炉灶AE-C/D各44个已记录更新未落到完整checkpoint，故从完整step0恢复，各600有效/644已记录实际计算；另各最多一个旧在途更新无法确认。前44步采样、loss与梯度日志复现一致，不声称已核对未保存参数的逐位相等。
2. **缓存**：故障抽屉缓存保留但不验收，在新版本目录完整重提取。所有最终sidecar均来自验收后的缓存和最终AE；未用失败缓存补齐。
3. **wrapper返回状态**：初期修改仍在等待Python子进程的launcher，造成子进程正常完成、验收文件写入后出现shell尾部解析错误。已单独核查完整缓存/训练验收和严格重载，没有按shell退出码盲目放行；后续launcher改为exec。原日志保留。
4. **资源门槛与开销**：初期恢复wrapper记录的GPU0最低检查值仍为15GiB，但当次实际启动空闲显存超过已授权24.6GiB；后续显式区分GPU0=24.6、GPU1=15GiB。完整batch预检仍决定能否训练，15GiB不是容量保证。成本中的约19.03GiB为PyTorch peak allocated，不含全部reserved/driver开销，不等同最低物理显存需求。
5. **排程**：GPU1释放显存后，独立arm提前并行；原调度join其既有完成记录，未重复更新。不同卡/并行负载导致耗时不同，比较的是匹配有效更新预算，不声称GPU小时完全相同。初始化、数据、seed、dtype、batch、Math SDPA、optimizer、LR、scheduler、冻结系数和最终checkpoint规则未因此改变。
6. **继承的非论文精确复现条件**：官方LIBERO AE step90000而非论文step40000；配对h1..8、600步小预算及派生配对RGB沿用前轮冻结recipe。B/C官方head结构不同，C−B是完整RLA接入差异；C/D结构、初始化相同。共同aux系数/归一化不保证相同梯度强度。
7. **评估覆盖**：历史cookie使用一种未见材质，新任务预先固定三种并按状态7/7/6分配；跨任务未见背景均值不是同一精确材质分布。三个任务仍属取碗放盘，抽屉初始已打开。新reset状态不是封存独立最终测试。零SR地板、满分天花板和经验bootstrap退化均保留，不改协议。
8. **实际batch溯源**：legacy reference bootstrap batch及isolated_smoke目录命名保留；它不是正式forward的样本。实际训练使用独立dataset/batch，每个模型batch检查任务token，正式权重scope/global_step600和optimizer状态验收。

新增有效闭环360、无效0；闭环阶段没有基础设施重试、任务失败重跑或额外训练seed。所有历史数据、权重和报告保留；当前研究分支辅助源码的修改以Git提交和来源快照追溯，未覆盖官方固定worktree源码。
