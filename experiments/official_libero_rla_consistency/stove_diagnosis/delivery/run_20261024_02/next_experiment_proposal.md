## 8. 唯一下一步：只改变D target中心偏移

建议一个冻结AE的两臂对照：D-control维持(z_D−mu_C)/sigma_C；D-recenter仅改为(z_D−mu_D_train)/sigma_C。mu_D只用已有20条训练的两外观sidecar计算；保留相同sigma_C、lambda_aux、AE-D权重、结构、初始化、所有数据、seed、LR/optimizer和600步预算。不能同时改变scale、辅助系数、夹爪权重或阈值。两臂均从同一原始step0开始，不续训退化策略；固定step0/100/300/600，final600为唯一比较。

放行前离线验证：target差异必须是逐坐标常数平移；所有转移间差分完全不变；共享初始参数/RNG/action输出一致；数据及batch逐项一致。后续若授权训练，统一评估冻结development，不挑checkpoint或状态；关注关闭行为、抓取位姿、实际SR。若recenter无改善，不追加调参。本轮没有执行该干预。

理由：这是隔离已观测到的固定均值混杂的最小改变；无须预设AE丢失物理信息，也无须增加物理loss。闭式读出目前未支持加入物理后果loss作为修复结论。
