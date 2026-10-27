# 共享Clean参考目标：结果解释

AE-C冻结，只有Background样本的辅助target改为同一物理转移的Clean target。旧μC/sC和λaux保持不变。不是AE-D方法，也没有教师策略或显式一致性loss。

| 顺序 | 方法 | Clean | 已见背景 | 未见背景 | 三外观均值 |
|---|---|---|---|---|---|
| 20261101 | C | 20/20 | 20/20 | 20/20 | 100.00% |
| 20261101 | D-center | 16/20 | 18/20 | 16/20 | 83.33% |
| 20261101 | C-shared-target | 19/20 | 20/20 | 20/20 | 98.33% |
| 20261102 | C | 13/20 | 16/20 | 13/20 | 70.00% |
| 20261102 | D-center | 3/20 | 0/20 | 1/20 | 6.67% |
| 20261102 | C-shared-target | 1/20 | 2/20 | 3/20 | 10.00% |

## 主比较与限制

- 20261101: shared minus C, equal appearance: -1.67个百分点；物理状态聚类95%区间[-5.00, +0.00]。
- 20261102: shared minus C, equal appearance: -60.00个百分点；物理状态聚类95%区间[-78.33, -40.00]。
- two orders: shared minus C: -30.83个百分点；物理状态聚类95%区间[-40.00, -20.83]。
- 20261101: shared minus D-center, equal appearance: +15.00个百分点；物理状态聚类95%区间[+3.33, +26.67]。
- 20261102: shared minus D-center, equal appearance: +3.33个百分点；物理状态聚类95%区间[-10.00, +16.67]。
- two orders: shared minus D-center: +9.17个百分点；物理状态聚类95%区间[+0.00, +18.33]。
- 两种顺序均低于对应C：本轮没有观察到共享参考target的控制增益，保留负结果，不自动调权重。
- development已反复用于开发；仅两个顺序、固定初始化与训练RNG、一个任务。区间将同一物理状态的全部外观和顺序一起重采样，不包含完整训练seed或任务不确定性。
- 新方法及D参考的闭环在GPU0；历史C闭环在GPU1。设备差异明确记录，未声称执行硬件完全相同。

| 顺序 | 外观 | 新方法成功/C失败 | C成功/新方法失败 |
|---|---|---|---|
| 20261101 | clean | 0 | 1 |
| 20261101 | seen_background | 0 | 0 |
| 20261101 | unseen_background | 0 | 0 |
| 20261102 | clean | 1 | 13 |
| 20261102 | seen_background | 1 | 15 |
| 20261102 | unseen_background | 1 | 11 |

## 背景交互

- 20261101: seen_background shared minus C background interaction: +5.00个百分点，95%区间[+0.00, +15.00]。
- 20261101: unseen_background shared minus C background interaction: +5.00个百分点，95%区间[+0.00, +15.00]。
- 20261102: seen_background shared minus C background interaction: -10.00个百分点，95%区间[-45.00, +25.00]。
- 20261102: unseen_background shared minus C background interaction: +10.00个百分点，95%区间[-15.00, +35.00]。
- two orders: seen_background shared minus C background interaction: -2.50个百分点，95%区间[-22.50, +17.50]。
- two orders: unseen_background shared minus C background interaction: +7.50个百分点，95%区间[-7.50, +22.56]。
- 交互必须结合上表绝对成功率：Clean下降不能解释为鲁棒性提升。

## 实际预测表示（不是目标相等性）

固定每条训练/development示范首个及中间有效窗口，两种方法读相同观测。以下为development均值，均在既有C规范化坐标计算；误差参考为同一Clean target。

| 顺序 | 方法 | 跨外观预测L1 | 相对L2 | Clean参考误差L1 | Background参考误差L1 |
|---|---|---|---|---|---|
| 20261101 | C | 0.003778 | 0.035547 | 0.431011 | 0.431106 |
| 20261101 | C-shared-target | 0.004332 | 0.040451 | 0.430187 | 0.430034 |
| 20261102 | C | 0.005728 | 0.051379 | 0.425357 | 0.425574 |
| 20261102 | C-shared-target | 0.007040 | 0.057847 | 0.425617 | 0.425603 |
- 目标相同是人为定义，不能作为学到不变性的证据；预测差异减小也可能伴随预测误差或控制能力变差。不能用表示指标替代闭环结果。
- loss_contribution_comparison.csv提供原C/新方法各训练区间的动作loss、原始/加权辅助loss和已记录梯度范数。两个方法监督目标有变化，不按辅助loss绝对值给方法排名。
- failure_stage_summary.csv保留抓取/接触和抬升后/放置等冻结代理分类；这些不是精确物理因果标签。

## 执行与保护

- 两组各600次成功更新，共1200；实际更新计算1200，重放0。120次有效闭环、120次尝试、无效0。
- AE/C/D更新0，封存集读取0。关闭共享模式输出/loss/梯度与旧C路径精确一致；Background图像保留；target无梯度且不进入推理输出。
- 两组初始参数与RNG、600步实际采样ID及除方法名外的执行配置均与对应C匹配。历史受保护文件哈希未变。完成后停止，不自动扩量或搜索。
- 一次冻结表示诊断因独立随机补齐的动作token ID不相同而断言停止。已保留失败日志；官方RLDSBatchTransform会random.choices补齐64个动作token位置，模型随后将这些位置替换为action queries，连续BC标签另取batch actions。修正仅匹配诊断的两路预处理Python RNG及方法间样本随机序列，并验证原差异只位于替换位置；没有修改训练、权重、部署或闭环结果。重试新增训练更新0、backward0、闭环0。
