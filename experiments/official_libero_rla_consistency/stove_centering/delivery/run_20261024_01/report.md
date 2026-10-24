# 炉灶 D target 中心化：最终结论

本轮中心化干预**没有改善最终任务成功率**：两组各外观均0/20。不能因此说中心化没有任何行为作用；需结合夹爪误差和失败阶段。也不能声称找到了唯一根因或证明外观一致性优于历史C。

## 成功率与配对统计

| 外观 | D-old | D-center | 差值 | 配对95%CI（比例） | 历史C |
|---|---:|---:|---:|---|---:|
| clean | 0/20 | 0/20 | +0.0 pp | [0.0, 0.0] | 20/20 |
| seen_background | 0/20 | 0/20 | +0.0 pp | [0.0, 0.0] | 19/20 |
| unseen_background | 0/20 | 0/20 | +0.0 pp | [0.0, 0.0] | 20/20 |

上表依次为外观、D-old成功次数、D-center成功次数、差值、配对bootstrap区间、历史C成功次数；每格分母20。共120有效闭环，1次基础设施中断尝试单独保留，未重试任务失败。网络中断后保留前101次有效结果，仅恢复剩余19个槽位；中断槽位使用新目录，训练没有重放。以20个物理状态为跨外观聚类单位，不把60种状态/外观组合当60个独立状态。

全部观测均失败，因此配对翻转为0、背景交互为0，bootstrap区间退化为[0,0]，这不是“真实差异精确为0”的证明。单个外观每组0/20的二项精确95%区间上限约16.8%；地板效应和小样本限制明显。补充边际精确区间仅用于说明地板不确定性，不替换预定的状态聚类主分析。

## 行为阶段，而非仅最终标签

- D-old：发出关闭命令 12/60；达到抬升≥3cm 0/60；阶段分布 {'near_target_grasp_or_contact_failure': 55, 'approach_failure': 5}。
- D-center：发出关闭命令 60/60；达到抬升≥3cm 60/60；阶段分布 {'post_lift_or_placement_failure': 60}。

阶段规则沿用既定EE距离/目标抬升代理，不将接触代理冒充独立抓取真值。关闭命令不等于实际成功抓取；抬升也不等于最终任务完成。逐轨迹检查和视频保存在本地。

`rollout_gripper_timing.csv`覆盖全部120条保存轨迹：预测chunk经冻结后处理后与实际执行动作逐项一致。只恢复保存状态并执行mj_forward，未新增控制步。D-center首次关闭均不早于首次进入10cm接近范围；D-old有1次提前关闭。固定选择首个状态、全部三种外观的均匀时间点视频拼图，仅作视觉参考，不替代统一阶段统计。

训练验收耗时：D-old 2263.16秒，D-center 2080.16秒；各进程峰值显存20,397,878,784字节（约19.0GiB），无新增ECC。恢复评估峰值约3.72GiB，结束后本任务GPU进程全部退出。历史ECC计数保留，不声称硬件已修复。

## 离线动作与梯度

全部固定development窗口（动作采用训练归一化空间，夹爪保持0/1语义）：

| 组 | 窗口×外观数 | 动作L1 | 夹爪L1 |
|---|---:|---:|---:|
| D-old | 1406 | 0.208111 | 0.088801 |
| D-center | 1406 | 0.207499 | 0.083801 |

关闭事件锚点的当前步闭合预测：

| 组 | 划分 | 预测闭合/观测数 |
|---|---|---:|
| D-old | train | 29/40 |
| D-old | validation | 9/10 |
| D-center | train | 34/40 |
| D-center | validation | 10/10 |

与覆盖事件附近的固定诊断样本分开报告，不能只挑局部改善。完整逐维/逐预测跨度指标见diagnostics目录。

最终验证固定batch的辅助残差分解：

| 组 | target平方能量 | 残差均值能量 | 去均值残差能量 | 原始latent坐标L1 |
|---|---:|---:|---:|---:|
| D-old | 0.241512 | 0.014019 | 0.020176 | 0.374196 |
| D-center | 0.032622 | 0.011472 | 0.019638 | 0.355792 |

目标平方能量、实际残差能量和梯度范数是不同量；训练损失仍是官方L1。去中心后误差同时还原到统一原始AE-D latent坐标。

最终验证梯度（全部共享模块）：

| 组 | 动作梯度范数 | 加权辅助梯度范数 | 余弦 |
|---|---:|---:|---:|
| D-old | 2.351836 | 0.167867 | 0.053160 |
| D-center | 2.394308 | 0.100305 | -0.037444 |

另提供VLM、共享head主干、proprio分项；不把较小辅助loss或更正的梯度夹角直接当作物理信息改善。

## 复现、预算与限制

D-old在0/1/100/300/600全部参数是否逐位复现历史D：True。两组共同step0参数与所有训练RNG检查通过。各600成功更新、实际600、重放0；AE更新0。训练仅使用20条训练示范，development不参与均值拟合。单seed结论不外推至其他任务或独立测试。

本轮到此停止；不自动调LR、辅助系数、阈值或增加物理loss。下面是完整方法与复现记录。

---

# Stove D target centering: single-variable development experiment

## Scope and identity

D-old `(z_D−mu_C)/s_C`; D-center `(z_D−mu_D)/s_C`. Same frozen AE-D/raw sidecars,20 training episodes,5 offline development episodes,2555 training complete windows. Both start from the common original Prismatic/RLA policy step0, not a trained policy.600 successful updates each;4800 physical-window and9600 appearance exposures each. No AE updates or sealed-test access. Final600 selected in advance.

## Closed-loop results

| Condition | D-old | D-center | Difference | Paired95%CI (fraction) | Historical C |
|---|---:|---:|---:|---|---:|
| clean | 0/20 | 0/20 | +0.0 pp | [0.0, 0.0] | 20/20 |
| seen_background | 0/20 | 0/20 | +0.0 pp | [0.0, 0.0] | 19/20 |
| unseen_background | 0/20 | 0/20 | +0.0 pp | [0.0, 0.0] | 20/20 |

120 valid rollouts,20 physical development states, three appearances. State-cluster bootstrap preserves appearances within each state. These are single-training-seed development results, not independent test evidence. Historical C's state/XML/language hashes match. Interactions and paired flips are in `paired_statistics.json`; absolute Clean success must be considered with any background interaction.

## Reproduction and normalization

Historical D parameter bitwise reproduction at0/1/100/300/600: **True**. Full per-checkpoint evidence is in `initialization_and_reproduction_audit.json`. If this is false, changes between old historical D and D-old must not be attributed to target centering. The primary comparison is the new matched D-center versus D-old.

mu_D uses the original extractor's FP32-output/FP64-accumulated training moments, not development.40 episode/appearance sources and40880 complete-window cumulative-horizon rows. Independent FP16-cache recomputation differs slightly from pre-storage moments; see `normalization.json`. s_C remains the exact official float32 scalar. Fixed-offset and pairwise-difference checks pass within explicitly documented float32 rounding bounds. Aux output has a bias; no automatic center compensation, bias reset or architecture change was introduced.

## Offline evidence and gradient interpretation

`offline_summary.csv` and `diagnostics/isolated_smoke/action_by_dimension_horizon.csv` separate train/development, dimension and prediction horizon. `closure_predictions.csv` records first closing prediction and missed/early closes within the fixed8-position chunk; no-close outside that window is censored, not invented as a delay.

`auxiliary_residual_energy.csv` separates target squared energy, actual residual mean energy and demeaned residual energy. This is a squared-error decomposition for diagnosis, not the trained mean-L1 objective. Raw-coordinate predictions use s_C times prediction plus the appropriate center. Lower normalized loss alone is not evidence of better physical information/control.

`gradient_contributions.csv` measures action and weighted-auxiliary derivatives on matching frozen train/development batches at0/1/100/300/600. Scopes separately include trainable VLM, shared head trunk, proprio projector and their union; branch-specific output projections are excluded. Each objective has an independent, matched-RNG forward. Diagnostics use eval-with-grad and FP32 accumulation, so they describe frozen-model derivatives, not a stochastic training-step replay. No optimizer is constructed in diagnostics. Equal lambda does not imply equal gradient strength.

## Failure analysis and limitations

`failure_stage_counts.csv` uses the frozen EE-target distance and lift proxies, not unverified visual ground truth. All trials record whether a positive simulator closing command was actually executed. Videos/trajectories remain local. Small counts and discrete intervals are not proof of stable benefit; no unique root cause or superiority over C is inferred from this single intervention.

## Execution and protection

Math SDPA, BF16, effective16 appearance examples, microbatch8/accumulation2, accepted non-reentrant activation checkpointing; fixed lambda0.22408536585365854 and lr2e−4. Full optimizer/scheduler/RNG/sample-position checkpoints. No extra updates or replays. Two initial strict-loading gates failed because the new launcher omitted the official RLA_STEPS switch; corrected before any update, failed logs preserved. All subsequent settings match the frozen experiment.

Historical protected hashes unchanged. No weights, sidecars, raw trajectories or credentials are included in GitHub delivery. Completion releases GPU resources. No automatic follow-up training is authorized by this result.
