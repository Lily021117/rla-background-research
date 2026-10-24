# 方法身份与任务定义

固定官方RLA commit：`0fc6e97212442ebccffa06f1d42f5481e5f22950`；VLA-Adapter上游`23fa0c9c`。本研究为基于LIBERO示范的派生配对训练与自定义背景材质评估，不是官方LIBERO-Plus或论文精确复现。

| 组 | AE初始化与目标 | 策略初始化与目标 | 辅助target | 任务组织 |
|---|---|---|---|---|
| B | 不使用AE辅助监督 | 共同官方Prismatic多模态预训练＋原官方动作head初始化；动作L1 | 无 | 每任务独立20条训练示范 |
| C | 官方LIBERO双视角16×64 step90000；600步，两外观各自未来DINO重建(L1+MSE)的平均 | 原共同策略step0＋官方RLA head；600步，动作L1＋固定λ_aux×辅助L1 | `(z_C−μ_C)/s_C` | 同上，Clean/Background均训练 |
| D / D-old | 同一step90000；600步，同C重建＋0.009127798272072366×配对latent MSE | 同C共同step0、结构及600步预算 | `(z_D−μ_C)/s_C` | 同上，非cookie权重迁移 |
| D-center | 同一已冻结AE-D，不再更新AE | 重新从共同step0训练600步，不续训D | `(z_D−μ_D)/s_C` | 同上，仅改训练均值中心 |

共同`λ_aux=0.22408536585365854`，`s_C=2.6870148181915283`。均值为16×64逐槽位训练统计，缩放为C的全局标量；D-center不重算缩放/系数。C并非仅Clean数据：每个物理窗口两种外观各一次，共4800物理窗口、9600外观曝光。策略B原head与C/D官方RLA head不同，C−B不是结构完全一致的比较。C/D共享同一原始初始化，不从已训练B开始。AE两组更新encoder和匹配decoder，DINO冻结。

来源：本地`run_20261023_10/stove/ae_run/training_plan.json`、`policy_run/run_plan.json`、`frozen_dataset_manifest.json`及中心化实验`run_plan.json`、`checkpoint_inventory.csv`。本轮只复用这些已通过的身份验收，核对实际C/D-center最终文件哈希，不重训或重新挑checkpoint。

## 真实任务

- Suite：`libero_spatial`。
- 完整task ID：`pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`。
- 实际模型语言：`pick up the black bowl on the stove and place it on the plate`。
- BDDL语言：`Pick the akita black bowl on the stove and place it on the plate`；词面不同，但指向同一目标物体/任务，沿用冻结输入，不本轮改词。
- BDDL：`${LIBERO_ROOT}/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl`。
- 唯一goal：`(And (On akita_black_bowl_1 plate_1))`。第二个黑碗、炉灶开关和柜门不属于goal。

源码链：`envs/env_wrapper.py:check_success` → `envs/problems/libero_tabletop_manipulation.py:_check_success/_eval_predicate` → `envs/predicates/base_predicates.py:On.__call__` → `envs/object_states/base_object_states.py:ObjectState.check_ontop`。

实际三条件（必须同时成立）：

1. `plate_z <= bowl_z`：目标碗中心不低于盘子中心。
2. `plate.check_contact(bowl)`：盘与目标碗的碰撞几何接触。
3. `norm(plate_xy−bowl_xy)<0.03`：水平中心间距严格小于3cm。

源码**没有**额外要求手松开、稳定保持若干步、特定碗朝向、抬升3cm或炉灶开关状态。不要把“抓起碗”当最终成功，也不要把手未释放直接当判定失败原因。3cm抬升和10cm接近均为历史描述性阶段阈值，与官方3cm水平目标阈值不是同一量。

保存状态仅有积分时刻qpos/qvel，历史接触/缓存位姿可能来自最后子步的计算相位；离线mj_forward重建条件不得覆盖历史成功标签。接触是几何代理，不能单独证明持续稳定抓持或精确抓持力。
