# 冻结现有方法：完整新背景对照

复用上一轮20个development物理状态与全部4套新背景，固定现有600步权重。C的新背景160次与各方法Clean120次直接复用；本轮新增D-center/共享target背景320次。

| 顺序 | 方法 | Clean | 背景1 | 背景2 | 背景3 | 背景4 | 四背景均值 |
|---|---|---|---|---|---|---|---|
| 20261101 | C | 20/20 | 20/20 | 15/20 | 20/20 | 18/20 | 91.25% |
| 20261101 | D-center | 16/20 | 19/20 | 18/20 | 17/20 | 17/20 | 88.75% |
| 20261101 | C-shared-target | 19/20 | 20/20 | 17/20 | 20/20 | 19/20 | 95.00% |
| 20261102 | C | 13/20 | 15/20 | 10/20 | 16/20 | 12/20 | 66.25% |
| 20261102 | D-center | 3/20 | 1/20 | 3/20 | 1/20 | 0/20 | 6.25% |
| 20261102 | C-shared-target | 1/20 | 1/20 | 3/20 | 4/20 | 2/20 | 12.50% |

## 相对C的配对差异

- 20261101 D-center minus C: four backgrounds equal weight: -2.50pp，物理状态聚类95%区间[-12.50, +7.50]。
- 20261101 D-center minus C: clean: -20.00pp，物理状态聚类95%区间[-40.00, -5.00]。
- 20261101 C-shared-target minus C: four backgrounds equal weight: +3.75pp，物理状态聚类95%区间[-5.00, +12.50]。
- 20261101 C-shared-target minus C: clean: -5.00pp，物理状态聚类95%区间[-15.00, +0.00]。
- 20261102 D-center minus C: four backgrounds equal weight: -60.00pp，物理状态聚类95%区间[-73.75, -45.00]。
- 20261102 D-center minus C: clean: -50.00pp，物理状态聚类95%区间[-70.00, -30.00]。
- 20261102 C-shared-target minus C: four backgrounds equal weight: -53.75pp，物理状态聚类95%区间[-75.00, -28.75]。
- 20261102 C-shared-target minus C: clean: -60.00pp，物理状态聚类95%区间[-85.00, -35.00]。
- two orders equal weight D-center minus C: four backgrounds: -31.25pp，物理状态聚类95%区间[-41.25, -21.25]。
- two orders equal weight D-center minus C: clean: -35.00pp，物理状态聚类95%区间[-52.50, -20.00]。
- two orders equal weight C-shared-target minus C: four backgrounds: -25.00pp，物理状态聚类95%区间[-36.25, -11.25]。
- two orders equal weight C-shared-target minus C: clean: -32.50pp，物理状态聚类95%区间[-47.50, -17.50]。

## Clean代价与背景效应

complete_comparison_table.csv列出各条件相对各自Clean的变化；paired_statistics.json给出相应区间及相对C的背景交互。Clean变差造成的相对降幅变小不能被解释为更鲁棒。背景2是在看过C结果后受到关注的局部条件，属于事后探索，不是独立确认性证据。

| 顺序 | 方法 | 条件 | 方法成功/C失败 | C成功/方法失败 | 双成功 | 双失败 |
|---|---|---|---|---|---|---|
| 20261101 | D-center | clean | 0 | 4 | 16 | 0 |
| 20261101 | D-center | new_background_1 | 0 | 1 | 19 | 0 |
| 20261101 | D-center | new_background_2 | 4 | 1 | 14 | 1 |
| 20261101 | D-center | new_background_3 | 0 | 3 | 17 | 0 |
| 20261101 | D-center | new_background_4 | 2 | 3 | 15 | 0 |
| 20261101 | C-shared-target | clean | 0 | 1 | 19 | 0 |
| 20261101 | C-shared-target | new_background_1 | 0 | 0 | 20 | 0 |
| 20261101 | C-shared-target | new_background_2 | 4 | 2 | 13 | 1 |
| 20261101 | C-shared-target | new_background_3 | 0 | 0 | 20 | 0 |
| 20261101 | C-shared-target | new_background_4 | 2 | 1 | 17 | 0 |
| 20261102 | D-center | clean | 0 | 10 | 3 | 7 |
| 20261102 | D-center | new_background_1 | 0 | 14 | 1 | 5 |
| 20261102 | D-center | new_background_2 | 1 | 8 | 2 | 9 |
| 20261102 | D-center | new_background_3 | 0 | 15 | 1 | 4 |
| 20261102 | D-center | new_background_4 | 0 | 12 | 0 | 8 |
| 20261102 | C-shared-target | clean | 1 | 13 | 0 | 6 |
| 20261102 | C-shared-target | new_background_1 | 1 | 15 | 0 | 4 |
| 20261102 | C-shared-target | new_background_2 | 3 | 10 | 0 | 7 |
| 20261102 | C-shared-target | new_background_3 | 2 | 14 | 2 | 2 |
| 20261102 | C-shared-target | new_background_4 | 1 | 11 | 1 | 7 |

## 研究决定

- D-center未达到两个顺序均有可靠背景收益的判据。结合上表Clean代价，停止该实现的继续调参，保留负结果归档；不否定所有外观不变性方法。
  局部/单顺序正差仍单独保留为探索性候选：正均值顺序[]；正差背景条件['20261101 D-center minus C: new_background_2']。不能宣称整体有效，也不针对背景2调参。
- C-shared-target未达到两个顺序均有可靠背景收益的判据。结合上表Clean代价，停止该实现的继续调参，保留负结果归档；不否定所有外观不变性方法。
  局部/单顺序正差仍单独保留为探索性候选：正均值顺序['20261101 C-shared-target minus C: four backgrounds equal weight']；正差背景条件['20261101 C-shared-target minus C: new_background_2', '20261101 C-shared-target minus C: new_background_4']。不能宣称整体有效，也不针对背景2调参。

## 来源、可比性与范围

checkpoint_inventory.json固定实际文件与SHA-256，含D-center顺序20261102迁移恢复后的最终600步文件。comparability_audit.json核对动作/proprio统计、共享训练配置、部署设置及逐状态XML/权重身份。历史D-center训练发生的100步重放属于历史成本，本轮不重训。

reuse_sources.json与all_episode_metrics.csv逐条注明复用来源和新执行来源。本轮新增执行及选用的历史Clean/C结果均来自GPU0；更早GPU1的C结果未冒充同轮执行。既有C同轮Clean已与更早结果逐状态核验一致。所有Clean均为复用，不是本轮重跑。

沿用原部署：220控制步、chunk8、20Hz、双相机/FK proprio/STOP与动作后处理、反射1、MSAA4及env.check_success()。状态/资产/XML原样引用上一轮，四套全部保留，没有调整背景2。failure_stage_summary.csv仅沿用接近/接触及抬升后放置代理分类，不新增物理根因诊断。

本轮320次有效闭环、320次尝试、无效0；optimizer updates=0、AE updates=0、backward=0；未读取封存集。全部有效失败保留，历史保护哈希一致。

## 不确定性

- 以同一物理状态为聚类单位，将所有方法、外观与两个顺序一起重采样；20,000次bootstrap，seed20261028。两个顺序分别报告，再等权汇总。
- 两个顺序不是两个完整初始化/训练seed；区间不含训练seed和任务不确定性。20个状态及这些背景已经反复用于开发。
- 所有背景都报告，不能只选择背景2或赢家顺序；低Clean成功率下的相对背景退化可能受到地板效应影响。
- 单个或少量翻转的区间有离散性，零端点或全零bootstrap不能证明精确无效应。
