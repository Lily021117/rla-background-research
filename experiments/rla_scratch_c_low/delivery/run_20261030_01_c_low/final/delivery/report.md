# C-low配对补全结果

|顺序|方法|Clean|背景1|背景2|背景3|背景4|背景平均|
|---|---|---|---|---|---|---|---|
|20261101|C-fixed|19/20|17/20|18/20|16/20|19/20|87.50%|
|20261101|C-low|0/20|0/20|0/20|0/20|0/20|0.00%|
|20261101|D-low|16/20|14/20|3/20|17/20|15/20|61.25%|
|20261102|C-fixed|1/20|12/20|3/20|1/20|2/20|22.50%|
|20261102|C-low|16/20|16/20|18/20|18/20|19/20|88.75%|
|20261102|D-low|17/20|6/20|18/20|15/20|15/20|67.50%|

## 配对比较（百分点）
- 20261101 C-low minus C-fixed four-background equal mean: -87.50；状态聚类95%区间 [-93.75, -81.25]。
- 20261102 C-low minus C-fixed four-background equal mean: +66.25；状态聚类95%区间 [+53.75, +78.75]。
- two-order equal C-low minus C-fixed clean: -10.00；状态聚类95%区间 [-20.00, +0.00]。
- two-order equal C-low minus C-fixed background: -10.62；状态聚类95%区间 [-17.50, -3.75]。
- two-order equal C-low minus C-fixed background-clean interaction: -0.62；状态聚类95%区间 [-13.75, +11.25]。
- 20261101 D-low minus C-low four-background equal mean: +61.25；状态聚类95%区间 [+50.00, +71.25]。
- 20261102 D-low minus C-low four-background equal mean: -21.25；状态聚类95%区间 [-36.25, -7.50]。
- two-order equal D-low minus C-low clean: +42.50；状态聚类95%区间 [+27.50, +57.50]。
- two-order equal D-low minus C-low background: +20.00；状态聚类95%区间 [+11.25, +28.75]。
- two-order equal D-low minus C-low background-clean interaction: -22.50；状态聚类95%区间 [-33.12, -11.88]。

## 范围与限制
仅新增两个C-low，各600更新、共200闭环；C-fixed/D-low共400历史匹配结果复用，未重跑。所有原始配置和历史产物哈希保持不变。标签、统计、初始化、模型训练RNG、完整采样索引、动作标签及部署一致；唯一干预为C辅助权重固定减半。AE更新0。

C-low对C-fixed检验降低C权重；D-low对C-low为共同半权重下的新AE表示比较。需分别看两个顺序、Clean绝对能力、四背景及相对Clean变化。不能把Clean下降造成的较小背景降幅称为更鲁棒。两个采样顺序不是完整训练seed；反复使用的development状态不是独立测试，区间不包含任务和完整seed不确定性。

完成即停止，不自动搜索权重或追加训练、试验。
