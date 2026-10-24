# 两个新增任务：固定AE与B/C/D训练及development闭环

## 本轮结论：未复现一致性增量的跨任务收益

历史cookie任务中微小的D−C正差异没有在新增任务重复：抽屉处于成功率地板；炉灶C为20/19/20次成功，D为0/0/0（顺序Clean/已见/未见背景，分母均20）。当前配方的正向方法结论不获支持；这是负面development结果，不通过改权重或换checkpoint消除。

三任务等权D−C分别为Clean −33.33、已见背景 −31.67、未见背景 −33.33个百分点；再对三种外观等权为−32.78个百分点。三个任务各自的跨外观平均D−C为cookie +1.67、drawer −1.67、stove −98.33个百分点，异质性极强。下文状态聚类区间只条件于这三个固定任务、一个训练seed和当前development状态，不能外推为所有LIBERO任务的置信区间。

炉灶已见背景交互项为+5个百分点，但D的Clean和背景均为0，属于地板效应，不能叫鲁棒性收益。抽屉B/D各0/60，C仅1/60，当前预算下不能可靠区分该任务的方法增量。没有重新挑选任务、状态或材质。

AE-D确实降低归一化配对距离（drawer 0.416→0.231；stove 0.435→0.241），同时Clean latent范数明显收缩（85.5→38.8；85.9→36.1），有效秩降低（67.5→42.1；72.4→44.0），重建L1略变差。非零跨episode距离和有效秩不支持“所有转移已变常量”的断言；也没有据此证明物理运动信息已保留或损失。

炉灶最终offline-development动作L1（示范等权，归一化空间）C为Clean 0.2126/Background 0.2119，D为0.2020/0.2011；D离线误差更低却闭环0/60、C为59/60。不能凭L1挑模型或解释闭环收益。固定共同aux系数也未匹配梯度强度：炉灶预检C/D共享backbone辅助梯度范数约0.391/0.476，是潜在尺度/监督强度混杂，不把D−C全部解释为抽象“不变性”的因果效应。

统一阶段代理显示：炉灶D有55次进入目标中心10cm范围但未达到3cm抬升，另5次未进入该范围；B有48次抬升后未完成；C为59次成功。抽屉B/D各60次接近阶段失败；C有42次近目标未完成、14次接近失败、3次抬升后失败、1次成功。这些是轨迹代理和固定规则视频面板，不是接触/抓取真值，也未定位唯一实现或失效机制。

本轮止于这些固定结果。下一轮是否执行剩余两个匹配seed由用户决定；若执行必须同时改变AE和策略训练seed。不能将只换策略seed称为完整方法复现，也不自动新增loss或调参。

本实验研究背景材质干预下的视觉鲁棒性，不是官方LIBERO-Plus。主比较D−C；单训练seed、三个固定取碗放盘任务，仅提供探索性证据。

## 成功次数（每项20个物理状态）

|任务|外观|B|C|D|D−C|
|---|---|---|---|---|---|
|cookie_historical|clean|19/20|20/20|20/20|+0 pp|
|cookie_historical|seen_background|17/20|19/20|20/20|+5 pp|
|cookie_historical|unseen_background|17/20|20/20|20/20|+0 pp|
|drawer|clean|0/20|0/20|0/20|+0 pp|
|drawer|seen_background|0/20|1/20|0/20|-5 pp|
|drawer|unseen_background|0/20|0/20|0/20|+0 pp|
|stove|clean|2/20|20/20|0/20|-100 pp|
|stove|seen_background|2/20|19/20|0/20|-95 pp|
|stove|unseen_background|5/20|20/20|0/20|-100 pp|

## 主比较与不确定性

- cookie_historical / clean: +0.00 pp；状态聚类bootstrap95% CI=[0.0, 0.0]；成功翻转+0/−0。
- cookie_historical / seen_background: +5.00 pp；状态聚类bootstrap95% CI=[0.0, 15.0]；成功翻转+1/−0。
- cookie_historical / unseen_background: +0.00 pp；状态聚类bootstrap95% CI=[0.0, 0.0]；成功翻转+0/−0。
- cookie_historical / three_appearance_mean: +1.67 pp；状态聚类bootstrap95% CI=[0.0, 5.0]；成功翻转+None/−None。
- drawer / clean: +0.00 pp；状态聚类bootstrap95% CI=[0.0, 0.0]；成功翻转+0/−0。
- drawer / seen_background: -5.00 pp；状态聚类bootstrap95% CI=[-15.0, 0.0]；成功翻转+0/−1。
- drawer / unseen_background: +0.00 pp；状态聚类bootstrap95% CI=[0.0, 0.0]；成功翻转+0/−0。
- drawer / three_appearance_mean: -1.67 pp；状态聚类bootstrap95% CI=[-5.0, 0.0]；成功翻转+None/−None。
- stove / clean: -100.00 pp；状态聚类bootstrap95% CI=[-100.0, -100.0]；成功翻转+0/−20。
- stove / seen_background: -95.00 pp；状态聚类bootstrap95% CI=[-100.0, -85.0]；成功翻转+0/−19。
- stove / unseen_background: -100.00 pp；状态聚类bootstrap95% CI=[-100.0, -100.0]；成功翻转+0/−20。
- stove / three_appearance_mean: -98.33 pp；状态聚类bootstrap95% CI=[-100.0, -95.0]；成功翻转+None/−None。

三个任务等权平均（仅条件于这三个任务，不是跨任务总体置信区间）：

```json
{
  "results": [
    {
      "appearance": "clean",
      "D_minus_C": -0.3333333333333333,
      "fixed_three_task_state_cluster_CI95": [
        -0.3333333333333333,
        -0.3333333333333333
      ],
      "per_task": [
        0.0,
        0.0,
        -1.0
      ]
    },
    {
      "appearance": "seen_background",
      "D_minus_C": -0.31666666666666665,
      "fixed_three_task_state_cluster_CI95": [
        -0.3666666666666667,
        -0.26666666666666666
      ],
      "per_task": [
        0.05,
        -0.05,
        -0.95
      ]
    },
    {
      "appearance": "unseen_background",
      "D_minus_C": -0.3333333333333333,
      "fixed_three_task_state_cluster_CI95": [
        -0.3333333333333333,
        -0.3333333333333333
      ],
      "per_task": [
        0.0,
        0.0,
        -1.0
      ]
    },
    {
      "appearance": "three_appearance_mean",
      "D_minus_C": -0.3277777777777777,
      "fixed_three_task_state_cluster_CI95": [
        -0.3444444444444445,
        -0.31111111111111106
      ],
      "per_task": [
        0.016666666666666666,
        -0.016666666666666666,
        -0.9833333333333332
      ]
    }
  ],
  "interactions": [
    {
      "background": "seen_background",
      "D_C_interaction": 0.016666666666666666,
      "fixed_three_task_state_cluster_CI95": [
        -0.03333333333333333,
        0.06666666666666667
      ],
      "per_task": [
        0.05,
        -0.05,
        0.05
      ]
    },
    {
      "background": "unseen_background",
      "D_C_interaction": 0.0,
      "fixed_three_task_state_cluster_CI95": [
        0.0,
        0.0
      ],
      "per_task": [
        0.0,
        0.0,
        0.0
      ]
    }
  ],
  "limits": [
    "Three fixed tasks only: conditional state-cluster intervals do not establish generalization across tasks",
    "Single matched AE/policy training seed",
    "Discrete success flips and empirical zero lower bounds do not establish stable positive gains",
    "Historical cookie task has one unseen palette, new tasks have frozen three-palette mix",
    "New reset development states are not independent final test; no sealed contents read"
  ]
}
```

## 数据、训练与恢复

每个新增任务20条训练＋5条development示范，按源episode隔离。每任务20个固定合法reset状态，Clean/已见/未见材质配对；未见材质为预先指定的三种调色板，按状态序号固定分配。原cookie历史结果未重跑，其未见背景仅一种，因此跨任务OOD均值并非完全相同的材质分布。

四组AE各600有效更新，总计2400；六组策略各600有效更新，总计3600；新增有效闭环360，任务失败保留为有效结果。主比较仅用最终600步。AE-C/D使用官方step90000共同初始化；策略均使用原始共同未训练初始化，不续训B20。B是官方action-only结构，C/D是官方RLA结构；C−B包含完整RLA接入差异，C/D结构与初始化一致。

GPU3此前出现correctable ECC增量后停止，未重新使用；GPU2始终排除。炉灶两AE各重放44个已记录更新，记录实际计算644步/组，另各有至多一个旧中断时在途更新无法确认。前44步loss、梯度范数及采样日志复现一致，但不声称对未保存参数进行了逐位验证。历史故障、部分缓存和checkpoint均保留。抽屉缓存在新目录完整重提取；已记录wrapper脚本编辑导致的提取结束后shell解析错误，其Python完整验收文件独立核验通过。

GPU资源释放后另附ECC终检。跨卡排程只改变运行时间，不改变数据、seed、batch、优化器、系数和最终checkpoint选择。已完成的并行arm由原调度核验复用，不重复执行更新。

## 表示与策略机制限制

详见representation_summary.csv、按示范/horizon的原始指标及decoder的zero/other-episode latent消融。必须同时看归一化距离、latent尺度/有效秩及重建相对copy-current基线；不把距离变小等同动作语义保真，也不因策略失败断言表示坍塌。

固定lambda_aux=0.22408536585365854（不按新任务重新校准）；lambda_cons=0.009127798272072366。C/D共享AE-C训练集归一化。共同系数并不保证共同辅助梯度强度，已保存实际辅助loss及共享backbone辅助梯度范数；D−C可能包含表示尺度和有效监督强度变化，不能全部归因于外观不变性。

失败阶段采用统一的目标抬升/末端距离轨迹代理，不把代理或视频观察当作接触物理真值。所有trial均保存日志、视频和轨迹；不根据失败重选状态或追加训练。

## 限制与下一步

仅一个匹配AE/策略训练seed；20个development状态/任务，非封存最终测试。三个任务仍是取碗放盘，操作场景不同但primitive覆盖有限。天花板/地板、少量成功翻转、离散bootstrap及经验零下界均不能证明稳定收益。若接近满分，应认定现有协议区分能力有限，而非方法已被证实。

后续每任务总计3个匹配seed，新增seed20261024/20261025须同时重训AE-C/AE-D和策略B/C/D，从共同原始初始化出发；只换策略seed只能说明给定AE下的稳定性。不在本轮执行。若需扩展状态/背景，必须先冻结新的合法reset种子及互斥未见材质清单，仅按合法性保留，不筛选D获胜样本；封存集仍不读取。


bootstrap退化警告：20个状态全无差异时可能得到[0,0]，全同向翻转时可能得到[-100,-100]或[100,100]个百分点。这只是经验重采样没有未观察事件，绝不是总体效应精确已知。绝对SR另提供Wilson区间；未开展等效性试验，不能把零差异解释为方法等效。

历史cookie任务沿用run09原结果及其原始恢复记录：原AE-D的80步重放、策略C的额外一次已记录计算均不并入本轮成本，也不被抹除。因此600步指有效更新预算，不声称三任务的实际计算次数完全相同。

成本CSV中的seconds为记录的更新/固定验证阶段耗时；observed_active_process_wall_seconds_approx使用日志出生时间至验收文件时间，包含初始化且不计无人运行的停机等待。并行进程耗时相加不是整体墙钟时间或精确GPU kernel busy time。

目录isolated_smoke是历史bootstrap命名，正式checkpoint由scope/global_step600及optimizer/scheduler状态验收。其legacy batch_audit.json描述的是未用于正式forward的reference bootstrap batch；真实训练数据由policy_training.execute中的独立dataset/batch构造，每个实际模型batch均检查任务token。formal_batch_provenance_audit.json明确区分二者，不以legacy参考batch推断正式任务来源。
