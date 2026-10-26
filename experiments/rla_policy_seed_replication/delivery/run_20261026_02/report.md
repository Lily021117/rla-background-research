# 固定 AE 的 C/D-center 策略 seed 配对复核

本轮无 AE 更新、无超参数搜索。新 seed 各 C/D-center600步，共2400成功更新；新闭环240次。

初始化范围：`all_random_modules_new_seed`。实际源权重与随机化模块见各seed初始化验收。

|策略seed|方法|Clean|已见背景|未见背景|等权平均|
|---|---|---:|---:|---:|---:|
|20261023|C|20/20|19/20|20/20|0.9833|
|20261023|D-center|0/20|0/20|0/20|0.0000|
|20261029|C|2/20|2/20|0/20|0.0667|
|20261029|D-center|6/20|2/20|3/20|0.1833|
|20261030|C|10/20|11/20|13/20|0.5667|
|20261030|D-center|0/20|1/20|1/20|0.0333|

## 统计与限制

同seed内按20个物理状态配对，三外观联合重采样。逐seed的D−C、翻转及背景交互全部保留。三个seed平均的状态聚类区间条件于这三次训练，不包含完整训练seed不确定性；不能把360条新旧trial当独立episode。

{
  "mean_D_minus_C": -0.4666666666666667,
  "conditional_state_cluster_CI95": [
    -0.5222222222222223,
    -0.4111111111111111
  ],
  "seed_effects": [
    {
      "policy_seed": 20261023,
      "C": 0.9833333333333333,
      "D_center": 0.0,
      "D_minus_C": -0.9833333333333333
    },
    {
      "policy_seed": 20261029,
      "C": 0.06666666666666667,
      "D_center": 0.18333333333333332,
      "D_minus_C": 0.11666666666666667
    },
    {
      "policy_seed": 20261030,
      "C": 0.5666666666666667,
      "D_center": 0.03333333333333333,
      "D_minus_C": -0.5333333333333333
    }
  ],
  "method_seed_variability": {
    "C": {
      "seed_values": [
        0.9833333333333333,
        0.06666666666666667,
        0.5666666666666667
      ],
      "range": 0.9166666666666666,
      "sample_SD": 0.4589642122738418
    },
    "D_center": {
      "seed_values": [
        0.0,
        0.18333333333333332,
        0.03333333333333333
      ],
      "range": 0.18333333333333332,
      "sample_SD": 0.09765775461803858
    }
  },
  "mean_absolute_within_seed_gap": 0.5444444444444444,
  "interpretation": "CI conditions on these three training runs, not uncertainty over training seeds. Only three seeds; report individual runs, no precise random-seed population inference."
}

固定AE、单任务、重复development状态，不能评价AE seed稳定性或独立泛化。历史设备/恢复差异保留。历史细粒度阶段字段缺失时标未知，不当作无事件。3cm抬升和10cm接近仅为诊断代理，成功始终由官方判定。

训练初始化为原始预训练策略，不从任何C/D成品、D-inv03、低LR或E系列续训。两方法各自已验收训练均值、共同s_C与lambda_aux保持不变。全部600次实际LR2e-4，scheduler在optimizer后调用，保存的下一步LR2e-5不代表第600次更新LR。

数据20train/5development；2555有效训练窗口，每组4800物理窗口曝光、9600外观曝光，约1.879遍窗口；不同seed顺序不同，同seed内C/D完全匹配。不新增轨迹或读取封存内容。

## 新 seed 与历史参照的直接回答

- seed 20261023：C等权成功率98.3%，D-center0.0%，D−C=-98.3个百分点。
- seed 20261029：C等权成功率6.7%，D-center18.3%，D−C=+11.7个百分点。
- seed 20261030：C等权成功率56.7%，D-center3.3%，D−C=-53.3个百分点。
两个新seed未同时复现D−C为负，必须保留方向变化，不概括为稳定退化。

|seed|方法|失败数|抬升后失败|接近目标但未抬升|接近失败|
|---|---|---:|---:|---:|---:|
|20261023|C|1|0|1|0|
|20261023|D-center|60|60|0|0|
|20261029|C|56|56|0|0|
|20261029|D-center|49|49|0|0|
|20261030|C|26|23|3|0|
|20261030|D-center|58|54|4|0|

阶段仅是冻结代理规则下的描述：抬升后失败不等同于已证明的精细放置控制根因。

C的跨seed成功率极差为91.7个百分点；D-center为18.3个百分点。同seed内方法差异绝对值平均为54.4个百分点。不同统计量只作描述，三个seed不足以精确区分训练随机性与方法效应。

## 初始化与硬件恢复

每seed从相同严格加载的Prismatic预训练权重构建原baseline，再用seed+1的forked RNG构建RLA overlay并复制共享baseline head参数。LoRA-A按原Gaussian规则；LoRA-B、门控及LayerNorm常量保持原规则。历史seed使用20261023/20261024相同构造规则；本轮同时改变随机模块初始化与训练随机性，不分离二者贡献。

有效更新2400；实际成功更新计算2606，其中恢复重放206；失败backward尝试1。新有效闭环240/240，AE更新0，封存集读取0。

GPU1发生新增不可纠正ECC与cuBLAS故障后停用。seed20261030的C从故障前完整step300恢复，301–506的206次未持久化更新重放。seed20261029的D在完整step100验证阶段安全暂停，短暂GPU1恢复未更新即停止，最终使用原GPU0 checkpoint恢复。全部余下训练和评估回到已授权GPU0。设备与恢复差异不隐瞒；权重、optimizer、scheduler、RNG及采样位置均完整恢复。

[
  {
    "seed": 20261029,
    "method": "D-center",
    "resume": {
      "source": "${PROJECT_ROOT}/experiments/rla_policy_seed_replication/run_20261026_02/seed_20261029/D-center/formal/step0100.pt",
      "restored_step": 100,
      "model_optimizer_scheduler_RNG_sampler_restored": true,
      "AMP_scaler": null,
      "bfloat16": true
    },
    "repeated_validation_rows": 61,
    "max_cross_device_fixed_validation_metric_delta": 0.0,
    "effective_updates": 600,
    "actual_update_computations": 600,
    "replayed_updates": 0
  },
  {
    "seed": 20261030,
    "method": "C",
    "resume": {
      "source": "${PROJECT_ROOT}/experiments/rla_policy_seed_replication/run_20261026_02/seed_20261030/C/formal/step0300.pt",
      "restored_step": 300,
      "model_optimizer_scheduler_RNG_sampler_restored": true,
      "AMP_scaler": null,
      "bfloat16": true
    },
    "repeated_validation_rows": 1406,
    "max_cross_device_fixed_validation_metric_delta": 0.0,
    "effective_updates": 600,
    "actual_update_computations": 806,
    "replayed_updates": 206
  }
]

本轮结束，不追加训练、seed或任务。所有结果仍为固定AE下的单任务development探索，不是独立测试或完整方法多AE-seed复现。
