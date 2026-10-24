# 当前实验与固定官方recipe差异

依据固定commit下`rebuttal/exp2_vla_adapter/README.md`、`configs/rla/16x64_libero.yaml`、`vla_adapter_overlay/tools/train_eval{,_rla}.sh`和`rla/sidecar.py:_scaling`；不以当前main替代固定版本。

| 项目 | 固定官方文件 | 当前实验 | 分类与影响 |
|---|---|---|---|
| AE数据 | 四个LIBERO suites，训练从各轨迹索引10开始，验证0–9 | 单任务20训练/5development、同状态双外观 | 有意小规模适配，不是多任务AE复现 |
| AE起点 | YAML load_dir=null；论文集成sidecar step40000 | 已核验作者step90000，再微调600 | 有意复用另一发布步，不称论文step40000 |
| AE结构 | 双相机、Plücker、16×64，DINO1024、224px | 保留 | 结构一致不意味着数据/训练等价 |
| AE目标 | λ_L1=λ_MSE=1 | C两外观重建平均；D额外latent一致性 | 明确方法改动 |
| AE跨度 | train h2…64 | h1…8，官方累计sidecar仍8项 | 有意匹配策略窗口、缩窄运动覆盖 |
| AE预算/批量 | YAML max_steps=1,000,000，batch128 | 600更新，effective8物理/16外观 | 计算受限，不宣称收敛 |
| AE优化 | AdamW lr1e−4, wd0, mixedFP16 master；动态loss scale | 同lr/wd，固定loss scale256 | 稳定性适配，共同应用C/D |
| 策略数据 | suite级no_noops RLDS、多任务；官方launcher image_aug=True | 单任务完整派生窗口，Clean/Background各一次；直接固定图像预处理，无该随机增强 | 有意派生数据路径，统计来源和曝光不同 |
| 策略起点 | 官方Prismatic VLM＋head；RLA launcher也支持显式warm-start | 共同原始Prismatic＋head，不用Spatial-Pro成品策略 | 符合冷启动辅助路线；非所有官方可选阶段组合 |
| 策略预算 | Spatial 20,000更新；每task评估50次 | 600更新；20development状态×3外观 | 小预算探索，不是官方得分复现 |
| 策略优化 | LR2e−4，LoRA64；baseline BATCH8/ACCUM2，多GPU数影响有效batch；RLA默认ACCUM1 | LR2e−4、LoRA64、micro8/accum2，单GPU；固定Math SDPA与activation checkpointing | C/D共同采用；与官方总batch/计算不同 |
| 辅助归一化 | 各sidecar自己的逐槽mean＋该sidecar全局std | C均值/缩放用于历史D；D-center只改D均值，保留C缩放 | 实验性跨组共享统计；非官方每库独立center语义，已明确测试，不静默修复 |
| 辅助系数 | auto:0.25，依target统计解析 | 共同固定0.22408536585365854 | 消除独立自动系数改变的混杂，非相同梯度强度保证 |
| 输入/时序 | 官方双图、语言、8维proprio、8×7 action | 保留接口，FK同步proprio、派生身份、显式STOP/mask修复 | 已验收必要适配，不冒用原RLDS内容身份 |
| 评估 | 标准LIBERO初始状态/成功谓词 | 自定义材质、原反射/MSAA；development种子冻结 | 不是独立最终测试 |

当前检查未据此认定新的实现错误。可能的错误必须由实际执行证据证明；预算少、归一化选择和数据覆盖变化本身不能直接被称为bug。已有错误修复与偏离保留来源记录，不在本轮更改采样、loss或阈值。
