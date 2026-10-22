from pathlib import Path
import csv,json,collections,subprocess,hashlib,importlib.metadata
import numpy as np
O=Path(__file__).parent
def rr(n):return list(csv.DictReader(open(O/n)))
def js(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
f=rr('failure_case_analysis.csv');summ=[]
for arm in 'ABCD':
 for app in ['clean','background']:
  rows=[r for r in f if r['model']==arm and r['appearance']==app]
  for cat,count in collections.Counter(r['descriptive_stage'] for r in rows).items():summ.append({'arm':arm,'appearance':app,'descriptive_stage':cat,'trials':count,'method':'state-derived description; 10cm approach/2cm lift thresholds fixed before trajectories inspected; no contact/grasp truth'})
with (O/'failure_summary.csv').open('x') as h:w=csv.DictWriter(h,fieldnames=list(summ[0]));w.writeheader();w.writerows(summ)
labels=rr('physical_label_availability.csv');ls={}
for split in ['train','validation']:
 ls[split]={}
 for lab in ['bowl','gap','drawer','stove']:
  rows=[r for r in labels if r['split']==split and r['label']==lab];amp=np.array([np.linalg.norm(np.asarray(json.loads(r['delta']))) for r in rows]);ls[split][lab]={'available':len(rows),'total_transitions':190 if split=='train' else 30,'coverage':1.,'magnitude_q0_q50_q95_q100':np.quantile(amp,[0,.5,.95,1]).tolist(),'source':'matched XML+exact saved state endpoints; body origin/finger qpos/fixture joint, not independently measured external truth'}
js('physical_label_summary.json',ls)
probe=rr('fixed_input_action_probe.csv');st=rr('stop_token_intervention.csv');ps=[]
for key in sorted({(r['arm'],r['step']) for r in probe}):
 rows=[r for r in probe if (r['arm'],r['step'])==key]
 ps.append({'arm':key[0],'step':int(key[1]),'samples':len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in ['training_path_action_L1','deployment_path_action_L1','train_deploy_normalized_action_L1','deployment_shift_from_A_L1']}})
js('fixed_probe_summary.json',{'rows':ps,'stop_augmented_max_abs':max(float(r['stop_augmented_vs_deploy_max_abs']) for r in st),'prefix_IDs_equal':all(r['prefix_token_IDs_equal']=='True' for r in st),'sample_unit':'14 fixed task/split/appearance representatives, not independent demonstration statistics'})
versions={}
for k in ['torch','transformers','peft','numpy','mujoco','tensorflow','robosuite','h5py','Pillow','imageio']:
 try:versions[k]=importlib.metadata.version(k)
 except importlib.metadata.PackageNotFoundError:versions[k]='missing metadata'
js('runtime_versions.json',versions)
opt=json.loads((O/'optimizer_and_sampling_audit.json').read_text());act=json.loads((O/'action_pipeline_audit.json').read_text())
failure_table='\n'.join(f'|{a}|'+ '|'.join(str(sum(int(r['trials']) for r in summ if r['arm']==a and r['descriptive_stage']==cat)) for cat in ['SUCCESS_ENV_SIGNAL','APPROACH_NOT_WITHIN_10CM','NEAR_TARGET_WITHOUT_2CM_LIFT','LIFT_OCCURRED_BUT_NO_SUCCESS'])+'|' for a in 'ABCD')
report=f'''# Step 5.9 基础策略退化诊断

**诊断流程完成；B严重退化的唯一确定原因仍未找到。** 已确认训练/部署序列末尾STOP不一致，并通过冻结对照解释了小幅前向差异；没有证据证明它解释了70–75个百分点的闭环下降。最有证据的候选是大范围微调在稀疏、小规模数据上的能力遗忘/过拟合，仍缺少单变量训练干预。本轮0 optimizer updates、0 backward、0新增闭环、0保留集新增评分；所有闭环证据来自已有Step5.8。

研究路线为ManiSkill-derived RLA自定义跨域策略辅助监督。不能将该实现称为作者Original LIBERO RLA原样复现。下文路径为相对实验路径或占位符，实际来源SHA保存在本地protection/source inventory。

## 1. 已确认事实与候选原因

| 假设类别 | 本轮证据 | 结论层级 |
|---|---|---|
| 动作/数据处理错误 | 440/440 batch语言、动作target、FK归一化复算通过；220/220 source h8与actions[s:s+8]匹配；第一6维Q99裁剪数按维为{act['clipped_action_values_by_dimension'][:6]} /1760个chunk步 | 未发现顺序、符号、统计错配或一帧偏移；这不是对所有物理/传感器语义的证明 |
| 训练/部署不一致 | 缓存训练序列缺末尾STOP，部署加入STOP2；14个固定输入上A/B100补STOP后action max_abs=0；训练phase另有原生0.02高斯扰动 | STOP差异确认；两种差异对真实退化的贡献未确定 |
| 更新过强/遗忘 | 可训练270,861,855参数；constant AdamW1e-4；B首步部署动作相对A平均L1漂移0.15526；训练loss下降、三条验证误差上升，A闭环远好于B | 有证据候选，不能据参数数量或一次动作漂移认定因果 |
| 数据覆盖不足 | 19训练episode、190窗口；每组400次曝光中173物理窗口/250外观实例被采样；只来自4任务，验证3条缺炉灶 | 覆盖受限确认；不能仅凭未采样17窗口解释整个失败 |
| 一致性损害物理信息 | B根本没有一致性/辅助梯度也退化；历史Step2.5总体运动probe B/C相近且CI跨0 | 未建立；不得以此预设加入物理loss可修复B |

## 2. 实际监督与数据路径

完整带代码位置图见`code_path_audit.md`。当前观测/语言/FK经VLM及原生Pro action trunk，输出动作并从最后隐藏特征预测一个32×64转移latent。**辅助头没有示范动作或预测动作向量输入**；它读取与动作输出共享的隐藏特征。示范动作只进入动作L1监督；learned action queries是模型参数，不是实际动作序列。

当前损失仅`L_action + weight*L_aux`，B权重0，C/D均0.05；C的teacher为Step2 B Encoder，D为Step2 C Encoder。teacher从真实DINO未来减当前残差生成，缓存npy，用各自既有训练统计减mean除scale；模型与teacher没有跨组混用。未来target不进入policy forward。**没有显式物理后果loss，也没有动作条件未来latent预测器。** 当前FK输入不等于物理变化监督。

## 3. 动作与时序审查

动作维度为xyz增量控制命令、三维旋转增量控制命令、夹爪；OSC_POSE控制器内部再缩放。第一6维沿用公开Spatial-Pro的Q01/Q99归一化并clip到[-1,1]，不能要求被裁剪动作完全可逆。夹爪原始−1开/+1闭，经训练standardizer变成0闭/1开；部署为`-sign(2*p_open-1)`，正好0.5时输出0。模型动作本身没有为改善loss额外clamp，控制器有其原生输入范围裁剪。

本轮重算440个batch动作最大误差{act['max_training_target_recompute_error']:.3g}，FK归一化最大误差{act['max_FK_proprio_normalization_error']:.3g}。`action_roundtrip_examples.csv`保存44个固定chunk首/尾动作的原始→标准化→训练target→反归一化→部署命令。当前状态为state[s]、未来state[s+8]、动作半开区间[s,s+8)，时间0.4s；原始raw obs[s−1]不作为policy proprio输入。FK由同一state计算，外部同期末端真值仍缺失。

所有chunk均有8个真实动作，没有episode边界padding。L1对batch×8×7取均值；语言序列right-padding使用attention_mask及IGNORE_INDEX，动作token位置通过labels识别。trainable action queries替换示范token embedding，不向policy提供专家动作内容。实际部署chunk8执行完再观测，max220步、settle10步、语言与相机/背景协议固定。

## 4. 冻结前向与参数漂移

固定14输入覆盖4训练任务×2外观和3验证任务×2外观；只作定位，不作为14个独立示范统计。原路径训练/部署差异平均L1：A0.00159、B1000.00147，补齐STOP后A和B100两路径精确相等。前缀token IDs均相同，因此该控制明确定位了这批样本的序列差异；未修历史batch或checkpoint。

相对A的部署动作平均L1漂移：B step1=0.15526、step50=0.07934、step100=0.07357；C100=0.07414，D100=0.08281。漂移大小不是性能指标，也不是越小越好；B首步是优先定位的线索。参数文件严格加载，step100对step0的B VLM可训练子集相对L2=4.24%、head=0.54%、proprio=2.54%；这里VLM基准含随机LoRA因子，不能解释为全VLM功能变化比例。head中辅助参数B不更新；本轮没有证据表明B被辅助梯度污染。

## 5. 优化与数据覆盖

LoRA/action queries51,954,432参数、head218,095,647参数（含231,424辅助参数）、proprio811,776参数。B辅助参数虽在optimizer中，但加权梯度0且weight decay0。AdamW lr1e-4，betas(.9,.999)，eps1e-8，weight_decay0，constant scheduler；microbatch2×accumulation2，有效batch4，三组各100次成功更新、无重放/跳过。正式Math SDPA、BF16 native head/base与FP32 LoRA，未使用gradient clipping或GradScaler。

相同采样表：clean/background各200次，总400，19条示范均出现。任务曝光cookie-box81、ramekin86、stove109、drawer124。重复外观不算新物理轨迹；单个被采样外观窗口1–5次，并非每条可用样本完整曝光一轮。

B训练动作loss前10步平均0.2081→末10步0.1133；C0.2069→0.1088；D0.2077→0.1093。固定验证示范等权动作L1：共同step0 clean/background0.15339/0.15417；B1000.17142/0.17198，C1000.16024/0.16222，D1000.17021/0.17038。训练和验证口径/数据不同，不能直接相减作泛化gap估计；但三组验证均较各自step0变差，与能力保留问题一致。

## 6. 辅助监督尺度混杂

同一190个训练clean窗口，C对应teacher原始norm均值162.83、D128.75；各自归一化后45.28/45.72，global variance1.012/1.038。中心化PR为20.37/15.04，表明D的变化方向较集中，不证明物理信息被丢弃。逐维尺度分布、两外观、train/validation分别见`auxiliary_target_statistics.csv`。

现有同一step0 batch加权辅助backbone gradient norm：C0.001540、D0.001686（D约+9.5%）；native action branch0.010118/0.010799，辅助head0.045319/0.047931。C/D相同权重不代表有效梯度相同。本轮直接复用已验收梯度记录，没有新增backward。既有全数据target统计与原归一化来自有限训练子集，未重新拟合，不能把D−C所有差异直接归因于背景一致性机制。

## 7. 既有轨迹失败阶段

逐条分析160份轨迹，以匹配XML和state执行mj_forward取末端/命名目标碗/plate位置、夹爪qpos、抽屉/炉灶关节；没有模拟推进或新渲染。下表合并两外观，仅描述观察到的阶段：10cm接近阈值、2cm抬升阈值在读取轨迹前固定在脚本里，不用于修改成功标签或筛选测试状态。靠近、抬升不等于真实抓取，目标碗中心到grip site距离不是表面接触距离。

|模型|成功信号|未进入10cm|接近但未抬升2cm|抬升但未成功|
|---|---:|---:|---:|---:|
{failure_table}

所有失败原标签均为220步超时；无invalid action、sim异常。不能将超时本身当具体机制。没有每步contact event/force及成功谓词子条件，因此碰撞、滑落、抓取失败和成功判定边缘机制仍部分UNRESOLVED。

视频选择规则先固定为每类别最小state ID、先clean再background，只看4组配对。state0 clean：A成功，B/C已有12–14cm抬升但到盘上放置未完成，D成功；B/C不是一开始完全不运动。state7 clean：C成功，D抬升15.8cm后未成功、末端碗离盘xy约7cm，并发生23次夹爪命令切换，对照C为1次；这是夹爪时序/接触后偏离的候选证据，不是已证明接触因果。抽屉state15：B/C/D均靠近目标但碗最大抬升不足0.5cm，C/D夹爪命令切换42/40次，抽屉关节也发生不同变化；不能只归为语言错误或定位失败。contact sheet检查支持“交互/放置阶段存在差异”的视觉观察，没有把视频推测当作物理事件真值。

## 8. 物理后果标签与约束建议

19训练/190与3验证/30窗口均可从已匹配XML与state端点可靠计算：目标`akita_black_bowl_1_main` body原点位移、两指qpos间距变化、`wooden_cabinet_1_top_level`与`flat_stove_1_button`关节变化，四类覆盖均220/220。具体目标碗由这些四任务BDDL goal命名，不能推广为所有LIBERO任务固定同一碗。单位分别m/m/m/rad，关节有标签不表示该任务需要改变它；例如stove任务是取碗，不是开炉灶。标签覆盖不等于非零运动样本充分，见`physical_label_summary.json`的幅度分布。

这些是记录仿真状态的模型真值，没有独立外部位姿精度真值；图像来自同一state，时序可靠。物体位移没有用末端位移替代。当前policy训练未消费这些标签。

**值得保留物理后果监督作为后续受控候选，但当前不应优先添加。** 可考虑保留重建、同转移多外观一致性，再从最终latent监督真实物体/夹爪/任务关节变化，最后单独检验动作条件前向可预测性。它针对“表示是否保留交互运动”这个待验假设，不能据本轮承诺修复BC-only退化。历史Step2.5总体probe B/C相近、夹爪/物体有取舍，未建立“一致性已损害真实运动信息”。

## 9. 唯一优先下一步

建议**B首步更新的组件置换冻结诊断**，详见`next_experiment_proposal.md`。使用现存step0/step1权重，在同样固定离线输入上比较VLM可训练子集、native head、proprio的8种组合；先让两前向路径采用同一STOP布局并通过max0回归。没有新训练或rollout，可直接定位首步动作漂移主要由哪个模块或交互产生。在定位前不同时搜索LR、加物理loss或改背景协议。本轮未执行这个新组件置换实验。

## 10. 保护、异常与同步

原始数据、代码、checkpoint及Step5.7/5.8结果只读。CPU参数诊断初次因旧PyTorch的mmap只接受str路径而退出，保留run.log；改诊断脚本路径类型后从未完成段恢复，不重复生成既有结果，不涉及优化更新。两次冻结GPU探针使用GPU1、26GiB门槛及UUID/ECC核验，参数前后hash一致、ECC无增量，峰值分配约3.68GiB。报告中的数据和假设均未用于修改历史评估或checkpoint选择。

GitHub同步另见`github_sync_report.md`。只上传可分享源码、配置模板和聚合诊断，不上传模型、数据、视频、原始轨迹或凭证。同步身份必须是已核验的Lily021117，不能使用之前连接的其他账号。本地完整验收见`delivery_verification.json`。
'''
with (O/'report.md').open('x') as h:h.write(report)
proposal='''# 最优先下一步：B首步更新的组件置换冻结诊断

唯一主实验：以现存共同step0和B-step1参数，做VLM可训练子集、native action head、proprio projector的2×2×2组件置换，共8组合。辅助head不影响action，保持step0即可。各组合只用于离线诊断，不作为新正式checkpoint或策略成绩。

动机：B首次更新后的固定输入动作漂移约0.155，而step100仍约0.074；同时训练下降/验证上升。现有证据尚不知道漂移来自head、VLM、proprio还是交互。补STOP控制已经解释小幅路径差，未解释控制能力下降。

固定现有14个真实输入及图像/FK/语言，不读历史保留数据生成新分数，不加新窗口。比较共同的训练和部署两前向路径、标准化动作L1、相对A漂移、第一chunk与夹爪变化。分析按训练/验证、任务/外观分别报告，不选最有利样本。

放行条件：初始化8组合中的全step0准确返回A；统一STOP布局后训练/部署action max_abs=0；相同current输入与归一化；严格state_dict键名；optimizer updates=0、backward=0、rollout=0、无新checkpoint；UUID/ECC和历史hash通过。

若某单模块重现大部分首步漂移，可据此只提出一项模块范围或优化数值修复实验，并在新的训练授权后执行。若组合交互主导或结果不清晰，报告证据不足，不能因此直接加物理损失。本轮仅提出，不自动启动。

# 后续物理后果保真候选（当前不实施）

保留原始RLA重建，匹配同物理转移的多外观latent；以训练集统计归一化的真实目标物体位移、两指间距变化及确属该任务语义的关节变化监督最终latent。不能只监督独立且不影响latent的head。用相同数据、初始化、预算比较是否加该项，并报告真实运动与静止窗口取舍。后续另做动作条件前向可预测性，不混为当前策略辅助监督的已有能力。

可用标签：当前190训练、30验证窗口各有四类状态派生label，时间为state[s]→state[s+8]；没有独立外部状态精度真值。标签存在但某任务关节几乎不动，不能用大量静止误差支持运动保真。先看训练运动分布、按episode评估，不用held-out定尺度/阈值。

该方案检验一致性是否牺牲运动信息；当前没有充分证据证明这种牺牲已发生。也不能修复错误的动作归一化、时间索引或低数据微调遗忘，不能视为BC-only退化的既定解决方案。
'''
with (O/'next_experiment_proposal.md').open('x') as h:h.write(proposal)
print('REPORTS_COMPLETE',flush=True)
