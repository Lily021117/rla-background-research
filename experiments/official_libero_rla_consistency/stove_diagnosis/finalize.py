"""Summaries, protected-input recheck and sanitized research delivery."""
import argparse,collections,subprocess,re,os,io
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_csvsave=csvsave
def csvsave(path,rows):
 if path.exists():
  buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows);assert path.read_text()==buf.getvalue().replace('\r\n','\n');return
 _csvsave(path,rows)
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--delivery',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);pol=a.out/'policy_verified';policy=read(pol/'verification.json');assert policy['optimizer_updates']==0 and all(x['before']==x['after'] for x in policy['parameter_checks']);assert read(a.out/'checkpoint_state_audit.json')['common_C_D_initial_exact'];assert read(a.out/'representation_verification.json')['status']=='PASS_FROZEN_DIAGNOSTIC'
data=csvread(pol/'action_by_dimension_horizon.csv');grad=csvread(pol/'gradient_conflict.csv');groups=collections.defaultdict(list)
for r in data:groups[r['model'],int(r['step']),r['split'],r['appearance'],int(r['chunk_position'])].append(r)
summary=[]
for key,rs in groups.items():
 row=dict(model=key[0],step=key[1],split=key[2],appearance=key[3],chunk_position=key[4],n=len(rs),action_L1=float(np.mean([float(r['action_L1']) for r in rs])))
 for d in range(7):row[f'dim{d}_L1']=float(np.mean([float(r[f'dim{d}_L1']) for r in rs]))
 row['gripper_binary_error']=float(np.mean([(float(r['dim6_prediction'])<.5)!=(float(r['dim6_target'])<.5) for r in rs]));closed=[r for r in rs if float(r['dim6_target'])<.5];row['true_close_count']=len(closed);row['false_open_given_close']=float(np.mean([float(r['dim6_prediction'])>=.5 for r in closed])) if closed else None
 summary.append(row)
csvsave(a.out/'action_by_split_appearance_horizon.csv',summary)
closure=[]
for model in ('C','D'):
 for step in (0,1,100,300,600):
  for split in ('train','validation'):
   for app in ('clean','background'):
    for h in (0,7):
     rs=[r for r in data if r['model']==model and int(r['step'])==step and r['split']==split and r['appearance']==app and int(r['chunk_position'])==h and int(r['t'])==int(r['closure_t'])]
     if rs:closure.append(dict(model=model,step=step,split=split,appearance=app,chunk_position=h,n=len(rs),predicted_close=sum(float(r['dim6_prediction'])<.5 for r in rs),target_close=sum(float(r['dim6_target'])<.5 for r in rs),prediction_mean=float(np.mean([float(r['dim6_prediction']) for r in rs])),L1=float(np.mean([float(r['dim6_L1']) for r in rs]))))
csvsave(a.out/'closure_anchor_predictions.csv',closure)
agg=[]
for model in ('C','D'):
 for step in (0,1,100,300,600):
  for split in ('train','validation'):
   for app in ('clean','background'):
    rs=[r for r in summary if r['model']==model and r['step']==step and r['split']==split and r['appearance']==app];agg.append(dict(model=model,step=step,split=split,appearance=app,action_L1=float(np.mean([r['action_L1'] for r in rs])),translation_L1=float(np.mean([[r[f'dim{d}_L1'] for d in range(3)] for r in rs])),rotation_L1=float(np.mean([[r[f'dim{d}_L1'] for d in range(3,6)] for r in rs])),gripper_L1=float(np.mean([r['dim6_L1'] for r in rs]))))
csvsave(a.out/'offline_action_summary.csv',agg)
fig,axs=plt.subplots(2,2,figsize=(11,7))
for i,split in enumerate(('train','validation')):
 for model in ('C','D'):
  rs=[r for r in agg if r['model']==model and r['split']==split and r['appearance']=='clean'];axs[i,0].plot([r['step'] for r in rs],[r['action_L1'] for r in rs],marker='o',label=model)
  rs=[r for r in summary if r['model']==model and r['split']==split and r['step']==600 and r['appearance']=='clean'];axs[i,1].plot([r['chunk_position'] for r in rs],[r['dim6_L1'] for r in rs],marker='o',label=model)
 axs[i,0].set_title(split+' clean action L1, fixed diagnostic samples');axs[i,1].set_title(split+' clean gripper L1 by chunk position');axs[i,0].set_xlabel('saved successful update');axs[i,1].set_xlabel('chunk index');axs[i,0].legend();axs[i,1].legend()
fig.tight_layout()
if not (a.out/'action_diagnostic_curves.png').exists():fig.savefig(a.out/'action_diagnostic_curves.png')
plt.close(fig)
original=csvread(a.out/'policy/gradient_conflict.csv');assert sha(a.out/'policy/action_by_dimension_horizon.csv')==sha(pol/'action_by_dimension_horizon.csv')
for x,y in zip(original,grad):
 for key in ('model','step','split','shared_action_gradient_norm','shared_weighted_aux_gradient_norm'):assert x[key]==y[key]
 assert abs(float(x['cosine'])-float(y['cosine']))<1e-14 # float64 dot sum over Python sets may differ in accumulation order
protected=read(a.out/'protected_input_hashes.json');changed=[]
for x in protected:
 if sha(x['path'])!=x['sha256']:changed.append(x['path'])
assert not changed
GPU='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419';after=subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True);before=(pol/'ecc_before.txt').read_text();filterecc=lambda s:[x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))];assert filterecc(before)==filterecc(after)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv'],text=True);assert GPU not in processes
(a.out/'ecc_after.txt').write_text(after);save(a.out/'resource_release.json',dict(GPU_UUID=GPU,ECC_increment=False,diagnostic_GPU_processes_remaining=0,other_processes_untouched=True,peak_allocated_GiB=policy['peak_allocated_bytes']/1024**3,peak_reserved_GiB=policy['peak_reserved_bytes']/1024**3))
save(a.out/'delivery_verification.json',dict(status='PASS_DIAGNOSTIC_ROOT_CAUSE_UNRESOLVED',optimizer_updates=0,AE_updates=0,policy_updates=0,diagnostic_backward_calls=320,closed_form_ridge_probes=True,new_closed_loop_trials=0,sealed_contents_read=False,protected_files_rehashed=len(protected),changed_historical_files=changed,policy_parameter_hashes_unchanged=True,repeat_actions_and_gradient_norms_exact=True,repeat_cosine_max_absolute_roundoff=max(abs(float(x['cosine'])-float(y['cosine'])) for x,y in zip(original,grad)),GPU_released=True,diagnostic_attempts=['initial preparation requested a formal batch audit file saved only by preflight; corrected path in new run02, run01 retained','AE-env lacks MuJoCo; reused existing VLA environment, no install','AE source FP32 vs configured mixedFP16 strict-load rounding: verified exact cast, not mismatched initialization','rollout cached xpos not equal poststep state reconstruction, max0.98mm; retained and disclosed, no offset applied','second identical policy pass logs raw auxiliary directly before BF16 coefficient multiplication; first pass retained;320 total diagnostic backward calls','final summary first exact cosine check failed at6.7e-16 from float64 unordered-set dot reduction; no tensor/weight change; declared numerical rounding and preserved summaries'],GitHub_sync='pending_remote_verification'))
dist=csvread(a.out/'latent_distribution.csv');probe=csvread(a.out/'motion_probe_metrics.csv');timing=csvread(a.out/'rollout_details/rollout_timing_summary.csv');ck=csvread(a.out/'checkpoint_inventory.csv');ident=read(a.out/'identity_summary.json');prefix='experiments/official_libero_rla_consistency/'
lines=['# 炉灶 D 全失败：冻结权重诊断','', '结论：已定位到接近/抓取阶段的行为差异，尚未确认唯一训练因果根因。未发现错任务训练、cookie 成品模型复用、错误 sidecar 或错误评估 checkpoint。D 在48/60次已有闭环中从未发出关闭命令，60/60未达到3 cm抬升；C在60/60发出关闭命令，59/60成功。不能把这个现象单独归咎于夹爪head。','', '## 1. 训练身份与保护','', 'B/C/D各自使用同一炉灶20条训练示范（2,555个完整h=8窗口），同一4800次物理窗口采样、9600次外观曝光；不是20个窗口。5条离线development示范有703个完整窗口。闭环另有20个冻结development初始状态，每个三种外观，各模型60次；三种外观不是三个独立状态。','', 'AE-C/AE-D使用完全相同的炉灶25条来源清单，仅20条train用于更新，5条用于固定诊断。策略sampling_order所有身份均在20条train中，实际预检模型tokens包含炉灶指令，正式代码每个batch也断言任务tokens。旧bootstrap cookie batch记录不参与正式forward。','', '策略B从原始未训练B step0开始，C/D从共同原始未训练RLA step0开始；参数与Python/NumPy/CPU/CUDA RNG逐项匹配。AE step0与官方LIBERO step90000经官方配置的mixed-FP16加载后逐参数一致；源文件FP32与实际FP16躯干有精度转换，不是换模型。AE不是论文step40000。','', '证据：checkpoint_state_audit.json、checkpoint_inventory.csv、protected_input_hashes.json、training_episode_identity.csv、target_join_verification.json。100个sidecar文件的SHA-256与冻结provenance一致，所有25×2模型×2外观的proprio身份键逐项通过；AE Encoder哈希与sidecar记录匹配，180次原评估记录的checkpoint哈希与实际最终文件匹配。','', '|阶段/组|实际冻结文件|SHA-256|','|---|---|---|']
for r in ck:lines.append(f"|{r['stage']}/{r['arm']}/{r['module']}|{r['path']}|{r['sha256']}|")
lines+=['','## 2. 唯一预期差异及恢复','', 'AE-C：两路各自未来特征L1+MSE重建平均；AE-D另加latent MSE，lambda_cons=0.009127798272072366。两AE共同初始化、数据/horizon顺序、AdamW1e-4、600有效更新及解冻范围一致。C/D策略同结构/初始化/样本/动作/语言/两路图像/FK/STOP mask，AdamW2e-4、WD.01、Math SDPA BF16、appearance micro8×accum2、600有效更新、activation checkpointing、末步600 checkpoint选择相同；target分别来自AE-C/AE-D。辅助系数共同固定0.22408536585365854。','', '正式调用：`ae_consistency/policy_training.py:execute -> DerivedEpisodeDataset.sample -> RlaStore.frame -> official rla.loss.make_run_forward_pass -> finetune.run_forward_pass`。动作归一化与proprio来自同一炉灶训练统计；target统一采用 `(z−mu_C)/sigma_C_global`，sigma_C=2.68701489155。STOP修复/完整8步mask来自coverage_gripper/dataset.py，不用tail重复帧充当真实target。未来图像只用于AE/sidecar，实际策略输入仍是当前双视角、语言和proprio。','', 'B/C/D评估调用同一policy_evaluation.py，使用各自最终600步、同一状态/seed/XML/动作后处理/220控制步上限。所有180条已有执行动作与模型chunk按官方夹爪公式−sign(2p−1)后的结果逐元素一致。','', '炉灶AE两组旧故障均从完整step0恢复，各重放44个已记录更新；最终各600有效、644已记录计算，旧故障可能各有1个未确认在途update。FP32 master、optimizer、RNG、采样位置恢复；AE无scheduler/动态AMP scaler，固定256 loss scale。44步标量前缀精确复现，不声称未持久化step44参数逐位已验证。策略B/C/D没有本轮中断恢复，final optimizer计数均600、scheduler600、采样位置4800。没有D独有的恢复配置变化证据。','', '## 3. 全部已有配对闭环的失败阶段','', '主阶段分类沿用评分前冻结的目标抬升>=3cm、EE到碗中心<=10cm代理。D：55/60到过附近但未抬升，5/60接近失败；0/60进入可靠抬升后的搬运阶段。C：59成功，1附近未抬升失败。失败都不是单纯完成搬运后漏掉末端判定。','', '|模型/外观|成功|曾发关闭命令|先于10cm附近关闭|重算手指接触|抬升>=3cm|','|---|---:|---:|---:|---:|---:|']
for r in timing:lines.append(f"|{r['model']}/{r['appearance']}|{r['successes']}/20|{r['ever_close']}/20|{r['close_before_10cm']}/20|{r['ever_recomputed_finger_contact']}/20|{r['ever_lift_3cm']}/20|")
lines+=['', 'C/D在同一初态的第一动作chunk已不同，全部60对在记录的第3个零基控制步后末端分离超过2cm。D实际发出关闭的12次，其首关时水平偏差平均约8–9cm；C约3–4cm。D通常未形成可成功抓取的位置与关闭组合。接触计数来自已记录state的mj_forward重算，只能说明几何接触候选，不能证明稳定夹持或接触力。','', '固定选取每外观第一个seed2026102500、沿C执行时长五个均匀视频帧，已人工查看：C完成抓取/抬升向盘移动，D保持在炉灶侧附近，未形成相同搬运轨迹。这是视觉观察，与全部轨迹量化分开；完整面板仅保存在本地video_panels_local_only。','', '发现记录精度限制：原日志在env.step后保存cached xpos，与记录qpos/qvel经mj_forward的末端/碗位置不逐位相同，抽查最大0.980mm。没有平移时间索引或修改原日志；阶段原统计保留，新增接触/关闭位置明确是同state重算值。这一共同记录相位限制不证明D特异故障，也不能用小差值断言严格同步已通过。训练派生FK与同state重算在25条示范上通过1e-7检查，不能混淆这两项。','', '## 4. 相同固定batch：动作与梯度','', '预先冻结每条示范首帧/首次关闭anchor/最后有效anchor，20train+5development，共150个外观观测。读取step0/1/100/300/600，全程eval+no-grad动作诊断；梯度在eval模式开启autograd，每split固定4条示范的关闭anchor×2外观，共8样本，micro2累积。梯度分别来自官方action L1和加权aux L1，作用于全部可训练VLM/backbone参数；不包括head内部共享块。它是确定性局部导数，不是重演训练噪声、优化器动量或完整训练分布。','', 'action_by_split_appearance_horizon.csv / action_by_dimension_horizon.csv保留8个chunk位置和7个维度。全部chunk真实有效、mean reduction，无padding进入loss。归一化动作误差与Q99反变换后的控制命令误差分别记录，命令坐标不能解释为米或弧度，夹爪阈值另算。','', '|组/step|split|Clean action L1|Background action L1|Clean gripper L1|Background gripper L1|','|---|---|---:|---:|---:|---:|']
for model in ('C','D'):
 for step in (0,1,100,300,600):
  for split in ('train','validation'):
   c=next(r for r in agg if (r['model'],r['step'],r['split'],r['appearance'])==(model,step,split,'clean'));b=next(r for r in agg if (r['model'],r['step'],r['split'],r['appearance'])==(model,step,split,'background'));lines.append(f"|{model}/{step}|{split}|{c['action_L1']:.5f}|{b['action_L1']:.5f}|{c['gripper_L1']:.5f}|{b['gripper_L1']:.5f}|")
lines+=['', 'C/D step0输出精确一致；首步两者都暂时变差，随后恢复。D的夹爪误差在100、300、600步均高于C，但最终总体development L1低于C。没有中间checkpoint闭环，不能确认哪一步开始闭环全失败，也不能把偏离step0直接定义为退化。','', '|组/step/split|action loss|raw aux|weighted aux|backbone action grad norm|weighted aux grad norm|cosine|','|---|---:|---:|---:|---:|---:|---:|']
for r in grad:
 if int(r['step']) in (0,600):lines.append(f"|{r['model']}/{r['step']}/{r['split']}|{float(r['action_loss']):.5f}|{float(r['auxiliary_raw_loss']):.5f}|{float(r['weighted_auxiliary_loss']):.5f}|{float(r['shared_action_gradient_norm']):.5f}|{float(r['shared_weighted_aux_gradient_norm']):.5f}|{float(r['cosine']):+.4f}|")
lines+=['', '这批样本不支持D存在持续更强的反向梯度冲突：step0 C为负cosine而D为正；step1两者为负；后期D为正。D final raw aux更低，却在development的weighted auxiliary梯度norm约0.03497，C约0.00853。共同系数不等于相同有效梯度；只由aux loss大小判断学习强弱不成立。','', '## 5. AE表示、重建与尺度','', '|组/clean|latent norm|逐维方差均值|entropy有效秩|unit跨外观距|unit跨episode同h距|within/between|','|---|---:|---:|---:|---:|---:|---:|']
for arm in ('C','D'):
 r=next(x for x in dist if x['model']==arm and x['appearance']=='clean' and x['normalization']=='raw');u=next(x for x in dist if x['model']==arm and x['appearance']=='clean' and x['normalization']=='unit_L2');lines.append(f"|{arm}|{float(r['norm_mean']):.3f}|{float(r['variance_mean']):.5f}|{float(r['effective_rank']):.3f}|{float(u['within_mean']):.5f}|{float(u['between_same_horizon_different_episode_mean']):.5f}|{float(u['within_between_ratio']):.5f}|")
lines+=['', '同一400个development转移比较；同时给raw、每样本unit L2、官方共同C中心归一化三种结果。D尺度/方差/有效秩降低，归一化within/between也下降，说明并非只有等比例缩小，但不能据此称语义改善或坍塌。','', '未来DINO L1：C clean0.04751/background0.05223，D0.04859/0.05326；copy-current为0.0810/0.0822。D有小幅重建代价，且两者明显优于复制当前。AE_reconstruction_by_camera_horizon_phase.csv进一步按双相机、h1..8、approach/closure/transport代理阶段拆分，使用原来的完全相同样本；不是动作条件未来模型。','', 'target_scale_and_offset.csv使用全部训练/验证有效sidecar rows：共同C居中后，D固定均值分量占平方能量83.6–84.3%，C约0.6–1.2%；D动态逐维方差约为C的12%。这不是aux L1或梯度的84%，不能作这样的换算。共同归一化符合冻结方案，尚未定位代码错误；但它使C/D监督的固定偏移、尺度与不变性一起变化，是明确的表示统计混杂。','', '## 6. 同容量冻结latent物理变化读出','', '从已验收的同state XML/qpos/qvel恢复碗world xyz和两指qpos；25条全部状态的FK proprio逐维核验通过。不是从图像猜标签。无独立真实世界位姿真值；本任务是炉灶上取碗，不是操作炉灶开关，因此未用无关旋钮变化替代任务后果。','', '每episode10个固定h8窗口：20train共200 clean样本拟合闭式ridge，5development共50转移分别评估clean/background。相同1024维输入、固定alpha=1（平均平方loss约定）、同样训练均值/逐维std标准化、同一输出容量；不调参，不对AE/策略新增监督或反向。标准化有意消除简单尺度差异，衡量线性可读信息，而不是策略实际aux学习难度。','', '|组/外观|碗位移MAE(mm)|碗位移R²|两指qpos变化MAE(mm)|','|---|---:|---:|---:|']
for arm in ('C','D'):
 for c in ('clean','background'):
  r=next(x for x in probe if x['model']==arm and x['appearance']==c and x['label']=='bowl_xyz' and x['episode']=='ALL');g=next(x for x in probe if x['model']==arm and x['appearance']==c and x['label']=='finger_qpos' and x['episode']=='ALL');lines.append(f"|{arm}/{c}|{1000*float(r['MAE_m']):.3f}|{float(r['R2_pooled']):.4f}|{1000*float(g['MAE_m']):.3f}|")
lines+=['', 'D−C碗位移MAE：clean+0.135mm，episode bootstrap95%CI[−0.088,+0.372]mm；background−0.395mm，[−0.715,−0.075]mm。只5条development，小样本相关窗口不能当独立轨迹；结果不支持“D碗运动信息普遍不可读”，不排除更复杂语义、长时后果或闭环分布下的信息损失。未以EEF位移冒充物体后果。','', '## 7. 事实、排除与假设','', '- 已确认：炉灶各自正确数据/初始化/sidecar/最终checkpoint；配对部署规则一致；D在接近/抓取阶段偏离、48次从不关闭、无抬升；夹爪离线误差较C高；D表示尺度与有效秩降低；共同C中心引入显著D均值分量。','- 在现有证据范围内排除：错用cookie已训练模型、跨任务target串用、错评checkpoint、后处理吞掉关闭命令、D独有恢复预算变化、仅因总体离线action L1更差、仅因线性物体运动标签不可读。','- 仍待验证：目标中心/尺度变化如何影响共享head与策略优化；局部夹爪/接近时序误差怎样通过闭环状态偏移放大；单seed优化路径偶然性；head共享块/优化器动量层面的干扰。当前backbone局部cosine不能排除所有梯度干扰。','', '## 8. 唯一下一步：只改变D target中心偏移','', '建议一个冻结AE的两臂对照：D-control维持(z_D−mu_C)/sigma_C；D-recenter仅改为(z_D−mu_D_train)/sigma_C。mu_D只用已有20条训练的两外观sidecar计算；保留相同sigma_C、lambda_aux、AE-D权重、结构、初始化、所有数据、seed、LR/optimizer和600步预算。不能同时改变scale、辅助系数、夹爪权重或阈值。两臂均从同一原始step0开始，不续训退化策略；固定step0/100/300/600，final600为唯一比较。','', '放行前离线验证：target差异必须是逐坐标常数平移；所有转移间差分完全不变；共享初始参数/RNG/action输出一致；数据及batch逐项一致。后续若授权训练，统一评估冻结development，不挑checkpoint或状态；关注关闭行为、抓取位姿、实际SR。若recenter无改善，不追加调参。本轮没有执行该干预。','', '理由：这是隔离已观测到的固定均值混杂的最小改变；无须预设AE丢失物理信息，也无须增加物理loss。闭式读出目前未支持加入物理后果loss作为修复结论。','', '## 9. 范围、复现和技术记录','', 'optimizer updates=0；AE/policy权重变化=0；新增闭环=0；未读取封存样本内容。诊断backward总320次（两次相同固定流程；第二次仅修正raw辅助loss的日志取值为乘系数前值，两次动作及梯度完全一致）。不把闭式ridge拟合称为策略训练。','', 'GPU1 UUID与ECC在启动/结束核验，历史correctable2、uncorrectable parity1/SM1保持原值，无新增；GPU已释放，其他进程未干预。protected_input_hashes列出的历史输入全部复算一致。详细源代码和运行方式见README.md。','', '诊断过程保留了失败尝试：首次prepare误查仅preflight保存的batch证据，改为真实preflight路径后在新run02重做；AE Python环境没有MuJoCo，切回既有VLA环境；初次AE逐位比较未考虑官方FP16转换，随后验证确切cast等价；cached rollout姿态不逐位匹配被保留为记录限制，未放宽阈值宣称通过。','']
lines=[s.replace('两次动作及梯度完全一致','两次动作和梯度范数完全一致，float64 cosine因参数名集合求和顺序相差最多6.7e-16') for s in lines]
at=lines.index('## 5. AE表示、重建与尺度')
lines[at:at]=['示范首次关闭anchor的当前步：D在固定5条development上Clean正确关闭5/5、Background4/5，后续第7步两外观均5/5；训练示范当前步为14/20、15/20。这与闭环48/60不关闭并存，支持检查接近轨迹/闭环观测偏移及动作联合时序，不能说D完全不会输出关闭。具体结果见closure_anchor_predictions.csv。','']
(a.out/'report.md').write_text('\n'.join(lines))
nexttext='\n'.join(lines[lines.index('## 8. 唯一下一步：只改变D target中心偏移'):lines.index('## 9. 范围、复现和技术记录')]);(a.out/'next_experiment_proposal.md').write_text(nexttext)
a.delivery.mkdir(parents=True,exist_ok=False)
def sanitize(s):return s.replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}').replace('/data/lyc','${DATA_ROOT}')
names=['report.md','next_experiment_proposal.md','configuration.json','fixed_sample_manifest.json','checkpoint_inventory.csv','checkpoint_state_audit.json','identity_summary.json','protected_input_hashes.json','training_episode_identity.csv','failure_stage_counts.csv','paired_rollout_audit.csv','offline_action_summary.csv','action_by_split_appearance_horizon.csv','closure_anchor_predictions.csv','AE_reconstruction_by_camera_horizon_phase.csv','latent_distribution.csv','physical_label_availability.csv','motion_probe_metrics.csv','motion_probe_paired_uncertainty.json','target_scale_and_offset.csv','target_join_verification.json','representation_verification.json','delivery_verification.json','resource_release.json']
for name in names:(a.delivery/name).write_text(sanitize((a.out/name).read_text()))
for name in ('gradient_conflict.csv','verification.json'):(a.delivery/('policy_'+name)).write_text(sanitize((pol/name).read_text()))
for name in ('rollout_timing_summary.csv','rollout_grasp_timing.csv','rollout_diagnostic_verification.json'):(a.delivery/name).write_text(sanitize((a.out/'rollout_details'/name).read_text()))
print('REPORT AND SANITIZED DELIVERY READY',a.out,flush=True)
