"""Fixed600 paired state-cluster statistics and same-sample offline continuation curves."""
import argparse,json,csv,collections,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def write(n,rs):
 with (o/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
plan=read(o/'run_plan.json');proto=read(o/'evaluation_protocol.json');models=proto['models'];arms=list(plan['arms']);conditions=proto['conditions'];seeds=proto['states'];short=Path(plan['short_source']);long=Path(plan['long_run_source']);attempts=read(o/'attempt_registry_v4.json');registry=read(o/'policy_registry_v4.json');trial=read(o/'per_trial_results.json')
assert len(trial)==120 and all(r['status']=='COMPLETE' for r in trial)
phase_summary=[]
for r in trial:
 timeline=read(Path(r['trajectory']).with_name('phase_timeline.json'));assert len(timeline)==r['steps']
 phase_summary.append(dict(model=r['model'],seed=r['seed'],appearance=r['condition'],success=r['success'],close_command=r['issued_close_command'],finger_contact_observed=any(x['finger_contact'] for x in timeline),first_finger_contact_step=next((x['step'] for x in timeline if x['finger_contact']),None),first_lift_step=r['first_lift_step'],descent_horizontal_path_m=r['descent_horizontal_path_m'],descent_xy_error_change_m=r['descent_xy_error_change_m'],final_centered=timeline[-1]['xy_condition'],final_contact=timeline[-1]['contact_condition'],final_height=timeline[-1]['height_condition'],final_goal=timeline[-1]['goal'],failure_stage=r['stage']))
write('per_episode_phase_summary.csv',phase_summary)
idx={(r['model'],r['seed'],r['condition']):r for r in trial};assert len(idx)==120
for s in seeds:
 for c in conditions:
  for k in ('state_sha256','xml_sha256','language'):assert len({idx[m,s,c][k] for m in models})==1
scores=[];stages=[]
for m in models:
 for c in conditions:
  rs=[idx[m,s,c] for s in seeds];n=sum(r['success'] for r in rs);ph=n/20;z=1.959963984540054;den=1+z*z/20;mid=(ph+z*z/40)/den;half=z*np.sqrt(ph*(1-ph)/20+z*z/(4*400))/den
  scores.append(dict(model=m,appearance=c,successes=n,trials=20,success_rate=ph,Wilson95_lower=mid-half,Wilson95_upper=mid+half))
  lifted=[r for r in rs if r['first_lift_step'] is not None]
  stages.append(dict(model=m,appearance=c,trials=20,successes=n,close_command=sum(r['issued_close_command'] for r in rs),lifted=len(lifted),approach_failure=sum(r['stage']=='approach_failure' for r in rs),near_target_failure=sum(r['stage']=='near_target_grasp_or_contact_failure' for r in rs),postlift_failure=sum(r['stage']=='post_lift_or_placement_failure' for r in rs),conditional_descent_n=len(lifted),descent_horizontal_path_m=float(np.mean([r['descent_horizontal_path_m'] for r in lifted])) if lifted else None,final_xy_distance_m=float(np.mean([r['final_xy_m'] for r in rs])),postlift_contact=sum(r['postlift_plate_contact'] for r in rs),postlift_centered=sum(r['postlift_centered'] for r in rs),postlift_contact_height_xy=sum(r['postlift_contact_height_xy'] for r in rs)))
write('success_rates.csv',scores);write('stage_summary.csv',stages)
stats=[];paired=[];draw=np.random.default_rng(20261025).integers(0,20,(10000,20))
def compare(left,right,label):
 d=np.array([[int(idx[left,s,c]['success'])-int(idx[right,s,c]['success']) for c in conditions] for s in seeds],float)
 for j,c in enumerate(conditions):
  x=d[:,j];stats.append(dict(comparison=label,appearance=c,difference=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist(),wins=int((x>0).sum()),losses=int((x<0).sum())))
  paired.extend(dict(comparison=label,seed=s,appearance=c,left=left,right=right,left_success=idx[left,s,c]['success'],right_success=idx[right,s,c]['success'],difference=int(x[i])) for i,s in enumerate(seeds))
 x=d.mean(1);stats.append(dict(comparison=label,appearance='equal_mean_state_clustered',difference=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist()))
 for j in (1,2):
  x=d[:,j]-d[:,0];stats.append(dict(comparison=label,appearance=conditions[j]+'_interaction_vs_clean',difference=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist()))
compare(models[1],models[0],'D-center_minus_C_lowLR600')
histroot=Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency';historical=[];ha=[]
for label,root,name in [('historical_C',Path(plan['historical_source'])/'policy_run','C'),('historical_D-center',histroot/'stove_centering/run_20261024_01','D-center')]:
 hp=read(root/'run_plan.json');hp_eval=read(root/'evaluation_protocol.json');rs=[r for r in read(root/'per_trial_results.json') if r['model']==name];assert len(rs)==60
 for r in rs:
  ref=idx[models[0],r['seed'],r['condition']]
  for k in ('state_sha256','xml_sha256','language'):assert r[k]==ref[k]
  idx[label,r['seed'],r['condition']]=r
 for k in ('conditions','states','max_steps','chunk_execution','reflection','MSAA','bddl','task'):assert hp_eval[k]==proto[k],k
 assert hp['common_step0_sha256']==plan['common_step0_sha256'] and hp.get('seed',hp['shared_protocol']['seed'])==plan['seed']
 for k in ('data_manifest','action_proprio_statistics'):assert hp['inputs'][k]==plan['inputs'][k]
 assert read(hp['inputs']['B20_sampling_order'])[:4800]==read(plan['inputs']['B20_sampling_order'])[:4800]
 for k in ('lambda_aux','microbatch_appearance','accumulation','backend','dtype','weight_decay'):assert hp['shared_protocol'][k]==plan['shared_protocol'][k],k
 path=Path(read(root/'policy_registry.json')[name]);train=rows(path.parent/'training_metrics.csv');first600=[r for r in train if int(r['step'])<=600];oldlrs=sorted({float(r['lr']) for r in first600});assert oldlrs==[.0002]
 ha.append(dict(model=label,checkpoint=str(path),sha256=sha(path),source_config_sha256=sha(root/'run_plan.json'),same_state_XML_seed_language=True,same_data_normalization_initialization_seed=True,sampling_first4800_equal=True,original_LRs_first600=oldlrs,new_LR_first600=2e-5,original_scheduler='milestone600 after final update',new_scheduler='milestone1200',same_first600_actual_LR=False,source_implementation_identical=False,limitations='historical implementation uses prior entry; current uses accepted Interface; no bitwise end-to-end reproduction asserted; C continuation has16 known replayed computations'))
 for c in conditions:historical.append(dict(model=label,appearance=c,successes=sum(r['success'] for r in rs if r['condition']==c),trials=20))
 compare(models[0] if name=='C' else models[1],label,'new_minus_'+label)
write('historical_success_rates.csv',historical);save('historical_comparability_audit.json',ha);write('paired_results.csv',paired);save('paired_statistics.json',dict(physical_state_clusters=20,bootstrap_draws=10000,bootstrap_seed=20261025,comparisons=stats,development_used_for_tuning=True,single_seed=True,no_independent_test_claim=True))
training=[];offline=[];inventory=[];accept=[];recovery=[]
keys=['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']+[f'dim{d}_L1' for d in range(7)]+[f'h{h}_L1' for h in range(8)]
for arm in arms:
 folder=o/arm/attempts[arm];acc=read(folder/'training_acceptance.json');accept.append(dict(model=arm,**acc));assert acc['successful_updates']==600
 train=rows(folder/'training_metrics.csv');start=acc['restored_step'];assert [int(r['step']) for r in train]==list(range(start+1,601))
 assert all(abs(float(r['lr'])-2e-5)<1e-12 and float(r['weighted_keep'])==0 for r in train)
 training.extend(dict(model=arm,source='continued',**r) for r in train)
 original=long/arm/'formal' if arm=='C_m0p1' else short/arm/'formal'
 training.extend(dict(model=arm,source='historical_confirmed_including_unpersisted_C401_416' if arm=='C_m0p1' else 'historical',**r) for r in rows(original/'training_metrics.csv'))
 offline.extend(dict(model=arm,**r) for r in rows(folder/'validation_metrics.csv'))
 if arm=='D-center_m0p1':
  prior=o/arm/'formal_gpu0_retry1';priorrows=rows(prior/'training_metrics.csv');assert [int(r['step']) for r in priorrows]==list(range(201,397))
  training.extend(dict(model=arm,source='prior_attempt_including_unpersisted_D301_396',**r) for r in priorrows)
  offline.extend(dict(model=arm,**r) for r in rows(prior/'validation_metrics.csv') if int(r['step'])<300)
  replaykeys=['action_loss','auxiliary_loss','total_loss','gradient_vla','gradient_head','gradient_proprio','gradient_aux_head'];replay=[dict(step=int(r['step']),max_abs_difference=max(abs(float(r[k])-float(train[int(r['step'])-301][k])) for k in replaykeys)) for r in priorrows if int(r['step'])>=301]
  save('D_replay_equivalence_audit.json',dict(known_replay_count=96,scalar_and_norm_comparisons=replay,bitwise_full_parameter_equivalence_not_inferred=True))
 ckpath=Path(registry[arm+'_s600']);ck=torch.load(ckpath,map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['scheduler']['last_epoch']==600 and ck['scheduler']['milestones']=={1200:1}
 assert ck['AMP_scaler'] is None and all(int(s['step'])==600 for s in ck['optimizer']['state'].values());assert ck['optimizer']['param_groups'][0]['lr']==2e-5
 inventory.append(dict(model=arm,step=600,path=str(ckpath),sha256=sha(ckpath),bytes=ckpath.stat().st_size,optimizer_states=len(ck['optimizer']['state'])));recovery.append(read(folder/'resume_provenance.json'));del ck
write('training_metrics_all.csv',training);write('checkpoint_inventory.csv',inventory)
def aggregate(cols):
 d=collections.defaultdict(list)
 for r in offline:d[tuple(r[k] for k in cols)].append(r)
 return [dict(zip(cols,k))|dict(n=len(rs),**{f:float(np.mean([float(r[f]) for r in rs])) for f in keys}) for k,rs in d.items()]
summary=aggregate(['model','step','split','appearance']);demo=aggregate(['model','step','split','appearance','episode'])
write('offline_continuation_summary.csv',summary);write('per_demo_metrics.csv',demo)
phasefile=histroot/'stove_postlift/run_20261025_01/demo_phase_coverage.csv';phase={r['episode']:int(r['first_lift']) if r['first_lift'] else None for r in rows(phasefile)}
for r in offline:r['phase']=('postlift' if int(r['timestep'])>=phase[r['episode']] else 'prelift') if phase.get(r['episode']) is not None else 'unknown'
write('coarse_phase_metrics.csv',aggregate(['model','step','split','appearance','phase']))
oldsummary=[r for r in rows(short/'fixed_offline_summary.csv') if r['model'] in arms and r['step']=='200'];si={(r['model'],r['step'],r['split'],r['appearance']):r for r in summary};changes=[]
for r in oldsummary:
 s=si[r['model'],'600',r['split'],r['appearance']];changes.append(dict(model=r['model'],split=r['split'],appearance=r['appearance'],**{k+'_200':float(r[k]) for k in keys if not k.startswith('h')},**{k+'_600':s[k] for k in keys if not k.startswith('h')},**{k+'_change':s[k]-float(r[k]) for k in keys if not k.startswith('h')}))
write('offline_200_to600.csv',changes)
figdir=o/'figures';figdir.mkdir(exist_ok=False)
def figure(fig,name):fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=150);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
for metric in ('action_L1','translation_L1','rotation_L1','gripper_L1'):
 fig,axs=plt.subplots(2,2,figsize=(10,7))
 for i,split in enumerate(('train','validation')):
  for j,c in enumerate(('clean','background')):
   for arm,col in zip(arms,('#2678b2','#b64b58')):
    rs={int(r['step']):float(r[metric]) for r in oldsummary+summary if r['model']==arm and r['split']==split and r['appearance']==c};x=sorted(rs);axs[i,j].plot(x,[rs[k] for k in x],'-o',label=arm,color=col)
   axs[i,j].set(title=split+' '+c,xlabel='effective optimizer step',ylabel='normalized '+metric);axs[i,j].legend(fontsize=7);axs[i,j].grid(alpha=.2)
 figure(fig,'fixed_'+metric)
fig,axs=plt.subplots(2,2,figsize=(12,7))
for j,arm in enumerate(arms):
 effective={int(r['step']):r for r in training if r['model']==arm and r['source']!='continued'};effective.update({int(r['step']):r for r in training if r['model']==arm and r['source']=='continued'});rs=[effective[k] for k in sorted(effective)];x=[int(r['step']) for r in rs]
 for i,key in enumerate(('action_loss','weighted_auxiliary')):
  y=np.array([float(r[key]) for r in rs]);axs[i,j].plot(x,y,alpha=.3,label='raw batch');axs[i,j].plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),label='trailing25');axs[i,j].set(title=arm,ylabel=key,xlabel='effective optimizer step');axs[i,j].legend()
figure(fig,'instantaneous_training_curves')
protected=[dict(path=x['path'],unchanged=sha(x['path'])==x['sha256']) for x in read(o/'protected_inputs.json')];assert all(x['unchanged'] for x in protected)
ea=read(o/'evaluation/formal/evaluation_acceptance.json');assert ea['valid_trials']==120
save('delivery_verification.json',dict(status='PASS_EXECUTION_NOT_METHOD',effective_final_updates={arm:600 for arm in arms},new_computed_successful_updates=sum(x['actual_computed_updates'] for x in accept)+196,new_distinct_updates=584,known_replayed_updates=112,C_replay=16,D_replay=96,unconfirmed_historical_inflight417=True,unconfirmed_D_inflight397=True,training=accept,recovery=recovery,valid_rollouts=120,invalid_rollouts=ea['invalid_trials'],actual_rollout_attempts=120+ea['invalid_trials'],additional_diagnostic_backward=0,AE_updates=0,teacher_updates=0,sealed_reads=0,protected_inputs=protected,preupdate_incident=read(o/'preupdate_incident.json'),cuda_incident=read(o/'cuda_incident.json'),driver_pause=read(o/'gpu0_driver_pause/delivery_verification.json'),no_automatic_extension=True))
lines=['# 低LR C / D-center续训600步与配对development闭环','', '过程PASS，不代表方法有效性PASS。两组统一使用最终600步，20个已参与调参的物理状态，每状态三种外观。','', '|模型|Clean|Seen|Unseen|等权平均|','|---|---:|---:|---:|---:|']
for m in models:
 ns=[next(r['successes'] for r in scores if r['model']==m and r['appearance']==c) for c in conditions];lines.append('|'+m+'|'+ '|'.join(str(n)+'/20' for n in ns)+f'|{sum(ns)/60:.4f}|')
for m in ('historical_C','historical_D-center'):
 ns=[next(r['successes'] for r in historical if r['model']==m and r['appearance']==c) for c in conditions];lines.append('|'+m+'|'+ '|'.join(str(n)+'/20' for n in ns)+f'|{sum(ns)/60:.4f}|')
lines+=['','## 恢复与计算','', 'C有416步成功日志，但完整状态仅到400；D完整状态200。实际续算C200步、D400步，共600次成功计算，其中C401–416为16次已知重放，新增未完成更新步584。C历史可能执行中的417步未确认，不计入。BF16无AMP scaler，checkpoint显式None；模型、optimizer、scheduler、CPU/CUDA/Python/NumPy RNG及采样位置恢复。不是从权重单独重启。','', '保留1200步MultiStepLR里程碑，无warmup；本轮实际LR始终2e-5，未进入衰减阶段。辅助系数、中心、缩放、seed和有效batch不变，AE/教师更新0。初次恢复因str/Path类型错误在状态加载前停止，updates=0，原日志保留，修正读取后新attempt运行。','', '## 可比性与边界','', '历史状态/XML/seed/语言/部署条件以及训练数据、初始化、归一化、样本顺序和共同辅助系数已核对。历史前600步LR=2e-4，本轮=2e-5；历史scheduler在600步后衰减，本轮在1200步后衰减，前600步均没有低LR衰减更新。不同历史入口未做逐位端到端重现，因此历史对比为参照，不声称严格单变量因果识别。','', '推理动作不使用状态真值修正；状态仅用于记录阶段和官方成功判定。抬升3cm、接近10cm为沿用的描述代理；接触/高度/水平条件来自既有任务指标。条件下降均值仅针对已抬升试验，存在阶段进入选择，不单独证明控制改善。','', '状态聚类bootstrap共10000次，同一物理状态的三种外观联合抽样，见paired_statistics.json；20状态而非60独立样本。单seed，development已参与调参，不是独立测试。','', '固定离线评估复用short200样本/预处理/mask；C400/500/600和D200/300/400/500/600。C300缺失固定节点未补造；C401–416旧日志与重放日志均保留，主训练曲线使用恢复后有效路径。逐维、逐chunk、逐示范及可靠抬升前后指标分别保存；没有精确放置动作标签时不编造。','', '本轮无额外诊断backward，无AE训练，无封存测试读取，不自动延长至1200、不追加LR或seed。']
lines=[x.replace('实际续算C200步、D400步，共600次成功计算，其中C401–416为16次已知重放，新增未完成更新步584。','实际续算C200步、D先196步后300步，共696次成功计算，其中C401–416重放16次、D301–396重放96次，共112次已知重放；新增未完成更新步584。D首次GPU1恢复在201步backward报cuDNN错误，新增成功更新0；随后GPU0在396步后出现驱动查询阻塞，仅300步完整状态可恢复，后从300步恢复。两次故障原日志均保留，不推断为OOM或表示问题。') for x in lines]
lines+=['','## 设备与恢复差异','','C在GPU1完成，D在GPU0完成；D第二次GPU0启动时驱动响应正常、空闲约79.2GiB，volatile ECC为0，aggregate correctable=6且uncorrectable=0。没有将计数重置视为硬件修复证明。各attempt分别检查ECC增量，先前故障单独保留。D重复301–396步的scalar/gradient norm比较见D_replay_equivalence_audit.json；不以此冒充所有参数逐位比较。']
with (o/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('CONTINUATION_ANALYSIS_COMPLETE',scores,flush=True)
