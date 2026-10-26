"""Report each matched seed; preserve physical-state clusters across appearances."""
import argparse,json,csv,hashlib,collections
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def write(n,rs):
 with (o/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
protocol=read(o/'replication_protocol.json');first=read(o/f"seed_{protocol['seeds'][0]}/run_plan.json");S=Path(first['historical_source']);H=Path(first['historical_center_source']);ep=read(o/f"seed_{protocol['seeds'][0]}/evaluation_protocol.json");states=ep['states'];conditions=ep['conditions'];allrows=[];inventory=[];init=[];train=[];validation=[];accept=[]
for seed in protocol['seeds']:
 root=o/f'seed_{seed}';plan=read(root/'run_plan.json');evaluation=read(root/'evaluation_protocol.json');assert evaluation['states']==states and evaluation['conditions']==conditions
 r=read(root/'per_trial_results.json');assert len(r)==120 and all(x['status']=='COMPLETE' for x in r);assert read(root/'evaluation/formal/evaluation_acceptance.json')['invalid_trials']==0
 allrows.extend(dict(**x,policy_seed=seed,method=x['model'].removesuffix('_s600'),historical=False) for x in r)
 starts=[];val0=[]
 for method in protocol['methods']:
  run=root/method/'formal';ck=torch.load(run/'step0000.pt',map_location='cpu');starts.append(ck);assert ck['global_step']==0
  audit=read(run/'initialization_audit.json');assert audit['all_trainable_parameters_equal_common_step0'] and audit['CPU_CUDA_RNG_equal_common_step0'];init.append(dict(seed=seed,method=method,**audit))
  rs=rows(run/'training_metrics.csv');assert [int(x['step']) for x in rs]==list(range(1,601));assert all(float(x['lr'])==.0002 for x in rs);train.extend(dict(policy_seed=seed,method=method,**x) for x in rs)
  vv=rows(run/'validation_metrics.csv');validation.extend(dict(policy_seed=seed,method=method,**x) for x in vv);val0.append([{k:v for k,v in x.items() if k!='auxiliary_L1'} for x in vv if int(x['step'])==0])
  final=run/'step0600.pt';ckfinal=torch.load(final,map_location='cpu');assert ckfinal['global_step']==600 and ckfinal['next_physical_sample']==4800 and ckfinal['scheduler']['last_epoch']==600 and ckfinal['scheduler']['milestones']=={600:1};assert all(int(x['step'])==600 for x in ckfinal['optimizer']['state'].values());del ckfinal
  inventory.append(dict(seed=seed,method=method,path=str(final),sha256=sha(final),global_step=600,common_step0_sha256=plan['common_step0_sha256']));accept.append(dict(seed=seed,method=method,**read(run/'training_acceptance.json')))
 for k in starts[0]['parameters']:
  assert starts[0]['parameters'][k].keys()==starts[1]['parameters'][k].keys()
  for n in starts[0]['parameters'][k]:assert torch.equal(starts[0]['parameters'][k][n],starts[1]['parameters'][k][n])
 assert torch.equal(starts[0]['torch_rng'],starts[1]['torch_rng']) and all(torch.equal(x,y) for x,y in zip(starts[0]['cuda_rng'],starts[1]['cuda_rng']))
 assert val0[0]==val0[1],'C/D actual step0 action validation differs';del starts
for method,root,key in [('C',S/'policy_run','C'),('D-center',H,'D-center')]:
 he=read(root/'evaluation_protocol.json')
 for k in ['states','conditions','max_steps','chunk_execution','reflection','MSAA','bddl','task']:assert he[k]==ep[k],k
 rr=[x for x in read(root/'per_trial_results.json') if x['model']==key];assert len(rr)==60;allrows.extend(dict(**x,policy_seed=20261023,method=method,historical=True) for x in rr)
index={(r['policy_seed'],r['method'],r['seed'],r['condition']):r for r in allrows};assert len(index)==360
for state in states:
 for condition in conditions:
  rr=[r for r in allrows if r['seed']==state and r['condition']==condition]
  for k in ['state_sha256','xml_sha256','language']:assert len({r[k] for r in rr})==1
summary=[];paired=[];stats=[];seed_effects=[];stages=[];draw=np.random.default_rng(20261026).integers(0,20,(20000,20));delta=[]
for seed in [20261023]+protocol['seeds']:
 for method in protocol['methods']:
  for c in conditions:
   rr=[index[seed,method,s,c] for s in states];n=sum(r['success'] for r in rr);ph=n/20;z=1.959963984540054;den=1+z*z/20;mid=(ph+z*z/40)/den;half=z*np.sqrt(ph*(1-ph)/20+z*z/1600)/den
   summary.append(dict(policy_seed=seed,method=method,appearance=c,successes=n,trials=20,SR=ph,Wilson95_lower=mid-half,Wilson95_upper=mid+half,historical=seed==20261023))
   st=dict(policy_seed=seed,method=method,appearance=c,**collections.Counter(r['stage'] for r in rr),lifted=sum(r['max_lift_m']>=.03 for r in rr))
   for k in ['issued_close_command','postlift_plate_contact','postlift_centered','postlift_contact_height_xy','postlift_open']:
    vv=[r[k] for r in rr if k in r];st[k+'_available']=len(vv);st[k+'_count']=sum(vv) if vv else None
   for k in ['descent_horizontal_path_m','descent_xy_error_change_m','final_xy_m']:
    vv=[r[k] for r in rr if k in r and (not k.startswith('descent') or r.get('first_lift_step') is not None)];st[k+'_available']=len(vv);st[k+'_mean']=float(np.mean(vv)) if vv else None
   stages.append(st)
 d=np.array([[int(index[seed,'D-center',s,c]['success'])-int(index[seed,'C',s,c]['success']) for c in conditions] for s in states]);delta.append(d)
 for j,c in enumerate(conditions):
  x=d[:,j];stats.append(dict(policy_seed=seed,appearance=c,D_minus_C=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist(),D_wins=int((x>0).sum()),C_wins=int((x<0).sum())));paired.extend(dict(policy_seed=seed,state=s,appearance=c,C=index[seed,'C',s,c]['success'],D_center=index[seed,'D-center',s,c]['success'],D_minus_C=int(x[i])) for i,s in enumerate(states))
 for name,x in [('equal_appearance_mean',d.mean(1)),('seen_interaction',d[:,1]-d[:,0]),('unseen_interaction',d[:,2]-d[:,0])]:stats.append(dict(policy_seed=seed,appearance=name,D_minus_C=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist()))
 seed_effects.append(dict(policy_seed=seed,C=sum(index[seed,'C',s,c]['success'] for s in states for c in conditions)/60,D_center=sum(index[seed,'D-center',s,c]['success'] for s in states for c in conditions)/60,D_minus_C=float(d.mean())))
effects=np.array([x['D_minus_C'] for x in seed_effects]);variability={m:dict(seed_values=[r[m] for r in seed_effects],range=float(np.ptp([r[m] for r in seed_effects])),sample_SD=float(np.std([r[m] for r in seed_effects],ddof=1))) for m in ['C','D_center']}
pooled=np.mean(delta,axis=(0,2));conditional=dict(mean_D_minus_C=float(pooled.mean()),conditional_state_cluster_CI95=np.quantile(pooled[draw].mean(1),[.025,.975]).tolist(),seed_effects=seed_effects,method_seed_variability=variability,mean_absolute_within_seed_gap=float(np.abs(effects).mean()),interpretation='CI conditions on these three training runs, not uncertainty over training seeds. Only three seeds; report individual runs, no precise random-seed population inference.')
write('success_rates.csv',summary);write('paired_results.csv',paired);write('failure_stage_summary.csv',stages);write('per_trial_results.csv',allrows);write('training_metrics.csv',train);write('validation_metrics.csv',validation);write('checkpoint_inventory.csv',inventory);save('initialization_matching_audit.json',dict(status='PASS',actual_C_D_step0_parameters_RNG_action_validation_exact=True,audits=init));save('paired_statistics.json',dict(draws=20000,resampling_unit='physical state, all appearances jointly; per seed separately',individual_seed_statistics=stats,conditional_three_seed_summary=conditional,no_independent_test=True,AE_seed_variability_not_tested=True))
fig=o/'figures';fig.mkdir(exist_ok=False)
for seed in protocol['seeds']:
 f,ax=plt.subplots(figsize=(8,4))
 for method in protocol['methods']:
  rr=[r for r in train if r['policy_seed']==seed and r['method']==method];x=np.array([int(r['step']) for r in rr]);y=np.array([float(r['action_loss']) for r in rr]);ax.plot(x,y,alpha=.2);ax.plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),label=method+' trailing25')
 ax.set(xlabel='successful update',ylabel='normalized training action L1',title=str(seed));ax.legend();f.tight_layout();f.savefig(fig/f'action_{seed}.png',dpi=150);plt.close(f)
protected=[dict(path=k,unchanged=sha(k)==v) for k,v in read(o/'protected_inputs.json').items()];assert all(x['unchanged'] for x in protected)
save('delivery_verification.json',dict(status='PASS_EXECUTION_NOT_METHOD',policy_successful_updates=2400,actual_update_computations=len(train),replayed_updates=0,AE_updates=0,teacher_updates=0,valid_new_rollouts=240,actual_new_attempts=240,invalid_new_rollouts=0,retries=0,sealed_reads=0,protected_inputs=protected,training_acceptances=accept,scope=protocol['initialization_scope'],no_automatic_extension=True))
lines=['# 固定 AE 的 C/D-center 策略 seed 配对复核','','本轮无 AE 更新、无超参数搜索。新 seed 各 C/D-center600步，共2400成功更新；新闭环240次。','',f"初始化范围：`{protocol['initialization_scope']}`。实际源权重与随机化模块见各seed初始化验收。",'','|策略seed|方法|Clean|已见背景|未见背景|等权平均|','|---|---|---:|---:|---:|---:|']
for seed in [20261023]+protocol['seeds']:
 for m in protocol['methods']:
  ns=[next(r['successes'] for r in summary if r['policy_seed']==seed and r['method']==m and r['appearance']==c) for c in conditions];lines.append(f'|{seed}|{m}|'+ '|'.join(str(n)+'/20' for n in ns)+f'|{sum(ns)/60:.4f}|')
lines+=['','## 统计与限制','','同seed内按20个物理状态配对，三外观联合重采样。逐seed的D−C、翻转及背景交互全部保留。三个seed平均的状态聚类区间条件于这三次训练，不包含完整训练seed不确定性；不能把360条新旧trial当独立episode。','',json.dumps(conditional,ensure_ascii=False,indent=2),'','固定AE、单任务、重复development状态，不能评价AE seed稳定性或独立泛化。历史设备/恢复差异保留。历史细粒度阶段字段缺失时标未知，不当作无事件。3cm抬升和10cm接近仅为诊断代理，成功始终由官方判定。','', '训练初始化为原始预训练策略，不从任何C/D成品、D-inv03、低LR或E系列续训。两方法各自已验收训练均值、共同s_C与lambda_aux保持不变。全部600次实际LR2e-4，scheduler在optimizer后调用，保存的下一步LR2e-5不代表第600次更新LR。','', '数据20train/5development；2555有效训练窗口，每组4800物理窗口曝光、9600外观曝光，约1.879遍窗口；不同seed顺序不同，同seed内C/D完全匹配。不新增轨迹或读取封存内容。']
with (o/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('REPLICATION_ANALYSIS_COMPLETE',seed_effects,flush=True)
