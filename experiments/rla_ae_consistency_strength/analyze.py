"""Fixed state-cluster comparisons, historical AE metrics, and complete provenance."""
import argparse,json,csv,hashlib,collections
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;ae=o/'ae_run'
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
plan=read(o/'run_plan.json');protocol=read(o/'experiment_protocol.json');ep=read(o/'evaluation_protocol.json');S=Path(plan['historical_source']);H=Path(plan['historical_center_source']);low=Path(plan['inputs']['project'])/'experiments/rla_lr_budget_sweep/run_20261025_01/extend600'
new=read(o/'per_trial_results.json');assert len(new)==60 and all(r['status']=='COMPLETE' for r in new)
model=ep['models'][0];assert model=='D-inv03_s600';states=ep['states'];conditions=ep['conditions'];idx={(r['seed'],r['condition']):r for r in new};assert len(idx)==60
historical=[('C',S/'policy_run','C'),('D-center',H,'D-center'),('C-lowLR',low,'C_m0p1_s600'),('D-center-lowLR',low,'D-center_m0p1_s600')]
datasets={'D-inv03':new};comparability=[];allpaths=[]
for label,root,key in historical:
 hp=read(root/'run_plan.json');he=read(root/'evaluation_protocol.json');rs=[r for r in read(root/'per_trial_results.json') if r['model']==key];assert len(rs)==60
 for k in ['states','conditions','max_steps','chunk_execution','reflection','MSAA','bddl','task']:assert he[k]==ep[k],(label,k)
 assert hp['common_step0_sha256']==plan['common_step0_sha256']
 for k in ['data_manifest','action_proprio_statistics']:assert hp['inputs'][k]==plan['inputs'][k]
 assert read(hp['inputs']['B20_sampling_order'])[:4800]==read(plan['inputs']['B20_sampling_order'])
 for r in rs:
  for k in ['state_sha256','xml_sha256','language']:assert r[k]==idx[r['seed'],r['condition']][k]
 registry=read(root/('policy_registry_v4.json' if root==low else 'policy_registry.json'));cp=Path(registry[key]);lr=sorted({float(r['lr']) for r in rows(cp.parent/'training_metrics.csv')});assert lr==([.00002] if root==low else [.0002])
 comparisons={k:hp['shared_protocol'][k]==plan['shared_protocol'][k] for k in ['lambda_aux','microbatch_appearance','accumulation','dtype','backend','weight_decay']};assert all(comparisons.values())
 comparability.append(dict(model=label,checkpoint=str(cp),checkpoint_sha256=sha(cp),same_initialization=True,same_data_order_first4800=True,same_normalization=True,same_evaluation_states_seed_render_control=True,actual_LR=lr,lowLR_sensitivity_reference=root==low,shared_fields=comparisons,source_entry_same_as_historical_center_training=root==H,limitation='historical devices/recovery and numeric histories differ; intentional new AE coefficient and derived target/own mean; not bitwise multi-device replication'))
 datasets[label]=rs
write('historical_comparability.csv',[{**r,'shared_fields':json.dumps(r['shared_fields'])} for r in comparability])
rates=[];paired=[];stats=[];draw=np.random.default_rng(20261026).integers(0,20,(10000,20))
for label,rs in datasets.items():
 for c in conditions:
  n=sum(r['success'] for r in rs if r['condition']==c);ph=n/20;z=1.959963984540054;den=1+z*z/20;mid=(ph+z*z/40)/den;half=z*np.sqrt(ph*(1-ph)/20+z*z/1600)/den
  rates.append(dict(model=label,appearance=c,successes=n,trials=20,success_rate=ph,Wilson95_lower=mid-half,Wilson95_upper=mid+half))
for label in list(datasets)[1:]:
 old={(r['seed'],r['condition']):r for r in datasets[label]};d=np.array([[int(idx[s,c]['success'])-int(old[s,c]['success']) for c in conditions] for s in states])
 for j,c in enumerate(conditions):
  x=d[:,j];stats.append(dict(comparison='D-inv03 minus '+label,appearance=c,difference=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist(),wins=int((x>0).sum()),losses=int((x<0).sum())))
  paired.extend(dict(reference=label,seed=s,appearance=c,new_success=idx[s,c]['success'],reference_success=old[s,c]['success'],difference=int(x[i])) for i,s in enumerate(states))
 for name,x in [('equal_appearance_mean',d.mean(1)),('seen_interaction',d[:,1]-d[:,0]),('unseen_interaction',d[:,2]-d[:,0])]:stats.append(dict(comparison='D-inv03 minus '+label,appearance=name,difference=float(x.mean()),CI95=np.quantile(x[draw].mean(1),[.025,.975]).tolist()))
write('success_rates.csv',rates);write('paired_results.csv',paired);save('paired_statistics.json',dict(seed=20261026,draws=10000,physical_state_clusters=20,comparisons=stats,development_used_for_tuning=True,single_training_seed=True,no_independent_test=True))
stages=[]
for label,rs in datasets.items():
 for c in conditions:
  ss=[r for r in rs if r['condition']==c];ct=collections.Counter(r['stage'] for r in ss)
  close_known=[r for r in ss if 'issued_close_command' in r];descent_known=[r for r in ss if r.get('first_lift_step') is not None and 'descent_horizontal_path_m' in r]
  stages.append(dict(model=label,appearance=c,successes=sum(r['success'] for r in ss),close_commands=sum(r['issued_close_command'] for r in close_known) if close_known else None,close_command_records_available=len(close_known),lifted=sum(r['max_lift_m']>=.03 for r in ss),**ct,descent_metrics_available=all('descent_horizontal_path_m' in r for r in ss),descent_horizontal_path_m=float(np.mean([r['descent_horizontal_path_m'] for r in descent_known])) if descent_known else None))
write('failure_stage_summary.csv',stages)
assert read(ae/'representation_sample_manifest.json')==read(S/'ae_run/representation_sample_manifest.json')
rep=[];dist=[];pairs=[];abl=[]
for root,allowed in [(S/'ae_run',('reference','C','D')),(ae,('D',))]:
 for file,dest in [('reconstruction_metrics.csv',rep),('latent_distribution_metrics.csv',dist),('decoder_ablation_metrics.csv',abl)]:
  for r in rows(root/file):
   if r['model'] in allowed:dest.append({**r,'model':'D-inv03' if root==ae else r['model']})
 for m in allowed:pairs.extend({**r,'model':'D-inv03' if root==ae else m} for r in rows(root/f'latent_pairs_{m}.csv'))
write('AE_reconstruction_comparison.csv',rep);write('AE_distribution_comparison.csv',dist);write('AE_paired_distance_comparison.csv',pairs);write('AE_decoder_ablation_comparison.csv',abl)
summary=[]
for label in ['reference','C','D','D-inv03']:
 rr=[r for r in rep if r['model']==label];pp=[r for r in pairs if r['model']==label];dd=[r for r in dist if r['model']==label]
 for c in ['clean','background']:
  dr=next(r for r in dd if r['appearance']==c);ss=[r for r in rr if r['appearance']==c];mean=lambda key:float(np.mean([float(r[key]) for r in ss]))
  summary.append(dict(model=label,appearance=c,reconstruction_L1=mean('reconstruction_L1'),reconstruction_MSE=mean('reconstruction_MSE'),copy_current_L1=mean('copy_current_L1'),copy_current_MSE=mean('copy_current_MSE'),latent_MSE=float(np.mean([float(r['latent_MSE']) for r in pp])),relative_L2=float(np.mean([float(r['relative_L2']) for r in pp])),unit_L2=float(np.mean([float(r['unit_L2']) for r in pp])),**{k:float(dr[k]) for k in ['norm_mean','dimension_variance_mean','entropy_effective_rank','participation_ratio','different_episode_L2_mean','different_episode_unit_L2_mean']}))
write('AE_representation_summary.csv',summary)
curve=[]
for arm in ['C','D']:
 active=Path(protocol['historical_AE_active'][arm]);effective={int(r['step']):r for r in rows(S/'ae_run'/('AE_'+arm)/'training_metrics.csv')};effective.update({int(r['step']):r for r in rows(active/'training_metrics.csv')});curve.extend(dict(model=arm,**r) for _,r in sorted(effective.items()))
ae_train=rows(ae/'AE_D_inv03/training_metrics.csv');assert [int(r['step']) for r in ae_train]==list(range(1,601));curve.extend(dict(model='D-inv03',**r) for r in ae_train);write('AE_training_curves.csv',curve)
policy=o/'D-inv03/formal';pt=rows(policy/'training_metrics.csv');assert [int(r['step']) for r in pt]==list(range(1,601));assert all(float(r['lr'])==.0002 for r in pt);write('policy_training_metrics.csv',pt)
pv=rows(policy/'validation_metrics.csv');write('policy_validation_metrics.csv',pv)
gg=collections.defaultdict(list)
for r in pv:gg[int(r['step']),r['appearance']].append(r)
write('policy_validation_summary.csv',[dict(step=k[0],appearance=k[1],n=len(rs),**{f:float(np.mean([float(x[f]) for x in rs])) for f in ['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']}) for k,rs in sorted(gg.items())])
inventory=[]
for stage,paths in [('AE',list((ae/'AE_D_inv03').glob('*step0600.pt'))),('policy',[policy/'step0600.pt'])]:
 for path in paths:inventory.append(dict(stage=stage,path=str(path),sha256=sha(path),bytes=path.stat().st_size))
ck=torch.load(policy/'step0600.pt',map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['scheduler']['last_epoch']==600 and ck['scheduler']['milestones']=={600:1}
assert all(int(s['step'])==600 for s in ck['optimizer']['state'].values());assert np.isclose(ck['optimizer']['param_groups'][0]['lr'],.00002);del ck
write('checkpoint_inventory.csv',inventory)
fig=o/'figures';fig.mkdir(exist_ok=False)
for key in ['reconstruction','consistency','weighted_consistency']:
 f,ax=plt.subplots(figsize=(8,4))
 for label in ['C','D','D-inv03']:
  rs=[r for r in curve if r['model']==label];x=[int(r['step']) for r in rs];y=np.array([float(r[key]) for r in rs]);ax.plot(x,y,alpha=.2);ax.plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),label=label+' trailing25')
 ax.set(xlabel='effective update',ylabel=key);ax.legend();f.tight_layout();f.savefig(fig/(key+'.png'),dpi=150);plt.close(f)
f,axes=plt.subplots(1,2,figsize=(12,4))
for key in ['action_loss','auxiliary_loss','weighted_auxiliary']:
 x=np.array([int(r['step']) for r in pt]);y=np.array([float(r[key]) for r in pt]);axes[0].plot(x,y,alpha=.18);axes[0].plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),label=key+' trailing25')
for appearance in ['clean','background']:
 keys=sorted(k for k in gg if k[1]==appearance);axes[1].plot([k[0] for k in keys],[np.mean([float(r['action_L1']) for r in gg[k]]) for k in keys],marker='o',label=appearance)
axes[0].set(xlabel='effective update',ylabel='training loss; raw plus trailing25');axes[1].set(xlabel='fixed checkpoint',ylabel='development normalized action L1')
for ax in axes:ax.legend()
f.tight_layout();f.savefig(fig/'policy_losses.png',dpi=150);plt.close(f)
protected=[dict(path=k,unchanged=sha(k)==v) for k,v in read(o/'protected_inputs.json').items()];assert all(r['unchanged'] for r in protected)
for r in new:
 for k in ['video','trajectory']:assert Path(r[k]).is_file() and Path(r[k]).stat().st_size>0
 assert read(Path(r['trajectory']).with_name('result.json'))==r
acc=read(o/'evaluation/formal/evaluation_acceptance.json');assert acc['valid_trials']==60 and acc['invalid_trials']==0
save('delivery_verification.json',dict(status='PASS_PROCESS_NOT_METHOD',AE_successful_updates=600,policy_successful_updates=600,AE_actual_computed_updates=len(ae_train),policy_actual_computed_updates=len(pt),known_replay_updates=0,valid_rollouts=60,actual_attempts=60,invalid_rollouts=0,sealed_test_reads=0,teacher_updates=0,protected_inputs=protected,policy_initialization=read(policy/'initialization_audit.json'),target_readiness=read(o/'target_readiness.json')['status'],AE_acceptance=read(ae/'AE_D_inv03/acceptance.json'),policy_acceptance=read(policy/'training_acceptance.json'),no_auto_extension=True))
lines=['# 炉灶AE一致性强度D-inv03单变量实验','','执行PASS不代表方法有效。仅新增一组AE600、策略600、development60。','',f"lambda_inv: {protocol['lambda_inv_old']} → {protocol['lambda_inv_new']}；lambda_aux固定{protocol['lambda_aux']}；策略600次实际LR均2e-4。",'','|模型|Clean|已见背景|未见背景|等权平均|','|---|---:|---:|---:|---:|']
for label in datasets:
 ns=[next(r['successes'] for r in rates if r['model']==label and r['appearance']==c) for c in conditions];lines.append('|'+label+'|'+'|'.join(str(n)+'/20' for n in ns)+f'|{sum(ns)/60:.4f}|')
lines+=['','## 配置与边界','','AE从官方step90000共同初始化，重建=两外观各自L1+MSE平均，一致性=16×64 latent均方差，h1..8、AdamW1e-4、固定loss scale256，DINO冻结。新AE采样4800物理转移与历史逐项一致。不是从已训练D继续。','', '策略直接复用历史stove_centering训练入口及原始共同step0；除新AE target及其训练集自身均值外，保留共同s_C、lambda_aux、Math BF16、checkpointing、STOP/mask、batch16、optimizer与LR。scheduler在每次optimizer.step后调用，600次update都用2e-4；600步checkpoint保存的LR2e-5是最后scheduler调用后的值，不是第600次实际LR。','', '新sidecar保留内部D目录别名，但指向本次D-inv03 encoder，不是历史D。训练均值为20条train、两外观、完整有效窗口全部8跨度的FP32输出/FP64累加；development不参与拟合。读取与源RGB重算检查见target_readiness。真实未来仅作为target生成与监督，不作为策略推理条件。','', '本次使用GPU0，历史AE/策略存在GPU1、GPU0与中断恢复差异；保留历史证据，不能声称跨设备路径逐位相同。共同初始参数、数据与RNG按实际入口核验，设备差异单列，不静默改变优化配置。','', '20个development状态已反复用于探索与调参，单seed。配对bootstrap以物理状态为聚类单位，三外观联合重采样；零翻转或极少翻转导致离散/退化区间，不代表确定等价或稳定泛化。未见背景并未用于AE/策略训练。','', 'latent距离、均值、尺度、有效秩及重建需一起解释：尺度缩小不等于不变性改善，有效秩降低不直接叫坍塌。重建不是动作条件前向预测。结果不触发追加lambda、LR、seed、任务或训练步数。']
(o/'report.md').write_text('\n'.join(lines)+'\n')
print('ANALYSIS_COMPLETE',rates)
