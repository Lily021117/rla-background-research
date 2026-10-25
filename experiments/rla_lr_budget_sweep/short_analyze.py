"""Aggregate fixed offline curves, actual computation and historical AE evidence."""
import argparse,json,csv,collections,hashlib,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;figdir=o/'figures';figdir.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def csvsave(n,rs):
 with (o/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def figsave(fig,name):
 fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=160);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
plan=read(o/'run_plan.json');long=Path(plan['long_run_source']);arms=list(plan['arms']);metrics=[];grad=[];updates=[];train=[];verifications=[];initial=[];inventory=[];sampling=[]
manifest=read(plan['inputs']['data_manifest']);epis={e['episode']:e for e in manifest['episodes']};order=read(plan['inputs']['B20_sampling_order']);pool=sum(e['valid_chunks'] for e in epis.values() if e['split']=='train');assert pool==2555
for arm in arms:
 folder=(long/arm/'formal') if arm=='C_m0p1' else o/arm/'formal';rs=rows(folder/'training_metrics.csv');expected=416 if arm=='C_m0p1' else 200
 assert [int(r['step']) for r in rs]==list(range(1,expected+1))
 for r in rs:
  if 'preclip_total_gradient_norm' not in r:r['preclip_total_gradient_norm']=sum(float(r[k])**2 for k in ('gradient_vla','gradient_head','gradient_proprio'))**.5;r['clipping_applied']=False;r['norm_source']='derived from disjoint recorded module gradient norms'
  else:r['norm_source']='recorded total before optimizer; no clipping configured'
 train.extend(dict(model=arm,**r) for r in rs)
 diagnostic_dir=o/arm/(read(o/'diagnostic_registry.json').get(arm,'diagnostics') if (o/'diagnostic_registry.json').exists() else 'diagnostics')
 initial.append(read(folder/'initial_state_audit.json'));v=read(diagnostic_dir/'diagnostic_verification.json');assert v['status']=='PASS' and v['optimizer_updates']==0;verifications.append(dict(model=arm,**v))
 metrics.extend(rows(diagnostic_dir/'fixed_sample_metrics.csv'));grad.extend(rows(diagnostic_dir/'gradient_contributions.csv'));updates.extend(rows(diagnostic_dir/'parameter_update_statistics.csv'))
 draws=order[:expected*8];counts=collections.Counter(tuple(x) for x in draws);eps=collections.Counter(x[0] for x in draws)
 sampling.append(dict(model=arm,confirmed_updates=expected,physical_draws=len(draws),appearance_draws=len(draws)*2,dual_camera_image_exposures=len(draws)*4,unique_drawn_windows=len(counts),independent_train_episodes=len(eps),available_train_windows=pool,equivalent_pool_passes=len(draws)/pool,max_window_multiplicity=max(counts.values()),sampler='seeded full-pool permutations; without replacement within2555-window pass, repeats across passes',matched_prefix_first200=True))
 for path in sorted(folder.glob('step*.pt')):inventory.append(dict(model=arm,step=int(path.stem[4:]),path=str(path),sha256=sha(path),bytes=path.stat().st_size))
assert len({x['parameter_hash'] for x in initial})==1 and all(x['parameter_hash']==x['common_parameter_hash'] for x in initial)
csvsave('checkpoint_inventory.csv',inventory);csvsave('training_metrics_all.csv',train);csvsave('gradient_contributions.csv',grad);csvsave('parameter_update_statistics.csv',updates);csvsave('data_exposure.csv',sampling)
runtime=[]
for arm in arms:
 rs=[r for r in train if r['model']==arm];ns=np.array([float(r['preclip_total_gradient_norm']) for r in rs]);assert np.isfinite(ns).all()
 runtime.append(dict(model=arm,logged_successful_updates=len(rs),clipping_threshold='none',clipping_trigger_fraction=0.,total_gradient_mean=float(ns.mean()),total_gradient_median=float(np.median(ns)),total_gradient_p95=float(np.quantile(ns,.95)),total_gradient_max=float(ns.max()),actual_LR_values=sorted({float(r['lr']) for r in rs}),nonfinite_recorded_updates=0,peak_allocated_GiB=max(float(r['allocated']) for r in rs)/1024**3,peak_reserved_GiB=max(float(r['reserved']) for r in rs)/1024**3,elapsed_logged_seconds=max(float(r['seconds']) for r in rs)))
csvsave('optimizer_runtime_summary.csv',runtime)
keys=['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1','weighted_auxiliary']+[f'dim{d}_L1' for d in range(7)]
def aggregate(columns):
 groups=collections.defaultdict(list)
 for r in metrics:groups[tuple(r[k] for k in columns)].append(r)
 return [dict(zip(columns,k))|dict(elements=len(rs),**{f:float(np.mean([float(r[f]) for r in rs])) for f in keys}) for k,rs in groups.items()]
summary=aggregate(['model','step','split','appearance']);demo=aggregate(['model','step','split','appearance','episode']);horizon=aggregate(['model','step','split','appearance','horizon'])
csvsave('fixed_offline_summary.csv',summary);csvsave('per_demo_metrics.csv',demo);csvsave('horizon_metrics.csv',horizon)
phasefile=Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency/stove_postlift/run_20261025_01/demo_phase_coverage.csv';phase={r['episode']:(int(r['first_lift']) if r['first_lift'] else None) for r in rows(phasefile)}
for r in metrics:r['phase']=('postlift' if int(r['timestep'])>=phase[r['episode']] else 'prelift') if phase.get(r['episode']) is not None else 'unknown'
csvsave('coarse_phase_metrics.csv',aggregate(['model','step','split','appearance','phase']))
save('stage_label_limitations.json',dict(source=str(phasefile),source_sha256=sha(phasefile),usable='historically verified first3cm lift before/after only',unavailable=['precise approach','stable grasp','transport','descent placement'],no_new_labels_invented=True))
idx={(r['model'],int(r['step']),r['split'],r['appearance']):r for r in summary};hidx={(r['model'],int(r['step']),r['split'],r['appearance'],int(r['horizon'])):r for r in horizon};signals=[]
for arm in arms:
 for c in ('clean','background'):
  t1=idx[arm,100,'train',c];t2=idx[arm,200,'train',c];d1=idx[arm,100,'validation',c];d2=idx[arm,200,'validation',c]
  signals.append(dict(model=arm,appearance=c,train_action_change100_200=t2['action_L1']-t1['action_L1'],development_action_change100_200=d2['action_L1']-d1['action_L1'],development_gripper_change100_200=d2['gripper_L1']-d1['gripper_L1'],development_far_h7_change100_200=hidx[arm,200,'validation',c,7]['action_L1']-hidx[arm,100,'validation',c,7]['action_L1'],train_down_development_up=t2['action_L1']<t1['action_L1'] and d2['action_L1']>d1['action_L1'],aux_down_action_up=d2['auxiliary_L1']<d1['auxiliary_L1'] and d2['action_L1']>d1['action_L1'],numerical_fault=False))
csvsave('learning_signals.csv',signals)
intervals=[]
for arm in arms:
 nodes=sorted({int(r['step']) for r in summary if r['model']==arm})
 for start,end in zip(nodes,nodes[1:]):
  for app in ('clean','background'):
   t0,t1=idx[arm,start,'train',app],idx[arm,end,'train',app];d0,d1=idx[arm,start,'validation',app],idx[arm,end,'validation',app]
   intervals.append(dict(model=arm,appearance=app,start_step=start,end_step=end,train_action_change=t1['action_L1']-t0['action_L1'],dev_action_change=d1['action_L1']-d0['action_L1'],dev_gripper_change=d1['gripper_L1']-d0['gripper_L1'],dev_h7_change=hidx[arm,end,'validation',app,7]['action_L1']-hidx[arm,start,'validation',app,7]['action_L1'],train_down_dev_up=t1['action_L1']<t0['action_L1'] and d1['action_L1']>d0['action_L1'],aux_down_action_up=d1['auxiliary_L1']<d0['auxiliary_L1'] and d1['action_L1']>d0['action_L1']))
csvsave('interval_learning_signals.csv',intervals)
paired=[]
for m in (.1,.3,1,3):
 token=str(m).replace('.','p')
 for step in (0,50,100,200):
  for split in ('train','validation'):
   for c in ('clean','background'):
    x=idx.get((f'C_m{token}',step,split,c));y=idx.get((f'D-center_m{token}',step,split,c))
    if x is None or y is None:continue
    paired.append(dict(multiplier=m,step=step,split=split,appearance=c,**{f'D_minus_C_{k}':y[k]-x[k] for k in keys}))
csvsave('matched_C_D_offline_comparison.csv',paired)
colors=['#2678b2','#e58626','#3b9657','#b64b58'];muls=[.1,.3,1,3]
for method in ('C','D-center'):
 for metric in ('action_L1','translation_L1','rotation_L1','gripper_L1'):
  fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
  for i,split in enumerate(('train','validation')):
   for j,c in enumerate(('clean','background')):
    ax=axs[i,j]
    for m,col in zip(muls,colors):
     arm=f'{method}_m{str(m).replace(".","p")}';rs=sorted([r for r in summary if r['model']==arm and r['split']==split and r['appearance']==c],key=lambda r:int(r['step']));ax.plot([int(r['step']) for r in rs],[r[metric] for r in rs],'-o',label=f'LR x{m}',color=col)
    ax.set(title=f'{method}: fixed {split} / {c}',xlabel='successful updates',ylabel=metric);ax.legend(fontsize=8);ax.grid(alpha=.2)
  figsave(fig,method+'_'+metric+'_fixed')
 fig,axs=plt.subplots(2,4,figsize=(15,6),sharex=True)
 for j,m in enumerate(muls):
  arm=f'{method}_m{str(m).replace(".","p")}';rs=[r for r in train if r['model']==arm];x=np.array([int(r['step']) for r in rs])
  for i,key in enumerate(('action_loss','weighted_auxiliary')):
   y=np.array([float(r[key]) for r in rs]);axs[i,j].plot(x,y,alpha=.3,label='raw instantaneous batch');smooth=np.convolve(y,np.ones(25)/25,mode='valid');axs[i,j].plot(x[24:],smooth,label='trailing25 mean');axs[i,j].axvline(200,color='gray',ls=':');axs[i,j].set(title=f'{method} LR x{m}',xlabel='updates',ylabel=key);axs[i,j].legend(fontsize=6)
 figsave(fig,method+'_instantaneous_training')
fig,axs=plt.subplots(2,4,figsize=(16,7),sharex=True)
for j,m in enumerate(muls):
 for i,split in enumerate(('train','validation')):
  for method,col in [('C','#2678b2'),('D-center','#b64b58')]:
   arm=f'{method}_m{str(m).replace(".","p")}';
   for c,ls in [('clean','-'),('background','--')]:
    rs=sorted([r for r in summary if r['model']==arm and r['split']==split and r['appearance']==c],key=lambda r:int(r['step']));axs[i,j].plot([int(r['step']) for r in rs],[r['action_L1'] for r in rs],ls+'o',color=col,label=method+' '+c)
  axs[i,j].set(title=f'LR x{m}: fixed {split}',xlabel='updates',ylabel='normalized action L1');axs[i,j].legend(fontsize=6);axs[i,j].grid(alpha=.2)
figsave(fig,'matched_C_D_action_curves')
for method in ('C','D-center'):
 fig,axs=plt.subplots(2,4,figsize=(16,6),sharex=True)
 for j,m in enumerate(muls):
  arm=f'{method}_m{str(m).replace(".","p")}';
  for i,c in enumerate(('clean','background')):
   for h,col in zip((0,3,7),colors):
    rs=sorted([r for r in horizon if r['model']==arm and r['split']=='validation' and r['appearance']==c and int(r['horizon'])==h],key=lambda r:int(r['step']));axs[i,j].plot([int(r['step']) for r in rs],[r['action_L1'] for r in rs],'-o',color=col,label=f'chunk position{h}')
   axs[i,j].set(title=f'{method} x{m}: {c}',xlabel='updates',ylabel='fixed dev action L1');axs[i,j].legend(fontsize=7)
 figsave(fig,method+'_chunk_positions')
fig,axs=plt.subplots(1,3,figsize=(14,4))
for arm in arms:
 rs=sorted([r for r in grad if r['model']==arm],key=lambda r:int(r['step']))
 for ax,key in zip(axs,('action_gradient_norm','norm_ratio','cosine')):ax.plot([int(r['step']) for r in rs],[float(r[key]) if r[key] else np.nan for r in rs],'-o',label=arm);ax.set(xlabel='checkpoint',ylabel=key)
axs[0].legend(fontsize=6);figsave(fig,'isolated_shared_gradient_diagnostics')
ae=rows(o/'AE_training_curves.csv');fig,axs=plt.subplots(2,2,figsize=(11,7))
for i,task in enumerate(('stove','cookie')):
 for arm,col in [('C','#2678b2'),('D','#b64b58')]:
  rs=[r for r in ae if r['task']==task and r['model']==arm];x=np.array([int(r['step']) for r in rs])
  for j,key in enumerate(('reconstruction','weighted_consistency')):
   y=np.array([float(r[key]) for r in rs]);axs[i,j].plot(x,y,color=col,alpha=.2);axs[i,j].plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),color=col,label=arm+' trailing25');axs[i,j].set(title=task+' historical AE',xlabel='AE updates',ylabel=key);axs[i,j].legend()
figsave(fig,'historical_AE_training')
ae_rec=rows(o/'AE_development_reconstruction.csv');ae_dist=rows(o/'AE_latent_distribution.csv');ae_pair=rows(o/'AE_paired_distances.csv');ae_summary=[]
for task in ('stove','cookie'):
 for model in ('reference','C','D'):
  pairs=[r for r in ae_pair if r['task']==task and r['model']==model]
  for app in ('clean','background'):
   rs=[r for r in ae_rec if r['task']==task and r['model']==model and r['appearance']==app];d=next(r for r in ae_dist if r['task']==task and r['model']==model and r['appearance']==app)
   ae_summary.append(dict(task=task,model=model,appearance=app,reconstruction=float(np.mean([float(r['reconstruction_L1'])+float(r['reconstruction_MSE']) for r in rs])),copy_current=float(np.mean([float(r['copy_current_L1'])+float(r['copy_current_MSE']) for r in rs])),latent_norm=float(d['norm_mean']),participation_ratio=float(d['participation_ratio']),entropy_effective_rank=float(d['entropy_effective_rank']),dimension_variance=float(d['dimension_variance_mean']),paired_MSE=float(np.mean([float(r['latent_MSE']) for r in pairs])),paired_unit_L2=float(np.mean([float(r['unit_L2']) for r in pairs])),paired_relative_L2=float(np.mean([float(r['relative_L2']) for r in pairs]))))
csvsave('AE_summary.csv',ae_summary)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,task in zip(axs,('stove','cookie')):
 for model,col in [('C','#2678b2'),('D','#b64b58')]:
  for app,marker in [('clean','o'),('background','x')]:
   ref=next(r for r in ae_summary if r['task']==task and r['model']=='reference' and r['appearance']==app);fin=next(r for r in ae_summary if r['task']==task and r['model']==model and r['appearance']==app)
   ax.scatter([0,600],[ref['reconstruction'],fin['reconstruction']],marker=marker,color=col,label=model+' '+app)
 ax.set(title=task+': only reference/final measurements',xlabel='additional AE updates',ylabel='fixed dev L1+MSE');ax.legend(fontsize=7)
figsave(fig,'historical_AE_dev_endpoints_only')
protected=[dict(path=x['path'],unchanged=sha(x['path'])==x['sha256']) for x in read(o/'protected_inputs.json')];assert all(x['unchanged'] for x in protected)
save('delivery_verification.json',dict(status='PASS_SHORT_TRAINING_AND_OFFLINE_DIAGNOSTICS',confirmed_successful_updates={arm:416 if arm=='C_m0p1' else 200 for arm in arms},confirmed_total=1816,comparable_first200_total=1600,post_revision_updates=1400,unpersisted_logged_updates16=True,inflight417_success_unconfirmed=True,replayed_updates=0,diagnostic_backward_calls=sum(x['backward_calls'] for x in verifications),diagnostic_optimizer_updates=0,diagnostic_incident=read(o/'diagnostic_incident.json'),AE_updates=0,teacher_updates=0,new_closed_loop=0,sealed_reads=0,initial_parameters_equal=True,protected_inputs=protected,diagnostics=verifications,missing_C_m0p1_step50=True))
lines=['# 炉灶短程学习率比较与两阶段曲线','', '8组均已有200步比较节点。C×0.1在用户修订前确认416次更新，保留400步完整恢复状态及416步日志；其余7组各200。累计确认1816次成功计算，无重放；暂停时可能正在执行417步，其成功未获日志确认，不计入。新增闭环0，AE/教师更新0。','', '|模型|Train Clean|Train BG|Dev Clean|Dev BG|Dev gripper Clean|Dev gripper BG|','|---|---:|---:|---:|---:|---:|---:|']
for arm in arms:lines.append('|'+arm+'|'+ '|'.join(f"{idx[arm,200,s,c][k]:.5f}" for s,c,k in [('train','clean','action_L1'),('train','background','action_L1'),('validation','clean','action_L1'),('validation','background','action_L1'),('validation','clean','gripper_L1'),('validation','background','gripper_L1')])+'|')
lines+=['','固定样本：20条训练示范各5个均匀anchor、5条development各20个anchor；各自Clean/Background，完整8步动作。固定评估与即时训练batch loss分开；评估以FP32误差汇总，正式BC依官方BF16动作标签计算。C×0.1没有50步checkpoint，缺点原样保留；没有重训补图。','', '策略AdamW单组322589983参数，基础LR2e-4×倍率，WD0.01，无裁剪，Math SDPA/BF16，micro8×累积2。保留1200步MultiStepLR里程碑，无warmup；本轮没有进入衰减阶段。原λ_aux=0.22408536585365854；AE-D λ_inv=0.009127798272072366保持冻结。','', '每个200步配置抽取1600物理窗口，3200外观、6400相机图像曝光；20独立示范、2555有效窗口，等效0.626遍。按固定seed逐遍随机排列，遍内不放回、跨遍重复。C×0.1额外已执行部分单独计数，不用来取代200步匹配比较。','', '梯度诊断使用同一8外观训练样本，eval模式、各loss独立forward，原RNG复位，无optimizer；范围含VLA共享参数、proprio和共享head层，详见scope清单。无梯度参数和近零分母单列。负余弦或大范数比仅是局部证据。参数绝对/相对变化与readout小范数限制单列。','', '完整四阶段可靠标签缺失；只复用已验收抬升前/后划分，不能把该划分叫作精确抓取/放置标签。development每条示范结果见per_demo_metrics.csv。','', '历史AE开发集统计只有共同step90000参考和最终600步，未补造中间验证点。保留原始曲线，平滑仅为尾随25步均值。距离含原始、clean范数归一化及单位向量L2，有效秩从中心化样本SVD计算；尺度缩小不是语义改善证明。历史物理读出复用既有状态来源和5示范bootstrap，没有新增探针。','', '此前λ_aux降为0.1倍损害C，不能由局部负余弦直接推荐继续降权。本轮只筛查固定AE下的早期策略优化，不能回答AE λ_inv、AE充分训练或AE seed稳定性，也不证明D的闭环能力恢复。','', '具体训练/验证背离、远期/夹爪恶化及aux下降但action上升信号见learning_signals.csv；未依据这些信号修改训练。下一步仅在审阅这些结果后提出一个最小匹配实验，不自动执行。']
with (o/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('SHORT_ANALYSIS_COMPLETE',flush=True)
