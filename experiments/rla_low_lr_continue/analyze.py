import sys,csv,json,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_low_lr_continue.common import *
import numpy as np
def main():
 out=ROOT/'delivery';out.mkdir(exist_ok=False);train=[];val=[];inventory=[];sampling=[]
 for seed in SEEDS:
  expected=[json.loads(x) for x in (ROOT/f'seed_{seed}/expected_sample_ids.jsonl').read_text().splitlines()][600:]
  for arm in ARMS:
   root=ROOT/f'seed_{seed}/{arm}';p=root/arm/'formal';plan=read(root/'run_plan.json');accept=read(p/'training_acceptance.json');assert accept['successful_updates']==1800 and accept['new_successful_updates']==1200 and accept['strict_reload']
   assert read(p/'resume_forward_equivalence.json')['status']=='PASS'
   trace=[json.loads(x) for x in (p/'actual_sample_ids.jsonl').read_text().splitlines()];assert len(trace)==1200 and all(a['step']==b['step'] and a['actual_sample_ids']==b['actual_sample_ids'] for a,b in zip(trace,expected))
   sampling.append(dict(seed=seed,method=arm,matched1200=True,trace_sha256=sha(p/'actual_sample_ids.jsonl')))
   rr=list(csv.DictReader((p/'training_metrics.csv').open()));assert len(rr)==1200 and all(float(r['lr'])==2e-5 and float(r['lambda_RLA'])==LAMBDA for r in rr)
   train.extend(dict(seed=seed,method=arm,**r) for r in rr)
   source=Path(plan['source_step600_directory'])
   for q,endpoint in [(source/'validation_metrics.csv',600),(p/'validation_metrics.csv',None)]:
    for r in csv.DictReader(q.open()):
     if endpoint is None or int(r['step'])==endpoint:val.append(dict(seed=seed,method=arm,weighted_auxiliary_L1=LAMBDA*float(r['auxiliary_L1']),**r))
   for step in [900,1200,1500,1800]:
    q=p/f'step{step:04}.pt';inventory.append(dict(seed=seed,method=arm,step=step,path=str(q),sha256=sha(q)))
 old=read(SOURCE/'delivery/per_trial_results.json');new=read(ROOT/'evaluation/per_trial_results.json');assert len(old)==len(new)==600 and all(r['status']=='COMPLETE' for r in new)
 states=sorted({r['seed'] for r in new});conditions=['clean']+[f'new_background_{i}' for i in range(1,5)];assert len(states)==20
 arrays={};rates=[];metric=[];phases=[];pairs=[];flips=[]
 for seed in SEEDS:
  for arm in ARMS:
   bystep={}
   for step,rows in [(600,old),(1800,new)]:
    rr={(r['seed'],r['condition']):r for r in rows if r['model']==f'{arm}_seed{seed}_s{step}'};assert len(rr)==100;bystep[step]=rr
    a=np.array([[int(rr[s,c]['success']) for c in conditions] for s in states]);arrays[seed,arm,step]=a
    for j,c in enumerate(conditions):
     rates.append(dict(seed=seed,method=arm,step=step,appearance=c,successes=int(a[:,j].sum()),trials=20,SR=float(a[:,j].mean())))
     for stage,n in collections.Counter(rr[s,c]['stage'] for s in states).items():phases.append(dict(seed=seed,method=arm,step=step,appearance=c,stage=stage,count=n))
    metric.append(dict(seed=seed,method=arm,step=step,clean=float(a[:,0].mean()),background=float(a[:,1:].mean()),background_minus_clean=float(a[:,1:].mean()-a[:,0].mean())))
   for key in bystep[600]:
    for k in ['state_sha256','xml_sha256','language']:assert bystep[600][key][k]==bystep[1800][key][k]
 rng=np.random.default_rng(31001);idx=rng.integers(0,20,(20000,20));summaries=[]
 comparisons=[('C-low1800 minus600',('C-low',1800),('C-low',600)),('D-low1800 minus600',('D-low',1800),('D-low',600)),('D-low minus C-low at1800',('D-low',1800),('C-low',1800))]
 for label,ma,mb in comparisons:
  dd=[]
  for seed in SEEDS:
   a,b=arrays[(seed,)+ma],arrays[(seed,)+mb];d=a-b;dd.append(d)
   for j,c in enumerate(conditions):flips.append(dict(seed=seed,comparison=label,appearance=c,new_only=int(((a[:,j]==1)&(b[:,j]==0)).sum()),reference_only=int(((a[:,j]==0)&(b[:,j]==1)).sum()),both_success=int(((a[:,j]==1)&(b[:,j]==1)).sum()),both_failure=int(((a[:,j]==0)&(b[:,j]==0)).sum())))
   for key,v in [('clean',d[:,0]),('background',d[:,1:].mean(1)),('interaction',d[:,1:].mean(1)-d[:,0])]:pairs.append(dict(seed=seed,comparison=label,metric=key,difference=float(v.mean()),state_cluster_CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist()))
  d=np.mean(dd,0)
  for key,v in [('clean',d[:,0]),('background',d[:,1:].mean(1)),('interaction',d[:,1:].mean(1)-d[:,0])]:pairs.append(dict(seed='equal_three_seed',comparison=label,metric=key,difference=float(v.mean()),state_cluster_CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist()))
  for key in ['clean','background','interaction']:
   v=[r['difference'] for r in pairs if r['comparison']==label and r['metric']==key and r['seed'] in SEEDS];summaries.append(dict(comparison=label,metric=key,mean=float(np.mean(v)),SD=float(np.std(v,ddof=1)),minimum=min(v),maximum=max(v),per_seed=v))
 for arm in ARMS:
  for step in [600,1800]:
   for key in ['clean','background','background_minus_clean']:
    v=[r[key] for r in metric if r['method']==arm and r['step']==step];summaries.append(dict(comparison=f'{arm} step{step}',metric=key,mean=float(np.mean(v)),SD=float(np.std(v,ddof=1)),minimum=min(v),maximum=max(v),per_seed=v))
 table(out/'training_metrics.csv',train);table(out/'validation_metrics.csv',val);table(out/'checkpoint_inventory.csv',inventory);table(out/'success_summary.csv',rates);table(out/'per_seed_metrics.csv',metric);table(out/'paired_flips.csv',flips);table(out/'failure_stage_summary.csv',phases)
 save(out/'sampling_verification.json',sampling);save(out/'paired_statistics.json',dict(results=pairs,cluster='physical state; appearances, models, endpoints and seeds jointly resampled; NOT training-seed uncertainty'));save(out/'seed_variability.json',summaries);save(out/'per_trial_results.json',new)
 assert all(sha(p)==h for p,h in read(ROOT/'protected_inputs.json').items())
 save(out/'delivery_verification.json',dict(status='COMPLETE',new_successful_updates=7200,new_actual_computations=7200,new_replays=0,prior_replays_preserved=49,new_valid_clean=120,new_valid_background=480,new_attempts=600,new_invalid=0,AE_updates=0,sealed_read=False,historical_hashes_unchanged=True))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for seed in SEEDS:
  fig,ax=plt.subplots(1,3,figsize=(12,4))
  for arm in ARMS:
   rr=[r for r in train if r['seed']==seed and r['method']==arm]
   for a,k in zip(ax,['action_loss','auxiliary_loss','weighted_auxiliary']):a.plot([int(r['step']) for r in rr],[float(r[k]) for r in rr],label=arm,lw=.5);a.set_title(k)
  ax[0].legend();fig.suptitle(f'seed{seed}, continuation only, raw batch curves');fig.tight_layout();fig.savefig(out/f'training_seed{seed}.png');plt.close(fig)
 lines=['# 固定AE、固定半权重：低学习率续训','', '新增7200次成功更新，六组均从完整step600继续到1800；LR2e-5沿用原scheduler已触发的低LR阶段。新闭环600次，原600步闭环只复用。原先49次重放单独保留。本轮不是纯训练步数消融，而是更多更新与低学习率阶段的组合。','', '|seed|方法|step|Clean|背景1|背景2|背景3|背景4|背景均值|','|---|---|---|---|---|---|---|---|---|']
 for seed in SEEDS:
  for arm in ARMS:
   for step in [600,1800]:
    a=arrays[seed,arm,step];lines.append(f'|{seed}|{arm}|{step}|'+ '|'.join(f'{n}/20' for n in a.sum(0))+f'|{a[:,1:].mean():.2%}|')
 lines+=['','## 三seed差异与波动']
 for x in summaries:lines.append(f"- {x['comparison']} {x['metric']}: mean={x['mean']:+.4f}, SD={x['SD']:.4f}, range=[{x['minimum']:+.4f},{x['maximum']:+.4f}], per-seed={x['per_seed']}.")
 lines+=['','## 解释边界','1800步不是充分收敛证明。先看每seed完整能力与近全失败情况，再看背景绝对成功率和相对Clean变化；不能把Clean下降当鲁棒收益。若改善，不能区分额外更新与低LR的各自贡献。反复使用的development不是独立测试；3seed不能精确估计训练随机性，状态聚类区间不包含训练seed不确定性。固定结束，不自动继续系数/loss搜索或后续训练。']
 with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('CONTINUATION_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
