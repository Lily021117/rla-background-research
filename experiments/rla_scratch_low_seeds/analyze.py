import sys,json,csv,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_low_seeds.common import *
import numpy as np
def main():
 out=ROOT/'delivery';out.mkdir(exist_ok=False);protocol=read(ROOT/'protocol.json');seeds=protocol['seeds'];training=[];validation=[];inventory=[];matching=[]
 for seed in seeds:
  root=ROOT/f'seed_{seed}';expected=[json.loads(x) for x in (root/'expected_sample_ids.jsonl').read_text().splitlines()];initial=[]
  for arm in ARMS:
   old=root/arm/'formal';p=root/arm/('recovery_gpu0_01' if seed==31002 and arm=='D-low' else 'formal');accept=read(p/'training_acceptance.json');assert accept['successful_updates']==600 and accept['strict_reload'];assert accept['replayed_updates']==0
   initial.append(read(p/'initialization_audit.json'))
   trace=[json.loads(x) for x in (p/'actual_sample_ids.jsonl').read_text().splitlines()]
   if p!=old:trace=[json.loads((old/'actual_sample_ids.jsonl').read_text().splitlines()[0])]+trace
   assert len(trace)==600 and all(a['step']==b['step'] and a['actual_sample_ids']==b['actual_sample_ids'] for a,b in zip(trace,expected))
   rr=list(csv.DictReader((p/'training_metrics.csv').open()))
   if p!=old:rr=[next(csv.DictReader((old/'training_metrics.csv').open()))]+rr
   assert len(rr)==600 and all(float(r['lambda_RLA'])==LAMBDA for r in rr)
   training.extend(dict(seed=seed,method=arm,**r) for r in rr)
   validation.extend(dict(seed=seed,method=arm,**r) for r in csv.DictReader((p/'validation_metrics.csv').open()))
   if p!=old:validation.extend(dict(seed=seed,method=arm,**r) for r in csv.DictReader((old/'validation_metrics.csv').open()) if int(r['step'])==0)
   for step in [0,1,100,300,600]:
    q=(old if p!=old and step==0 else p)/f'step{step:04}.pt';inventory.append(dict(seed=seed,method=arm,step=step,path=str(q),sha256=sha(q)))
  for key in ['parameter_sha256','RNG_hash','common_step0_sha256','sampling_hash']:assert initial[0][key]==initial[1][key]
  matching.append(dict(seed=seed,paired_initialization_and_RNG_exact=True,sample_IDs_exact_all600=True,initialization=initial[0]))
 init=[read(ROOT/f'seed_{s}/module_initialization_audit.json') for s in seeds]
 assert len({x['frozen_pretrained_sha256'] for x in init})==1
 assert len({x['initialization']['parameter_sha256'] for x in matching})==3
 rows=read(ROOT/'evaluation/per_trial_results.json');assert len(rows)==600 and all(r['status']=='COMPLETE' for r in rows)
 conditions=['clean']+[f'new_background_{i}' for i in range(1,5)];states=sorted({r['seed'] for r in rows});assert len(states)==20
 arrays={};rates=[];phases=[];flips=[];metrics=[];paired=[]
 rng=np.random.default_rng(31001);idx=rng.integers(0,20,(20000,20))
 def stat(seed,label,d):
  v=d.mean(1) if d.ndim==2 else d
  return dict(seed=seed,metric=label,difference=float(v.mean()),state_cluster_CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist())
 for seed in seeds:
  sets={}
  for arm in ARMS:
   model=f'{arm}_seed{seed}_s600';rr={(r['seed'],r['condition']):r for r in rows if r['model']==model};assert len(rr)==100;sets[arm]=rr
   a=np.array([[int(rr[s,c]['success']) for c in conditions] for s in states]);arrays[seed,arm]=a
   for j,c in enumerate(conditions):
    rates.append(dict(seed=seed,method=arm,appearance=c,successes=int(a[:,j].sum()),trials=20,SR=float(a[:,j].mean())))
    for stage,n in collections.Counter(rr[s,c]['stage'] for s in states).items():phases.append(dict(seed=seed,method=arm,appearance=c,stage=stage,count=n))
   metrics.append(dict(seed=seed,method=arm,clean=float(a[:,0].mean()),background=float(a[:,1:].mean()),background_minus_clean=float(a[:,1:].mean()-a[:,0].mean())))
  for key in sets['C-low']:
   for k in ['state_sha256','xml_sha256','language']:assert sets['C-low'][key][k]==sets['D-low'][key][k]
  a,b=arrays[seed,'D-low'],arrays[seed,'C-low'];d=a-b
  for j,c in enumerate(conditions):
   paired.append(stat(seed,c,d[:,j]));flips.append(dict(seed=seed,appearance=c,D_only=int(((a[:,j]==1)&(b[:,j]==0)).sum()),C_only=int(((a[:,j]==0)&(b[:,j]==1)).sum()),both_success=int(((a[:,j]==1)&(b[:,j]==1)).sum()),both_fail=int(((a[:,j]==0)&(b[:,j]==0)).sum())))
  paired.extend([stat(seed,'background_equal_mean',d[:,1:]),stat(seed,'background_clean_interaction',d[:,1:].mean(1)-d[:,0])])
 summary=[]
 for arm in ARMS+['D-low minus C-low']:
  for key in ['clean','background','background_minus_clean']:
   by={(r['seed'],r['method']):r[key] for r in metrics}
   values=[by[s,arm] if arm in ARMS else by[s,'D-low']-by[s,'C-low'] for s in seeds]
   summary.append(dict(method=arm,metric=key,mean=float(np.mean(values)),sample_SD=float(np.std(values,ddof=1)),minimum=min(values),maximum=max(values),per_seed=values))
 d=np.mean([arrays[s,'D-low']-arrays[s,'C-low'] for s in seeds],0)
 for key,x in [('clean',d[:,0]),('background',d[:,1:]),('background_clean_interaction',d[:,1:].mean(1)-d[:,0])]:paired.append(stat('three_seed_equal_mean',key,x))
 table(out/'training_metrics.csv',training);table(out/'validation_metrics.csv',validation);table(out/'checkpoint_inventory.csv',inventory);table(out/'success_summary.csv',rates);table(out/'per_seed_metrics.csv',metrics);table(out/'paired_flips.csv',flips);table(out/'failure_stage_summary.csv',phases)
 save(out/'matching_audit.json',matching);save(out/'seed_variability.json',summary);save(out/'paired_statistics.json',dict(results=paired,unit='physical state with all models/appearances/seeds jointly resampled; NOT training seed uncertainty',historical_not_pooled=True));save(out/'per_trial_results.json',rows)
 assert all(sha(p)==h for p,h in read(ROOT/'protected_inputs.json').items())
 save(out/'delivery_verification.json',dict(status='COMPLETE_FIXED_BUDGET',policy_successful_updates=3600,actual_update_computations=3649,replays=49,valid_clean=120,valid_background=480,actual_attempts=600,invalid_trials=0,AE_updates=0,historical_inputs_unchanged=True,sealed_read=False,recovery='seed31002 D-low resumed full step1 after correctable ECC increment at step50; original50 plus resumed599 computations; first49 lost updates replayed'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for seed in seeds:
  fig,axes=plt.subplots(1,3,figsize=(12,4))
  for arm in ARMS:
   rr=[r for r in training if r['seed']==seed and r['method']==arm]
   for ax,key in zip(axes,['action_loss','auxiliary_loss','weighted_auxiliary']):ax.plot([int(r['step']) for r in rr],[float(r[key]) for r in rr],label=arm,lw=.7);ax.set_title(key)
  axes[0].legend();fig.suptitle(f'Policy seed {seed}; raw batch losses, no smoothing');fig.tight_layout();fig.savefig(out/f'curves_seed{seed}.png');plt.close(fig)
 lines=['# 固定AE下三个新策略seed：C-low / D-low','', '六组各600次成功更新，600次有效闭环；AE更新0。历史两个采样顺序不并入本轮统计。GPU0可纠正ECC增量触发停止；授权复核后seed31002 D-low从完整step1恢复，重放49次，实际计算3649次。原故障记录与原50步日志保留；主曲线合并原step1与恢复后的step2–600，不重复计数。','', '|seed|方法|Clean|背景1|背景2|背景3|背景4|背景均值|','|---|---|---|---|---|---|---|---|']
 for seed in seeds:
  for arm in ARMS:
   a=arrays[seed,arm];lines.append(f'|{seed}|{arm}|'+ '|'.join(f'{n}/20' for n in a.sum(0))+f'|{a[:,1:].mean():.2%}|')
 lines+=['','## 配对差异与随机性']
 for x in summary:lines.append(f"- {x['method']} {x['metric']}: mean {x['mean']:+.4f}, sample SD {x['sample_SD']:.4f}, range [{x['minimum']:+.4f}, {x['maximum']:+.4f}], seeds {x['per_seed']}.")
 lines+=['','## 解释限制','逐seed展示方向，不用平均值掩盖C单seed失败。Clean收益不是背景不变性收益；必须同时看背景绝对成功率与相对Clean交互。三seed未观察到差异不证明等效。状态bootstrap不包含训练seed不确定性。同一任务、固定一对AE、反复使用development，非独立泛化验证。','', '本轮只有初始化与训练随机条件改变；AE-C/D自身训练统计仍不同。未增加方法、参数或试验，结束后停止。']
 with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
