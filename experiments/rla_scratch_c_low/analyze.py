"""New C-low vs frozen C-fixed and D-low, preserving orders and physical-state clusters."""
import sys,csv,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_c_low.common import *
import numpy as np
def main():
 out=ROOT/'delivery';out.mkdir(exist_ok=False);training=[];validation=[];inventory=[];matches=[]
 for order in ORDERS:
  path=ROOT/f'seed_{order}/C-low/formal';done=read(path/'training_acceptance.json');assert done['successful_updates']==600 and done['strict_reload']
  ini=read(path/'initialization_audit.json');ref=read(BASE/f'seed_{order}/C-fixed/formal/initialization_audit.json')
  for k in ['parameter_sha256','RNG_hash','common_step0_sha256']:assert ini[k]==ref[k]
  trace=[json.loads(x) for x in (path/'actual_sample_ids.jsonl').read_text().splitlines()]
  reference=[json.loads(x) for x in (BASE/f'seed_{order}/C-fixed/formal/actual_sample_ids.jsonl').read_text().splitlines()];assert trace==reference and len(trace)==600
  metrics=list(csv.DictReader((path/'training_metrics.csv').open()));assert len(metrics)==600 and all(float(x['lambda_RLA'])==LAMBDA/2 for x in metrics)
  training.extend(dict(order=order,method='C-low',**r) for r in metrics)
  validation.extend(dict(order=order,method='C-low',**r) for r in csv.DictReader((path/'validation_metrics.csv').open()))
  matches.append(dict(order=order,parameters_RNG_exact=True,sampling_exact_600_rows=True,constant_half_weight_exact=True,trace_sha256=sha(path/'actual_sample_ids.jsonl')))
  for step in [0,1,100,300,600]:
   p=path/f'step{step:04}.pt';inventory.append(dict(order=order,method='C-low',step=step,path=str(p),sha256=sha(p)))
 old=read(BASE/'delivery/per_trial_results.json');new=read(ROOT/'evaluation/per_trial_results.json');assert len(new)==200 and all(r['status']=='COMPLETE' for r in new)
 reused=[dict(r,result_source='prior completed run; not rerun') for r in old if r['arm'] in ['C-fixed','D-low']];assert len(reused)==400
 for r in new:r.update(arm='C-low',order=int(r['model'].split('_order')[1].split('_s')[0]),result_source='new C-low completion')
 rows=reused+new;conditions=['clean']+[f'new_background_{i}' for i in range(1,5)];states=sorted({r['seed'] for r in new});assert len(states)==20
 arrays={};rates=[];phases=[]
 for order in ORDERS:
  for method in ['C-fixed','C-low','D-low']:
   rr={(r['seed'],r['condition']):r for r in rows if r['order']==order and r['arm']==method};assert len(rr)==100
   arrays[order,method]=np.array([[int(rr[s,c]['success']) for c in conditions] for s in states])
   for j,c in enumerate(conditions):
    rates.append(dict(order=order,method=method,appearance=c,successes=int(arrays[order,method][:,j].sum()),trials=20,SR=float(arrays[order,method][:,j].mean())))
    for stage,n in collections.Counter(rr[s,c]['stage'] for s in states).items():phases.append(dict(order=order,method=method,appearance=c,stage=stage,count=n))
   for r in rr.values():
    paired=next(x for x in new if x['order']==order and x['seed']==r['seed'] and x['condition']==r['condition'])
    for k in ['state_sha256','xml_sha256','language']:assert r[k]==paired[k]
 rng=np.random.default_rng(20261029);idx=rng.integers(0,20,(20000,20));statistics=[];flips=[]
 def stat(label,d):
  v=d.mean(1) if d.ndim==2 else d
  return dict(comparison=label,difference=float(v.mean()),CI95=np.quantile(v[idx].mean(1),[.025,.975]).tolist())
 for method,ref in [('C-low','C-fixed'),('D-low','C-low')]:
  differences=[]
  for order in ORDERS:
   a,b=arrays[order,method],arrays[order,ref];d=a-b;differences.append(d)
   for j,c in enumerate(conditions):
    statistics.append(stat(f'{order} {method} minus {ref} {c}',d[:,j]));flips.append(dict(order=order,method=method,reference=ref,appearance=c,method_only=int(((a[:,j]==1)&(b[:,j]==0)).sum()),reference_only=int(((a[:,j]==0)&(b[:,j]==1)).sum()),both_success=int(((a[:,j]==1)&(b[:,j]==1)).sum()),both_fail=int(((a[:,j]==0)&(b[:,j]==0)).sum())))
   statistics.append(stat(f'{order} {method} minus {ref} four-background equal mean',d[:,1:]));statistics.append(stat(f'{order} {method} minus {ref} background-clean interaction',d[:,1:].mean(1)-d[:,0]))
  d=np.mean(differences,0)
  for label,x in [('clean',d[:,0]),('background',d[:,1:]),('background-clean interaction',d[:,1:].mean(1)-d[:,0])]:statistics.append(stat(f'two-order equal {method} minus {ref} {label}',x))
 table(out/'training_metrics.csv',training);table(out/'validation_metrics.csv',validation);table(out/'checkpoint_inventory.csv',inventory);save(out/'matching_audit.json',matches);table(out/'success_summary.csv',rates);table(out/'paired_flips.csv',flips);table(out/'failure_stage_summary.csv',phases);save(out/'paired_statistics.json',dict(unit='physical state; all appearances and both orders jointly resampled; excludes full training-seed/task uncertainty',results=statistics))
 cols=sorted(set().union(*(r.keys() for r in rows)));table(out/'per_trial_results.csv',[{k:r.get(k) for k in cols} for r in rows]);save(out/'per_trial_results.json',rows)
 assert all(sha(p)==h for p,h in read(ROOT/'protected_inputs.json').items())
 save(out/'delivery_verification.json',dict(status='COMPLETE_BOUNDED_COMPLETION',new_effective_updates=1200,actual_update_computations=1200,replays=0,new_valid_rollouts=200,actual_attempts=200,invalid_trials=0,reused_valid_trials=400,AE_updates=0,DINO_feature_updates=0,protected_historical_inputs_unchanged=True,sealed_read=False))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for order in ORDERS:
  fig,axes=plt.subplots(1,3,figsize=(12,4))
  for method in ['C-fixed','C-low','D-low']:
   rr=[r for r in training if r['order']==order] if method=='C-low' else list(csv.DictReader((BASE/f'seed_{order}'/method/'formal/training_metrics.csv').open()))
   for ax,k in zip(axes,['action_loss','auxiliary_loss','weighted_auxiliary']):ax.plot([int(r['step']) for r in rr],[float(r[k]) for r in rr],label=method,lw=.6);ax.set_title(k)
  axes[0].legend();fig.suptitle(f'Order {order}, raw batch curves, no smoothing');fig.tight_layout();fig.savefig(out/f'curves_order{order}.png');plt.close(fig)
 lines=['# C-low配对补全结果','', '|顺序|方法|Clean|背景1|背景2|背景3|背景4|背景平均|','|---|---|---|---|---|---|---|---|']
 for order in ORDERS:
  for method in ['C-fixed','C-low','D-low']:
   a=arrays[order,method];lines.append(f'|{order}|{method}|'+ '|'.join(f'{int(n)}/20' for n in a.sum(0))+f'|{a[:,1:].mean():.2%}|')
 lines+=['','## 配对比较（百分点）']
 for s in statistics:
  if 'two-order' in s['comparison'] or 'equal mean' in s['comparison']:lines.append(f"- {s['comparison']}: {100*s['difference']:+.2f}；状态聚类95%区间 [{100*s['CI95'][0]:+.2f}, {100*s['CI95'][1]:+.2f}]。")
 lines+=['','## 范围与限制','仅新增两个C-low，各600更新、共200闭环；C-fixed/D-low共400历史匹配结果复用，未重跑。所有原始配置和历史产物哈希保持不变。标签、统计、初始化、模型训练RNG、完整采样索引、动作标签及部署一致；唯一干预为C辅助权重固定减半。AE更新0。','', 'C-low对C-fixed检验降低C权重；D-low对C-low为共同半权重下的新AE表示比较。需分别看两个顺序、Clean绝对能力、四背景及相对Clean变化。不能把Clean下降造成的较小背景降幅称为更鲁棒。两个采样顺序不是完整训练seed；反复使用的development状态不是独立测试，区间不包含任务和完整seed不确定性。','', '完成即停止，不自动搜索权重或追加训练、试验。']
 with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('C_LOW_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
