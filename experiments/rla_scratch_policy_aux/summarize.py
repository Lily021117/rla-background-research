"""All arms/orders retained; paired bootstrap resamples physical states jointly."""
import sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.common import *
import numpy as np
def main():
 dest=ROOT/'delivery';dest.mkdir(exist_ok=False);training=[];validation=[];grad=[];inventory=[];checks=[]
 for order in ORDERS:
  initial=[];samplehash=[]
  for arm in ARMS:
   r=ROOT/f'seed_{order}'/arm/'formal';accept=read(r/'training_acceptance.json');assert accept['successful_updates']==600 and accept['strict_reload']
   ini=read(r/'initialization_audit.json');initial.append((ini['parameter_sha256'],ini['RNG_hash']))
   rows=list(csv.DictReader((r/'training_metrics.csv').open()));assert [int(x['step']) for x in rows]==list(range(1,601))
   assert all(float(x['lambda_RLA'])==weight(arm,int(x['step'])-1) for x in rows)
   training.extend(dict(order=order,arm=arm,**x) for x in rows)
   validation.extend(dict(order=order,arm=arm,**x) for x in csv.DictReader((r/'validation_metrics.csv').open()))
   for p in sorted(r.glob('gradient_step*.json')):grad.extend(dict(order=order,arm=arm,**x) for x in read(p)['statistics'])
   actual=[readline for readline in map(json.loads,(r/'actual_sample_ids.jsonl').read_text().splitlines())]
   expected=[readline for readline in map(json.loads,(OLD/f'seed_{order}/C/formal/actual_sample_ids.jsonl').read_text().splitlines())]
   assert actual==expected
   samplehash.append(sha(r/'actual_sample_ids.jsonl'))
   for step in [0,1,100,300,600]:
    p=r/f'step{step:04}.pt';inventory.append(dict(order=order,arm=arm,step=step,path=str(p),sha256=sha(p),bytes=p.stat().st_size))
  assert len(set(initial))==1 and len(set(samplehash))==1
  checks.append(dict(order=order,all_four_initial_parameters_RNG_equal=True,all600_actual_sample_rows_exact=True,sample_hash=samplehash[0]))
 table(dest/'training_metrics.csv',training);table(dest/'validation_metrics.csv',validation);table(dest/'gradient_statistics.csv',grad);table(dest/'checkpoint_inventory.csv',inventory);save(dest/'matching_audit.json',checks)
 clean=read(ROOT/'evaluation_clean/per_trial_results.json');assert len(clean)==160 and all(r['status']=='COMPLETE' for r in clean)
 gate=read(ROOT/'clean_gate.json');rows=clean+ (read(ROOT/'evaluation_background/per_trial_results.json') if gate['passed'] else [])
 assert len(rows)==(800 if gate['passed'] else 160) and all(r['status']=='COMPLETE' for r in rows)
 for r in rows:
  method,remain=r['model'].split('_order');r['arm']=method;r['order']=int(remain.split('_')[0])
 save(dest/'per_trial_results.json',rows)
 columns=sorted(set().union(*(r.keys() for r in rows)));table(dest/'per_trial_results.csv',[{c:r.get(c) for c in columns} for r in rows])
 conditions=['clean']+([f'new_background_{i}' for i in range(1,5)] if gate['passed'] else []);states=sorted({r['seed'] for r in rows});assert len(states)==20
 rates=[];arrays={};phase=[]
 for order in ORDERS:
  for arm in ARMS:
   group={(r['seed'],r['condition']):r for r in rows if r['arm']==arm and r['order']==order};assert len(group)==20*len(conditions)
   arrays[order,arm]=np.array([[int(group[s,c]['success']) for c in conditions] for s in states])
   for j,c in enumerate(conditions):
    v=arrays[order,arm][:,j];rates.append(dict(order=order,arm=arm,appearance=c,successes=int(v.sum()),trials=20,success_rate=float(v.mean())))
    for stage,n in collections.Counter(group[s,c]['stage'] for s in states).items():phase.append(dict(order=order,arm=arm,appearance=c,stage=stage,count=n))
 table(dest/'success_summary.csv',rates);table(dest/'failure_stage_summary.csv',phase)
 rng=np.random.default_rng(20261029);ix=rng.integers(0,20,size=(20000,20));statistics=[];flips=[]
 def statistic(label,delta):
  v=delta.mean(1) if delta.ndim==2 else delta
  return dict(comparison=label,mean=float(v.mean()),CI95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),unit='20 physical states; all appearances and both orders retained together; excludes complete training-seed/task uncertainty')
 for new,ref in read(ROOT/'protocol.json')['comparisons']:
  diffs=[]
  for order in ORDERS:
   d=arrays[order,new]-arrays[order,ref];diffs.append(d)
   for j,c in enumerate(conditions):
    statistics.append(statistic(f'{order} {new} minus {ref} {c}',d[:,j]))
    a=arrays[order,new][:,j];b=arrays[order,ref][:,j];flips.append(dict(order=order,new=new,reference=ref,appearance=c,new_only=int(((a==1)&(b==0)).sum()),reference_only=int(((a==0)&(b==1)).sum()),both_success=int(((a==1)&(b==1)).sum()),both_fail=int(((a==0)&(b==0)).sum())))
   if gate['passed']:
    statistics.append(statistic(f'{order} {new} minus {ref} four-background equal mean',d[:,1:]))
    statistics.append(statistic(f'{order} {new} minus {ref} background-clean interaction',d[:,1:].mean(1)-d[:,0]))
  d=np.mean(diffs,0)
  statistics.append(statistic(f'two-order equal {new} minus {ref} clean',d[:,0]))
  if gate['passed']:
   statistics.append(statistic(f'two-order equal {new} minus {ref} background',d[:,1:]))
   statistics.append(statistic(f'two-order equal {new} minus {ref} background-clean interaction',d[:,1:].mean(1)-d[:,0]))
 save(dest/'paired_statistics.json',statistics);table(dest/'paired_flips.csv',flips)
 # Fixed offline per-episode summaries, not streaming training-batch loss.
 sums=collections.defaultdict(list)
 for r in validation:sums[r['order'],r['arm'],r['step'],r['episode'],r['appearance']].append(r)
 vm=[]
 for (order,arm,step,e,c),rr in sums.items():vm.append(dict(order=order,arm=arm,step=int(step),episode=e,appearance=c,**{k:float(np.mean([float(x[k]) for x in rr])) for k in ['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']}))
 table(dest/'per_demo_validation_summary.csv',vm)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for order in ORDERS:
  fig,axes=plt.subplots(2,2,figsize=(12,8))
  for arm in ARMS:
   rr=[r for r in training if r['order']==order and r['arm']==arm]
   for ax,k in zip(axes.flat,['action_loss','auxiliary_loss','weighted_auxiliary','lambda_RLA']):ax.plot([int(r['step']) for r in rr],[float(r[k]) for r in rr],label=arm,lw=.7);ax.set_title(k);ax.set_xlabel('successful optimizer updates')
  axes[0,0].legend();fig.suptitle(f'Order {order}: raw training batch curves, no smoothing');fig.tight_layout();fig.savefig(dest/f'training_order{order}.png');plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(11,4))
  for j,c in enumerate(['clean','background']):
   for arm in ARMS:
    xs=[0,100,300,600];ys=[np.mean([r['action_L1'] for r in vm if r['order']==order and r['arm']==arm and r['step']==s and r['appearance']==c]) for s in xs];axes[j].plot(xs,ys,'o-',label=arm)
   axes[j].set_title('fixed development action L1 '+c);axes[j].legend()
  fig.tight_layout();fig.savefig(dest/f'validation_order{order}.png');plt.close(fig)
 protected=read(ROOT/'protected_inputs.json');assert all(sha(p)==h for p,h in protected.items())
 targetmanifest=read(ROOT/'target_manifest.json');assert all(sha(r['path'])==r['sha256'] for r in targetmanifest)
 save(dest/'delivery_verification.json',dict(status='COMPLETE_FIXED_BUDGET_NOT_METHOD_VALIDATION',policy_updates=4800,actual_update_computations=4800,replays=0,AE_updates=0,DINO_updates=0,Clean_valid_trials=160,background_valid_trials=640 if gate['passed'] else 0,actual_attempts=len(rows),invalid_trials=0,background_gate=gate,protected_input_hashes_unchanged=True,all_targets_rehashed=True,sealed_read=False))
 lines=['# 从零训练 AE 的下游策略验证与辅助权重对照','', '|顺序|方法|Clean|'+('背景1|背景2|背景3|背景4|' if gate['passed'] else ''),'|---|---|---|'+('---|---|---|---|' if gate['passed'] else '')]
 for order in ORDERS:
  for arm in ARMS:lines.append('|'+str(order)+'|'+arm+'|'+'|'.join(str(int(x))+'/20' for x in arrays[order,arm].sum(0))+'|')
 lines+=['',f"背景门槛：{'通过，完成全部640次背景评估' if gate['passed'] else '未通过，停止背景阶段；不能判断外观鲁棒性'}。",'', '## 固定条件与边界','新AE-C/D均为共同未训练初始化后正式2000步最终模型；不是step90000微调版。各自训练标签均值及全局尺度重新计算，development不参与拟合。策略八组各600次成功更新、共同历史step0、两个均衡采样顺序。lambda0=0.22408536585365854；D-low固定一半，D-decay从首步lambda0线性到末步0。两者平均系数相同，不等于累计梯度影响相同。','', '两种顺序并非完整训练seed，20个development状态反复用于开发。配对区间按物理状态聚类，不含训练seed/任务不确定性。辅助loss受各自target分布影响，不可直接作质量排名。梯度来自共同可训练模块，未缩放，不是AdamW更新比例。','', '训练/validation曲线、逐维误差、共享梯度和双向成功翻转均保留。总loss或动作L1下降不能替代完整任务成功率。无AE/DINO更新、无教师/keep、无额外一致性或未来重建loss，无封存集读取。','', '## 解释原则','D-fixed与C-fixed比较新表示；D-low与D-fixed比较固定系数；D-decay与D-low比较时间安排。单顺序优势不能称稳定收益。若均无基础能力，应记为当前下游设置未通过，不能自动归因一致性。本轮结束，不自动延长或调参。']
 with (dest/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('SUMMARY_COMPLETE',flush=True)
if __name__=='__main__':main()
