"""All tested intensities retained; no models or simulator loaded."""
from common import *
def main():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=BASE/'delivery';out.mkdir();clean=read(BASE/'clean_reuse.json');states=read(BASE/'state_manifest.json');rows=[];metrics=[];flips=[];intervals=[];counts=[]
 rng=np.random.default_rng(20261102);draws=rng.integers(0,10,(20000,10))
 for stage in ['calibration','combination','review']:
  for task in TASK_NAMES:
   p=BASE/stage/task/'per_trial_results.json'
   if not p.exists():continue
   rr=read(p);assert all(r['status']=='COMPLETE' for r in rr);rows.extend([dict(r,stage=stage) for r in rr])
   counts.append(dict(stage=stage,task=task,**read(BASE/stage/task/'completion.json'),diagnostic_policy_forwards=read(BASE/stage/task/'clean_identity.json')['policy_forwards']))
   ss=[s for s in states if s['task']==task and s['subset']==('review' if stage=='review' else 'calibration')];ids=[s['seed'] for s in ss];cc=read(BASE/stage/'stage_conditions.json')[task]
   a=np.array([[next(r['success'] for r in clean if r['task']==task and r['policy_seed']==seed and r['seed']==state) for state in ids] for seed in SEEDS],dtype=float)
   for condition in cc:
    b=np.array([[next(r['success'] for r in rr if r['policy_seed']==seed and r['seed']==state and r['condition']==condition) for state in ids] for seed in SEEDS],dtype=float)
    for i,seed in enumerate(SEEDS):
     metrics.append(dict(stage=stage,task=task,condition=condition,policy_seed=seed,successes=int(b[i].sum()),denominator=10,rate=float(b[i].mean()),clean_successes=int(a[i].sum()),delta_pp=float((b[i]-a[i]).mean()*100)))
     flips.append(dict(stage=stage,task=task,condition=condition,policy_seed=seed,clean_success_intervention_fail=int(((a[i]==1)&(b[i]==0)).sum()),clean_fail_intervention_success=int(((a[i]==0)&(b[i]==1)).sum()),both_success=int(((a[i]==1)&(b[i]==1)).sum()),both_fail=int(((a[i]==0)&(b[i]==0)).sum())))
    delta=(b-a)[:,draws].mean(axis=(0,2))*100;rates=b[:,draws].mean(axis=(0,2))*100
    intervals.append(dict(stage=stage,task=task,condition=condition,mean_success_pct=float(b.mean()*100),clean_pct=float(a.mean()*100),delta_pp=float((b-a).mean()*100),delta_ci95=np.quantile(delta,[.025,.975]).tolist(),success_ci95=np.quantile(rates,[.025,.975]).tolist(),seed_rates_pct=(b.mean(axis=1)*100).tolist(),seed_sd_pct=float(b.mean(axis=1).std(ddof=1)*100),seed_range_pct=[float(b.mean(axis=1).min()*100),float(b.mean(axis=1).max()*100)]))
 table(out/'per_trial_results.csv',rows);save(out/'per_trial_results.json',rows);table(out/'success_by_seed_condition.csv',metrics);table(out/'paired_flips.csv',flips);save(out/'cluster_intervals.json',intervals)
 for task in TASK_NAMES:
  fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True)
  for ax,family in zip(axs,['blur','noise','crop']):
   cs=[c for c in CONDITIONS if c.startswith(family)];xx=np.arange(3)
   for seed in SEEDS:
    yy=[next((100*r['rate'] for r in metrics if r['task']==task and r['stage']=='calibration' and r['condition']==c and r['policy_seed']==seed),np.nan) for c in cs];ax.plot(xx,yy,'o-',label=str(seed))
   yy=[next((r['mean_success_pct'] for r in intervals if r['task']==task and r['stage']=='calibration' and r['condition']==c),np.nan) for c in cs];ax.plot(xx,yy,'k--',label='equal mean');ax.axhspan(40,60,alpha=.1,color='green');ax.set_xticks(xx,cs,rotation=25);ax.set_ylim(0,105);ax.set_title(family);ax.legend(fontsize=7)
  fig.suptitle(task+' calibration: all intensities');fig.tight_layout();fig.savefig(out/(task+'_strength_curve.png'));plt.close(fig)
 selection=read(BASE/'selected_candidates.json');report='# 冻结C-low观测退化强度校准\n\n这是已使用development状态上的开发难度校准，不是独立测试，不检验D。C-low不是作者论文完整配方复现。全部单因素完成后按预登记优先级选取候选；没有追加搜索。\n\n'
 for task in TASK_NAMES:
  report+=f'## {task}\n\n|阶段|条件|31001|31002|31003|平均%|相对Clean pp|差值状态聚类95%区间|\n|---|---|---:|---:|---:|---:|---:|---|\n'
  for r in intervals:
   if r['task']!=task:continue
   mm=[next(m for m in metrics if all(m[k]==r[k] for k in ['task','stage','condition']) and m['policy_seed']==seed) for seed in SEEDS]
   report+=f"|{r['stage']}|{r['condition']}|"+'|'.join(str(m['successes'])+'/10' for m in mm)+f"|{r['mean_success_pct']:.2f}|{r['delta_pp']:.2f}|{r['delta_ci95']}|\n"
  if task in selection:
   c=selection[task][0];r=next(r for r in intervals if r['task']==task and r['stage']=='review');report+=f"\n选中{c}。本轮复核平均{r['mean_success_pct']:.2f}%；低于60%：{r['mean_success_pct']<60}；落入40%—60%（不含60%）：{40<=r['mean_success_pct']<60}。复核后不返回重新选择。\n"
  else:report+='\n未命中目标档位；没有候选复核，不插值或增加强度。完整结果保留。\n'
 report+='\n## 解释与限制\n\n40%—60%是开发目标，不是总体成功率的保证。每阶段仅10个状态，保留所有seed/条件进行状态聚类重采样；区间不涵盖完整训练seed、AE、噪声实现、任务不确定性。复核状态此前也参与开发，不能称独立验证。没有找到档位也是有效结果。图像可见性检查不证明可学习；有效失败不因评分后观察到目标出画而删除。不能推断普通增强可解决或D有效。\n\nClean复用120次历史匹配结果，没有新增Clean闭环；每阶段相对Clean使用对应10状态子集。原始在线视频记录Clean渲染，扰动在主干归一化前施加，实际输入变换由冻结代码和噪声键完全确定；代表性对比图位于本地image_acceptance。\n'
 totals=dict(new_valid=len(rows),actual_attempts=sum(c['actual_attempts'] for c in counts),invalid=sum(c['invalid'] for c in counts),reused_clean=120,rollout_policy_forwards=sum(c['policy_forwards'] for c in counts),diagnostic_policy_forwards=sum(c['diagnostic_policy_forwards'] for c in counts),optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0)
 assert totals['new_valid']<=660
 save(out/'execution_counts.json',dict(totals=totals,stages=counts));report+='\n## 计数\n\n```json\n'+json.dumps(totals,indent=2)+'\n```\n';(out/'report.md').write_text(report)
if __name__=='__main__':main()
