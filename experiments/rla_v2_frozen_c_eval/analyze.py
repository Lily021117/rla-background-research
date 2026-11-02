"""Task-separated paired summaries; no model/simulator imports."""
from common import *
def main():
 out=ROOT/'delivery';out.mkdir(exist_ok=False);rows=read(ROOT/'clean_reuse.json')
 for task in ['stove','drawer']:
  new=read(ROOT/task/'per_trial_results.json');assert len(new)==240 and all(r['status']=='COMPLETE' for r in new);rows+=new
 assert len(rows)==600
 table(out/'per_trial_results.csv',rows);save(out/'per_trial_results.json',rows)
 success=[];contrasts=[];flips=[];intervals=[];conditions=['clean']+CONDITIONS
 rng=np.random.default_rng(20261102);draws=rng.integers(0,20,size=(20000,20))
 for task in ['stove','drawer']:
  stateids=[s['seed'] for s in read(ROOT/'state_manifest.json') if s['task']==task];arr=np.zeros((3,20,5))
  for si,seed in enumerate(SEEDS):
   for ci,c in enumerate(conditions):
    rr=[r for r in rows if r['task']==task and r['policy_seed']==seed and r['condition']==c];assert len(rr)==20
    for ti,state in enumerate(stateids):arr[si,ti,ci]=next(r['success'] for r in rr if r['seed']==state)
    success.append(dict(task=task,policy_seed=seed,condition=c,successes=int(arr[si,:,ci].sum()),denominator=20,success_rate=float(arr[si,:,ci].mean()),reused=c=='clean'))
   for label,idx in [('all_four',[1,2,3,4]),('object_mean',[1,2]),('light_mean',[3,4])]+[(c,[i]) for i,c in enumerate(conditions[1:],1)]:
    diff=arr[si][:,idx].mean(axis=1)-arr[si,:,0];contrasts.append(dict(task=task,policy_seed=seed,contrast=label+'-Clean',delta_pp=float(100*diff.mean())))
   for ci,c in enumerate(conditions[1:],1):flips.append(dict(task=task,policy_seed=seed,condition=c,clean_success_variant_fail=int(((arr[si,:,0]==1)&(arr[si,:,ci]==0)).sum()),clean_fail_variant_success=int(((arr[si,:,0]==0)&(arr[si,:,ci]==1)).sum()),both_success=int(((arr[si,:,0]==1)&(arr[si,:,ci]==1)).sum()),both_fail=int(((arr[si,:,0]==0)&(arr[si,:,ci]==0)).sum())))
  for label,idx in [('all_four',[1,2,3,4]),('object_mean',[1,2]),('light_mean',[3,4])]+[(c,[i]) for i,c in enumerate(conditions[1:],1)]:
   diff=arr[:,:,idx].mean(axis=2)-arr[:,:,0];bs=diff[:,draws].mean(axis=(0,2))*100;intervals.append(dict(task=task,contrast=label+'-Clean',delta_pp=float(diff.mean()*100),ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),seed_delta_pp=(diff.mean(axis=1)*100).tolist(),cluster='physical state; all seeds/conditions retained',n=20))
 table(out/'success_by_seed_condition.csv',success);table(out/'paired_flips.csv',flips);table(out/'seed_contrasts.csv',contrasts);save(out/'cluster_intervals.json',intervals)
 report='# 冻结C策略：V2对象外观与静态光照development评估\n\n仅检验四种干预对当前冻结C-low的影响，不检验增强训练能否解决或D是否有效。炉灶和开中层抽屉分别分析；后者不是抽屉取碗。原有负结果保持不变。\n\n'
 for task in ['stove','drawer']:
  report+=f'## {task}\n\n|seed|Clean（复用）|对象1|对象2|光强|光向|\n|---|---:|---:|---:|---:|---:|\n'
  for seed in SEEDS:report+='|'+str(seed)+'|'+'|'.join(str(next(r['successes'] for r in success if r['task']==task and r['policy_seed']==seed and r['condition']==c))+'/20' for c in conditions)+'|\n'
  report+='\n|干预−Clean|三seed等权差（pp）|状态聚类95%区间|各seed差（pp）|\n|---|---:|---|---|\n'
  for r in intervals:
   if r['task']==task:report+=f"|{r['contrast']}|{r['delta_pp']:.2f}|[{r['ci_low']:.2f}, {r['ci_high']:.2f}]|{r['seed_delta_pp']}|\n"
 report+='\n## 身份、计数与边界\n\n新增480次有效闭环；历史同checkpoint/初态/Clean XML/语言/归一化/部署结果复用120次，未重跑。每任务20个已用development状态，不是新测试。四条件来自V2 revision2原配置，没有根据结果调整；抽屉对象2较弱仍保留。策略输入为在线双相机和同步机器人FK proprio，不播放示范图像、不逐帧恢复状态、不执行示范动作。\n\n源XML变体按对象材质名称及body映射，冻结资产哈希；ControlEnv reset后验证配置生效，闭环中监测材质/灯光不变。炉灶220、抽屉300控制步，20Hz，chunk8，Math SDPA/BF16，官方动作后处理和成功判据不变。\n\n区间以任务内20个物理状态配对重采样，保留全部seed和条件；不包括完整训练seed、AE、干预抽样或任务不确定性。区间跨0不是等效证据。对象/光照均值及每条件完整列出，没有事后挑最差条件或设10pp/三seed同向门槛。\n\n训练、AE、DINO更新和backward均为0。任何观察到的下降只说明当前冻结策略受影响，不能据此宣称C-aug能解决或D-aug有增量。历史纹理/新状态结果不并入本轮主要统计。\n'
 counts={t:read(ROOT/t/'completion.json') for t in ['stove','drawer']};report+='\n执行计数：\n\n```json\n'+json.dumps(counts,indent=2)+'\n```\n';(out/'report.md').write_text(report)
 save(out/'delivery_verification.json',dict(new_valid=480,reused_clean=120,actual_attempts=sum(r['actual_attempts'] for r in counts.values()),invalid=sum(r['invalid'] for r in counts.values()),policy_forwards=sum(r['policy_forwards'] for r in counts.values()),optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0))
if __name__=='__main__':main()
