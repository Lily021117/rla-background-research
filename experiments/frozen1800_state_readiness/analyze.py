"""Pre-scoring fixed state-cluster analysis; no development pooling."""
import argparse,csv,json,collections
from pathlib import Path
import numpy as np
def read(p):return json.loads(p.read_text())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def table(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main(root):
 protocol=read(root/'evaluation/evaluation_protocol.json');rows=read(root/'evaluation/per_trial_results.json');N=len(protocol['states']);assert len(rows)==30*N and all(r['status']=='COMPLETE' for r in rows)
 states=protocol['states'];conds=protocol['conditions'];seeds=[31001,31002,31003];arms=['C-low','D-low'];arrays={};counts=[];summary=[];flips=[];fail=[];phases=[]
 for seed in seeds:
  for arm in arms:
   rr={(r['seed'],r['condition']):r for r in rows if r['model']==f'{arm}_seed{seed}_s1800'};assert len(rr)==N*5
   a=np.array([[int(rr[s,c]['success']) for c in conds] for s in states]);arrays[seed,arm]=a
   for j,c in enumerate(conds):
    counts.append(dict(seed=seed,method=arm,condition=c,successes=int(a[:,j].sum()),trials=N,rate=float(a[:,j].mean())))
    for k,n in collections.Counter(rr[s,c]['stage'] for s in states).items():phases.append(dict(seed=seed,method=arm,condition=c,stage=k,count=n))
   summary.append(dict(seed=seed,method=arm,clean=float(a[:,0].mean()),background=float(a[:,1:].mean()),background_minus_clean=float(a[:,1:].mean()-a[:,0].mean())))
   fail.extend(dict(seed=seed,method=arm,state=s,condition='clean',stage=rr[s,'clean']['stage']) for s in states if not rr[s,'clean']['success'])
 rng=np.random.default_rng(31001);idx=rng.integers(0,N,(20000,N));paired=[];diffs=[]
 def interval(v):return np.quantile(v[idx].mean(1),[.025,.975]).tolist()
 for seed in seeds:
  c,d=arrays[seed,'C-low'],arrays[seed,'D-low'];delta=d-c;diffs.append(delta)
  for j,condition in enumerate(conds):
   flips.append(dict(seed=seed,condition=condition,D_success_C_failure=int(((d[:,j]==1)&(c[:,j]==0)).sum()),C_success_D_failure=int(((c[:,j]==1)&(d[:,j]==0)).sum()),both_success=int(((c[:,j]==1)&(d[:,j]==1)).sum()),both_failure=int(((c[:,j]==0)&(d[:,j]==0)).sum())))
  for key,v in [('clean',delta[:,0]),('background',delta[:,1:].mean(1)),('interaction',delta[:,1:].mean(1)-delta[:,0])]+[(condition,delta[:,j]) for j,condition in enumerate(conds[1:],1)]:
   paired.append(dict(seed=seed,metric=key,difference=float(v.mean()),CI95=interval(v)))
 delta=np.mean(diffs,0)
 for key,v in [('clean',delta[:,0]),('background',delta[:,1:].mean(1)),('interaction',delta[:,1:].mean(1)-delta[:,0])]+[(condition,delta[:,j]) for j,condition in enumerate(conds[1:],1)]:
  paired.append(dict(seed='equal_three_seed',metric=key,difference=float(v.mean()),CI95=interval(v)))
 variability=[]
 for arm in arms+['D-minus-C']:
  for key in ['clean','background','background_minus_clean']:
   if arm in arms:v=[r[key] for r in summary if r['method']==arm]
   else:v=[next(r[key] for r in summary if r['seed']==s and r['method']=='D-low')-next(r[key] for r in summary if r['seed']==s and r['method']=='C-low') for s in seeds]
   variability.append(dict(method=arm,metric=key,mean=float(np.mean(v)),sample_SD=float(np.std(v,ddof=1)),range=[min(v),max(v)],per_seed=v))
 out=root/'delivery';out.mkdir()
 table(out/'success_summary.csv',counts);table(out/'per_seed_metrics.csv',summary);table(out/'paired_flips.csv',flips);table(out/'failure_stage_summary.csv',phases)
 save(out/'clean_failure_states.json',fail);save(out/'paired_statistics.json',dict(results=paired,bootstrap='physical-state cluster,20k draws,all appearances/models/seeds together; not training-seed/AE/task uncertainty'));save(out/'seed_variability.json',variability)
 table(out/'per_trial_summary.csv',[{k:r[k] for k in ['model','seed','condition','checkpoint_sha256','state_sha256','xml_sha256','status','success','steps','ending','stage','seconds']} for r in rows])
 lines=['# 冻结1800步策略：新生成仿真状态验证','',f'按冻结初始化分布新生成的仿真状态验证；N={N}，有效闭环={len(rows)}。并非官方封存示范测试；四背景已用于development，不称为新外观泛化。','', '|seed|method|Clean|背景1|背景2|背景3|背景4|背景平均|','|---|---|---|---|---|---|---|---|']
 for s in seeds:
  for arm in arms:
   a=arrays[s,arm];lines.append(f'|{s}|{arm}|'+ '|'.join(f'{x}/{N}' for x in a.sum(0))+f'|{a[:,1:].mean():.2%}|')
 lines+=['','## 配对主要结果']
 for x in paired:
  if x['metric'] in ['clean','background','interaction']:lines.append(f"- {x['seed']} {x['metric']} D−C={100*x['difference']:+.2f}pp；状态聚类95%CI={np.asarray(x['CI95'])*100}pp。")
 bg=[r for r in paired if r['metric']=='background' and r['seed']!='equal_three_seed'];cl=[r for r in paired if r['metric']=='clean' and r['seed']!='equal_three_seed']
 lines+=['','## 解释与边界',f"各seed背景差异方向/幅度：{[round(x['difference']*100,2) for x in bg]}pp；Clean差异：{[round(x['difference']*100,2) for x in cl]}pp。所有seed和四背景均展示，不选择赢家。",
  '正的背景交互不单独证明鲁棒性：必须同时看绝对背景成功率与Clean代价。Clean区间跨零不证明等效。每套背景的贡献、双向翻转和Clean具体失败状态见附表。',
  '本项目新生成状态未用于训练或调参；与已知训练/development初态做物理签名去重，但无法保证与全部上游预训练来源独立。固定一对AE、一个任务、三个策略seed；区间只反映当前模型下评估状态的不确定性。',
  '来源审计读取过候选封存内容，已登记；这些旧内容没有进入本轮评分。生成v1在未落稳时检查接触谓词；v2采用官方10步no-op稳定等待但仍要求所有桌面小区域On谓词成立；v3保持原采样器不变，采用任务碗支持关系及桌面实际接触验收，并记录其余区域谓词。所有修订均在策略评分前，保留69个先前候选尝试，合计预算不超过500。',
  '不与历史development合并，不追加训练或再次评分。若优势未保留，停止针对这批数据调参；若保留，只建议未来独立AE初始化及跨任务验证，不自动执行。']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 save(out/'delivery_verification.json',dict(status='COMPLETE',N=N,valid_rollouts=len(rows),attempts=len(rows),invalid=0,optimizer_updates=0,AE_updates=0,backward=0,sealed_contents_read_for_audit=True,sealed_old_states_scored=0,new_state_set_only=True))
 print('ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
