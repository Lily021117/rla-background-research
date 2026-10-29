"""Read-only aggregation of this experiment; no model import or GPU use."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data/lyc/projects/rla-wm/experiments/rla_ae_from_scratch/run_20261029_01')
O=R/'delivery';O.mkdir(exist_ok=False)
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
 return h.hexdigest()
def write(name,rows):
 with (O/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plan=json.loads((R/'frozen_plan.json').read_text());agg=[];reps=[];epvalues={};invent=[]
for arm in ['C','D']:
 completion=json.loads((R/arm/'completion.json').read_text());assert completion['successful_updates']==2000
 for p in sorted((R/arm).glob('evaluation_step*.csv')):
  step=int(p.stem.removeprefix('evaluation_step'));groups=defaultdict(list)
  for row in read(p):groups[(row['task'],row['split'],row['appearance'],row['decoder_z'])].append(row)
  for (task,split,app,z),rows in groups.items():
   per=defaultdict(list)
   for r in rows:per[r['episode']].append(r)
   metrics={k:float(np.mean([np.mean([float(v[k]) for v in rr]) for rr in per.values()])) for k in ['l1','mse','rec','copy_current_l1','copy_current_mse']}
   agg.append(dict(arm=arm,step=step,task=task,split=split,appearance=app,z=z,episodes=len(per),**metrics))
   if step==2000 and split=='validation':
    for ep,rr in per.items():epvalues[arm,task,ep,app,z]=float(np.mean([float(r['rec']) for r in rr]))
 for p in sorted((R/arm).glob('representation_step*.csv')):
  step=int(p.stem.removeprefix('representation_step'))
  for r in read(p):reps.append(dict(arm=arm,step=step,**r))
 for p in sorted((R/arm).glob('step*.pt')):invent.append(dict(arm=arm,step=int(p.stem.removeprefix('step')),path=str(p),sha256=sha(p),bytes=p.stat().st_size))
write('reconstruction_summary.csv',agg);write('representation_summary.csv',reps);write('checkpoint_inventory.csv',invent)
rg=np.random.default_rng(20261029);paired=[]
for task in ['cookie','drawer_bowl','stove']:
 eps=sorted(set(k[2] for k in epvalues if k[1]==task))
 for app in ['clean','background']:
  d=np.array([epvalues['D',task,ep,app,'real']-epvalues['C',task,ep,app,'real'] for ep in eps]);boot=d[rg.integers(0,len(d),(10000,len(d)))].mean(1)
  paired.append(dict(task=task,appearance=app,metric='D_minus_C_reconstruction_L1_plus_MSE',episodes=len(d),mean=float(d.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),scope='paired episode bootstrap; h and cameras retained together; no AE seed/task uncertainty'))
write('paired_episode_statistics.csv',paired)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for arm,col in [('C','#2271b5'),('D','#d14936')]:
 rows=read(R/arm/'training_metrics.csv');x=np.array([int(r['step']) for r in rows])
 for ax,key in zip(axes.flat,['reconstruction','consistency','weighted_consistency','gradient_encoder']):
  y=np.array([float(r[key]) for r in rows]);ax.plot(x,y,color=col,alpha=.13,lw=.5);ax.plot(x,[y[max(0,i-49):i+1].mean() for i in range(len(y))],label=arm,color=col)
  ax.set(title=key,xlabel='Successful update');ax.legend();ax.grid(alpha=.2)
fig.suptitle('From-untrained AE: raw steps and trailing50 mean');fig.tight_layout();fig.savefig(O/'training_curves.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,task in zip(axes,['cookie','drawer_bowl','stove']):
 for arm,col in [('C','#2271b5'),('D','#d14936')]:
  for app,style in [('clean','-'),('background','--')]:
   rows=[r for r in agg if r['task']==task and r['split']=='validation' and r['arm']==arm and r['appearance']==app and r['z']=='real']
   ax.plot([r['step'] for r in rows],[r['rec'] for r in rows],style,marker='o',label=arm+' '+app,color=col)
 ax.set(title=task,xlabel='Checkpoint step',ylabel='Development L1+MSE');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'development_curves.png',dpi=160);plt.close(fig)
protected=json.loads((R/'protected_inputs.json').read_text());unchanged={p:sha(Path(p))==v for p,v in protected.items()};assert all(unchanged.values())
summary=dict(status='COMPLETED_FEASIBILITY_NOT_CONTROL_VALIDATION',formal_updates={'C':2000,'D':2000},pilot_updates=60,policy_updates=0,DINO_updates=0,rollouts=0,replayed_updates=0,history_protection=unchanged,lambda_cons=plan['lambda_cons'],initialization='official constructors, no pretrained AE loaded',paired_statistics=paired,limits=['three fixed bowl pick-place tasks','60 train15 reused exploratory dev episodes','one common initialization seed','small bounded budget relative to official pretraining','no policy evidence'])
(O/'delivery_verification.json').write_text(json.dumps(summary,indent=2))
lines=['# 共同未训练 AE 初始化：C/D 有限预算可行性实验','',
 '与历史官方step90000+600步微调分开。Encoder和feature decoder由官方构造器创建；DINO仅复用冻结预训练特征。',
 f"共同初始化seed={plan['seed']}；正式各2000次更新；独立重建校准60次。lambda_cons={plan['lambda_cons']:.10g}，在训练集固定样本及预实验10/30/60步确定，未使用development调权。",'',
 '## 数据与范围','', 'Cookie、抽屉取碗、炉灶三个任务，各20train+5development，共60/15；未读取封存集，无新渲染、轨迹或策略。训练窗口7231，h1..8共57848转移；每组16000物理样本、32000外观曝光。不是完整官方四套件预训练，不能称已收敛。','',
 '## 最终固定development重建（按episode等权）','', '|任务|组|外观|L1|MSE|和|打乱z的和|零z的和|复制当前的和|','|---|---|---|---:|---:|---:|---:|---:|---:|']
for r in agg:
 if r['step']==2000 and r['split']=='validation' and r['z']=='real':
  def other(kind):return next(x['rec'] for x in agg if x['step']==2000 and x['split']=='validation' and x['task']==r['task'] and x['arm']==r['arm'] and x['appearance']==r['appearance'] and x['z']==kind)
  lines.append(f"|{r['task']}|{r['arm']}|{r['appearance']}|{r['l1']:.6f}|{r['mse']:.6f}|{r['rec']:.6f}|{other('shuffled'):.6f}|{other('zero'):.6f}|{r['copy_current_l1']+r['copy_current_mse']:.6f}|")
lines+=['','完整逐相机/跨度/示范数据在各组evaluation CSV；latent范数、方差、有效秩、归一化距离及余弦在representation_summary.csv。有效秩受固定样本数上限限制。真实z优于打乱/零z只说明z被decoder使用，不证明控制价值。','',
 '协议偏离：保持官方结构/初始化及原重建定义，使用已有配对缓存、h1..8、有效物理batch8而非官方128；mixedFP16+FP32 master固定loss-scale256而非动态增长；无scheduler，AdamW1e-4/WD0；未加载图像RGB decoder。','',
 '历史保护输入哈希全部复核一致。无策略训练、闭环或新数据。正式检查点固定，不按development选模；本轮不自动追加训练或调整权重。']
(O/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
