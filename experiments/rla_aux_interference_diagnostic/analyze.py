"""All batches retained; descriptive statistics, not independent-seed inference."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1]);worker=out/'C'/(sys.argv[2] if len(sys.argv)>2 else 'diagnostic_scope_corrected')
def save(n,x):
 with (out/n).open('x') as f:json.dump(x,f,indent=2)
def csvout(n,df):
 assert not (out/n).exists()
 if isinstance(df.columns,pd.MultiIndex):df=df.copy();df.columns=['_'.join(str(x) for x in c if x!='') for c in df.columns]
 df.to_csv(out/n,index=False)
valid=pd.read_csv(out/'historical_validation_samples.csv');paired=pd.read_csv(out/'paired_validation_samples.csv');train=pd.read_csv(out/'historical_training_curves.csv')
metrics=['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']
csvout('validation_summary.csv',valid.groupby(['order','method','step','appearance'])[metrics].agg(['mean','std','count']).reset_index())
perdemo=valid.groupby(['order','method','step','episode','appearance'],as_index=False)[metrics].mean();csvout('per_demo_validation.csv',perdemo)
csvout('time_third_validation.csv',valid.groupby(['order','method','step','appearance','time_third'],as_index=False)[metrics].mean())
diffcols=[x+'_difference' for x in metrics]
diff=paired.groupby(['order','step','appearance'],as_index=False)[diffcols].mean();csvout('paired_validation_summary.csv',diff)
earliest=[]
for order in (20261101,20261102):
 for metric in metrics:
  q=paired[paired.order==order].groupby('step')[metric+'_difference'].agg(['mean','std',lambda x:float(np.max(np.abs(x)))])
  q.columns=['mean','std','max_abs'];nonzero=q[q.max_abs>0]
  earliest.append(dict(order=order,metric=metric,first_observed_nonzero_step=int(nonzero.index[0]) if len(nonzero) else None,available_steps=list(map(int,q.index)),means={str(k):float(v) for k,v in q['mean'].items()},interpretation='first observed offline difference; no step1 validation, not a closed-loop collapse time'))
save('earliest_observed_offline_difference.json',earliest)
(out/'figures').mkdir(exist_ok=False)
for order in (20261101,20261102):
 fig,ax=plt.subplots(2,3,figsize=(14,7))
 for method in ('C','shared'):
  q=train[(train.order==order)&(train.method==method)]
  for a,m in zip(ax.flat,['action_loss','auxiliary_loss','weighted_auxiliary','gradient_vla','gradient_head','gradient_proprio']):a.plot(q.step,q[m],label=method,alpha=.7,lw=.8);a.set_title(m);a.set_xlabel('successful update');a.grid(alpha=.2)
 ax[0,0].legend();fig.suptitle(f'Order {order}: raw instantaneous training curves (no smoothing)');fig.tight_layout();fig.savefig(out/'figures'/f'training_{order}.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(10,7))
 for method in ('C','shared'):
  for app in ('clean','background'):
   q=valid[(valid.order==order)&(valid.method==method)&(valid.appearance==app)].groupby('step')[metrics].mean()
   for a,m in zip(ax.flat,metrics[:4]):a.plot(q.index,q[m],marker='o',label=f'{method}/{app}');a.set_title(m);a.grid(alpha=.2)
 ax[0,0].legend();fig.suptitle(f'Order {order}: fixed development samples');fig.tight_layout();fig.savefig(out/'figures'/f'validation_{order}.png',dpi=150);plt.close(fig)
if not (worker/'diagnostic_acceptance.json').exists():
 print('HISTORICAL_ANALYSIS_COMPLETE_GPU_DIAGNOSTICS_PENDING');sys.exit(0)
g=pd.read_csv(worker/'gradient_statistics.csv')
old=pd.read_csv(out/'C/diagnostic_retry1/gradient_statistics.csv')
csvout('earlier_valid_VLM_proprio_gradients.csv',old[old.module.isin(['vla','proprio'])])
csvout('superseded_mixed_head_statistics.csv',old[old.module.isin(['ALL_SHARED','head'])])
ident=['snapshot','snapshot_method','snapshot_order','step','batch','batch_order','module']
aa=g[g.comparison=='action_vs_aux_C'].set_index(ident)
bb=g[g.comparison=='action_vs_aux_shared'].set_index(ident)
cc=g[g.comparison=='aux_C_vs_aux_shared'].set_index(ident)
switch=aa[['action_norm','action_loss','aux_C_loss','aux_shared_loss','weighted_aux_C','weighted_aux_shared']].copy()
for metric in ['aux_norm','norm_ratio','cosine','projection_coefficient','projection_on_action_unit']:
 switch[metric+'_C']=aa[metric];switch[metric+'_shared']=bb[metric];switch[metric+'_delta']=bb[metric]-aa[metric]
switch['aux_switch_cosine']=cc.cosine;switch['aux_switch_norm_ratio']=cc.norm_ratio
switch=switch.reset_index();csvout('target_switch_per_batch.csv',switch)
numeric=['cosine_C','cosine_shared','cosine_delta','norm_ratio_C','norm_ratio_shared','norm_ratio_delta','projection_coefficient_C','projection_coefficient_shared','projection_coefficient_delta','aux_switch_cosine','aux_switch_norm_ratio']
summary=switch.groupby(['snapshot','snapshot_method','snapshot_order','step','batch_order','module'])[numeric].agg(['mean','std','min','max']).reset_index()
csvout('target_switch_summary.csv',summary)
overall=switch[switch.module=='ALL_SHARED'].groupby(['snapshot','snapshot_method','snapshot_order','step'],as_index=False)[numeric].mean();csvout('snapshot_summary.csv',overall)
fig,ax=plt.subplots(1,3,figsize=(16,5))
for a,metric in zip(ax,['cosine_delta','norm_ratio_delta','aux_switch_cosine']):
 q=switch[switch.module=='ALL_SHARED'];names=list(dict.fromkeys(q.snapshot));a.boxplot([q[q.snapshot==n][metric].dropna() for n in names],labels=names);a.tick_params(axis='x',rotation=90,labelsize=7);a.set_title(metric);a.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'figures/target_switch.png',dpi=150);plt.close(fig)
accept=json.loads((worker/'diagnostic_acceptance.json').read_text())
lines=['# 冻结策略的辅助监督—动作学习机制诊断','', '状态：PARTIAL（原计划的严格共享梯度覆盖未完整完成）；已完成的修正范围计算及配对检查通过。','',
 '本轮保持原有两种实现停止调参、负结果归档的决定。无 optimizer 更新、AE 更新或新增闭环。','',
 '## 设计与证据范围','',
 '从已核验提交 b404cde 新建分支。20 个文件 checkpoint 均存在，四个 step0 的参数与完整保存RNG相同，原冻结清单去重后13份。step1只核验身份与读取已有日志。',
 '每个采样顺序固定第1、200、400、600次真实成功更新的完整batch；共8batch，均16个外观样本、两个8样本微批。C与shared全600步实际样本ID逐条相同。选择只使用元数据。',
 '首个计算尝试完成73单元后发现head范围包含仅动作路径的layer_norm1/fc1；单向attention mask阻断了它们对aux的影响。已停止该尝试；其VLM/proprio分项有效，但混合head/global指标仅保留审计，不作为主结论。第74单元中断时逐目标计数未持久化，保守计入整个3目标单元预算，不能声称精确实际调用数已全部恢复。',
 '为保持原318个有效batch目标提取上限，修正后仅计算四个最终600快照×原清单前7batch，以及共同step0×两个顺序的首batch，共30单元。这是实现范围修正导致的覆盖不足，不能声称原13×8全部完成。截断按原清单前缀，未按loss/梯度/成功率筛选。100/300步完整共享head对比与final第8batch缺失；不自动追加计算补齐。',
 '训练模式、BF16、Math SDPA、原activation checkpointing与λ_aux=0.22408536585365854。每个目标重置共同step0的Python/NumPy/CPU/CUDA RNG；包含随机动作token补齐和action/RLA训练噪声；重计算保留RNG。',
 '动作loss与辅助nn.L1Loss直接从正式官方wrapper计算图捕获，各微批/2，保持正式累积语义。修正范围排除action/RLA最终readout及action-only layer_norm1/fc1；报告VLM、proprio、共享MLP blocks及合并范围。无梯度参数在unit审计逐项列出，零梯度保留；接近零范数标记不可解释。共享是参数张量级定义，不声称查询位置专用行逐元素完全相同依赖。',
 '原两个额外验收单元检查共享target下动作梯度逐元素一致、正式总梯度与分量和，涵盖修正后的参数子集。原宽范围BF16总梯度分解相对L2为0.3689%–0.3798%，事前容差≤2%；未另加窄范围总梯度提取。不能将原始梯度解释为AdamW更新方向。','',
 '## 已有离线曲线','',
 '开发日志按episode/timestep/appearance逐行配对，共11248行。节点0/100/300/600，step1没有固定验证。训练即时loss与固定验证曲线分开绘制；不补造历史节点。已有训练即时日志无平移/旋转/夹爪逐步字段，分项来自固定验证。',
 '按有效窗口时间三等分，不称为抓取/抬升/放置阶段。没有本轮可复用的精确示范阶段标签。不同target的辅助loss只表示各自拟合情况。','',
 'D-center仅复制已有日志作旁证，没有加载其权重或新增其前向/梯度。','', '逐采样顺序固定开发集均值（shared−C，归一化动作空间）：','',
 '|顺序|步数|外观|动作L1差|平移差|旋转差|夹爪差|','|---|---:|---|---:|---:|---:|---:|']
for _,r in diff.iterrows():lines.append(f"|{int(r.order)}|{int(r.step)}|{r.appearance}|{r.action_L1_difference:.6f}|{r.translation_L1_difference:.6f}|{r.rotation_L1_difference:.6f}|{r.gripper_L1_difference:.6f}|")
lines+=['','## 同参数 target 切换结果','','下表为修正范围：step0使用2batch，四个final各用相同7batch。batch不是训练seed。两个batch来源顺序的分表、所有模块和逐batch结果均保存。','','|参数快照|动作/原aux余弦|动作/shared aux余弦|原aux/action范数比|shared aux/action范数比|两aux余弦|','|---|---:|---:|---:|---:|---:|']
for _,r in overall.iterrows():lines.append(f'|{r.snapshot}|{r.cosine_C:.5f}|{r.cosine_shared:.5f}|{r.norm_ratio_C:.5f}|{r.norm_ratio_shared:.5f}|{r.aux_switch_cosine:.5f}|')
lines+=['','## 计算与保护','',f"修正尝试实际计数：{json.dumps(accept['counters'],ensure_ascii=False)}",f"修正尝试耗时{accept['seconds']:.1f}秒；GPU峰值allocated={accept['peak_allocated']/1024**3:.2f}GiB。",'累计确认315个完整有效batch目标提取、630个微批前向与反向；另有一个中断单元，至少一次且最多6次微批前向/反向，其精确数未持久化。累计上界318个目标提取、636个微批前向/反向，包含验收6个目标提取，未超授权上限。保留启动语法错误（模型前向/反向0）和一次范围修正中断；基础设施故障重试0。','完整梯度仅在内存中用于归约，未保存/上传。完成的快照保护检查通过；被中断快照未做完整post-check，应以历史输入哈希和隔离进程无更新路径作为保护证据。修正尝试全部参数快照前后可训练参数哈希一致，registered buffers逐元素一致，完整RNG恢复。训练checkpoint与优化器历史只读。','', '## 限制','', '这是单任务、两个采样排列、固定训练batch的机制诊断，且因统计范围实现错误而未完成原计划覆盖。负余弦不证明根因；最终参数上的异常可能是训练结果；有限节点不能定位闭环崩溃时刻。development已反复使用，不是独立测试。不新增探针、损失、梯度投影、任务或背景。','', '## 最终机制判断与唯一后续建议','', '待结合全部逐batch和已有轨迹证据完成审阅。']
with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('ANALYSIS_COMPLETE')
