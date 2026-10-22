import csv,json,hashlib,subprocess,math,collections
from pathlib import Path
import numpy as np
W=Path(__file__).parent;E=Path('${RLA_PROJECT_ROOT}/experiments/rla_background_invariance');O=E/'step5_8_closed_loop/run_20261022_01'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def cs(n,rows):
 with (O/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
protocol=json.loads((O/'evaluation_protocol.json').read_text());manifest=list(csv.DictReader(open(O/'evaluation_manifest.csv')));states=json.loads((O/'state_manifest.json').read_text());results=[json.loads((O/'trials'/r['trial_id']/'result.json').read_text()) for r in manifest if (O/'trials'/r['trial_id']/'result.json').exists()]
assert len(results)==160 and all(r['status']=='COMPLETE' for r in results),'Incomplete/invalid: no full paired conclusions'
accept=json.loads((O/'rollout_acceptance.json').read_text());assert accept['status']=='PASS'
frozen=json.loads((O/'manifest_frozen.json').read_text());assert sha(O/'evaluation_manifest.csv')==frozen['manifest_SHA256'] and sha(O/'evaluation_protocol.json')==frozen['protocol_SHA256']
byid={r['trial_id']:r for r in results};assert len(byid)==160
arr=np.zeros((20,4,2),dtype=float);paired=[];taskmap={};apps=['clean','background']
for i,s in enumerate(states):
 row={'state_id':s['state_id'],'task':s['task'],'episode':s['episode'],'seed':s['seed'],'state_use':'exploratory_reused_states'};taskmap.setdefault(s['task'],[]).append(i)
 for j,a in enumerate('ABCD'):
  for k,c in enumerate(apps):
   r=byid[f'{a}_{int(s["state_id"]):02d}_{c}']
   for key in ['task','source_hdf5','source_demo','initial_state_SHA256','clean_xml','background_xml']:assert r[key]==s[key]
   assert int(r['seed'])==int(s['seed']) and r['initial_geometry_passed']
   arr[i,j,k]=r['success'];row[a+'_'+c]=int(r['success'])
 paired.append(row)
cs('paired_state_results.csv',paired)
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]
summary={};taskrows=[]
for j,a in enumerate('ABCD'):
 summary[a]={c:{'successes':int(arr[:,j,k].sum()),'trials':20,'SR':float(arr[:,j,k].mean()),'Wilson95_descriptive':wilson(int(arr[:,j,k].sum()),20)} for k,c in enumerate(apps)}
 summary[a]['background_minus_clean']=float((arr[:,j,1]-arr[:,j,0]).mean())
 for t,ids in taskmap.items():
  for k,c in enumerate(apps):
   success=int(arr[ids,j,k].sum());taskrows.append({'model':a,'task':t,'appearance':c,'successes':success,'trials':5,'SR':success/5,'raw_count':f'{success}/5','floor_exact':success==0,'ceiling_exact':success==5})
cs('task_stratified_metrics.csv',taskrows);js('success_rate_summary.json',summary)
rng=np.random.default_rng(20261022);boot_indices=np.concatenate([rng.choice(ids,size=(10000,len(ids)),replace=True) for ids in taskmap.values()],axis=1);boot=arr[boot_indices].mean(axis=1)
statistics={};flips=[]
for left,right in [('B','A'),('C','B'),('D','C')]:
 l='ABCD'.index(left);r='ABCD'.index(right);key=left+'-'+right;statistics[key]={}
 for k,c in enumerate(apps):
  values=arr[:,l,k]-arr[:,r,k];bv=boot[:,l,k]-boot[:,r,k];up=int((values==1).sum());down=int((values==-1).sum());n=up+down
  p=min(1.,2*sum(math.comb(n,j) for j in range(min(up,down)+1))/2**n) if n else 1.
  statistics[key][c]={'difference':float(values.mean()),'task_stratified_bootstrap_95':np.quantile(bv,[.025,.975]).tolist(),'paired_improvements':up,'paired_regressions':down,'both_success':int(((arr[:,l,k]==1)&(arr[:,r,k]==1)).sum()),'both_failure':int(((arr[:,l,k]==0)&(arr[:,r,k]==0)).sum()),'exact_McNemar_descriptive_p':p}
  flips.append({'contrast':key,'appearance':c,**statistics[key][c]})
 iv=(arr[:,l,1]-arr[:,l,0])-(arr[:,r,1]-arr[:,r,0]);bi=(boot[:,l,1]-boot[:,l,0])-(boot[:,r,1]-boot[:,r,0]);statistics[key]['interaction']={'difference':float(iv.mean()),'task_stratified_bootstrap_95':np.quantile(bi,[.025,.975]).tolist()}
js('interaction_statistics.json',{'contrasts':statistics,'replicates':10000,'seed':20261022,'resampling_unit':'initial state within each fixed task, preserving all 8 outcomes','no_iid_160_trials':True,'training_seeds':1});cs('paired_success_flips.csv',flips)
protected=json.loads((O/'protection_before.json').read_text());changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];assert not changed
before=json.loads((O/'git_before.json').read_text());after={p:{k:subprocess.check_output(['git','-C',p]+v,text=True) for k,v in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short']}.items()} for p in before};same=all(all(after[p][k]==before[p][k] for k in ['HEAD','branch','status']) for p in before if p!=str(W));assert same;js('git_after.json',after)
js('delivery_verification.json',{'status':'PASS','scope':'closed-loop evaluation pipeline, not automatically method efficacy','valid_trials':160,'invalid_trials':0,'reruns':0,'training_updates':0,'backward_calls':0,'checkpoint_parameters_unchanged':True,'protected_files_checked':len(protected),'historical_files_changed':changed,'historical_git_unchanged':same,'ECC_unchanged':accept['ECC_unchanged'],'state_use':'exploratory_reused_states','independent_new_states':0,'training_seed_count':1,'videos':len(list((O/'trials').glob('*/video.mp4'))),'trajectory_logs':len(list((O/'trials').glob('*/trajectory.npz'))),'four_smokes_counted_in_160':True})
table='\n'.join(f"| {a} | {summary[a]['clean']['successes']}/20 ({summary[a]['clean']['SR']:.0%}) | {summary[a]['background']['successes']}/20 ({summary[a]['background']['SR']:.0%}) | {summary[a]['background_minus_clean']*100:+.1f} pp |" for a in 'ABCD')
ct='\n'.join(f"| {key} | {v['clean']['difference']*100:+.1f} {np.round(np.array(v['clean']['task_stratified_bootstrap_95'])*100,1).tolist()} | {v['background']['difference']*100:+.1f} {np.round(np.array(v['background']['task_stratified_bootstrap_95'])*100,1).tolist()} | {v['interaction']['difference']*100:+.1f} {np.round(np.array(v['interaction']['task_stratified_bootstrap_95'])*100,1).tolist()} |" for key,v in statistics.items())
tasktable=[]
for t,ids in taskmap.items():tasktable.append('| '+t+' | '+' | '.join(f"{int(arr[ids,j,0].sum())}/5 ; {int(arr[ids,j,1].sum())}/5" for j in range(4))+' |')
report=f'''# Step 5.8 — A/B/C/D 配对闭环评估

**评估流程 PASS**。160/160有效试验，0 invalid，0重跑。训练/反向/optimizer updates均0。方法结论应结合下表及小样本限制，不因流程PASS自动判定方法有效。

## 冻结设置

状态标签：`exploratory_reused_states`。4任务×5历史状态；曾用于离线诊断和历史闭环，不是新的独立测试集。顺序在评分前固定：每state依次A/B/C/D Clean，再A/B/C/D Background。每次完整reset、恢复原state[1]、相同seed，执行10个原固定settle动作后与历史几何fingerprint按1e-8核对；所有试验通过。

研究定义：**背景材质干预下的视觉鲁棒性**。仅tex-table/tex-wall纹理变化；reflection=1、MSAA=4，允许固有反射/边界间接RGB响应。源XML只存在指定texture file差异；物理、前景材质、灯光、相机和控制条件不变。没有使用严格前景RGB固定方案。

四组使用统一wrapper、Math SDPA、相同双相机预处理、当前状态FK proprio、真实任务语言、action chunk8、20Hz控制和最多220策略步。成功由`env.check_success()`返回；超时为有效失败。异常预设规则是标记INVALID并停止、不自动重跑。本次无异常。

A是已核验公共Spatial-Pro初始化（附零增量LoRA/同wrapper，不改变原action路径）；B/C/D只使用Step5.7 final step100。加载前hash与step核对，加载后逐模块hash与Step5.7重载记录相同。未加载旧Step5.1/5.2 B/C策略。每组第一个Clean trial作为smoke并计入160，动作重复推理差0且合法。

## 成功率

| 模型 | Clean | Background | BG−Clean |
|---|---:|---:|---:|
{table}

## 配对差异及交互

下表单位为百分点；方括号为task-stratified bootstrap 95% CI。每次在各固定任务内重采样5个初始状态，并同时保留8个模型/外观结果，共10000次、seed20261022。不是对160条独立样本重采样，也不是对训练seed的置信区间。

| 对比 | Clean差异及CI | Background差异及CI | 背景交互及CI |
|---|---:|---:|---:|
{ct}

交互定义为左组(BG−Clean)减右组(BG−Clean)。D−C交互>0表示观察到较小背景退化；不自动证明稳定的因果机制。翻转数量、双方成功/失败及描述性exact McNemar见paired_success_flips.csv；每个state完整模式见paired_state_results.csv。

## 各任务原始计数

每格为Clean ; Background。

| 任务 | A | B | C | D |
|---|---:|---:|---:|---:|
{chr(10).join(tasktable)}

精确0/5或5/5标签保留在task_stratified_metrics.csv。地板的0→0不能证明鲁棒，天花板的5→5也不能证明方法无作用。必须结合基础任务能力、原始计数与配对翻转，不能只比较背景差。

## 证据与资源

所有160条trajectory.npz保存真实预测action chunk、实际执行动作、sim状态和当前FK；video.mp4每chunk采一帧，fps2.5，用于诊断而非逐控制步视频。每trial记录任务、state/seed、语言、checkpoint hash、外观、步数、timeout/invalid/exception和延迟。逐trial JSON是完整证据，CSV是汇总。

GPU1 UUID `{accept['GPU_UUID']}`，峰值allocated {accept['peak_GPU_allocated']/1024**3:.3f}GiB，CPU峰值RSS {accept['CPU_peak_RSS_KiB']/1024**2:.3f}GiB，闭环耗时{accept['seconds']/60:.2f}分钟（不含模型加载）。ECC无增量；模型参数评估前后hash一致。{len(protected)}个受保护文件未变，历史仓库状态未变。所有新文件在${DATA_ROOT}，无训练、无调参、无背景协议修改。

## 解释边界与下一步

本轮只有一个训练seed、每任务5个历史重复使用状态；bootstrap仅描述这些固定任务中的状态变异，不能反映训练seed方差、任务外泛化或未接触状态表现。离线L1没有用于选择checkpoint或rollout参数。

应优先根据D−C的配对结果与交互CI判断是否存在值得进一步验证的趋势；即使区间支持本组状态上的差异，也不能称为独立确认。若扩展seed，必须B/C/D三组同步运行相同新seed并冻结预算，同时增加来源可靠的新独立状态；本轮不自动开始任何新训练或评估。

复现入口为独立worktree中的run_step58.sh（新运行必须使用新的输出目录）。checkpoint_inventory.csv、evaluation_protocol.json与manifest_frozen.json固定模型和协议，delivery_verification.json记录交付验收。
'''
with (O/'report.md').open('x') as f:f.write(report)
print(json.dumps({'summary':summary,'contrasts':statistics,'report':str(O/'report.md')},ensure_ascii=False))
