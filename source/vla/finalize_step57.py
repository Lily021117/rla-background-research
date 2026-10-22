"""Read-only verification of completed training; writes only new run deliveries."""
from pathlib import Path
import json,csv,hashlib,subprocess,statistics,collections
W=Path(__file__).parent;E=Path('${RLA_PROJECT_ROOT}/experiments/rla_background_invariance');O=E/'step5_7_causal_training/run_20261022_01'
def read(n):return json.loads((O/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def csvout(n,rows):
 with (O/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
completion=read('training_completion.json');assert completion['status']=='PASS' and completion['updates']=={'B':100,'C':100,'D':100}
training=list(csv.DictReader(open(O/'training_metrics.csv')));val=list(csv.DictReader(open(O/'validation_metrics.csv')));inventory=list(csv.DictReader(open(O/'checkpoint_inventory.csv')))
assert len(training)==300
for a in 'BCD':assert [int(r['global_step']) for r in training if r['arm']==a]==list(range(1,101))
assert len(val)==360 and len({r['episode'] for r in val})==3
assert all(r['status']=='PASS' and r['global_step']==100 for r in read('checkpoint_reload_audit.json').values())
for r in inventory:assert sha(r['path'])==r['SHA256']
protected=read('protection_before.json')
for previous in ['step5_6_unified_trainer/run_20261022_02','step5_6_1_gradient_diagnosis/run_20261022_02']:
 protected.update(json.loads((E/previous/'code_inventory.json').read_text()))
protected.update(read('training_provenance.json')['sources'])
changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];assert not changed
before=read('git_before.json');after={p:{a:subprocess.check_output(['git','-C',p]+v,text=True) for a,v in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short']}.items()} for p in before}
same=all(all(after[p][a]==before[p][a] for a in ['HEAD','branch','status']) for p in before if p!=str(W));assert same;js('git_after.json',after)
bins=collections.defaultdict(list)
for r in val:bins[r['arm'],int(r['step']),r['episode'],r['appearance']].append(r)
episode=[]
for (a,s,e,c),rs in sorted(bins.items()):episode.append({'arm':a,'step':s,'episode':e,'appearance':c,'windows':len(rs),'action_L1':statistics.mean(float(r['action_L1']) for r in rs),'aux_L1':statistics.mean(float(r['aux_L1']) for r in rs)})
csvout('validation_per_episode.csv',episode)
summary=[]
for a in 'BCD':
 for s in [0,100]:
  x={c:[r for r in episode if r['arm']==a and r['step']==s and r['appearance']==c] for c in ['clean','background']}
  summary.append({'arm':a,'step':s,'episodes':3,'clean_action_L1':statistics.mean(r['action_L1'] for r in x['clean']),'background_action_L1':statistics.mean(r['action_L1'] for r in x['background']),'clean_aux_L1':statistics.mean(r['aux_L1'] for r in x['clean']),'background_aux_L1':statistics.mean(r['aux_L1'] for r in x['background'])})
  summary[-1]['background_minus_clean']=summary[-1]['background_action_L1']-summary[-1]['clean_action_L1']
csvout('validation_summary.csv',summary)
final={r['arm']:r for r in summary if r['step']==100};contrasts=[]
for left,right in [('C','B'),('D','C'),('D','B')]:
 contrasts.append({'contrast':left+'-'+right,**{m:final[left][m]-final[right][m] for m in ['clean_action_L1','background_action_L1','background_minus_clean']}})
csvout('validation_contrasts.csv',contrasts)
js('code_inventory.json',{str(W/n):sha(W/n) for n in ['train_math_step57.py','run_math_step57.sh','finalize_step57.py','causal_unified.py','validate_step0.py','prepare_acceptance.py']})
js('delivery_verification.json',{'status':'PASS','scope':'training pipeline and checkpoint acceptance, not method efficacy','updates':completion['updates'],'total_successful_updates':300,'preflight_updates':0,'skipped':0,'replayed':0,'historical_files_checked':len(protected),'historical_files_modified':changed,'historical_git_unchanged':same,'math_SDPA':True,'backend_fallback':False,'same_initialization_and_sampling':True,'validation_episodes':3,'validation_transitions':30,'missing_stove_validation':True,'closed_loop_trials':0,'heldout_scored':0,'new_independent_evaluation_states':0,'historical_exploratory_states':20,'no_second_seed':True,'checkpoint_reload':'PASS','A_updates':0,'ECC_unchanged':completion['ECC_unchanged']})
table='\n'.join(f"| {a} | {final[a]['clean_action_L1']:.6f} | {final[a]['background_action_L1']:.6f} | {final[a]['background_minus_clean']:+.6f} | {final[a]['clean_aux_L1']:.6f} / {final[a]['background_aux_L1']:.6f} |" for a in 'BCD')
ctable='\n'.join(f"| {r['contrast']} | {r['clean_action_L1']:+.6f} | {r['background_action_L1']:+.6f} | {r['background_minus_clean']:+.6f} |" for r in contrasts)
checkpoints='\n'.join(f"- {r['arm']}: `{r['path']}`\n  SHA-256 `{r['SHA256']}`" for r in inventory if r['step']=='100')
report=f'''# Step 5.7 — Math SDPA 统一 B/C/D 训练

**训练流程 PASS，不代表方法有效性 PASS。** B/C/D 各完成100个成功更新，共300个；A始终冻结。没有闭环、保留集评分或额外seed训练。

## 配置与实现

独立worktree `{W}`，分支 `exp/rla-background-causal-unified-trainer`，基线commit `23fa0c9c159e2aa04341cdd3e924f44061311060`。新增代码未自动提交，以code_inventory.json固定。历史代码与checkpoint保持只读。

只有attention backend改变：Flash和memory-efficient SDPA关闭，Math开启。训练前真实batch operator profile含attention_math且不含flash/efficient attention；训练每个microbatch继续断言backend设置。没有自动回退，没有将train模式改为eval。

共同seed20261021、同一Spatial-Pro公共初始化、同一LoRA/辅助头初始化、同一完整RNG及数据采样表。每microbatch使用共同seed规则20261021+step*10+offset，offset=0/2。common_rng.pt和common_step0.pt留存。没有从任何历史微调后的B/C checkpoint续训。

有效batch4，microbatch2，梯度累积2；AdamW lr1e-4、betas(0.9,0.999)、eps1e-8、weight_decay0，恒定LR；既有bf16/PEFT dtype不变；LoRA32/alpha64/dropout0，所有组可训练参数集合一致。各组400个分支/窗口曝光；19训练示范190真实转移，380种外观实例中按同一既有采样表取样。没有增加数据或增强。

B保留辅助头，权重0。C读取Step2 B普通Encoder target；D读取Step2 C一致性Encoder target；C/D系数均0.05。C/D唯一有意差别是target来源及已有配套train-only归一化值，使用同一归一化算法；不是把不同latent空间的raw auxiliary loss作为公共物理或视觉指标。完整32×64 latent、h=8、FK及任务语言沿用验收结果。这是ManiSkill-derived自定义跨域辅助监督，不是作者Original LIBERO方法原样复现。

## 正式更新前验收

440/440 batch再次通过CPU来源/语言/target核对。真实batch Math SDPA下，B/C/D从共同step0重载：参数hash、action和aux输出一致；action-only gradient逐张量完全一致；B weighted aux backbone梯度为0，C/D非零且有限。删除/改变target不影响action输入与输出。预检6次backward，optimizer更新0，未隐藏额外训练。

## 更新与资源

各100成功optimizer updates；成功global step=1..100连续，skipped=0，replayed=0。保存step0/1/50/100，固定最终step100，不按验证结果选checkpoint。原生训练函数run_training未改动；新增代码只封装Math backend、验收及日志。

GPU1 UUID `${GPU_UUID}`；峰值allocated {completion['GPU_peak_allocated']/1024**3:.3f}GiB，CPU峰值RSS {completion['CPU_max_RSS_KiB']/1024**2:.3f}GiB。从三组训练/验证开始计时 {completion['elapsed_seconds']/60:.2f}分钟，不含前置加载和step0检查。ECC没有增加，历史硬件风险没有被宣称修复。所有输出和缓存位于数据盘。每步loss、LR、梯度范数、显存、CPU RSS和耗时见training_metrics.csv；各组原始日志见B/C/D下updates.jsonl。

## 固定验证：示范等权

只有3条验证示范、30真实h=8转移，clean/background共60实例；缺炉灶验证。动作L1是既有动作归一化空间中的离线指标，不是成功率。先每示范聚合，再示范等权。

| 模型 | Clean action L1 | Background action L1 | BG−Clean | Aux L1 clean / BG |
|---|---:|---:|---:|---:|
{table}

| 配对差异 | Clean L1 | Background L1 | BG−Clean差异 |
|---|---:|---:|---:|
{ctable}

差值负表示对应离线误差更低。逐示范及step0参照见validation_per_episode.csv和validation_summary.csv；没有利用验证结果调参、早停、延长训练或更换seed。Auxiliary loss在B中仅记录，不参与训练；C/D target及其统计不同，不直接据其大小给表示优劣排名。

## 最终checkpoint与重载

{checkpoints}

每个checkpoint严格重载，global_step100，所有存在的Adam状态step100、scheduler step100；参数hash、optimizer状态hash及scheduler一致；动作与aux重载前后差0且有限。完整证据见checkpoint_reload_audit.json。checkpoint包含可训练VLM delta、native/aux head和proprio及优化器/RNG；冻结VLM由已核验公共base提供，不能脱离该base单独使用。对应base SHA、数据/target/config SHA均在training_provenance.json及各训练配置内。

## 是否进入后续闭环

现在具备用同一wrapper和固定渲染/控制协议评估A/B/C/D的训练与加载前提。A应继续使用已核验公共Spatial-Pro冻结初始化，不能替换为本轮B。闭环必须另行授权和冻结任务/状态/种子，不自动运行。

本轮是单共同seed、小训练集、3验证示范；新独立评估状态仍0，历史20状态仅探索性。离线L1趋势不证明背景鲁棒性或任务成功率改善，也不证明跨外观一致性机制成立。若后续闭环出现值得验证的D−C趋势，应对B/C/D同时增加相同的新seed，不能选择性重跑某一组。

## 保护与复现

{len(protected)}个受保护源文件与代码哈希复核无变化；历史仓库HEAD/分支/status一致。没有修改背景协议或历史结果。复现入口run_math_step57.sh，必须为另一次运行预设新的独立输出目录，不得复用现有目录。训练已完成并停止，最终交付文件清单见delivery_verification.json。
'''
with (O/'report.md').open('x') as f:f.write(report)
print(json.dumps({'status':'PASS','summary':final,'contrasts':contrasts,'report':str(O/'report.md')},ensure_ascii=False))
