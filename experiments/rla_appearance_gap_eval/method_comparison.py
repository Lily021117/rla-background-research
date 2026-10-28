"""Frozen methods on all four existing backgrounds, exactly320 new trials."""
import argparse,collections,csv,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_appearance_gap_eval.common import *
REFERENCE=R/'experiments/rla_appearance_gap_eval/run_20261027_01'
METHODS=['C','D-center','C-shared-target']
BACKGROUNDS=CONDITIONS[1:]

def method_model(method,order):return f'{method}_order{order}_s600'

def prepare(out):
 ref_protocol=read(REFERENCE/'evaluation_protocol.json');assert ref_protocol['trials']==200
 assert read(REFERENCE/'delivery_verification.json')['valid_trials']==200
 protected={str(REFERENCE/n):sha(REFERENCE/n) for n in ['evaluation_protocol.json','asset_manifest.json','per_trial_results.json','frozen_artifact_hashes.json','static_acceptance.json','preview_review.json','checkpoint_inventory.json','delivery_verification.json','evaluation_start.json']}
 # Byte-for-byte reuse of every previously frozen state, XML and asset.
 frozen=read(REFERENCE/'frozen_artifact_hashes.json')
 assert all(sha(p)==h for p,h in frozen.items())
 d_inventory=list(csv.DictReader((D/'checkpoint_inventory.csv').open()))
 s_inventory=list(csv.DictReader((SHARED/'checkpoint_inventory.csv').open()))
 previous_c=read(REFERENCE/'per_trial_results.json');assert len(previous_c)==200
 c_inventory={r['order']:r for r in read(REFERENCE/'checkpoint_inventory.json')}
 states=read(ref_protocol['frozen_state_protocol'])['new_development_states'];assert len(states)==20
 expected_states={s['seed']:s for s in states}
 inventory=[];registry={};reused=[];sources=[];checks=[]
 execution_keys=['max_steps','chunk_execution','reflection','MSAA','bddl','task','states']
 for order in ORDERS:
  cplan=read(C/f'seed_{order}/run_plan.json');c_exec=read(C/f'seed_{order}/C/formal/execution_configuration.json')
  for method in METHODS:
   model=method_model(method,order)
   if method=='C':
    path=Path(c_inventory[order]['checkpoint']);digest=c_inventory[order]['sha256'];root=REFERENCE;source=REFERENCE/'per_trial_results.json'
    rows=[r for r in previous_c if r['model']==model];assert len(rows)==100
    checkpoint_recomputed=False
   else:
    run=D if method=='D-center' else SHARED;root=run/f'seed_{order}';source=root/'per_trial_results.json'
    record=next(r for r in (d_inventory if method=='D-center' else s_inventory) if int(r['order_seed'])==order and (method!='D-center' or r['method']=='D-center'))
    path=Path(record['path']);digest=sha(path);assert digest==record['sha256'];checkpoint_recomputed=True
    acceptance=read(path.parent/'training_acceptance.json');assert acceptance['successful_updates']==600 and acceptance['strict_reload']
    registry[model]=str(path)
    old_protocol=read(root/'evaluation_protocol.json')
    for k in execution_keys:assert old_protocol[k]==ref_protocol[k],(method,order,k)
    plan=read(root/'run_plan.json');assert plan['shared_protocol']==cplan['shared_protocol']
    assert plan['inputs']['action_proprio_statistics']==cplan['inputs']['action_proprio_statistics']
    actual=read(path.parent/'execution_configuration.json')
    assert {k:v for k,v in actual.items() if k!='arm'}=={k:v for k,v in c_exec.items() if k!='arm'}
    clean_source=read(source);assert len(clean_source)==60 and all(r['status']=='COMPLETE' for r in clean_source)
    rows=[r for r in clean_source if r['condition']=='clean'];assert len(rows)==20
    checks.append(dict(method=method,order=order,checkpoint_sha_matches_prior_inventory=True,global_step600_strict_reload_accepted=True,action_proprio_statistics_match_C=True,training_execution_config_except_arm_matches_C=True,deployment_fields_match_C=True,clean_reused_not_rerun=True))
    for p in [path,source,root/'evaluation_protocol.json',root/'run_plan.json',path.parent/'execution_configuration.json',path.parent/'training_acceptance.json']:protected[str(p)]=sha(p)
   for r in rows:
    s=expected_states[r['seed']];condition=next(q for q in s['conditions'] if q['condition']==r['condition'])
    assert r['checkpoint_sha256']==digest and r['state_sha256']==s['state_sha256'] and r['xml_sha256']==condition['XML_sha256'] and r['language']==ref_protocol['task'] and r['status']=='COMPLETE'
    reused.append(dict(order=order,method=method,**r,comparison_model=model,origin='reused_C_previous_round' if method=='C' else 'reused_historical_Clean',source_results=str(source),source_results_sha256=sha(source),execution_GPU='GPU0'))
   sources.append(dict(method=method,order=order,results_file=str(source),sha256=sha(source),reused_trials=len(rows),appearances=CONDITIONS if method=='C' else ['clean'],execution_GPU='GPU0',reason='checkpoint/state/XML/task/deployment match; no new Clean or C rollout'))
   inventory.append(dict(method=method,order=order,model=model,checkpoint=str(path),sha256=digest,global_step=600,reused_only=method=='C',checkpoint_hash_recomputed_this_preparation=checkpoint_recomputed))
 # Confirm historical GPU0 evaluation evidence, not merely a guessed label.
 devices=[dict(method='C',evidence=str(REFERENCE/'evaluation_start.json'),GPU=read(REFERENCE/'evaluation_start.json')['GPU'])]
 for order in ORDERS:
  config=D/f'seed_{order}/gpu0_evaluate_config.json';x=read(config);assert x['stage']=='evaluate' and x['user_authorized']
  p=D/f'seed_{order}/evaluation/gpu0_evaluate/ecc_before.txt';assert '35:00.0' in p.read_text()
  devices.append(dict(method='D-center',order=order,evidence=str(p),authorization_config=str(config),GPU=GPU,PCI_evidence='35:00.0'));protected[str(p)]=sha(p);protected[str(config)]=sha(config)
  p=SHARED/f'seed_{order}/evaluation/formal/ecc_before.txt';assert '35:00.0' in p.read_text();devices.append(dict(method='C-shared-target',order=order,evidence=str(p),GPU=GPU));protected[str(p)]=sha(p)
 assert len(reused)==280 and len(registry)==4
 save(out/'checkpoint_inventory.json',inventory);save(out/'policy_registry.json',registry);save(out/'reused_results.json',reused);save(out/'reuse_sources.json',sources);save(out/'comparability_audit.json',checks);save(out/'device_provenance.json',dict(sources=devices,current_GPU=GPU,all_comparison_results_GPU0=True,earlier_pre_gap_C_evaluation_GPU1='not used as current C score; previous round GPU0 Clean matched it'))
 plan=read(REFERENCE/'run_plan.json');save(out/'run_plan.json',plan);save(out/'normalization.json',read(REFERENCE/'normalization.json'))
 for n in ['static_acceptance.json','preview_review.json']:save(out/n,read(REFERENCE/n))
 save(out/'static_reuse_provenance.json',dict(source=str(REFERENCE),source_static_acceptance_sha256=sha(REFERENCE/'static_acceptance.json'),source_asset_manifest_sha256=sha(REFERENCE/'asset_manifest.json'),same_state_files=True,same_XML_files=True,same_asset_files=True,no_new_render_configuration=True,no_new_static_policy_rollouts=True))
 protocol=dict(ref_protocol);protocol.update(frozen_at_unix=time.time(),models=list(registry),conditions=BACKGROUNDS,trials=320,order='state,background,method/order: D20261101,shared20261101,D20261102,shared20261102',method_comparison_evaluation=True,role='frozen existing-method comparison; repeated development states',checkpoint_selection='four specified existing final600 checkpoints only',Clean='reused, never rerun this round',C_background='reuse160 previous-round trials',source_protocol=str(REFERENCE/'evaluation_protocol.json'),source_assets=str(REFERENCE/'asset_manifest.json'),background2_status='post hoc focus after inspecting C results; exploratory, not independent confirmatory evidence')
 save(out/'evaluation_protocol.json',protocol)
 decision=dict(frozen_before_new_rollouts=True,primary='equal-weight mean over all four backgrounds, each method minus same-order C; show each order and each background',bootstrap=dict(draws=20000,seed=20261028,unit='20 physical states jointly preserving every appearance and both sampling orders',includes_training_seed_task_uncertainty=False),valuable_repeated_benefit_rule='method minus C background mean >0 and state-cluster CI lower bound >0 in each order; always display Clean cost and absolute SR',if_no_reliable_benefit_with_Clean_cost='archive current AE-consistency and shared-Clean-target implementations as negative results and stop tuning these implementations',if_local_or_order_dependent='describe exploratory local/order-dependent candidate only; no automatic background-specific tuning',if_repeated_benefit='propose evaluation on newly frozen independent development states, disclose Clean tradeoff; do not execute this round',no_universal_claim_about_invariance=True)
 save(out/'decision_protocol.json',decision);save(out/'protected_inputs.json',protected)
 frozen.update({str(out/n):sha(out/n) for n in ['checkpoint_inventory.json','policy_registry.json','run_plan.json','normalization.json','evaluation_protocol.json','decision_protocol.json','reuse_sources.json','comparability_audit.json']})
 save(out/'frozen_artifact_hashes.json',frozen);print('PREPARED_320_NEW_280_REUSED',flush=True)

def analyze(out):
 new=read(out/'per_trial_results.json');assert len(new)==320 and all(r['status']=='COMPLETE' for r in new);new_ledger_sha=sha(out/'per_trial_results.json')
 inventory=read(out/'checkpoint_inventory.json');by_model={r['model']:r for r in inventory};reused=read(out/'reused_results.json');rows=[]
 for r in new:
  meta=by_model[r['model']];assert meta['method']!='C' and r['condition'] in BACKGROUNDS and r['checkpoint_sha256']==meta['sha256']
  rows.append(dict(order=meta['order'],method=meta['method'],**r,comparison_model=r['model'],origin='new_background_rollout',source_results=str(out/'per_trial_results.json'),source_results_sha256=new_ledger_sha,execution_GPU='GPU0'))
 rows+=reused;assert len(rows)==600
 keys={(r['order'],r['method'],r['seed'],r['condition']):r for r in rows};assert len(keys)==600
 protocol=read(out/'evaluation_protocol.json');states=protocol['states'];frozen_states={r['seed']:r for r in read(protocol['frozen_state_protocol'])['new_development_states']}
 for r in rows:
  s=frozen_states[r['seed']];c=next(q for q in s['conditions'] if q['condition']==r['condition']);assert r['state_sha256']==s['state_sha256'] and r['xml_sha256']==c['XML_sha256'] and r['language']==protocol['task']
 arrays={(order,m):np.array([[int(keys[order,m,s,c]['success']) for c in CONDITIONS] for s in states]) for order in ORDERS for m in METHODS}
 rng=np.random.RandomState(20261028);ix=rng.randint(0,20,(20000,20))
 def stat(label,a):
  a=np.asarray(a);b=a[ix];return dict(comparison=label,difference=float(a.mean()),CI95=np.quantile(b.mean(axis=tuple(range(1,b.ndim))),[.025,.975]).tolist())
 rates=[];paired=[];phases=[];effects=[];degradation=[];interactions=[]
 for order in ORDERS:
  for m in METHODS:
   a=arrays[order,m]
   for j,c in enumerate(CONDITIONS):
    q=[keys[order,m,s,c] for s in states];d=a[:,j]-a[:,0]
    rates.append(dict(order=order,method=m,appearance=c,successes=int(a[:,j].sum()),trials=20,success_rate=float(a[:,j].mean()),difference_vs_own_Clean=float(d.mean()),origin=q[0]['origin']))
    for stage,n in collections.Counter(r['stage'] for r in q).items():phases.append(dict(order=order,method=m,appearance=c,stage=stage,count=n))
    if c!='clean':degradation.append(stat(f'{order} {m}: {c} minus own Clean',d))
   rates.append(dict(order=order,method=m,appearance='four_backgrounds_equal_weight',successes=int(a[:,1:].sum()),trials=80,success_rate=float(a[:,1:].mean()),difference_vs_own_Clean=float(a[:,1:].mean()-a[:,0].mean()),origin='reused_C_previous_round' if m=='C' else 'new_background_rollout'))
   degradation.append(stat(f'{order} {m}: four backgrounds minus own Clean',a[:,1:]-a[:,0,None]))
  for m in METHODS[1:]:
   d=arrays[order,m]-arrays[order,'C'];effects.append(stat(f'{order} {m} minus C: four backgrounds equal weight',d[:,1:]))
   for j,c in enumerate(CONDITIONS):
    effects.append(stat(f'{order} {m} minus C: {c}',d[:,j]))
    for s in states:
     a=keys[order,m,s,c];b=keys[order,'C',s,c]
     paired.append(dict(order=order,method=m,state=s,appearance=c,method_success=int(a['success']),C_success=int(b['success']),difference=int(a['success'])-int(b['success'])))
   interactions.append(stat(f'{order} {m} minus C: background-effect interaction',(arrays[order,m][:,1:]-arrays[order,m][:,0,None])-(arrays[order,'C'][:,1:]-arrays[order,'C'][:,0,None])))
 for m in METHODS[1:]:
  d=np.mean([arrays[o,m][:,1:]-arrays[o,'C'][:,1:] for o in ORDERS],axis=0);effects.append(stat(f'two orders equal weight {m} minus C: four backgrounds',d))
  for j,c in enumerate(CONDITIONS):effects.append(stat(f'two orders equal weight {m} minus C: {c}',np.mean([arrays[o,m][:,j]-arrays[o,'C'][:,j] for o in ORDERS],axis=0)))
  interactions.append(stat(f'two orders equal weight {m} minus C: background-effect interaction',np.mean([(arrays[o,m][:,1:]-arrays[o,m][:,0,None])-(arrays[o,'C'][:,1:]-arrays[o,'C'][:,0,None]) for o in ORDERS],axis=0)))
 table(out/'complete_comparison_table.csv',rates);table(out/'paired_state_results.csv',paired);table(out/'failure_stage_summary.csv',phases)
 flips=[]
 for order in ORDERS:
  for m in METHODS[1:]:
   for c in CONDITIONS:
    a=[r for r in paired if r['order']==order and r['method']==m and r['appearance']==c]
    flips.append(dict(order=order,method=m,appearance=c,method_success_C_failure=sum(r['difference']>0 for r in a),C_success_method_failure=sum(r['difference']<0 for r in a),both_success=sum(r['method_success']==1 and r['C_success']==1 for r in a),both_failure=sum(r['method_success']==0 and r['C_success']==0 for r in a)))
 table(out/'paired_flip_summary.csv',flips)
 save(out/'paired_statistics.json',dict(method_effects=effects,own_Clean_degradation=degradation,background_interactions=interactions,cluster_unit='same physical state; all appearances/methods/orders resampled together',bootstrap_seed=20261028,draws=20000,independent_training_seeds=False,task_seed_uncertainty_not_included=True))
 decisions=[]
 for m in METHODS[1:]:
  s=[r for r in effects if any(r['comparison']==f'{o} {m} minus C: four backgrounds equal weight' for o in ORDERS)];assert len(s)==2
  reliable=all(r['difference']>0 and r['CI95'][0]>0 for r in s);costs={str(o):float(arrays[o,m][:,0].mean()-arrays[o,'C'][:,0].mean()) for o in ORDERS}
  positive_orders=[r['comparison'] for r in s if r['difference']>0]
  local_positive=[r for r in effects if any(r['comparison']==f'{o} {m} minus C: {c}' for o in ORDERS for c in BACKGROUNDS) and r['difference']>0]
  decisions.append(dict(method=m,valuable_repeated_background_benefit=reliable,Clean_cost_by_order=costs,positive_order_mean_contrasts=positive_orders,local_positive_background_contrasts=local_positive,recommendation='propose new independent development-state replication, no execution' if reliable else 'archive current implementation as negative/limited exploratory result; stop tuning it when Clean cost remains; disclose any local/order-dependent advantages without claiming overall effectiveness'))
 save(out/'research_decision.json',dict(decisions=decisions,background2_post_hoc_focus=True,no_universal_claim_about_all_invariance_methods=True,no_automatic_followup=True))
 fields=['order','method','seed','condition','success','status','steps','ending','stage','origin','source_results','source_results_sha256','execution_GPU','checkpoint_sha256','state_sha256','xml_sha256','issued_close_command','max_lift_m','first_lift_step','descent_horizontal_path_m','postlift_contact_height_xy','final_xy_m','final_vertical_m']
 table(out/'all_episode_metrics.csv',[{k:r[k] for k in fields} for r in rows]);table(out/'video_trajectory_index.csv',[{k:r[k] for k in ['order','method','seed','condition','origin','video','trajectory']} for r in rows])
 assert all(sha(p)==h for p,h in read(out/'protected_inputs.json').items());assert all(sha(p)==h for p,h in read(out/'frozen_artifact_hashes.json').items());assert read(out/'evaluation/formal/no_update_audit.json')['backward']==0
 save(out/'delivery_verification.json',dict(status='PASS_EVALUATION_PROCESS_NOT_METHOD',new_valid_rollouts=320,actual_attempts=320,invalid_trials=0,reused_C_new_background_trials=160,reused_Clean_trials=120,total_comparison_records=600,optimizer_updates=0,AE_updates=0,backward=0,sealed_set_reads=0,historical_protected_hashes_unchanged=True,no_extra_backgrounds_tasks_seeds_checkpoints=True,policy_worker_exited=True,GPU_final=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True)))
 lines=['# 冻结现有方法：完整新背景对照','',
 '复用上一轮20个development物理状态与全部4套新背景，固定现有600步权重。C的新背景160次与各方法Clean120次直接复用；本轮新增D-center/共享target背景320次。','',
 '| 顺序 | 方法 | Clean | 背景1 | 背景2 | 背景3 | 背景4 | 四背景均值 |','|---|---|---|---|---|---|---|---|']
 for order in ORDERS:
  for m in METHODS:
   a=arrays[order,m];lines.append(f'| {order} | {m} | '+' | '.join(f'{n}/20' for n in a.sum(0))+f' | {a[:,1:].mean():.2%} |')
 lines+=['','## 相对C的配对差异','']
 for r in effects:
  if 'four backgrounds' in r['comparison'] or ': clean' in r['comparison']:lines.append(f"- {r['comparison']}: {100*r['difference']:+.2f}pp，物理状态聚类95%区间[{100*r['CI95'][0]:+.2f}, {100*r['CI95'][1]:+.2f}]。")
 lines+=['','## Clean代价与背景效应','',
 'complete_comparison_table.csv列出各条件相对各自Clean的变化；paired_statistics.json给出相应区间及相对C的背景交互。Clean变差造成的相对降幅变小不能被解释为更鲁棒。背景2是在看过C结果后受到关注的局部条件，属于事后探索，不是独立确认性证据。','',
 '| 顺序 | 方法 | 条件 | 方法成功/C失败 | C成功/方法失败 | 双成功 | 双失败 |','|---|---|---|---|---|---|---|']
 for r in flips:lines.append(f"| {r['order']} | {r['method']} | {r['appearance']} | {r['method_success_C_failure']} | {r['C_success_method_failure']} | {r['both_success']} | {r['both_failure']} |")
 lines+=['','## 研究决定','']
 for r in decisions:
  if r['valuable_repeated_background_benefit']:lines.append(f"- {r['method']}在两个顺序均达到预先定义的有价值背景收益判据。可提出新独立development状态复核方案，保留Clean取舍；本轮不执行。")
  else:
   lines.append(f"- {r['method']}未达到两个顺序均有可靠背景收益的判据。结合上表Clean代价，停止该实现的继续调参，保留负结果归档；不否定所有外观不变性方法。")
   if r['positive_order_mean_contrasts'] or r['local_positive_background_contrasts']:lines.append(f"  局部/单顺序正差仍单独保留为探索性候选：正均值顺序{r['positive_order_mean_contrasts']}；正差背景条件{[q['comparison'] for q in r['local_positive_background_contrasts']]}。不能宣称整体有效，也不针对背景2调参。")
 lines+=['','## 来源、可比性与范围','',
 'checkpoint_inventory.json固定实际文件与SHA-256，含D-center顺序20261102迁移恢复后的最终600步文件。comparability_audit.json核对动作/proprio统计、共享训练配置、部署设置及逐状态XML/权重身份。历史D-center训练发生的100步重放属于历史成本，本轮不重训。','',
 'reuse_sources.json与all_episode_metrics.csv逐条注明复用来源和新执行来源。本轮新增执行及选用的历史Clean/C结果均来自GPU0；更早GPU1的C结果未冒充同轮执行。既有C同轮Clean已与更早结果逐状态核验一致。所有Clean均为复用，不是本轮重跑。','',
 '沿用原部署：220控制步、chunk8、20Hz、双相机/FK proprio/STOP与动作后处理、反射1、MSAA4及env.check_success()。状态/资产/XML原样引用上一轮，四套全部保留，没有调整背景2。failure_stage_summary.csv仅沿用接近/接触及抬升后放置代理分类，不新增物理根因诊断。','',
 '本轮320次有效闭环、320次尝试、无效0；optimizer updates=0、AE updates=0、backward=0；未读取封存集。全部有效失败保留，历史保护哈希一致。','',
 '## 不确定性','',
 '- 以同一物理状态为聚类单位，将所有方法、外观与两个顺序一起重采样；20,000次bootstrap，seed20261028。两个顺序分别报告，再等权汇总。',
 '- 两个顺序不是两个完整初始化/训练seed；区间不含训练seed和任务不确定性。20个状态及这些背景已经反复用于开发。',
 '- 所有背景都报告，不能只选择背景2或赢家顺序；低Clean成功率下的相对背景退化可能受到地板效应影响。',
 '- 单个或少量翻转的区间有离散性，零端点或全零bootstrap不能证明精确无效应。']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('METHOD_COMPARISON_ANALYZED',flush=True)

def publish(out):
 dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
 names=['report.md','checkpoint_inventory.json','reuse_sources.json','device_provenance.json','comparability_audit.json','evaluation_protocol.json','decision_protocol.json','static_reuse_provenance.json','complete_comparison_table.csv','paired_state_results.csv','paired_flip_summary.csv','paired_statistics.json','failure_stage_summary.csv','all_episode_metrics.csv','research_decision.json','delivery_verification.json']
 if (out/'preparation_incident.json').exists():names.append('preparation_incident.json')
 for n in names:
  t=(out/n).read_text().replace(str(R),'<PROJECT_ROOT>').replace('/data/lyc/projects/LIBERO-plus','<LIBERO_ROOT>');t=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',t)
  assert len(t)<5_000_000 and not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',t)
  with (dest/n).open('x') as f:f.write(t)
 def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
 branch='experiments/rla-appearance-gap-eval';assert git('branch','--show-current')==branch;rel=str(dest.relative_to(REPO));git('add','--',rel);assert all(p.startswith(rel+'/') for p in git('diff','--cached','--name-only').splitlines())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report all frozen methods on four existing development backgrounds')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli');auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env);commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0]==commit
 save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',branch=branch,commit=commit,repository='https://github.com/Lily021117/rla-background-research'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run','analyze'],required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a.out)
 else:
  if a.mode=='run':subprocess.run([sys.executable,str(HERE/'launch.py'),'--out',str(a.out)],cwd=REPO,check=True)
  analyze(a.out);publish(a.out)
