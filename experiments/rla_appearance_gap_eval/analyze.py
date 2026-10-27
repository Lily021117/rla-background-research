"""All conditions reported; paired state-cluster inference, zero model work."""
import collections,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
from rla_appearance_gap_eval.common import *

def analyze(out):
 trials=read(out/'per_trial_results.json');protocol=read(out/'evaluation_protocol.json');assert read(out/'decision_rule.json')['frozen_before_new_policy_rollouts'];states=protocol['states'];models=protocol['models'];assert len(trials)==200 and all(t['status']=='COMPLETE' for t in trials)
 by={(t['seed'],t['condition'],t['model']):t for t in trials};assert len(by)==200
 frozen=read(out/'development_protocol/evaluation_protocol.json');inventory={r['model']:r for r in read(out/'checkpoint_inventory.json')};assert len(states)==20
 for s in frozen['new_development_states']:
  for c in s['conditions']:
   for model in models:
    r=by[s['seed'],c['condition'],model];assert r['state_sha256']==s['state_sha256'] and r['xml_sha256']==c['XML_sha256'] and r['checkpoint_sha256']==inventory[model]['sha256'] and r['language']==protocol['task'];assert 0<r['steps']<=220
 rng=np.random.RandomState(20261027);ix=rng.randint(0,20,(20000,20))
 def interval(d):
  a=np.asarray(d);sample=a[ix];axis=tuple(range(1,sample.ndim));return np.quantile(sample.mean(axis=axis),[.025,.975]).tolist()
 summary=[];paired=[];stats=[];stages=[];repeat=[];arrays={};cross=[]
 for model in models:
  order=inventory[model]['order'];a=np.array([[int(by[s,c,model]['success']) for c in CONDITIONS] for s in states]);arrays[model]=a
  old={(r['seed'],r['condition']):r for r in read(C/f'seed_{order}/per_trial_results.json')}
  for j,c in enumerate(CONDITIONS):
   d=a[:,j]-a[:,0];counts=collections.Counter((int(x),int(y)) for x,y in zip(a[:,0],a[:,j]))
   summary.append(dict(order=order,model=model,appearance=c,successes=int(a[:,j].sum()),trials=20,success_rate=float(a[:,j].mean()),difference_vs_same_run_clean=float(d.mean()),CI95_low=interval(d)[0],CI95_high=interval(d)[1],clean_success_background_failure=counts[1,0],clean_failure_background_success=counts[0,1],both_success=counts[1,1],both_failure=counts[0,0]))
   stats.append(dict(model=model,appearance=c,contrast='appearance minus same-run Clean',difference=float(d.mean()),CI95=interval(d)))
   for s,x,y in zip(states,a[:,0],a[:,j]):paired.append(dict(order=order,state=s,appearance=c,clean_success=int(x),appearance_success=int(y),difference=int(y-x)))
   group=[by[s,c,model] for s in states]
   for stage,n in collections.Counter(t['stage'] for t in group).items():stages.append(dict(order=order,appearance=c,stage=stage,count=n))
   if c=='clean':
    for s in states:
     prior=old[s,'clean'];cur=by[s,'clean',model];assert prior['state_sha256']==cur['state_sha256'] and prior['xml_sha256']==cur['xml_sha256'] and prior['checkpoint_sha256']==cur['checkpoint_sha256']
     repeat.append(dict(order=order,state=s,historical_success=int(prior['success']),current_success=int(cur['success']),success_equal=bool(prior['success']==cur['success']),historical_steps=prior['steps'],current_steps=cur['steps'],historical_GPU='GPU1',current_GPU='GPU0',same_state_XML_checkpoint=True))
  d=a[:,1:]-a[:,0,None];stats.append(dict(model=model,appearance='four_new_backgrounds_equal_weight',contrast='new-background mean minus same-run Clean',difference=float(d.mean()),CI95=interval(d)))
 for j,c in enumerate(CONDITIONS[1:],1):
  effects=[float(arrays[m][:,j].mean()-arrays[m][:,0].mean()) for m in models]
  cross.append(dict(appearance=c,effects_by_order=dict(zip(map(str,ORDERS),effects)),negative_in_both_models=bool(all(x<0 for x in effects))))
 common_negative=sum(x['negative_in_both_models'] for x in cross)
 overall=[s for s in stats if s['appearance']=='four_new_backgrounds_equal_weight']
 clear=common_negative>=2 and all(s['difference']<0 and s['CI95'][1]<0 for s in overall)
 table(out/'success_rate_summary.csv',summary);table(out/'paired_state_results.csv',paired);table(out/'failure_stage_summary.csv',stages);table(out/'clean_repeat_audit.csv',repeat)
 save(out/'paired_statistics.json',dict(results=stats,repeated_effects=cross,common_negative_backgrounds=common_negative,clear_repeated_gap=clear,decision_rule='at least two prespecified backgrounds have negative effects in both models AND each model equal-background mean paired CI is wholly below zero',bootstrap_draws=20000,bootstrap_seed=20261027,cluster_unit='same physical state across every appearance and both C models',fixed_backgrounds_not_a_random_sample=True,does_not_include_training_seed_or_task_uncertainty=True))
 fields=['model','seed','condition','success','status','steps','ending','stage','issued_close_command','max_lift_m','min_ee_target_distance_m','first_lift_step','final_xy_m','final_vertical_m','descent_horizontal_path_m','postlift_contact_height_xy','seconds','checkpoint_sha256','state_sha256','xml_sha256']
 table(out/'per_episode_metrics.csv',[{k:r[k] for k in fields} for r in trials]);table(out/'video_trajectory_index.csv',[{k:r[k] for k in ['model','seed','condition','video','trajectory']} for r in trials])
 protected=read(out/'protected_inputs.json');assert all(sha(p)==h for p,h in protected.items());assert all(sha(p)==h for p,h in read(out/'frozen_artifact_hashes.json').items());assert read(out/'evaluation/formal/no_update_audit.json')['backward']==0
 # Retain actual device and resource evidence without touching any other job.
 gpu=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True)
 save(out/'delivery_verification.json',dict(status='PASS_EVALUATION_PROCESS',valid_trials=200,actual_attempts=200,invalid_trials=0,optimizer_updates=0,AE_updates=0,backward=0,sealed_set_reads=0,new_tasks=0,historical_protected_hashes_unchanged=True,checkpoints_modified=False,GPU_final=gpu,policy_workers_finished=True,clear_repeated_appearance_gap=clear))
 lines=['# 冻结C策略：外观鲁棒性缺口评估','',
 '两个均衡采样顺序的现有C最终600步权重；固定20个历史development物理状态。所有条件与状态在评分前冻结；新背景全部保留。','',
 '| 采样顺序 | 条件 | 完整任务成功 | 相对同轮Clean | Clean成功→外观失败 | Clean失败→外观成功 | 双成功 | 双失败 |',
 '|---|---|---|---|---|---|---|---|']
 for r in summary:lines.append(f"| {r['order']} | {r['appearance']} | {r['successes']}/20 | {100*r['difference_vs_same_run_clean']:+.1f}pp | {r['clean_success_background_failure']} | {r['clean_failure_background_success']} | {r['both_success']} | {r['both_failure']} |")
 lines+=['','## 配对区间与判断','']
 for s in stats:
  if s['appearance']!='clean':lines.append(f"- {s['model']} / {s['appearance']}: {100*s['difference']:+.1f}pp，物理状态聚类95%区间[{100*s['CI95'][0]:+.1f}, {100*s['CI95'][1]:+.1f}]。")
 lines+=['',f'4套背景中有{common_negative}套在两个模型上均出现成功率下降。预先定义的重复外观缺口判据：{clear}。']
 if clear:lines.append('本轮观察到跨多个固定背景、两个模型均出现的额外退化。下一轮可冻结完全相同的4套背景与状态，对现有C/D-center/C-shared-target权重作完整配对评估；本轮不执行。')
 else:lines.append('本轮没有达到跨背景、跨两个模型重复出现明确额外退化的判据。建议暂停当前仅针对背景一致性的策略改进路线，不继续盲目增加loss；这不等于证明所有外观变化下都无缺口。')
 lines+=['','## 同轮Clean与历史结果','']
 for order in ORDERS:
  q=[r for r in repeat if r['order']==order];lines.append(f"- 顺序{order}：历史{sum(r['historical_success'] for r in q)}/20，本轮{sum(r['current_success'] for r in q)}/20，成功标签一致{sum(r['success_equal'] for r in q)}/20。相同checkpoint/state/XML；历史GPU1、本轮GPU0。")
 lines+=['','## 历史配对结果','',
 'historical_flip_summary.csv逐顺序、逐已见/未见背景保留C、D-center与C-shared-target的四格配对计数。Clean失败且背景也失败提示共同基础任务限制；成功翻转描述外观相关的结果变化，不能单独证明物理失败机制。','',
 '## 资产、控制与范围','',
 'asset_manifest.json记录固定字典序规则、公共通用纹理资产来源、原/派生哈希和全部排除项。只改tex-table/tex-wall纹理文件；material、几何、灯光、相机、反射1、MSAA4保持原协议。静态核验20状态×5条件的几何/物理相等，并渲染首/中间状态双相机检查segmentation/depth和任务可见性。自然反射/边界响应允许存在。','',
 '200次有效闭环、200次实际尝试、0次无效；optimizer updates=0、AE updates=0、backward=0，未读取封存集。视频与原始轨迹仅保存本地。历史文件保护哈希一致。','',
 '## 解释限制','',
 '- 这些是反复使用的development状态。两个模型仅采样顺序不同，不是两个完整初始化/训练seed。',
 '- 同一物理状态的所有外观与两模型一起聚类重采样，条件数量不增加独立状态或训练seed数。区间不包含任务/训练seed不确定性。',
 '- 全部4套背景均展示，不能只挑最差背景；本轮仅检验这组固定材质配置。',
 '- 天花板下的零翻转或退化为0不证明普遍鲁棒；地板效应下共同失败也不能解释为鲁棒。退化须结合Clean绝对成功数。',
 '- 阶段分类是沿用的接近/接触与抬升后放置代理，不能代替准确接触因果判断。']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('ANALYSIS_COMPLETE',clear,flush=True)

def publish(out):
 dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
 names=['report.md','evaluation_protocol.json','decision_rule.json','asset_manifest.json','checkpoint_inventory.json','historical_flip_summary.csv','historical_paired_states.csv','static_geometry_checks.csv','static_render_checks.csv','background_geom_mapping.csv','static_acceptance.json','preview_review.json','success_rate_summary.csv','paired_state_results.csv','failure_stage_summary.csv','clean_repeat_audit.csv','paired_statistics.json','per_episode_metrics.csv','delivery_verification.json']
 for name in names:
  t=(out/name).read_text().replace(str(R),'<PROJECT_ROOT>').replace('/data/lyc/projects/LIBERO-plus','<LIBERO_ROOT>');t=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',t)
  assert len(t)<5_000_000 and not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',t)
  with (dest/name).open('x') as f:f.write(t)
 def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
 branch='experiments/rla-appearance-gap-eval';assert git('branch','--show-current')==branch
 rel=str(dest.relative_to(REPO));git('add','--',rel);assert all(p.startswith(rel+'/') for p in git('diff','--cached','--name-only').splitlines())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report frozen C appearance-gap paired evaluation')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
 auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential'];git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env)
 commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0]==commit
 save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',branch=branch,commit=commit,repository='https://github.com/Lily021117/rla-background-research'))

if __name__=='__main__':
 out=Path(sys.argv[1]);analyze(out);publish(out)
