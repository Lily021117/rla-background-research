"""Complete-run receipts and sanitized delivery, only after all frozen trials."""
import argparse,csv,json,hashlib,gc,subprocess,collections
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--completion-name',default='completion');a=p.parse_args();out=a.run/'recovery_20261024'/a.completion_name;out.mkdir(exist_ok=False);a.delivery.mkdir(parents=True,exist_ok=False)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def csvread(path):return list(csv.DictReader(Path(path).open()))
def process_wall(log,end):
 if not log.exists():return None
 birth=int(subprocess.check_output(['stat','--format=%W',str(log)],text=True).strip())
 return max(0.,Path(end).stat().st_mtime-birth) if birth>0 else None
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('x') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def clean(x):
 if isinstance(x,str):return x.replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}').replace('/data/lyc','${DATA_ROOT}')
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def jsonsave(path,x):
 with path.open('x') as f:json.dump(x,f,indent=2)
inventory=[];cost=[];representations=[];validation=[];dataset=[];gradients=[];training=[];rec_groups=[]
for task in ('drawer','stove'):
 root=a.run/task;ae=root/('ae_run_recovery1' if task=='drawer' else 'ae_run');policy=root/'policy_run';manifest=read(root/'frozen_dataset_manifest.json');versions=sorted((ae/'model_registry_versions').glob('*.json'));registry=read(versions[-1]) if versions else {k:str(ae/('AE_'+k)) for k in ('C','D')}
 for label,path in [('dataset',root/'frozen_dataset_manifest.json'),('split',root/'split_integrity.json'),('task_protocol',root/'task_protocol.json'),('train_only_statistics',root/'normalization_statistics_expanded_train_only.json'),('AE_plan',ae/'training_plan.json'),('AE_initialization',ae/'official_initialization.json'),('feature_acceptance',ae/'feature_acceptance.json'),('sidecar_alignment',ae/'sidecar_alignment_audit.json'),('raw_source_recompute',ae/'raw_source_recompute_audit.json'),('sidecar_reader',ae/'official_sidecar_reader_audit.json'),('policy_plan',policy/'run_plan.json'),('evaluation_protocol',policy/'evaluation_protocol.json'),('initialization_and_RNG',policy/'initialization_and_rng_audit.json'),('target_gradient',policy/'preflight6/isolated_smoke/target_gradient_audit.json')]:jsonsave(a.delivery/(task+'_'+label+'.json'),clean(read(path)))
 assert read(policy/'evaluation/isolated_smoke/evaluation_acceptance.json')['valid_trials']==180
 assert read(policy/'initialization_and_rng_audit.json')['status']=='PASS_SAVED_C_D_STEP0_EXACT'
 assert read(ae/'raw_source_recompute_audit.json')['status']=='PASS' and read(ae/'official_sidecar_reader_audit.json')['status']=='PASS'
 dataset.append(dict(task=task,train=manifest['training_episodes'],development=manifest['development_episodes'],physical_frames=manifest['total_frames'],train_chunks=manifest['train_chunks'],development_chunks=manifest['development_chunks'],RGB_images=manifest['total_frames']*4,new_development_states=20))
 for arm in ('C','D'):
  folder=Path(registry[arm]);accept=read(folder/'acceptance.json');assert accept['successful_updates']==600 and accept['strict_matching_encoder_decoder_reload'];rs=csvread(folder/'training_metrics.csv');assert [int(x['step']) for x in rs]==list(range(1,601))
  prior=csvread(root/'ae_run'/('AE_'+arm)/'training_metrics.csv') if task=='stove' else [];cost.append(dict(task=task,stage='AE',arm=arm,effective_updates=600,logged_actual_updates=len(rs)+len(prior),replayed_logged_updates=len(prior),unconfirmed_inflight_update_at_old_fault=task=='stove',seconds=float(accept['seconds'])+(float(prior[-1]['seconds']) if prior else 0),peak_allocated_GiB=accept['peak_allocated']/1024**3))
  current=process_wall(root/(f'AE_{arm}_recovery1.log' if task=='stove' else f'AE_{arm}.log'),folder/'acceptance.json');previous=process_wall(root/f'AE_{arm}.log',root/'ae_run'/('AE_'+arm)/'training_metrics.csv') if prior else 0.;cost[-1]['observed_active_process_wall_seconds_approx']=None if current is None or previous is None else current+previous
  ck=torch.load(folder/'resume_step0600.pt',map_location='cpu',weights_only=False);assert ck['global_step']==600 and ck['next_sample']==4800 and ck['optimizer']['state'];del ck;gc.collect()
  for kind in ('encoder','decoder','resume'):
   path=folder/(kind+'_step0600.pt');inventory.append(dict(task=task,stage='AE',arm=arm,component=kind,path=str(path),bytes=path.stat().st_size,sha256=sha(path),global_step=600))
  latent=csvread(ae/('latent_pairs_'+arm+'.csv'));dist=[r for r in csvread(ae/'latent_distribution_metrics.csv') if r['model']==arm];rec=[r for r in csvread(ae/'reconstruction_metrics.csv') if r['model']==arm]
  for appearance in ('clean','background'):
   d=next(x for x in dist if x['appearance']==appearance);rr=[x for x in rec if x['appearance']==appearance];representations.append(dict(task=task,AE=arm,appearance=appearance,norm=float(d['norm_mean']),participation_ratio=float(d['participation_ratio']),effective_rank=float(d['entropy_effective_rank']),normalized_between_episode_distance=float(d['different_episode_unit_L2_mean']),paired_unit_L2=float(np.mean([float(x['unit_L2']) for x in latent])),paired_relative_L2=float(np.mean([float(x['relative_L2']) for x in latent])),DINO_L1=float(np.mean([float(x['reconstruction_L1']) for x in rr])),DINO_MSE=float(np.mean([float(x['reconstruction_MSE']) for x in rr])),copy_current_L1=float(np.mean([float(x['copy_current_L1']) for x in rr]))))
 ref_pairs=csvread(ae/'latent_pairs_reference.csv');ref_dist=csvread(ae/'latent_distribution_metrics.csv');allrec=csvread(ae/'reconstruction_metrics.csv')
 for appearance in ('clean','background'):
  d=next(x for x in ref_dist if x['model']=='reference' and x['appearance']==appearance);rr=[x for x in allrec if x['model']=='reference' and x['appearance']==appearance];representations.append(dict(task=task,AE='reference_step90000',appearance=appearance,norm=float(d['norm_mean']),participation_ratio=float(d['participation_ratio']),effective_rank=float(d['entropy_effective_rank']),normalized_between_episode_distance=float(d['different_episode_unit_L2_mean']),paired_unit_L2=float(np.mean([float(x['unit_L2']) for x in ref_pairs])),paired_relative_L2=float(np.mean([float(x['relative_L2']) for x in ref_pairs])),DINO_L1=float(np.mean([float(x['reconstruction_L1']) for x in rr])),DINO_MSE=float(np.mean([float(x['reconstruction_MSE']) for x in rr])),copy_current_L1=float(np.mean([float(x['copy_current_L1']) for x in rr]))))
 groups=collections.defaultdict(list)
 for r in allrec:groups[(r['model'],r['episode'],r['h'],r['appearance'],r['camera'])].append(r)
 for (model,episode,h,appearance,camera),rs in groups.items():rec_groups.append(dict(task=task,model=model,episode=episode,horizon=int(h),appearance=appearance,camera=int(camera),samples=len(rs),**{k:float(np.mean([float(x[k]) for x in rs])) for k in ('reconstruction_L1','reconstruction_MSE','copy_current_L1','copy_current_MSE','ee_displacement_m')}))
 for arm in ('B','C','D'):
  folder=policy/arm/'isolated_smoke';accept=read(folder/'training_acceptance.json');assert accept['successful_updates']==600 and accept['strict_reload'];rs=csvread(folder/'training_metrics.csv');assert [int(x['step']) for x in rs]==list(range(1,601));cost.append(dict(task=task,stage='policy',arm=arm,effective_updates=600,logged_actual_updates=len(rs),replayed_logged_updates=0,unconfirmed_inflight_update_at_old_fault=False,seconds=float(accept['seconds']),peak_allocated_GiB=accept['peak_allocated']/1024**3))
  log=root/f'policy_{arm}_parallel.log';log=log if log.exists() else root/f'policy_{arm}.log';cost[-1]['observed_active_process_wall_seconds_approx']=process_wall(log,folder/'training_acceptance.json')
  path=folder/'step0600.pt';ck=torch.load(path,map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['scheduler']['last_epoch']==600;assert all(int(v['step'])==600 for v in ck['optimizer']['state'].values());assert all(k in ck for k in ('torch_rng','cuda_rng','python_rng','numpy_rng'));del ck;gc.collect();inventory.append(dict(task=task,stage='policy',arm=arm,component='trainables_optimizer_scheduler_RNG',path=str(path),bytes=path.stat().st_size,sha256=sha(path),global_step=600))
  training.extend(dict(task=task,arm=arm,**r) for r in rs)
  vals=csvread(folder/'validation_metrics.csv');groups=collections.defaultdict(list)
  for r in vals:groups[(int(r['step']),r['appearance'],r['episode'])].append(r)
  for (step,appearance,episode),values in groups.items():validation.append(dict(task=task,arm=arm,step=step,appearance=appearance,episode=episode,**{k:float(np.mean([float(r[k]) for r in values])) for k in ('action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1')}))
 for r in read(policy/'preflight6/isolated_smoke/target_gradient_audit.json')['auxiliary_checks']:gradients.append(dict(task=task,**r))
success=read(a.run/'success_rate_summary.json');equal=read(a.run/'three_task_equal_summary.json');pairs=read(a.run/'paired_statistics.json');assert sum(x['effective_updates'] for x in cost if x['stage']=='AE')==2400 and sum(x['effective_updates'] for x in cost if x['stage']=='policy')==3600
for name,rows in [('checkpoint_inventory.csv',inventory),('training_cost.csv',cost),('representation_summary.csv',representations),('reconstruction_by_demo_horizon.csv',rec_groups),('validation_metrics.csv',validation),('training_metrics.csv',training),('dataset_summary.csv',dataset),('auxiliary_gradient_contribution.csv',gradients)]:csvsave(out/name,rows)
lines=['# 两个新增任务：固定AE与B/C/D训练及development闭环','', '本实验研究背景材质干预下的视觉鲁棒性，不是官方LIBERO-Plus。主比较D−C；单训练seed、三个固定取碗放盘任务，仅提供探索性证据。','', '## 成功次数（每项20个物理状态）','', '|任务|外观|B|C|D|D−C|','|---|---|---|---|---|---|']
for task in ('cookie_historical','drawer','stove'):
 for condition in ('clean','seen_background','unseen_background'):
  values={r['model']:r['successes'] for r in success if r['task']==task and r['appearance']==condition};lines.append(f"|{task}|{condition}|{values['B']}/20|{values['C']}/20|{values['D']}/20|{(values['D']-values['C'])*5:+d} pp|")
lines+=['','## 主比较与不确定性','']
for r in pairs:
 if r['comparison']=='D-C':lines.append(f"- {r['task']} / {r['appearance']}: {100*r['difference']:+.2f} pp；状态聚类bootstrap95% CI={[round(100*x,2) for x in r['CI95']]}；成功翻转+{r['gain']}/−{r['loss']}。")
lines+=['','三个任务等权平均（仅条件于这三个任务，不是跨任务总体置信区间）：','', '```json',json.dumps(equal,ensure_ascii=False,indent=2),'```','', '## 数据、训练与恢复','', '每个新增任务20条训练＋5条development示范，按源episode隔离。每任务20个固定合法reset状态，Clean/已见/未见材质配对；未见材质为预先指定的三种调色板，按状态序号固定分配。原cookie历史结果未重跑，其未见背景仅一种，因此跨任务OOD均值并非完全相同的材质分布。','', '四组AE各600有效更新，总计2400；六组策略各600有效更新，总计3600；新增有效闭环360，任务失败保留为有效结果。主比较仅用最终600步。AE-C/D使用官方step90000共同初始化；策略均使用原始共同未训练初始化，不续训B20。B是官方action-only结构，C/D是官方RLA结构；C−B包含完整RLA接入差异，C/D结构与初始化一致。','', 'GPU3此前出现correctable ECC增量后停止，未重新使用；GPU2始终排除。炉灶两AE各重放44个已记录更新，记录实际计算644步/组，另各有至多一个旧中断时在途更新无法确认。前44步loss、梯度范数及采样日志复现一致，但不声称对未保存参数进行了逐位验证。历史故障、部分缓存和checkpoint均保留。抽屉缓存在新目录完整重提取；已记录wrapper脚本编辑导致的提取结束后shell解析错误，其Python完整验收文件独立核验通过。','', 'GPU资源释放后另附ECC终检。跨卡排程只改变运行时间，不改变数据、seed、batch、优化器、系数和最终checkpoint选择。已完成的并行arm由原调度核验复用，不重复执行更新。','', '## 表示与策略机制限制','', '详见representation_summary.csv、按示范/horizon的原始指标及decoder的zero/other-episode latent消融。必须同时看归一化距离、latent尺度/有效秩及重建相对copy-current基线；不把距离变小等同动作语义保真，也不因策略失败断言表示坍塌。','', '固定lambda_aux=0.22408536585365854（不按新任务重新校准）；lambda_cons=0.009127798272072366。C/D共享AE-C训练集归一化。共同系数并不保证共同辅助梯度强度，已保存实际辅助loss及共享backbone辅助梯度范数；D−C可能包含表示尺度和有效监督强度变化，不能全部归因于外观不变性。','', '失败阶段采用统一的目标抬升/末端距离轨迹代理，不把代理或视频观察当作接触物理真值。所有trial均保存日志、视频和轨迹；不根据失败重选状态或追加训练。','', '## 限制与下一步','', '仅一个匹配AE/策略训练seed；20个development状态/任务，非封存最终测试。三个任务仍是取碗放盘，操作场景不同但primitive覆盖有限。天花板/地板、少量成功翻转、离散bootstrap及经验零下界均不能证明稳定收益。若接近满分，应认定现有协议区分能力有限，而非方法已被证实。','', '后续每任务总计3个匹配seed，新增seed20261024/20261025须同时重训AE-C/AE-D和策略B/C/D，从共同原始初始化出发；只换策略seed只能说明给定AE下的稳定性。不在本轮执行。若需扩展状态/背景，必须先冻结新的合法reset种子及互斥未见材质清单，仅按合法性保留，不筛选D获胜样本；封存集仍不读取。','']
lines+=['','历史cookie任务沿用run09原结果及其原始恢复记录：原AE-D的80步重放、策略C的额外一次已记录计算均不并入本轮成本，也不被抹除。因此600步指有效更新预算，不声称三任务的实际计算次数完全相同。','', '成本CSV中的seconds为记录的更新/固定验证阶段耗时；observed_active_process_wall_seconds_approx使用日志出生时间至验收文件时间，包含初始化且不计无人运行的停机等待。并行进程耗时相加不是整体墙钟时间或精确GPU kernel busy time。','', '目录isolated_smoke是历史bootstrap命名，正式checkpoint由scope/global_step600及optimizer/scheduler状态验收。其legacy batch_audit.json描述的是未用于正式forward的reference bootstrap batch；真实训练数据由policy_training.execute中的独立dataset/batch构造，每个实际模型batch均检查任务token。formal_batch_provenance_audit.json明确区分二者，不以legacy参考batch推断正式任务来源。','']
text='\n'.join(lines);(out/'report.md').write_text(text)
receipt=dict(status='PASS_PIPELINE_NOT_METHOD',AE_effective_updates=2400,AE_logged_computations=sum(r['logged_actual_updates'] for r in cost if r['stage']=='AE'),AE_old_inflight_updates_unconfirmed=True,policy_effective_updates=3600,valid_new_closed_loop_trials=360,invalid_trials=0,training_seed=20261023,heldout_read=False,original_outputs_preserved=True,commit_pending=True)
jsonsave(out/'delivery_verification.json',receipt)
for path in out.iterdir():
 if path.suffix in ('.json','.csv','.md'):(a.delivery/path.name).write_text(clean(path.read_text()))
for name in ('success_rate_summary.json','paired_statistics.json','interaction_statistics.json','three_task_equal_summary.json','task_stratified_metrics.csv','paired_comparisons.csv','failure_stage_analysis.csv','unseen_palette_metrics.csv','extension_protocol.json','development_identity_audit.json','formal_batch_provenance_audit.json'):(a.delivery/name).write_text(clean((a.run/name).read_text()))
print(text)
