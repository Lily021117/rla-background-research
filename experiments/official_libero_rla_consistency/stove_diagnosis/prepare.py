"""Freeze samples, rehash task provenance, audit existing rollouts. No GPU."""
import argparse,collections,os,subprocess
from common import *
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
root=a.source;ae=root/'ae_run';pol=root/'policy_run';manifest=read(root/'frozen_dataset_manifest.json');es=manifest['episodes'];plan=read(pol/'run_plan.json')
assert len([e for e in es if e['split']=='train'])==20 and len([e for e in es if e['split']=='validation'])==5
save(a.out/'configuration.json',dict(source=str(root),policy_steps=[0,1,100,300,600],policy_anchor_rule='first valid, first raw open-to-close (clamped to full valid chunk), last valid; deduplicate',gradient_rule='first four episodes in frozen manifest of each split, closure anchor, clean and background; effective batch8; eval mode with grad enabled; Math SDPA BF16; separate action/weighted auxiliary gradients',AE_samples='existing ten evenly spaced anchors per episode x h1..8; all matching samples',probe='fixed ridge alpha=1 on average squared loss; train clean h8,10anchors per20episodes; train-only feature zscore; validation5episodes both appearances; no hyperparameter selection',phase_rule='derived ground-state bowl lift>=.03m -> transport_or_placement; otherwise first raw closure +/-4 steps -> closure_window; earlier -> approach; later -> after_close_no_lift; descriptive only',video_rule='first seed in each appearance, five uniform C-timeline indices and corresponding clamped D times; all trials classified separately',optimizer_updates=0,new_rollouts=0,sealed_read=False))
samples=[];gradient=[];counts=[];inputs=[]
def protect(path,kind,expected=None):
 path=Path(path);h=sha(path)
 if expected is not None:assert h==expected,(path,h,expected)
 inputs.append(dict(path=str(path),kind=kind,bytes=path.stat().st_size,sha256=h));return h
for path in [root/'frozen_dataset_manifest.json',root/'split_integrity.json',root/'normalization_statistics_expanded_train_only.json',pol/'run_plan.json',pol/'sampling_order.json',pol/'policy_registry.json',pol/'evaluation_protocol.json',pol/'per_trial_results.json',pol/'initialization_and_rng_audit.json',ae/'training_plan.json',ae/'official_initialization.json',ae/'data_manifest.json',ae/'sidecar_provenance.json',root/'recovery_prefix_audit.json'] :protect(path,'configuration_or_evidence')
protect(plan['common_step0'],'policy_RLA_initial',plan['common_step0_sha256']);protect(plan['inputs']['B20_step0'],'policy_baseline_initial')
order=read(pol/'sampling_order.json');samplecounts=collections.Counter(e for e,t in order);assert len(order)==4800
train={e['episode'] for e in es if e['split']=='train'};val={e['episode'] for e in es if e['split']=='validation'};assert not train&val and set(samplecounts)==train
for e in es:
 lab=np.load(Path(e['root'])/'trajectory_labels.npz');protect(Path(e['root'])/'trajectory_labels.npz','derived_labels');raw=lab['actions'][:,6];event=np.flatnonzero((raw[:-1]<0)&(raw[1:]>0))+1;tclose=int(event[0]) if len(event) else (e['valid_chunks']-1)//2;anchor=min(tclose,e['valid_chunks']-1)
 for t in sorted({0,anchor,e['valid_chunks']-1}):
  for c in ('clean','background'):samples.append(dict(episode=e['episode'],split=e['split'],t=t,appearance=c,source_demo=e['source_demo'],closure_t=tclose))
 if sum(x['split']==e['split'] for x in gradient)<4:gradient.append(dict(episode=e['episode'],t=anchor,split=e['split']))
 counts.append(dict(episode=e['episode'],split=e['split'],source_demo=e['source_demo'],task=e['task'],frames=e['frames'],valid_full_chunks=e['valid_chunks'],sampled_physical_windows=samplecounts[e['episode']],appearance_exposures=2*samplecounts[e['episode']]))
 assert e['task']=='pick up the black bowl on the stove and place it on the plate'
 assert len(lab['states'])==len(lab['actions'])==e['frames'] and np.all(lab['valid_horizon_mask'][:e['valid_chunks']])
save(a.out/'fixed_sample_manifest.json',dict(policy=samples,gradient=gradient,source_manifest_sha256=sha(root/'frozen_dataset_manifest.json')));csvsave(a.out/'training_episode_identity.csv',counts)
side=read(ae/'sidecar_provenance.json');assert len(side)==100
reg=read(sorted((ae/'model_registry_versions').glob('*.json'))[-1]);enc={}
for arm in ('C','D'):
 for component in ('encoder','decoder','resume'):
  path=Path(reg[arm])/(component+'_step0600.pt');h=protect(path,'AE_'+arm+'_'+component)
  if component=='encoder':enc[arm]=h
 for c in ('clean','background'):
  for n in ('manifest.json','stats.json','libero_spatial/keys.json'):protect(ae/'sidecars'/arm/c/n,'sidecar_'+arm+'_'+c)
 assert read(ae/'sidecars'/arm/'clean/stats.json')==read(ae/'sidecars/C/clean/stats.json')
for s in side:
 protect(s['file'],'sidecar_tensor',s['sha256']);protect(s['mask_file'],'sidecar_mask');assert s['episode'] in train|val and s['encoder_sha256']==enc[s['model']]
registry=read(pol/'policy_registry.json');trial=read(pol/'per_trial_results.json');assert len(trial)==180
for arm in ('B','C','D'):
 for step in (0,1,100,300,600):protect(pol/arm/'isolated_smoke'/f'step{step:04}.pt','policy_'+arm+f'_step{step}')
 for n in ('execution_configuration.json','training_metrics.csv','training_acceptance.json'):protect(pol/arm/'isolated_smoke'/n,'policy_'+arm+'_execution')
 ckhash=next(x['sha256'] for x in inputs if x['path']==registry[arm]);assert all(r['checkpoint_sha256']==ckhash for r in trial if r['model']==arm)
 actualpath=pol/('preflight_B' if arm=='B' else 'preflight6')/'isolated_smoke/actual_task_batch_audit.json';protect(actualpath,'actual_formal_dataset_preflight_tokens');actual=read(actualpath);assert all(x['episode'] in train and x['language_verified_in_actual_tokens'] and 'on the stove' in x['instruction'] for x in actual)
save(a.out/'protected_input_hashes.json',inputs);save(a.out/'identity_summary.json',dict(status='PASS_FILE_IDENTITY_PENDING_CHECKPOINT_STATE_AUDIT',training_episodes=20,development_source_episodes=5,train_valid_windows=manifest['train_chunks'],development_valid_windows=manifest['development_chunks'],evaluated_physical_states=20,policy_physical_exposures=4800,policy_appearance_exposures=9600,AE_data_same_episodes=read(ae/'data_manifest.json')['episodes']==es,legacy_cookie_bootstrap='unused reference batch in old bootstrap, not formal forward; explicit actual-task tokens and sampling checked',no_trained_cookie_initialization=True,sidecar_files=100,policy_registry_matches_all180_trials=True))
rows=[];pairs=[];stages=collections.Counter()
for r in trial:
 d=np.load(r['trajectory']);pred=d['predicted_chunks'].reshape(-1,7)[:len(d['executed_actions'])].copy();pred[:,6]=-np.sign(2*pred[:,6]-1);assert np.array_equal(pred,d['executed_actions']);ee=d['eef_positions'];bowl=d['target_bowl_positions'];dist=np.linalg.norm(ee-bowl,axis=1);initialfile=next(x for x in read(ae/'development_protocol/evaluation_protocol.json')['new_development_states'] if x['seed']==r['seed'])['state_file']
 # Keep existing frozen classification, recheck numerical criteria independently below.
 stage='success' if r['success'] else 'post_lift_or_placement_failure' if r['max_lift_m']>=.03 else 'near_target_grasp_or_contact_failure' if dist.min()<=.10 else 'approach_failure';assert stage==r['stage'];stages[r['model'],r['condition'],stage]+=1
 near=np.flatnonzero(dist<=.10);closed=np.flatnonzero(d['executed_actions'][:,6]>0)
 rows.append(dict(model=r['model'],seed=r['seed'],appearance=r['condition'],success=r['success'],steps=r['steps'],stage_proxy=stage,min_distance_m=float(dist.min()),first_near_step=int(near[0]) if len(near) else -1,first_close_command_step=int(closed[0]) if len(closed) else -1,close_fraction=float(np.mean(d['executed_actions'][:,6]>0)),max_lift_m=r['max_lift_m'],translation_saturation=float(np.mean(abs(pred[:,:3])>=.999)),commands_equal_processed_predictions=True,trajectory=r['trajectory'],video=r['video'],state_hash=r['state_sha256'],xml_hash=r['xml_sha256']))
for seed in sorted({r['seed'] for r in trial}):
 for c in ('clean','seen_background','unseen_background'):
  rs={r['model']:r for r in rows if r['seed']==seed and r['appearance']==c};assert rs['C']['state_hash']==rs['D']['state_hash'] and rs['C']['xml_hash']==rs['D']['xml_hash'];ct=np.load(rs['C']['trajectory']);dt=np.load(rs['D']['trajectory']);n=min(len(ct['eef_positions']),len(dt['eef_positions']));div=np.linalg.norm(ct['eef_positions'][:n]-dt['eef_positions'][:n],axis=1);cross=np.flatnonzero(div>.02)
  pairs.append(dict(seed=seed,appearance=c,C_success=rs['C']['success'],D_success=rs['D']['success'],C_stage=rs['C']['stage_proxy'],D_stage=rs['D']['stage_proxy'],first_2cm_divergence_step=int(cross[0]) if len(cross) else -1,initial_chunk_action_L1=float(abs(ct['predicted_chunks'][0]-dt['predicted_chunks'][0]).mean()),C_first_near_step=rs['C']['first_near_step'],D_first_near_step=rs['D']['first_near_step'],C_first_close_step=rs['C']['first_close_command_step'],D_first_close_step=rs['D']['first_close_command_step']))
csvsave(a.out/'all_trial_stage_audit.csv',rows);csvsave(a.out/'paired_rollout_audit.csv',pairs);csvsave(a.out/'failure_stage_counts.csv',[dict(model=m,appearance=c,stage_proxy=s,count=n) for (m,c,s),n in stages.items()])
save(a.out/'git_start.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),status=subprocess.check_output(['git','status','--short'],text=True)))
print('PREPARED',a.out,'fixed observations',len(samples),flush=True)
