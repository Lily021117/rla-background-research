"""Read-only CPU diagnostics. Never imports a training entry or creates an optimizer."""
import os
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('${RLA_PROJECT_ROOT}');E=R/'experiments/rla_background_invariance'
W=R/'worktrees/rla-background-causal-unified-trainer'
T=E/'step5_7_causal_training/run_20261022_01';V=E/'step5_8_closed_loop/run_20261022_01'
F=E/'step4_5_3_feature_target_expansion/run_20261021_01'
S=E/'step5_1_policy_training/run_20261022_02';N=E/'step4_1_adapter_readiness/run_20261021_02'
for k,v in {'CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'disable','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}.items():os.environ[k]=v
import json,csv,hashlib,subprocess,collections,time,gc,re,math
import numpy as np,torch,h5py,mujoco
torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def cs(n,rows):
 with (O/n).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p):return json.loads(Path(p).read_text())
def git(p):return {k:subprocess.check_output(['git','-C',str(p)]+a,text=True) for k,a in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short'],'remote':['remote','-v']}.items()}
start=time.time();print('START CPU read-only audit',flush=True)
protected=read(V/'protection_before.json')
for folder in [T,V]:
 for p in folder.rglob('*'):
  if p.is_file() and not p.is_symlink() and '_runtime' not in str(p):protected[str(p)]=sha(p)
for p in W.glob('*.py'):protected[str(p)]=sha(p)
js('protection_before.json',protected);js('git_before.json',{str(p):git(p) for p in [R,W,R.parent/'vla-adapter-ref']})
assert read(T/'training_completion.json')['updates']==dict(B=100,C=100,D=100)
assert read(V/'delivery_verification.json')['valid_trials']==160
samples=read(S/'sample_manifest.json');trans=read(F/'selection_manifest.json')['transitions'];trby={t['transition_id']:t for t in trans}
stats=read(R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint/dataset_statistics.json')['libero_spatial_no_noops']
trainstat=E/'step4_1_adapter_readiness/run_20261021_03/runtime_metadata/dataset_statistics.json'
assert read(trainstat)['libero_spatial_no_noops']==stats
ain=stats['action'];pin=stats['proprio'];lo=np.asarray(ain['q01']);hi=np.asarray(ain['q99']);mask=np.asarray(ain.get('mask',[True]*7));plo=np.asarray(pin['q01']);phi=np.asarray(pin['q99']);pm=np.asarray(pin.get('mask',[True]*8))
inputs=np.load(F/'batch_inputs.npz');timing=[];examples=[];action_errors=[];proprio_errors=[];clipcount=np.zeros(7,int);totalpoints=0;short=0
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint',local_files_only=True)
for i,s in enumerate(samples):
 assert sha(s['path'])==s['SHA256'];b=torch.load(s['path'],map_location='cpu',weights_only=False);t=trby[s['transition_id']];idx=int(s['transition_index']);a=inputs[f'actions_{idx}'].astype(float);p=inputs[f'proprio_{idx}'][0].astype(float)
 with h5py.File(t['source_hdf5'],'r') as f:
  raw=f['data'][t['source_demo']];s0,u=t['source_start'],t['source_future'];original=raw['actions'][s0:u];elapsed=float(raw['states'][u,0]-raw['states'][s0,0]);length=len(raw['actions'])
 np.testing.assert_allclose(a,original,rtol=0,atol=3e-8)
 standardized=a.copy();standardized[:,-1]=1-np.clip(a[:,-1],0,1)
 unbounded=2*(standardized-lo)/(hi-lo+1e-8)-1
 expected=np.where(mask,np.clip(unbounded,-1,1),standardized)
 expected[:,np.asarray(ain['min'])==np.asarray(ain['max'])]=0
 actual=b['actions'][0].float().numpy();err=float(abs(actual-expected).max());action_errors.append(err);assert err<1e-6
 ep=np.where(pm,np.clip(2*(p-plo)/(phi-plo+1e-8)-1,-1,1),p);ep[np.asarray(pin['min'])==np.asarray(pin['max'])]=0
 pe=float(abs(b['proprio'][0].float().numpy()-ep).max());proprio_errors.append(pe);assert pe<1e-6
 text=tok.decode(b['input_ids'][0]);assert s['instruction'] in text.lower()
 if s['appearance']=='clean':
  clipcount+=((abs(unbounded)>1)&mask).sum(0);totalpoints+=8
  timing.append({'episode':s['episode'],'transition_id':s['transition_id'],'split':s['split'],'image_state':s0,'FK_state':s0,'action_start':s0,'action_stop_exclusive':u,'future_state':u,'horizon':u-s0,'elapsed_s':elapsed,'within_episode':u<len(raw['states']) if False else u<length,'padding_steps':0,'action_max_error':err,'proprio_max_error':pe,'language_pass':True,'status':'PASS'})
 if i%20==0:
  recovered=np.where(mask,.5*(actual+1)*(hi-lo+1e-8)+lo,actual);deployed=recovered.copy();deployed[:,-1]=-np.sign(2*recovered[:,-1]-1)
  for j in [0,7]:examples.append({'sample_id':i,'episode':s['episode'],'appearance':s['appearance'],'chunk_step':j,'raw_action':json.dumps(a[j].tolist()),'standardized_action':json.dumps(standardized[j].tolist()),'training_target':json.dumps(actual[j].tolist()),'unnormalized_action':json.dumps(recovered[j].tolist()),'deployment_input':json.dumps(deployed[j].tolist()),'raw_vs_deployment_first6_max':float(abs(a[j,:6]-deployed[j,:6]).max()),'gripper_matches_raw_sign':bool(np.sign(a[j,-1])==deployed[j,-1])})
 if i%100==0:print('BATCH',i,flush=True)
cs('temporal_alignment_audit.csv',timing);cs('action_roundtrip_examples.csv',examples)
js('action_pipeline_audit.json',{'samples_checked':len(samples),'unique_transitions':len(timing),'dimension_order':['delta_xyz_command_x','delta_xyz_command_y','delta_xyz_command_z','delta_rotation_command_x','delta_rotation_command_y','delta_rotation_command_z','gripper_command'],'controller':'OSC_POSE; world-frame delta position and relative axis-angle orientation command; gain scaling, not metres/radians directly','controller_output_max':[.05,.05,.05,.5,.5,.5],'control_delta':True,'training_action_statistics_equal_deployment':True,'action_normalization':'BOUNDS_Q99 first6 clip to[-1,1]; gripper 1-clip(raw,0,1); published mask respected','action_statistics':ain,'proprio_statistics':pin,'max_training_target_recompute_error':max(action_errors),'max_FK_proprio_normalization_error':max(proprio_errors),'clipped_action_values_by_dimension':clipcount.tolist(),'clean_unique_chunk_steps':totalpoints,'clipping_intentional_noninvertible':True,'gripper_deploy':'-sign(2*p_open-1); exactly p_open=.5 maps0','model_output_clipping':False,'controller_clipping':'scale_action clips controller first6 to input[-1,1] before rescale','chunk_size':8,'executed_before_replan':8,'horizon':8,'microbatch':2,'accumulation':2,'effective_batch':4,'loss':'mean L1 over batch x8 x7; no padded timesteps in accepted chunks','future_target_forward_input':False,'raw_proprio_used':False,'input_proprio':'FK from exact state[s], world grip_site xyz / hand body axis-angle / finger qpos','physical_state_change_loss':False,'explicit_action_input_to_auxiliary':False,'action_conditioned_forward_predictor':False,'known_native_train_inference_difference':'new random Normal(0,.02) tensor added to action-head zero query input only phase=Training; eval phase has none','A_wrapper_native_equivalence':'existing 8 task/appearance real-input step0 comparisons max0; no new full closed-loop equivalence test performed'})
print('ACTION_TIMING_PASS',flush=True)
# Targets: the policy consumes agentview only. Keep same physical transition IDs.
targets=list(csv.DictReader(open(F/'rla_target_manifest.csv')));targetrows=[]
for source,arm in [('B','C'),('C','D')]:
 n=np.load(N/f'normalization_{source}.npz');mean=n['mean'];scale=n['scale']
 for split in ['train','validation']:
  for app in ['clean','background']:
   rr=[r for r in targets if r['group']==source and r['camera']=='agentview' and r['split']==split and r['appearance']==app]
   raw=np.stack([np.load(r['path']).reshape(-1) for r in rr]).astype(float);normalized=(raw-mean.reshape(-1))/float(scale)
   for name,x in [('raw',raw),('normalized',normalized)]:
    centered=x-x.mean(0);gram=centered@centered.T;tr=float(np.trace(gram));pr=tr*tr/float((gram*gram).sum()) if tr else 0;std=x.std(0)
    targetrows.append({'policy_arm':arm,'encoder_source':source,'split':split,'appearance':app,'space':name,'windows':len(rr),'global_mean':float(x.mean()),'global_variance':float(x.var()),'mean_norm':float(np.linalg.norm(x,axis=1).mean()),'norm_std':float(np.linalg.norm(x,axis=1).std()),'per_dim_std_q05':float(np.quantile(std,.05)),'per_dim_std_median':float(np.median(std)),'per_dim_std_q95':float(np.quantile(std,.95)),'participation_ratio_centered':pr,'normalization_scale':float(scale),'normalization_fit':'existing Step4.1 train episodes only; not refitted on validation; limited support'})
cs('auxiliary_target_statistics.csv',targetrows)
order=read(S/'sampling_order.json');counts=collections.Counter(i for row in order for i in row);episodes=collections.defaultdict(collections.Counter);appcounts=collections.Counter()
for i,n in counts.items():
 s=samples[i];episodes[s['episode']]['exposures']+=n;episodes[s['episode']]['unique_appearance_windows']+=1;appcounts[s['appearance']]+=n
sampling=[{'episode':ep,'task':next(s['instruction'] for s in samples if s['episode']==ep),**dict(c)} for ep,c in episodes.items()];cs('sampling_by_episode.csv',sampling)
paramrows=list(csv.DictReader(open(T/'parameter_structure_comparison.csv')));trainable=[r for r in paramrows if r['arm']=='B' and r['trainable']=='True'];cs('trainable_parameters.csv',trainable)
modulecounts=collections.Counter()
for r in trainable:modulecounts[r['name'].split('.')[0]]+=int(r['numel'])
metrics=list(csv.DictReader(open(T/'training_metrics.csv')));vm=list(csv.DictReader(open(T/'validation_metrics.csv')))
curves=[]
for arm in 'BCD':
 rr=[r for r in metrics if r['arm']==arm]
 for a,b in [(1,10),(41,50),(91,100)]:
  rows=[r for r in rr if a<=int(r['global_step'])<=b]
  curves.append({'arm':arm,'steps':f'{a}-{b}','action_loss':float(np.mean([float(r['action_loss']) for r in rows])),'aux_loss':float(np.mean([float(r['aux_loss']) for r in rows])),'weighted_aux_loss':float(np.mean([float(r['aux_loss']) for r in rows]))*(0 if arm=='B' else .05),'shared_backbone_grad_norm':float(np.mean([float(r['shared_backbone_grad_norm']) for r in rows]))})
cs('training_curve_summary.csv',curves)
val=[]
for arm in 'BCD':
 for step in [0,100]:
  for app in ['clean','background']:
   rr=[r for r in vm if r['arm']==arm and int(r['step'])==step and r['appearance']==app]
   per_ep=[np.mean([float(r['action_L1']) for r in rr if r['episode']==ep]) for ep in {r['episode'] for r in rr}]
   val.append({'arm':arm,'step':step,'appearance':app,'action_L1_demo_equal':float(np.mean(per_ep)),'episodes':len(per_ep)})
cs('validation_curve_summary.csv',val)
pre=read(T/'step0_preflight.json')
js('optimizer_and_sampling_audit.json',{'optimizer':read(T/'training_config_B.json')['optimizer'],'scheduler':'constant LambdaLR','gradient_clipping':False,'amp_scaler':False,'precision':'BF16 native model/head/proprio, FP32 PEFT LoRA','trainable_parameter_counts':dict(modulecounts),'total_trainable':sum(int(r['numel']) for r in trainable),'full_names_file':'trainable_parameters.csv','microbatch':2,'accumulation':2,'effective_batch':4,'updates':read(T/'training_completion.json')['updates'],'replayed':0,'skipped':0,'exposures_per_arm':sum(counts.values()),'available_train_appearance_windows':380,'unique_sampled_appearance_windows':len(counts),'unique_physical_transitions_sampled':len({samples[i]['transition_id'] for i in counts}),'unique_episodes_sampled':len(episodes),'appearance_counts':dict(appcounts),'sample_exposure_min':min(counts.values()),'sample_exposure_max':max(counts.values()),'exposure_task_counts':dict(collections.Counter({task:sum(r['exposures'] for r in sampling if r['task']==task) for task in {r['task'] for r in sampling}})),'validation_episodes':3,'missing_validation_task':'stove','step0_weighted_aux_gradient_norms':{arm:pre['arms'][arm]['weighted_aux_gradients'] for arm in 'BCD'},'fixed_batch_target_loss':{arm:pre['arms'][arm]['aux_loss'] for arm in 'BCD'}})
# Compare already saved parameters only; no network forward/backward here.
base=torch.load(str(T/'common_step0.pt'),map_location='cpu',weights_only=False,mmap=True);changes=[]
for arm in 'BCD':
 for step in [1,50,100]:
  ck=torch.load(str(T/arm/f'step{step:04d}.pt'),map_location='cpu',weights_only=False,mmap=True)
  for group in ['vlm','head','proprio']:
   diffsq=refsq=0.;maxabs=0.;changed=0;n=0
   for name,x in ck[group].items():
    ref=base[group][name];delta=x.float()-ref.float();diffsq+=float(delta.double().square().sum());refsq+=float(ref.double().square().sum());maxabs=max(maxabs,float(delta.abs().max()));changed+=int(delta.count_nonzero());n+=x.numel()
   changes.append({'arm':arm,'step':step,'module':group,'numel':n,'changed_coordinates':changed,'delta_L2':math.sqrt(diffsq),'relative_L2':math.sqrt(diffsq/refsq) if refsq else 'UNDEFINED_ZERO_REFERENCE','max_abs':maxabs})
  del ck;gc.collect()
del base;gc.collect();cs('parameter_change_metrics.csv',changes);print('PARAMETER_PASS',flush=True)
# Physical labels from exact original states. Forward kinematics only, no env.step/render.
split=read(E/'step2_training/split_manifest.json')['splits'];entries={e['episode_id']:e for sp in ['train','validation'] for e in split[sp]};models={};labelrows=[];motion=[]
def model_for(ep):
 if ep not in models:
  p=Path(entries[ep]['geometry_path']).parent/'clean.xml';m=mujoco.MjModel.from_xml_path(str(p));models[ep]=(m,mujoco.MjData(m))
 return models[ep]
def physical(m,d,state):
 assert len(state)==1+m.nq+m.nv;d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d)
 ids={name:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name) for name in ['akita_black_bowl_1_main','plate_1_main']};assert min(ids.values())>=0
 site=m.site('gripper0_grip_site').id;fi=[int(m.jnt_qposadr[m.joint(f'gripper0_finger_joint{i}').id]) for i in [1,2]]
 return {'bowl':d.xpos[ids['akita_black_bowl_1_main']].copy(),'plate':d.xpos[ids['plate_1_main']].copy(),'eef':d.site_xpos[site].copy(),'gap':float(d.qpos[fi[0]]-d.qpos[fi[1]]),'drawer':float(d.qpos[m.jnt_qposadr[m.joint('wooden_cabinet_1_top_level').id]]),'stove':float(d.qpos[m.jnt_qposadr[m.joint('flat_stove_1_button').id]])}
for t in trans:
 m,d=model_for(t['episode']);entry=entries[t['episode']]
 with h5py.File(entry['source_hdf5'],'r') as f:
  ss=f['data'][entry['source_demo']]['states'];p0=physical(m,d,ss[t['source_start']]);p1=physical(m,d,ss[t['source_future']])
 for name in ['bowl','gap','drawer','stove']:
  delta=np.asarray(p1[name])-np.asarray(p0[name]);labelrows.append({'episode':t['episode'],'transition_id':t['transition_id'],'split':t['split'],'label':name,'available':True,'source':'exact recorded simulator state + matching XML/mj_forward','current_index':t['source_start'],'future_index':t['source_future'],'delta':json.dumps(delta.tolist()),'units':'m' if name!='stove' else 'rad','same_state_image_mapping':'prior source-ID audit + current actions/FK recheck','independent_external_ground_truth':False,'task_semantics':'black bowl_1 from BDDL task goal; drawer/stove joints contextual, not task success proxies'})
cs('physical_label_availability.csv',labelrows)
states=read(V/'state_manifest.json');state_by={str(x['state_id']):x for x in states};trials=list(csv.DictReader(open(V/'per_trial_results.csv')));fail=[];trialby={r['trial_id']:r for r in trials}
for row in trials:
 st=state_by[str(row['state_id'])];m=mujoco.MjModel.from_xml_path(st['clean_xml']);d=mujoco.MjData(m)
 with np.load(row['trajectory']) as z:
  sim=z['sim_states'];aa=z['executed_actions'];chunks=z['predicted_chunks'];first=np.asarray(chunks[0,0]);pp=[physical(m,d,x) for x in sim];bowl=np.stack([x['bowl'] for x in pp]);eef=np.stack([x['eef'] for x in pp]);plate=np.stack([x['plate'] for x in pp]);gap=np.array([x['gap'] for x in pp]);dist=np.linalg.norm(eef-bowl,axis=1)
  success=row['success']=='True';lift=float((bowl[:,2]-bowl[0,2]).max());near=float(dist.min());startdist=float(dist[0]);enddist=float(dist[-1]);displacement=float(np.linalg.norm(bowl[-1]-bowl[0]));close=int((aa[:,-1]>0).sum());switches=int((np.diff(aa[:,-1])!=0).sum())
  # Descriptive stage flags, not semantic grasp/collision ground truth.
  if success:category='SUCCESS_ENV_SIGNAL'
  elif near>=.10:category='APPROACH_NOT_WITHIN_10CM'
  elif lift<.02:category='NEAR_TARGET_WITHOUT_2CM_LIFT'
  else:category='LIFT_OCCURRED_BUT_NO_SUCCESS'
  fail.append({'trial_id':row['trial_id'],'model':row['model'],'task':row['task'],'state_id':row['state_id'],'appearance':row['appearance'],'success':success,'steps':row['policy_steps'],'failure_reason':row['failure_reason'],'descriptive_stage':category,'start_grip_bowl_distance_m':startdist,'min_grip_bowl_distance_m':near,'final_grip_bowl_distance_m':enddist,'max_bowl_lift_m':lift,'bowl_displacement_m':displacement,'final_bowl_plate_xy_distance_m':float(np.linalg.norm(bowl[-1,:2]-plate[-1,:2])),'finger_gap_min_m':float(gap.min()),'finger_gap_max_m':float(gap.max()),'close_command_steps':close,'gripper_command_switches':switches,'first_predicted_action':json.dumps(first.tolist()),'all_actions_finite':bool(np.isfinite(aa).all()),'drawer_qpos_change':float(pp[-1]['drawer']-pp[0]['drawer']),'stove_qpos_change':float(pp[-1]['stove']-pp[0]['stove']),'evidence':'saved simulator state/matching XML FK and executed commands; no contact force or contact event record','causal_failure_mechanism':'UNRESOLVED; thresholds descriptive only; no grasp/success relabeling'})
cs('failure_case_analysis.csv',fail)
selected=[]
for label,condition in [('A_success_B_failure',lambda sid,app:trialby[f'A_{sid:02d}_{app}']['success']=='True' and trialby[f'B_{sid:02d}_{app}']['success']=='False'),('C_failure_D_success',lambda sid,app:trialby[f'C_{sid:02d}_{app}']['success']=='False' and trialby[f'D_{sid:02d}_{app}']['success']=='True'),('C_success_D_failure',lambda sid,app:trialby[f'C_{sid:02d}_{app}']['success']=='True' and trialby[f'D_{sid:02d}_{app}']['success']=='False'),('drawer_common_failure',lambda sid,app:sid>=15 and all(trialby[f'{a}_{sid:02d}_{app}']['success']=='False' for a in 'BCD'))]:
 candidates=[(sid,app) for sid in range(20) for app in ['clean','background'] if condition(sid,app)]
 sid,app=candidates[0];selected.append({'category':label,'state_id':sid,'appearance':app,'selection_rule':'lexicographically first state ID then clean/background meeting predicate, fixed before video inspection','models':'AB' if label.startswith('A_') else 'BCD' if label.startswith('drawer') else 'CD'})
cs('selected_video_cases.csv',selected)
import imageio.v2 as imageio
from PIL import Image,ImageDraw
(O/'video_contact_sheets').mkdir()
for case in selected:
 tile=Image.new('RGB',(6*256,len(case['models'])*150),'white');draw=ImageDraw.Draw(tile)
 for ri,arm in enumerate(case['models']):
  p=V/'trials'/f'{arm}_{case["state_id"]:02d}_{case["appearance"]}'/'video.mp4';reader=imageio.get_reader(p);count=reader.count_frames();ids=np.linspace(0,count-1,6).astype(int)
  for ci,k in enumerate(ids):
   frame=Image.fromarray(reader.get_data(int(k))).resize((256,128));tile.paste(frame,(ci*256,ri*150+20));draw.text((ci*256+2,ri*150+2),f'{arm} frame {k}/{count}',fill='black')
  reader.close()
 tile.save(O/'video_contact_sheets'/f'{case["category"]}.png')
js('cpu_analysis_completion.json',{'status':'PASS','seconds':time.time()-start,'optimizer_updates':0,'backward_calls':0,'new_rollouts':0,'GPU_used':False,'batches':440,'transitions':220,'trajectories_analyzed':160,'physical_label_rows':len(labelrows),'video_selection_predeclared_before_inspection':True})
print('CPU_ANALYSIS_COMPLETE',flush=True)
