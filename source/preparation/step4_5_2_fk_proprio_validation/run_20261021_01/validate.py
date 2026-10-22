import os
from pathlib import Path
R=Path('${RLA_PROJECT_ROOT}'); O=Path(__file__).resolve().parent
T=Path('${RLA_CACHE_ROOT}/step4_5_2');T.mkdir(parents=True,exist_ok=True)
for k,sub in {'TMPDIR':'tmp','HF_HOME':'hf','HUGGINGFACE_HUB_CACHE':'hf/hub','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'mpl','NUMBA_CACHE_DIR':'numba'}.items():
 p=T/sub;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',RLA_STEPS='0',TOKENIZERS_PARALLELISM='false')
import json,csv,hashlib,subprocess,ast,math,sys,resource
import numpy as np,h5py,mujoco
from PIL import Image
W=R/'worktrees/vla-background-step4-1-23fa0c9';sys.path.insert(0,str(W))
P=R/'experiments/rla_background_invariance';P4=P/'step4_4_pilot/run_20261021_01';P5=P/'step4_5_feature_target_pilot/run_20261021_02';P1=P/'step4_1_adapter_readiness/run_20261021_03';OLD=P/'step4_1_adapter_readiness/run_20261021_02'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def csvout(n,x):
 with (O/n).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def git():return {k:subprocess.check_output(['git']+v,cwd=R,text=True) for k,v in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short']}.items()}
g=git();gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv'],text=True)
write('resource_preflight.json',{'gpu':gpu,'threshold_MiB':26624,'gpu1_free_MiB':int(gpu.splitlines()[2].split(',')[2].split()[0]),'GPU_used':False,'disk':subprocess.check_output(['df','-h',str(R)],text=True),'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)})
paths=[]
for d in [P1,P4,P5,P/'step4_5_1_proprio_alignment/run_20261021_01']:
 paths += [p for p in d.rglob('*') if p.is_file() and (p.suffix in ['.json','.csv','.md','.py','.png','.npz','.npy'])]
targets=list(csv.DictReader((P5/'rla_target_manifest.csv').open()));images=list(csv.DictReader((P4/'sample_manifest.csv').open()))
hdf=Path(images[0]['source_hdf5']);paths += [hdf,W/'sparse_rla/prepare.py',W/'sparse_rla/adapter.py']
for t in targets:
 assert sha(t['path'])==t['SHA256'];paths.append(Path(t['encoder']))
for a in ['B','C']:paths += [OLD/f'normalization_{a}.npz',R/f'experiments/rla_background_invariance/step2_training/{a}/ckpts/decoder_step0000200.pt']
hashes={str(p):sha(p) for p in set(paths)};write('protected_before.json',{'git':g,'sha256':hashes})
rs=Path('${LIBERO_SITE_PACKAGES}/robosuite')
fn=next(x for x in ast.parse((rs/'utils/transform_utils.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='quat2axisangle');fn.decorator_list=[]
ns={'np':np,'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'official_conversion','exec'),ns)
trans=json.loads((P4/'selection_manifest.json').read_text())['transitions']; fk={}; manifest=[];metrics=[];maps=[];actions={}
with h5py.File(hdf,'r') as f:
 for ep in sorted({t['episode'] for t in trans}):
  raw=f['data/demo_'+ep.split('_')[-1]];states=raw['states'][:];actions[ep]=raw['actions'][:].astype('float32')
  base=R/f'artifacts/action_conditioned_wm/TZ_three_seed_500/independent_data/pairs/{ep}';xml=base/'clean.xml';hashes[str(xml)]=sha(xml)
  m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);geometry=np.load(base/'geometry.npz')
  names=[f'robot0_joint{i}' for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']
  ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];assert min(ids)>=0
  ix=[int(m.jnt_qposadr[i]) for i in ids];site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot0_right_hand')
  assert min(site,body)>=0 and m.na==0
  maps.append({'episode':ep,'XML':str(xml),'sha256':sha(xml),'joint_names':names,'qpos_addresses':ix,'joint_types':[int(m.jnt_type[i]) for i in ids],'site':'gripper0_grip_site','orientation_body':'robot0_right_hand','site_local_position':m.site_pos[site].tolist(),'site_parent_body':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.site_bodyid[site])),'coordinate_frame':'world','units':'xyz m, axis-angle rad, finger slide qpos m','external_pose_truth':False})
  endpoints={t[k] for t in trans if t['episode']==ep for k in ['source_start','source_future']}
  needed=set(endpoints)
  for t in trans:
   if t['episode']==ep:needed.update(range(t['source_start'],t['source_future']))
  for sid in sorted(needed):
   s=states[sid];assert len(s)==1+m.nq+m.nv
   d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
   p=np.r_[d.site_xpos[site],ns['quat2axisangle'](np.roll(d.xquat[body].copy(),-1)),d.qpos[ix[-2:]]];fk[ep,sid]=p.astype('float32')
   if sid not in endpoints:continue
   k=int(np.flatnonzero(geometry['state_indices']==sid)[0]);np.testing.assert_array_equal(s,geometry['states'][k]);err=float(abs(d.xpos-geometry['body_positions'][k]).max());assert err<=1e-8
   for ci,cn in enumerate(['agentview','robot0_eye_in_hand']):
    cam=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,cn);pose=geometry['c2w_opencv'][k,ci]
    np.testing.assert_allclose(d.cam_xpos[cam],pose[:3,3],rtol=0,atol=1e-8);np.testing.assert_allclose(d.cam_xmat[cam].reshape(3,3)@np.diag([1,-1,-1]),pose[:3,:3],rtol=0,atol=1e-8)
    a=np.load(P4/f'arrays/{ep}_state{sid}_{cn}.npz');np.testing.assert_array_equal(a['state'],s);np.testing.assert_array_equal(a['replay_qpos'],d.qpos)
   old=np.r_[raw['obs/ee_states'][sid-1],raw['obs/gripper_states'][sid-1]]
   manifest.append({'episode':ep,'source_state_id':sid,'sim_time':float(s[0]),'proprio':json.dumps(p.tolist()),'source':'exact state FK, not shifted raw observation','image_source_state':sid,'external_pose_ground_truth':'MISSING'})
   metrics.append({'episode':ep,'state_id':sid,'body_pose_max_abs_m':err,'joint_raw_max_rad':float(abs(raw['obs/joint_states'][sid-1]-d.qpos[ix[:7]]).max()),'finger_raw_max_m':float(abs(old[6:]-p[6:]).max()),'raw_eef_position_difference_m':float(abs(old[:3]-p[:3]).max()),'raw_eef_axisangle_difference_rad':float(abs(old[3:6]-p[3:6]).max()),'independent_pose_accuracy':'NOT_MEASURED','geometry_implementation':'PASS'})
csvout('fk_proprio_manifest.csv',manifest);csvout('fk_alignment_metrics.csv',metrics);write('joint_and_frame_definitions.json',maps)
print('FK_PASS',len(manifest),flush=True)

# Actual Step4.1 standardizer / normalizer / chunker / transform / collator.
# Only current images are loaded; no synthetic intermediate images or future inputs.
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
import torch
from transformers import AutoConfig,AutoImageProcessor,AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor,PrismaticProcessor
from sparse_rla.adapter import SparseCollator,RlaBatchTransform,ActionTokenizer,PurePromptBuilder,libero_dataset_transform,chunk_act_obs,_normalize_with_rla,ACTION_PROPRIO_NORMALIZATION_TYPE,set_store
AutoConfig.register('openvla',OpenVLAConfig);AutoImageProcessor.register(OpenVLAConfig,PrismaticImageProcessor);AutoProcessor.register(OpenVLAConfig,PrismaticProcessor)
processor=AutoProcessor.from_pretrained(P1/'runtime_metadata',trust_remote_code=False,local_files_only=True)
stats=json.loads((P1/'runtime_metadata/dataset_statistics.json').read_text())['libero_spatial_no_noops'];metadata={k:{kk:tf.convert_to_tensor(v) for kk,v in stats[k].items()} for k in ['action','proprio']}
transform=RlaBatchTransform(ActionTokenizer(processor.tokenizer),processor.tokenizer,image_transform=processor.image_processor.apply_transform,prompt_builder_fn=PurePromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True)
collator=SparseCollator(processor.tokenizer.model_max_length,processor.tokenizer.pad_token_id)
class Store:
 def slot(self,x):np.testing.assert_array_equal(x,self.proprio);return 0
 def frame(self,slot,t):assert slot==0 and t==self.sid;return self.target
store=Store();set_store(store);rows=[];schema=None;pairchecks={}
for ti,tr in enumerate(trans):
 ep,s,u=tr['episode'],tr['source_start'],tr['source_future'];state=np.stack([fk[ep,j] for j in range(s,u)]);act=actions[ep][s:u];assert act.shape==(8,7)
 for group in ['B','C']:
  norm=np.load(OLD/f'normalization_{group}.npz')
  for appearance in ['clean','background']:
   target=next(t for t in targets if t['transition_id']==tr['transition_id'] and t['group']==group and t['camera']=='agentview' and t['appearance']==appearance)
   z=np.load(target['path']);assert z.shape==(32,64)
   store.proprio=state;store.sid=s;store.target=np.ascontiguousarray(((z-norm['mean'])/float(norm['scale'])).reshape(1,2048),dtype='float32')
   traj=libero_dataset_transform({'action':tf.convert_to_tensor(act),'observation':{'state':tf.convert_to_tensor(state)}});traj['observation']['proprio']=tf.convert_to_tensor(state)
   traj=_normalize_with_rla(traj,metadata,ACTION_PROPRIO_NORMALIZATION_TYPE);traj['observation']={k:traj['observation'][k] for k in ['proprio','rla_slot']};traj['observation']['timestep']=tf.range(s,u)
   traj['task']={'language_instruction':tf.repeat(b'pick up the black bowl on the cookie box and place it on the plate',8)};traj['dataset_name']=tf.repeat(b'libero_spatial_no_noops',8);traj['absolute_action_mask']=tf.tile([[False]*6+[True]],[8,1])
   expected=traj['action'].numpy().copy();traj=chunk_act_obs(traj,window_size=1,future_action_window_size=7);frame=tf.nest.map_structure(lambda x:x[0].numpy(),traj);np.testing.assert_array_equal(frame['action'],expected)
   for cam,key in [('agentview','image_primary'),('robot0_eye_in_hand','image_wrist')]:
    row=next(x for x in images if x['episode']==ep and int(x['source_state_id'])==s and x['camera']==cam)
    rgb=np.asarray(Image.open(row[appearance]).convert('RGB'));frame['observation'][key]=np.ascontiguousarray(rgb[:,::-1])[None] # top-left renderer -> native vertical flip -> legacy rotate180
   instance=transform(frame)
   batch=next(iter(torch.utils.data.DataLoader([instance],batch_size=1,num_workers=0,collate_fn=collator)))
   assert batch['actions'].shape==(1,8,7) and batch['proprio'].shape==(1,8) and batch['rla'].shape==(1,1,2048)
   assert all(torch.isfinite(v).all() for v in batch.values() if torch.is_tensor(v))
   np.testing.assert_array_equal(batch['rla'][0].numpy(),store.target)
   if appearance=='clean':pairchecks[ti,group]={k:batch[k].clone() for k in ['actions','proprio']}
   else:
    for k in ['actions','proprio']:assert torch.equal(batch[k],pairchecks[ti,group][k])
   schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in batch.items() if torch.is_tensor(v)}
   rows.append({'episode':ep,'transition':tr['transition_id'],'group':group,'appearance':appearance,'current_state':s,'future_state':u,'action_half_open':f'[{s},{u})','target_path':target['path'],'encoder_sha256':target['encoder_SHA256'],'finite':True,'real_pipeline':True})
csvout('batch_coverage.csv',rows)
write('batch_schema.json',{'schema':schema,'constructed_batches':len(rows),'unique_transitions':8,'teacher_camera':'agentview only; wrist target remains diagnostic, no concatenation','teacher_shape':[32,64],'horizons':[8],'input_cameras':['agentview','robot0_eye_in_hand'],'orientation':'horizontal flip top-left PNG = native vertical flip then legacy rotate180','proprio':'world grip-site xyz / hand body axis-angle / two finger qpos','normalization':'published Pro action/proprio; existing Step4.1 B/C target statistics, not refitted','pipeline':'libero_dataset_transform -> _normalize_with_rla -> chunk_act_obs -> current source RGB join -> RlaBatchTransform -> DataLoader(num_workers=0,SparseCollator)','scope':'sparse same-trajectory HDF5 path; not native RLDS identity acceptance'})
unchanged=all(sha(p)==h for p,h in hashes.items())
write('batch_smoke_test.json',{'CPU_data_pipeline':'PASS','GPU_forward':'BLOCKED_RESOURCE','gradient_test':'NOT_RUN','GPU1_required_free_MiB':26624,'optimizer_updates':0,'backward_calls':0,'model_loaded':False,'model_in_memory_hash':'not applicable, no model loaded','CPU_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
write('delivery_verification.json',{'status':'PARTIAL','FK_geometry':'PASS','external_EEF_ground_truth':'MISSING','CPU_real_batch':'PASS','batches':len(rows),'historical_sha256_unchanged':unchanged,'training_updates':0,'GPU_inference':0,'closed_loop':0,'heldout_scoring':0,'git_before':g,'git_after':git()})
print('CPU_BATCH_PASS',len(rows),'PROTECTED',unchanged,flush=True)
