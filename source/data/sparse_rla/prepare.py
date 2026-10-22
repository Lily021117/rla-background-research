"""Matched raw-replay dataset: offline FK only, never sim.step or env.reset."""
import os,json,csv,hashlib,ast,math,subprocess
from pathlib import Path
import numpy as np,h5py,mujoco,imageio.v2 as imageio
R=Path('${RLA_PROJECT_ROOT}');BASE=R/'experiments/rla_background_invariance'
OUT=Path(os.environ['RUN_DIR']);S2=BASE/'step2_training';S3=BASE/'step3_rla_wm_bc/run_20261021_04'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def write(n,v):
 p=OUT/n;assert not p.exists(),p;p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 assert not (OUT/'preparation.json').exists()
 inv=list(csv.DictReader((BASE/'step4_readiness/run_20261021_01/checkpoint_inventory.csv').open()))
 for x in inv:assert sha(x['path'])==x['sha256'],x['path']
 write('protected_checkpoints_before.json',inv)
 sourcepaths=[R/'AGENTS.md',S2/'training_records.json',S2/'split_manifest.json',S3/'final_delivery_sha256.json',BASE/'step4_readiness/run_20261021_01/report.md']
 write('protected_sources_before.json',{str(p):sha(p) for p in sourcepaths})
 write('git_before.json',{str(p):dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=p,text=True)) for p in [R,Path('${VLA_REFERENCE_ROOT}'),Path.cwd()]})
 join=list(csv.DictReader((BASE/'step4_readiness/run_20261021_01/episode_identity_join_audit.csv').open()))
 allowed={x['episode_id']:x for x in join if x['split']=='train' and x['exact_task_action_matches']=='1'};assert len(allowed)==3
 split=read(S2/'split_manifest.json')['splits'];entries={x['episode_id']:x for x in split['train'] if x['episode_id'] in allowed}
 records=read(S2/'training_records.json');rows=[(i,x) for i,x in enumerate(records) if x['episode_id'] in allowed and x['camera']=='agentview'];assert len(rows)==30
 # Execute the exact official axis-angle helper without importing rollout/model code.
 helper=Path.cwd()/'experiments/robot/libero/libero_utils.py';tree=ast.parse(helper.read_text())
 fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='quat2axisangle')
 ns={'np':np,'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'),ns)
 qq=ns['quat2axisangle'];epmeta=[];manifest=[];target_rows={g:[] for g in ['B','C']};alignment=[]
 for slot,(eid,e) in enumerate(entries.items()):
  directory=Path(e['geometry_path']).parent;geometry=np.load(e['geometry_path']);assert sha(e['geometry_path'])==e['geometry_sha256']
  full=geometry['states'];stateids=geometry['state_indices'];assert len(set(stateids.tolist()))==len(stateids)
  model=mujoco.MjModel.from_xml_path(str(directory/'clean.xml'));data=mujoco.MjData(model)
  assert len(full[0])==1+model.nq+model.nv and model.na==0
  names=geometry['body_names'].tolist();assert names==[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,k) for k in range(model.nbody)]
  grip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');hand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot0_right_hand')
  finger=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'gripper0_finger_joint'+str(k))] for k in [1,2]]
  cameras=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,n) for n in ['agentview','robot0_eye_in_hand']]
  proprio=[];errors=[]
  with h5py.File(e['source_hdf5'],'r') as f:
   d=f['data'][e['source_demo']];actions=np.asarray(d['actions'],dtype='float32');assert hashlib.sha256(actions.tobytes()).hexdigest()==e['actions_sha256']
   rawstates=d['states'][:];np.testing.assert_array_equal(full,rawstates[stateids])
   rlds_path=R/'artifacts/semantic_supervision_scaled_CEF/features'/allowed[eid]['matched_RLDS_ids']/'source.npz'
   state_discrepancy=None
   if rlds_path.exists():
    rlds=np.load(rlds_path);np.testing.assert_array_equal(actions,rlds['actions'])
    oldproprio=np.concatenate([d['obs/ee_states'][:],d['obs/gripper_states'][:]],1)
    state_discrepancy=float(np.max(np.abs(oldproprio-rlds['states'])))
   # The other allowed IDs remain exact matches from the sealed Step4 audit.
   # Missing regenerated caches are not replaced by another episode or guessed.
  for k,state in enumerate(full):
   data.time=state[0];data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:];mujoco.mj_forward(model,data)
   bodyerr=float(np.max(np.abs(data.xpos-geometry['body_positions'][k])));assert bodyerr<=1e-8
   for j,cam in enumerate(cameras):
    pose=geometry['c2w_opencv'][k,j]
    np.testing.assert_allclose(data.cam_xpos[cam],pose[:3,3],rtol=0,atol=1e-8)
    np.testing.assert_allclose(data.cam_xmat[cam].reshape(3,3)@np.diag([1,-1,-1]),pose[:3,:3],rtol=0,atol=1e-8)
   q=np.roll(data.xquat[hand].copy(),-1)
   proprio.append(np.concatenate([data.site_xpos[grip],qq(q),data.qpos[finger]]).astype('float32'));errors.append(bodyerr)
  proprio=np.stack(proprio);assert proprio.shape==(len(full),8)
  images=[];video_hash={}
  for cam in ['agentview','wrist']:
   meta=read(S2/f'features/{eid}_{cam}_clean.json');v=Path(meta['source_video']);assert sha(v)==meta['source_video_sha256']
   assert sha(meta['path'])==meta['sha256']
   reader=imageio.get_reader(v);frames=np.stack([np.rot90(frame,2).copy() for frame in reader]);reader.close()
   assert len(frames)==len(full) and frames.shape[1:]==(128,128,3)
   images.append(frames);video_hash[cam]=sha(v)
  current_actions=actions[stateids].copy();samplefile=OUT/f'episode_{slot}.npz'
  assert not samplefile.exists();np.savez(samplefile,images=images[0],wrist_images=images[1],proprio=proprio,
   actions=current_actions,source_state_ids=stateids,source_timestamps=full[:,0])
  epmeta.append(dict(slot=slot,episode_id=eid,matched_RLDS_id=allowed[eid]['matched_RLDS_ids'],source=str(samplefile),task=e['task'],
   source_hdf5=e['source_hdf5'],source_demo=e['source_demo'],geometry=e['geometry_path'],xml=str(directory/'clean.xml'),
   geometry_sha256=sha(e['geometry_path']),xml_sha256=sha(directory/'clean.xml'),video_sha256=video_hash,
   raw_state_RLDS_regenerated_state_max_difference=state_discrepancy,RLDS_observations_used=False,
   current_proprio_source='offline forward kinematics at exact recorded state; official eef site/body quaternion/finger-qpos definitions',max_body_error=max(errors)))
  for index,r in rows:
   if r['episode_id']!=eid:continue
   lo,hi=r['source_actions_halfopen'];assert lo==r['state_t'] and hi==r['state_future'] and hi-lo==8
   ci=int(np.flatnonzero(stateids==lo)[0]);fi=int(np.flatnonzero(stateids==hi)[0]);assert fi-ci==8
   assert ci==r['output_t'] and fi==r['output_future']
   np.testing.assert_array_equal(current_actions[ci:ci+8],actions[lo:hi])
   dt=float(full[fi,0]-full[ci,0]);assert abs(dt-.4)<1e-8
   for vcam in ['agentview','wrist']:
    meta=read(S2/f'features/{eid}_{vcam}_clean.json')
    assert meta['source_output_ids'][r['current_cache_index']]==ci and meta['source_output_ids'][r['future_cache_index']]==fi
   mi=len(manifest)
   item=dict(sample_index=mi,slot=slot,episode_id=eid,matched_RLDS_id=allowed[eid]['matched_RLDS_ids'],window=r['window'],
    raw_RLA_record_index=index,current_video_index=ci,future_video_index=fi,current_source_state=lo,future_source_state=hi,
    current_time=float(full[ci,0]),future_time=float(full[fi,0]),action_source_lo=lo,action_source_hi_exclusive=hi,
    action_timestamps=json.dumps(rawstates[lo:hi,0].tolist()),proprio_source_state=lo,horizon=8,action_identity_matched=True,
    full_state_camera_alignment=True,teacher_camera='agentview',policy_camera_order='agentview,wrist')
   manifest.append(item)
   for group in ['B','C']:
    cache=S3/group/'targets.npy';expected=read(S3/'final_delivery_sha256.json')['files'][str(cache.relative_to(S3))];assert sha(cache)==expected
    z=np.load(cache,mmap_mode='r')[index,0].copy();assert z.shape==(32,64) and np.isfinite(z).all();target_rows[group].append(z)
  print('ALIGNED',eid,len(full),'maxbody',max(errors),'rlds_state_discrepancy',state_discrepancy,flush=True)
 for g,zs in target_rows.items():
  z=np.stack(zs)[:,None];assert z.shape==(30,1,32,64)
  np.save(OUT/f'targets_{g}.npy',z)
  mu=z.mean(0);scale=float(np.sqrt(((z-mu)**2).mean()));assert scale>1e-8
  np.savez(OUT/f'normalization_{g}.npz',mean=mu,scale=np.asarray(scale,dtype='float32'))
  np.testing.assert_allclose(((z-mu)/scale)*scale+mu,z,rtol=1e-5,atol=1e-5)
 with (OUT/'data_manifest.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
 write('episodes.json',epmeta);write('samples.json',manifest)
 write('preparation.json',dict(status='PASS',matched_train_episodes=3,valid_physical_transitions=30,
  validation_used=False,heldout_used=False,FK_only_no_dynamics_steps=True,env_resets=0,
  tensor_targets_source='own Step2 frozen AE via verified Step3 posterior cache, not flow predictions',
  source_frame_mapping='video[k]=recorded state[k+1], commands[k+1:k+9], explicit source IDs',
  regenerated_RLDS_observations_rejected_due_physical_state_difference=True))
if __name__=='__main__':main()
