import os
from pathlib import Path
O=Path(__file__).resolve().parent;R=Path('${RLA_PROJECT_ROOT}');X=R/'experiments/rla_background_invariance';D=X/'step4_4_1_dataset_expansion/run_20261021_02'
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable',PYTHONDONTWRITEBYTECODE='1')
import json,csv,hashlib,ast,math,subprocess
import numpy as np,h5py,mujoco
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def tab(p,rs):
 with p.open('x') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
protected={str(D/n):h for n,h in json.loads((D/'frozen_manifest_hashes.json').read_text()).items()};assert all(sha(p)==h for p,h in protected.items())
eps=list(csv.DictReader((D/'episode_manifest.csv').open()));accepted={r['episode']:r for r in eps if r['status'].startswith('PASS')}
rows=list(csv.DictReader((D/'sample_manifest.csv').open()));trans=[t for t in csv.DictReader((D/'transition_manifest.csv').open()) if t['valid']=='True'];assert len(trans)==220 and len(rows)==876
assert all(r['episode'] in accepted for r in rows+trans);assert 'sealed_354_demo_3' not in accepted
for t in trans:
 for k in ['source_start','source_future','output_start','output_future','action_start_inclusive','action_stop_exclusive']:t[k]=int(t[k])
 assert t['split']==accepted[t['episode']]['split']
for r in rows:
 for k in ['clean','background','array']:assert sha(r[k])==r[k+'_SHA256'];protected[r[k]]=r[k+'_SHA256']
for g in ['B','C']:
 for part in ['encoder','decoder']:
  p=X/f'step2_training/{g}/ckpts/{part}_step0000200.pt';protected[str(p)]=sha(p)
cfg=Path('${DATA_ROOT}/models/rla-wm/official/rla/maniskill/20260323_21-41-25/config.yaml');protected[str(cfg)]=sha(cfg)
helper=Path('${LIBERO_SITE_PACKAGES}/robosuite/utils/transform_utils.py');fn=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quat2axisangle');fn.decorator_list=[];ns={'np':np,'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(helper),'exec'),ns)
split=json.loads((X/'step2_training/split_manifest.json').read_text())['splits'];entries={e['episode_id']:e for s in ['train','validation'] for e in split[s]};inputs={};audit=[]
for ep in accepted:
 e=entries[ep];m=mujoco.MjModel.from_xml_path(str(Path(e['geometry_path']).parent/'clean.xml'));d=mujoco.MjData(m);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot0_right_hand');fi=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'gripper0_finger_joint{i}')] for i in [1,2]]
 with h5py.File(e['source_hdf5'],'r') as f:
  raw=f['data'][e['source_demo']];states=raw['states'][:];actions=raw['actions'][:]
  def fk(sid):
   s=states[sid];d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
   return np.r_[d.site_xpos[site],ns['quat2axisangle'](np.roll(d.xquat[body].copy(),-1)),d.qpos[fi]]
  for ti,t in enumerate(trans):
   if t['episode']!=ep:continue
   s,u=t['source_start'],t['source_future'];assert u-s==8 and [s,u]==[t['action_start_inclusive'],t['action_stop_exclusive']]
   assert abs(states[u,0]-states[s,0]-.4)<1e-8
   np.testing.assert_array_equal(actions[s:u],np.asarray(json.loads(t['actions'])))
   maxerr=0.
   for sid in [s,u]:
    p=fk(sid)
    for camera in ['agentview','robot0_eye_in_hand']:
     r=next(r for r in rows if r['episode']==ep and int(r['source_state_id'])==sid and r['camera']==camera);a=np.load(r['array'])
     np.testing.assert_array_equal(a['state'],states[sid]);err=float(abs(p-a['fk_proprio']).max());assert err<=1e-8;maxerr=max(err,maxerr)
   inputs[f'proprio_{ti}']=np.stack([fk(j) for j in range(s,u)]).astype('float32');inputs[f'actions_{ti}']=actions[s:u].astype('float32')
   audit.append(dict(episode=ep,split=t['split'],transition_id=t['transition_id'],state_t=s,state_future=u,action_start=s,action_stop=u,elapsed_s=float(states[u,0]-states[s,0]),FK_max_difference=maxerr,same_state=True,paired_appearances=True,status='PASS',external_pose_truth='MISSING'))
 print('TIMING',ep,flush=True)
np.savez(O/'batch_inputs.npz',**inputs);tab(O/'sample_manifest.csv',rows);js(O/'selection_manifest.json',{'transitions':trans});tab(O/'proprio_alignment_audit.csv',audit)
(O/'preflight').mkdir();js(O/'preflight/delivery_verification.json',{'protected_sha256':protected});tab(O/'preflight/proprio_alignment_audit.csv',audit)
js(O/'protected_sources.json',{'sha256':protected,'git_status':subprocess.check_output(['git','status','--short'],cwd=R,text=True),'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True)})
# Audit only existing candidate metadata; no heldout simulation/scoring or reassignment.
inventory=X/'step4_2_data_audit/run_20261021_02/episode_inventory.csv';candidates=[r for r in csv.DictReader(inventory.open()) if 'stove' in r.get('task','').lower() and r.get('split') not in ['test','heldout']]
js(O/'validation_coverage.json',{'status':'GAP_REMAINS','validation_episodes':3,'stove_validation_episodes':0,'excluded':'sealed_354_demo_3','candidate_audit_source':str(inventory),'candidate_audit_sha256':sha(inventory),'existing_stove_metadata_records_checked':len(candidates),'metadata_status_counts':{s:sum(r['status']==s for r in candidates) for s in set(r['status'] for r in candidates)},'qualified_unused_validation_replacements':0,'reason':'fixed validation split has only excluded stove episode; other metadata candidates are not jointly certified as unused validation with current rendering and timing protocol; no resplit allowed','raw_candidates_not_promoted':True,'heldout_used':False,'candidate_records':candidates})
print('PREFLIGHT_PASS',len(trans),len(rows)*2)
