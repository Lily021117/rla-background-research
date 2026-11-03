"""Bounded CPU-only synchronization audit. No learned model imports."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', LP_NUM_THREADS='2')
import csv, json, hashlib
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path('/data/lyc/projects/rla-wm')
OLD=ROOT/'experiments/rla_official_ae_paired_rerender/run_20261103_01'
RUN=ROOT/'experiments/rla_official_ae_sync_repair/run_20261103_01'
SUITES=['libero_object','libero_spatial','libero_goal','libero_10']
CAMS=['agentview','robot0_eye_in_hand']
STATE=mujoco.mjtState.mjSTATE_INTEGRATION
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def packed(m,d):
 a=np.empty(mujoco.mj_stateSize(m,STATE));mujoco.mj_getState(m,d,a,STATE);return a
def restore(m,d,a):
 mujoco.mj_setState(m,d,a,STATE);mujoco.mj_forward(m,d)
def legacy(m,d,s):
 d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
def opt():
 o=mujoco.MjvOption();o.geomgroup[0]=0;o.geomgroup[1]=1;o.sitegroup[:]=0;return o
def render(r,d,o=None):
 views=[]
 for cam in CAMS:
  r.update_scene(d,camera=cam,scene_option=o or opt());views.append(r.render().copy()[:,::-1])
 return np.stack(views)
def metrics(a,b,prefix):
 delta=np.abs(a.astype(float)-b.astype(float));out={}
 for i,c in enumerate(['front','wrist']):
  out.update({f'{prefix}_{c}_mae':float(delta[i].mean()),f'{prefix}_{c}_p99':float(np.quantile(delta[i],.99)),f'{prefix}_{c}_max':float(delta[i].max())})
 return out
def geometry(d):return np.concatenate([d.xpos.ravel(),d.xquat.ravel(),d.cam_xpos.ravel(),d.cam_xmat.ravel(),d.site_xpos.ravel()])
def freeze():
 p=RUN/'frozen_protocol.json';assert not p.exists()
 old=json.loads((OLD/'frozen_protocol.json').read_text())
 manifest=ROOT/'experiments/rla_official_background_hue/run_20261102_01/audit/trajectory_manifest.csv'
 rows=list(csv.DictReader(open(manifest)));assert len(rows)==1693 and sum(x['split']=='train' for x in rows)==1653
 dump(p,dict(parent_commit='835cd3116f12d2b6a5f0c003e2be5ece6a68a918',original_protocol_sha256=sha(OLD/'frozen_protocol.json'),original_manifest_sha256=sha(manifest),
  train=1653,val=40,pilots=old['pilots'],acceptance_unchanged=old['acceptance'],
  self_restore_tolerance=dict(physical_state_max=0,geometry_max=1e-12,rgb_max=0),
  offline_first=True,max_extra_source_actions=741,max_extra_wait_steps=40,max_extra_replays_per_pilot=1,environment_search=0,
  synchronized_export='Copy runtime MjData; forward only copy; export full integration state and synchronized geometry/RGB; fresh load+forward check. Main replay unmodified.',
  old_hue='Preserved in parent; no overwrite. Expanded-white-list H-only control uses same parent trajectory hue.',
  new_color=dict(delta_h=.20,saturation_formula='S_new = S + 0.35 * (1-S)',value_formula='V_new = V',quantization='round to uint8; rendered brightness not guaranteed unchanged',same_parameter_all_trajectories=True),
  background_whitelist=['floor','table_visual','wall_leftcorner_visual','wall_rightcorner_visual','wall_left_visual','wall_right_visual','wall_rear_visual','wall_front_visual'],
  exclusions=['robot','gripper','task objects','basket','operable cabinets','table legs (not expanded this round)'],
  visibility=dict(background_visible_min_pixels=1,weak_if_visible_background_mean_rgb_change_below=1.,unit='uint8'),
  visual_fixed_frames='7 evenly spaced per complete trajectory; full-sequence numeric checks, comparison videos retained',
  batch_gate='all four original correspondence + self restore + visible color pass; otherwise stop systemic failures',
  all_learned_model_forwards_updates_backward_rollouts=0,backend='CPU OSMesa'))
 print(p)
if __name__=='__main__':freeze()
