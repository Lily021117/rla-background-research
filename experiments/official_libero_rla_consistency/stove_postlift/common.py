import json,csv,hashlib,os
from pathlib import Path
import numpy as np
R=Path('/data/lyc/projects/rla-wm/experiments/official_libero_rla_consistency')
S=R/'run_20261023_10/stove'; CENTER=R/'stove_centering/run_20261024_01'
BDDL=Path('/data/lyc/projects/LIBERO-plus/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def csvsave(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def trials():return [r for r in read(S/'policy_run/per_trial_results.json') if r['model']=='C']+[r for r in read(CENTER/'per_trial_results.json') if r['model']=='D-center']
def states():return read(S/'ae_run/development_protocol/evaluation_protocol.json')['new_development_states']
def env_make(state,condition):
 from libero.libero.envs.env_wrapper import ControlEnv
 e=ControlEnv(bddl_file_name=str(BDDL),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None,ignore_done=False)
 e.seed(state['seed']);e.reset();e.reset_from_xml_string((Path(state['state_file']).parent/(condition+'.xml')).read_text());e.set_init_state(np.load(state['state_file']));e.sim.forward();return e
def metrics(e):
 d=e.sim.data._data;m=e.sim.model._model;obj=e.env.object_states_dict
 bowl=obj['akita_black_bowl_1'].get_geom_state();plate=obj['plate_1'].get_geom_state();b=bowl['pos'];p=plate['pos'];ee=d.site_xpos[m.site('gripper0_grip_site').id];contact=bool(obj['plate_1'].check_contact(obj['akita_black_bowl_1']));xy=float(np.linalg.norm(b[:2]-p[:2]));height=bool(b[2]>=p[2]);goal=bool(e.check_success());assert goal==(height and contact and xy<.03)
 fingers=[]
 for c in d.contact[:d.ncon]:
  names=[m.body(int(m.geom_bodyid[g])).name or '' for g in (c.geom1,c.geom2)]
  if any('akita_black_bowl_1' in n for n in names) and any('finger' in n for n in names):fingers.append(names)
 r=dict(goal=goal,height_condition=height,contact_condition=contact,xy_condition=xy<.03,xy_distance_m=xy,vertical_bowl_plate_m=float(b[2]-p[2]),finger_contact=bool(fingers),ee_bowl_distance_m=float(np.linalg.norm(ee-b)))
 for label,v in [('bowl',b),('plate',p),('ee',ee),('bowl_quat',bowl['quat']),('ee_quat',d.xquat[m.body('robot0_right_hand').id])]:
  r.update({label+'_'+str(i):float(x) for i,x in enumerate(v)})
 return r
