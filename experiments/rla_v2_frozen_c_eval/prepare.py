"""CPU source/hash and initial-state integration checks; no policy scoring."""
import sys,copy,xml.etree.ElementTree as ET,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_v2_frozen_c_eval.common import *
import mujoco
from rla_paired_data_v2.render import PHYS,setup,restore,frame
from PIL import Image

def variant(xml,config):
 root=ET.fromstring(xml);original=ET.tostring(root);asset=root.find('asset');changes=config['changes']
 if changes and 'original_material' in changes[0]:
  for change in changes:
   old=change['original_material'];mat=change['new_material'];tex=change['texture'];assert sha(tex['file'])==change['texture_sha256']
   assert any(n.get('name')==old for n in root.iter('material'))
   matches=[n for n in root.iter('geom') if n.get('material')==old];assert matches
   # Restrict material substitution to the semantic target body's visual geometry.
   bodyname='akita_black_bowl_1_main' if old.startswith('akita') else 'wooden_cabinet_1_cabinet_middle'
   body=next(n for n in root.iter('body') if n.get('name')==bodyname);allowed=list(body.iter('geom'))
   assert all(n in allowed and n.get('group')=='1' for n in matches)
   asset.append(ET.Element('material',mat));asset.append(ET.Element('texture',tex))
   for n in matches:n.set('material',mat['name'])
  back=copy.deepcopy(root)
  for c in changes:
   for n in back.iter('geom'):
    if n.get('material')==c['new_material']['name']:n.set('material',c['original_material'])
   for n in list(back.find('asset')):
    if n.get('name') in [c['new_material']['name'],c['texture']['name']]:back.find('asset').remove(n)
 else:
  for c in changes:
   n=next(n for n in root.iter('light') if n.get('name')==c['name'])
   assert dict(n.attrib)==c['before'],('light source differs',n.attrib,c['before'])
   n.attrib.clear();n.attrib.update(c['after'])
  back=copy.deepcopy(root)
  for c in changes:
   n=next(n for n in back.iter('light') if n.get('name')==c['name']);n.attrib.clear();n.attrib.update(c['before'])
 assert ET.tostring(back)==original
 return ET.tostring(root,encoding='unicode')

def main():
 assert read(V2/'validation_results.json')['status']=='PASS_PROTOTYPE_WITH_LIMITATIONS'
 protocol={'version':'V2 frozen C development evaluation','prototype_commit':'91f5ebed6634b8443fab509aef4a3773a591193b','prototype_files':{n:sha(V2/n) for n in ['appearance_configs.json','prototype_protocol.json','validation_results.json']},'tasks':['stove','drawer'],'seeds':SEEDS,'conditions':CONDITIONS,'states_per_task':20,'new_valid_budget':480,'max_with_clean':600,'reuse_clean_expected':120,'order':'task stove/drawer; historical state manifest order; condition listed order; policy seeds31001/31002/31003','control_hz':20,'chunk':8,'max_steps':{'stove':220,'drawer':300},'image':{'cameras':['agentview','robot0_eye_in_hand'],'resolution':224,'flip':'horizontal','reflection':1,'MSAA':4},'backend':'BF16 Math SDPA; same bootstrap and action decoder','success':'unchanged official env.check_success','randomness':'reset Python/NumPy/torch/CUDA to historical environment seed each trial','technical_failure':'preserve failed attempt and stop; identical trial retry only after documented repair; task failure never retry','statistics':{'bootstrap_draws':20000,'seed':20261102,'cluster':'within task physical initial state, retaining all seeds and conditions','contrasts':['each intervention minus Clean','four-condition mean minus Clean','two-object mean minus Clean','two-light mean minus Clean'],'no_effect_threshold':True,'excluded_uncertainty':['full training seed','AE','condition sampling','task']},'interpretation':'used development states; interventions may enter future training; neither final unseen-appearance test nor D evaluation','updates':0,'backward':0,'no_AE_feature_generation':True}
 save(ROOT/'evaluation_protocol.json',protocol)
 inv=[]
 with (STOVE/'delivery/checkpoint_inventory.csv').open() as f:
  for r in csv.DictReader(f):
   if r['method']=='C-low' and int(r['step'])==1800:inv.append(dict(task='stove',policy_seed=int(r['seed']),path=r['path'],sha256=r['sha256'],step=1800))
 for r in read(DRAWER/'delivery/checkpoint_inventory.json'):
  if r['method']=='C-low' and r['step']==1800:inv.append(dict(task='drawer',policy_seed=r['policy_seed'],path=r['path'],sha256=r['sha256'],step=1800))
 assert len(inv)==6
 taskinfo={'stove':dict(root=STOVE,state_protocol=Path(read(STOVE/'evaluation/evaluation_protocol.json')['frozen_state_protocol']),statistics=Path(read(STOVE/'evaluation/run_plan.json')['inputs']['action_proprio_statistics']),language='pick up the black bowl on the stove and place it on the plate',bddl='libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl'), 'drawer':dict(root=DRAWER,state_protocol=DATA/'development_protocol/evaluation_protocol.json',statistics=DATA/'normalization_statistics_expanded_train_only.json',language='open the middle drawer of the cabinet',bddl='libero_goal/open_the_middle_drawer_of_the_cabinet.bddl')}
 for r in inv:
  assert sha(r['path'])==r['sha256'];info=taskinfo[r['task']];r.update(method='C-low',statistics=str(info['statistics']),statistics_sha256=sha(info['statistics']),architecture='official RLA VLA-Adapter, scratch2000 AE-C targets, lambda_aux 0.11204268292682927',model_id=f"{r['task']}_C-low_seed{r['policy_seed']}_s1800")
 save(ROOT/'checkpoint_inventory.json',inv)
 configs=read(V2/'appearance_configs.json');sources=read(V2/'source_manifest.json');states=[];clean=[];checks=[]
 for task,info in taskinfo.items():
  original=read(info['state_protocol']);ss=original['new_development_states'];assert len(ss)==20
  oldrows=read(info['root']/'evaluation/per_trial_results.json');representative=next(e for e in sources if e['task']==task)
  (ROOT/task).mkdir();save(ROOT/task/'run_plan.json',read(info['root']/'evaluation/run_plan.json'))
  for s in ss:
   initial=np.load(s['state_file']);assert digest(initial)==s['state_sha256'];oldxml=Path(s['state_file']).parent/'clean.xml';assert sha(oldxml)==next(x['XML_sha256'] for x in s['conditions'] if x['condition']=='clean')
   dest=ROOT/'states'/task/str(s['seed']);dest.mkdir(parents=True);xmls={'clean':str(oldxml)}
   for cond in CONDITIONS:
    p=dest/(cond+'.xml');p.write_text(variant(oldxml.read_text(),configs[representative['episode']][cond]));xmls[cond]=str(p)
   sm=dict(task=task,seed=s['seed'],state_file=s['state_file'],state_sha256=s['state_sha256'],state_file_sha256=sha(s['state_file']),source_protocol=str(info['state_protocol']),source_protocol_sha256=sha(info['state_protocol']),source=s.get('source','historical drawer development reset, not prototype demonstration'),xmls=xmls,xml_sha256={c:sha(p) for c,p in xmls.items()},language=info['language'],bddl=info['bddl']);states.append(sm)
   baseline=None
   for cond,p in xmls.items():
    m,d,r,opt=setup(p)
    try:
     restore(m,d,initial);physical={k:np.array(getattr(m,k)).copy() for k in PHYS};pose={k:np.array(getattr(d,k)).copy() for k in ['qpos','qvel','xpos','xquat','cam_xpos','cam_xmat']};rgb=np.stack([frame(r,d,opt,c) for c in ['agentview','robot0_eye_in_hand']])
     if baseline is None:baseline=(physical,pose,rgb)
     else:
      for k in physical:assert np.array_equal(physical[k],baseline[0][k]),k
      for k in pose:assert np.array_equal(pose[k],baseline[1][k]),k
     checks.append(dict(task=task,seed=s['seed'],condition=cond,physics_equal=True,pose_equal=True,rgb_mae=float(np.abs(rgb.astype(float)-baseline[2]).mean()),render_no_time_advance=bool(d.time==initial[0])))
     if s==ss[0]:Image.fromarray(np.concatenate(list(rgb),axis=1)).save(dest/(cond+'.png'))
    finally:r.close()
   for model in [r for r in inv if r['task']==task]:
    oldname=f"C-low_seed{model['policy_seed']}_s1800" if task=='stove' else f"{model['policy_seed']}_C-low"
    matches=[r for r in oldrows if r['model']==oldname and r['seed']==s['seed'] and r['condition']=='clean'];assert len(matches)==1
    r=matches[0];assert r['status']=='COMPLETE' and r['checkpoint_sha256']==model['sha256'] and r['state_sha256']==s['state_sha256'] and r['xml_sha256']==sha(oldxml) and r['language']==info['language']
    result=Path(r['trajectory']).with_name('result.json');assert read(result)==r
    clean.append(dict(r,task=task,policy_seed=model['policy_seed'],model=model['model_id'],reused=True,source_result=str(result),source_result_sha256=sha(result)))
 save(ROOT/'state_manifest.json',states);save(ROOT/'clean_reuse.json',clean);save(ROOT/'integration_static_checks.json',dict(status='PASS_RAW_MODEL_STATIC',checks=checks,policy_calls=0,render_images=400,state_restorations=200,physics_steps=0,prototype_configuration_by_task={t:next(e['episode'] for e in sources if e['task']==t) for t in taskinfo},runtime_ControlEnv_reset_check_required=True))
 save(ROOT/'clean_reuse_audit.json',dict(status='MATCHED_PENDING_RUNTIME_INTEGRATION',count=120,checkpoints_states_XML_language_results_verified=True,old_evaluators=['rla_scratch_policy_aux/evaluation.py with1800 adapter','rla_new_task_baseline/evaluation.py with drawer1800 adapter'],unchanged_inference_source=True,device='same authorized GPU UUID; current ECC baseline separately recorded'))
 print('PREPARED six policies,40states,480new,120Clean reuse',flush=True)
if __name__=='__main__':main()
