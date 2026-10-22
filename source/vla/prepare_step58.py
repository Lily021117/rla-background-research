import json,csv,hashlib,subprocess,os,xml.etree.ElementTree as ET
from pathlib import Path
R=Path('${RLA_PROJECT_ROOT}');E=R/'experiments/rla_background_invariance';OLD=E/'step5_2_closed_loop/run_20261022_02';T=E/'step5_7_causal_training/run_20261022_01';O=Path(os.environ['RUN_DIR']);W=Path(__file__).parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def cs(n,rows):
 with (O/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert json.loads((T/'delivery_verification.json').read_text())['status']=='PASS'
protected=json.loads((T/'protection_before.json').read_text())
protected.update(json.loads((T/'training_provenance.json').read_text())['sources'])
protected.update(json.loads((T/'code_inventory.json').read_text()))
protected.update(json.loads((OLD/'protection_before.json').read_text())['sha256'])
for p,h in protected.items():assert sha(p)==h,p
for folder in [T,OLD,E/'step5_4_baseline_audit/run_20261022_01']:
 for p in folder.iterdir():
  if p.is_file():protected[str(p)]=sha(p)
states=json.loads((OLD/'state_manifest.json').read_text());assert len(states)==20 and len({r['task'] for r in states})==4
for s in states:
 for name in ['clean_xml','background_xml','source_hdf5','bddl','geometry_source']:
  p=s[name]
  if p not in protected:protected[p]=sha(p)
 gp=OLD/'geometry'/f"{s['state_id']}_settled.npz";protected[str(gp)]=sha(gp)
 ac=list(ET.parse(s['clean_xml']).getroot().iter());ab=list(ET.parse(s['background_xml']).getroot().iter());assert len(ac)==len(ab)
 changed=[]
 for a,b in zip(ac,ab):
  assert a.tag==b.tag
  if a.attrib!=b.attrib:
   assert a.tag=='texture' and a.get('name') in ['tex-table','tex-wall'] and {k for k in set(a.attrib)|set(b.attrib) if a.get(k)!=b.get(k)}=={'file'}
   changed.append(a.get('name'))
   for el in [a,b]:protected[el.get('file')]=sha(el.get('file'))
 assert set(changed)=={'tex-table','tex-wall'}
base=R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint';prov=json.loads((E/'step5_4_baseline_audit/run_20261022_01/initialization_provenance.json').read_text())
for r in prov['release_file_checks']:
 if r['algorithm']=='sha256':assert sha(base/r['file'])==r['expected']
inventory=[{'arm':'A','step':0,'path':str(base/'model.safetensors'),'SHA256':sha(base/'model.safetensors'),'base_directory':str(base),'type':'frozen public Spatial-Pro + zero-delta wrapper'}]
for a in 'BCD':
 row=next(r for r in csv.DictReader(open(T/'checkpoint_inventory.csv')) if r['arm']==a and r['step']=='100');assert sha(row['path'])==row['SHA256'];protected[row['path']]=row['SHA256']
 inventory.append({'arm':a,'step':100,'path':row['path'],'SHA256':row['SHA256'],'base_directory':str(base),'type':'Step5.7 Math-trained unified final'})
cs('checkpoint_inventory.csv',inventory)
protocol=json.loads((OLD/'evaluation_protocol.json').read_text());protocol.update({'research_definition':'背景材质干预下的视觉鲁棒性','state_use':'exploratory_reused_states','models':list('ABCD'),'trials':160,'order':'state_id ascending; clean A B C D, then background A B C D','smoke':'first state Clean A/B/C/D; included in160; no parameter changes','inference_backend':'Math SDPA for all four','GPU_UUID':'${GPU_UUID}','invalid_rule':'record INVALID, stop all remaining trials; no automatic retry; invalid excluded from SR, never count as failure','bootstrap':{'replicates':10000,'seed':20261022,'unit':'resample five initial states with replacement within each of four fixed tasks, keeping all eight paired outcomes','limitations':'conditional on these four tasks; not a new-task or training-seed CI'},'no_training':True,'original_geometry_reference':str(OLD/'geometry')})
js('evaluation_protocol.json',protocol);js('state_manifest.json',states)
trials=[]
for s in states:
 for app in ['clean','background']:
  for a in 'ABCD':
   r=next(x for x in inventory if x['arm']==a);trials.append({**s,'trial_id':f'{a}_{int(s["state_id"]):02d}_{app}','model':a,'appearance':app,'checkpoint':r['path'],'checkpoint_SHA256':r['SHA256'],'state_use':'exploratory_reused_states','background_texture_ID':'tex-table/tex-wall' if app=='background' else 'original','order_index':len(trials)})
cs('evaluation_manifest.csv',trials);js('manifest_frozen.json',{'manifest_SHA256':sha(O/'evaluation_manifest.csv'),'protocol_SHA256':sha(O/'evaluation_protocol.json'),'before_any_model_scoring':True})
js('protection_before.json',protected)
js('git_before.json',{str(p):{k:subprocess.check_output(['git','-C',str(p)]+v,text=True) for k,v in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short']}.items()} for p in [R,Path('${VLA_REFERENCE_ROOT}'),W]})
js('code_inventory.json',{str(W/n):sha(W/n) for n in ['prepare_step58.py','rollout_step58.py','run_step58.sh','causal_unified.py']}|{str(OLD/'env_helpers.py'):sha(OLD/'env_helpers.py')})
print('FROZEN_160',flush=True)
