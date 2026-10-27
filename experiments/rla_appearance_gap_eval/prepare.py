"""Historical flip counts and outcome-independent asset/protocol freeze."""
import argparse,collections,copy,os,re,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_appearance_gap_eval.common import *

def prepare(out):
 protected={};rows=[];pairs=[]
 for order in ORDERS:
  for method,run in [('C',C),('D-center',D),('C-shared-target',SHARED)]:
   path=run/f'seed_{order}/per_trial_results.json';trials=read(path);protected[str(path)]=sha(path)
   assert len(trials)==60 and all(t['status']=='COMPLETE' for t in trials)
   by={(t['seed'],t['condition']):t for t in trials};states=sorted({t['seed'] for t in trials});assert len(states)==20
   for bg in ['seen_background','unseen_background']:
    counts=collections.Counter()
    for s in states:
     clean,b=by[s,'clean'],by[s,bg]
     assert clean['state_sha256']==b['state_sha256'] and clean['language']==b['language']
     label=('clean_success_' if clean['success'] else 'clean_failure_')+('background_success' if b['success'] else 'background_failure')
     counts[label]+=1;pairs.append(dict(order=order,method=method,state=s,background=bg,clean_success=int(clean['success']),background_success=int(b['success']),pattern=label))
    rows.append(dict(order=order,method=method,background=bg,states=20,**{k:counts[k] for k in ['clean_success_background_failure','clean_failure_background_success','clean_success_background_success','clean_failure_background_failure']}))
 table(out/'historical_flip_summary.csv',rows);table(out/'historical_paired_states.csv',pairs)
 plan=read(C/f'seed_{ORDERS[0]}/run_plan.json');old_protocol=read(C/f'seed_{ORDERS[0]}/evaluation_protocol.json');frozen_path=Path(plan['previous'])/'development_protocol/evaluation_protocol.json';old_states=read(frozen_path);protected[str(frozen_path)]=sha(frozen_path)
 # Fixed selection: generic, publicly shipped simulator color maps only.
 # No benchmark split, task trajectory, model result, or protected asset list
 # is used to select or rank these files.
 asset_root=Path('/data/lyc/projects/LIBERO-plus/libero/libero/assets/textures')
 excluded_paths=set()
 for s in old_states['new_development_states']:
  for c in s['conditions']:
   for change in c['changes']:excluded_paths.update([change['old'],change['new']])
 baseline=ET.parse(Path(old_states['new_development_states'][0]['state_file']).parent/'clean.xml')
 for n in baseline.iter('texture'):
  if n.get('name') in ['tex-table','tex-wall']:excluded_paths.add(n.get('file'))
 excluded_hashes={sha(p) for p in excluded_paths};families={}
 for p in sorted(asset_root.glob('*_COL*.png')):
  if 'backup' in p.name.lower():continue
  family=p.name.split('_COL')[0]
  if family not in families:families[family]=p
 selection_rule='Sort public generic textures/*_COL*.png, exclude backup files, group by prefix before _COL, keep lexicographically first file per family; exclude historical Clean/seen/unseen assets by path/SHA; take first eight eligible families. First four are table textures; next four are wall textures in matching rank. No policy scoring. Resize RGB to original target texture dimensions with LANCZOS; all material/scattering parameters fixed.'
 selected=[]
 for family,p in sorted(families.items()):
  digest=sha(p)
  if str(p) in excluded_paths or digest in excluded_hashes:continue
  selected.append((family,p,digest))
  if len(selected)==8:break
 assert len(selected)==8,'insufficient permitted development texture families; do not substitute perturbation type'
 (out/'assets').mkdir();asset_rows=[];configs=[]
 target_sizes={n.get('name'):Image.open(n.get('file')).size for n in baseline.iter('texture') if n.get('name') in ['tex-table','tex-wall']}
 for i in range(4):
  config=dict(id=CONDITIONS[i+1],textures={})
  for name,j in [('tex-table',i),('tex-wall',i+4)]:
   family,p,digest=selected[j];dest=out/'assets'/f'{CONDITIONS[i+1]}_{name}.png'
   with Image.open(p) as im:
    source_size=im.size;rgb=im.convert('RGB').resize(target_sizes[name],Image.Resampling.LANCZOS);rgb.save(dest)
   assert sha(dest) not in excluded_hashes
   protected[str(p)]=digest
   config['textures'][name]=str(dest)
   asset_rows.append(dict(condition=config['id'],texture=name,family=family,source=str(p),source_sha256=digest,source_size=list(source_size),output=str(dest),output_sha256=sha(dest),output_size=list(target_sizes[name]),role='generic development simulator material; not a sealed-test asset',selection_rank=j))
  configs.append(config)
 save(out/'asset_manifest.json',dict(selection_rule=selection_rule,source_root=str(asset_root),exclusion_scope='all texture resources in current task frozen training/seen/unseen development protocol, original clean resources; configurations disjoint by SHA',excluded_resources=[dict(path=p,sha256=sha(p)) for p in sorted(excluded_paths)],assets=asset_rows,configurations=configs,sealed_asset_inventory_read=False))
 inventory=[];registry={}
 for order in ORDERS:
  p=C/f'seed_{order}/C/formal/step0600.pt';accept=read(p.parent/'training_acceptance.json');assert accept['successful_updates']==600 and accept['strict_reload']
  model=f'C_order{order}_s600';digest=sha(p);registry[model]=str(p);protected[str(p)]=digest
  inventory.append(dict(order=order,model=model,checkpoint=str(p),sha256=digest,global_step=600,source_acceptance=accept,historical_evaluation_GPU='GPU1',current_evaluation_GPU='GPU0'))
  for q in [p.parent/'execution_configuration.json',C/f'seed_{order}/run_plan.json',C/f'seed_{order}/evaluation_protocol.json']:protected[str(q)]=sha(q)
 save(out/'checkpoint_inventory.json',inventory);save(out/'policy_registry.json',registry)
 save(out/'run_plan.json',plan);save(out/'normalization.json',read(C/f'seed_{ORDERS[0]}/normalization.json'))
 static_root=out/'development_protocol';static_root.mkdir();states=[]
 for s in old_states['new_development_states']:
  original=Path(s['state_file']);state=np.load(original);assert __import__('hashlib').sha256(state.tobytes()).hexdigest()==s['state_sha256']
  folder=static_root/f'seed_{s["seed"]}';folder.mkdir();np.save(folder/'state.npy',state);clean=original.parent/'clean.xml';text=clean.read_text();assert sha(clean)==s['conditions'][0]['XML_sha256'];protected[str(original)]=sha(original);protected[str(clean)]=sha(clean)
  (folder/'clean.xml').write_text(text);conditions=[dict(condition='clean',XML_sha256=sha(folder/'clean.xml'),changes=[])]
  for config in configs:
   tree=ET.fromstring(text);changes=[]
   for name,path in config['textures'].items():
    nodes=[n for n in tree.iter('texture') if n.get('name')==name];assert len(nodes)==1
    changes.append(dict(texture=name,old=nodes[0].get('file'),new=path));nodes[0].set('file',path)
   generated=ET.tostring(tree,encoding='unicode');dest=folder/(config['id']+'.xml');dest.write_text(generated)
   # Revert the only two file attributes; canonical XML must equal source.
   for change in changes:next(n for n in tree.iter('texture') if n.get('name')==change['texture']).set('file',change['old'])
   assert ET.tostring(tree)==ET.tostring(ET.fromstring(text))
   conditions.append(dict(condition=config['id'],XML_sha256=sha(dest),changes=changes,only_two_texture_file_attributes_changed=True))
  states.append(dict(seed=s['seed'],state_file=str(folder/'state.npy'),state_sha256=s['state_sha256'],source='unchanged historical frozen development reset state; reused',original_state_file=str(original),conditions=conditions,initial_success=False))
 save(static_root/'evaluation_protocol.json',dict(new_development_states=states,independence='exploratory reused development states; no independent test claim',status='FROZEN_BEFORE_STATIC_RENDER_AND_POLICY',conditions=CONDITIONS))
 protocol=dict(frozen=True,frozen_at_unix=time.time(),models=list(registry),states=[s['seed'] for s in states],conditions=CONDITIONS,trials=200,order='state,condition,model (20261101 then20261102)',max_steps=220,control_freq=20,chunk_execution=8,reflection=1,MSAA=4,bddl=old_protocol['bddl'],task=old_protocol['task'],role='appearance-gap evaluation; exploratory reused development states',frozen_state_protocol=str(static_root/'evaluation_protocol.json'),appearance_gap_evaluation=True,checkpoint_selection='existing two final600 C checkpoints only',retry_rule='stop on infrastructure error and retain evidence; documented repair can retry identical triplet; never rerun task failures',success='unchanged env.check_success()',optimizer_updates=0,AE_updates=0,backward=0,statistical_plan='20,000 physical-state clustered bootstrap draws with seed20261027; all appearances and both models stay together; each model reported separately; report all four backgrounds including favorable ones',decision_rule='report individual effects and whether additional failures repeat across multiple backgrounds in both C models; no inference from a selected worst background; no post hoc exclusion')
 save(out/'evaluation_protocol.json',protocol);save(out/'protected_inputs.json',protected)
 print('PREPARED_FROZEN_200_TRIALS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.out)
