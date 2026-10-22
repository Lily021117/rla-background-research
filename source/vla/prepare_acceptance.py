import os,json,csv,hashlib,subprocess
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
R=Path('${RLA_PROJECT_ROOT}');E=R/'experiments/rla_background_invariance';O=Path(os.environ['RUN_DIR']);T=E/'step5_1_policy_training/run_20261022_02';F=E/'step4_5_3_feature_target_expansion/run_20261021_01';N=E/'step4_1_adapter_readiness/run_20261021_02'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False))
protected=json.loads((E/'step5_5_causal_readiness/run_20261022_01/protection_before.json').read_text())
for p,h in protected.items():assert sha(p)==h,p
for p in (E/'step5_5_causal_readiness/run_20261022_01').glob('*'):
 if p.is_file():protected[str(p)]=sha(p)
js('protection_before.json',protected)
js('git_before.json',{str(repo):{k:subprocess.check_output(['git','-C',str(repo)]+v,text=True) for k,v in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short'],'worktrees':['worktree','list']}.items()} for repo in [R,Path('${VLA_REFERENCE_ROOT}'),Path(__file__).parent]})
samples=json.loads((T/'sample_manifest.json').read_text());ts={(r['group'],r['transition_id'],r['appearance']):r for r in csv.DictReader(open(F/'rla_target_manifest.csv')) if r['camera']=='agentview'}
tok=AutoTokenizer.from_pretrained(R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint',local_files_only=True)
old={r['sample_id']:r for r in csv.DictReader(open(T/'language_condition_audit.csv'))};rows=[]
for s in samples:
 assert sha(s['path'])==s['SHA256'];b=torch.load(s['path'],map_location='cpu',weights_only=False)
 decoded=tok.decode(b['input_ids'][0].tolist(),skip_special_tokens=False).lower();assert decoded==old[str(s['id'])]['actual_decoded_input'] and s['instruction'] in decoded
 for g in ['B','C']:
  r=ts[g,s['transition_id'],s['appearance']];assert sha(r['path'])==r['SHA256'];n=np.load(N/f'normalization_{g}.npz');z=((np.load(r['path'])-n['mean'])/n['scale']).reshape(1,1,2048)
  assert np.isfinite(z).all()
  if g=='C':np.testing.assert_allclose(z,b['rla'],rtol=0,atol=1e-6)
 for k,v in b.items():
  if torch.is_tensor(v):assert torch.isfinite(v).all()
 rows.append({'id':s['id'],'episode':s['episode'],'split':s['split'],'appearance':s['appearance'],'transition_id':s['transition_id'],'source_batch':s['path'],'source_SHA256':s['SHA256'],'actual_decoded_input':decoded,'language_pass':True,'all_current_fields_identical_across_arms':True,'ordinary_target':ts['B',s['transition_id'],s['appearance']]['path'],'consistency_target':ts['C',s['transition_id'],s['appearance']]['path'],'target_shapes':'1x1x2048','pass':True})
with (O/'batch_audit.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
common=json.loads((T/'common_config.json').read_text())
js('unified_model_config.json',{'architecture':'native Pro action trunk unchanged; side readout of final normalized 8x896 action hidden -> shared per-token Linear896->256 ->1x2048 target','auxiliary_parameters':231424,'target_shape':[1,32,64],'no_aux_queries_in_action_attention':True,'aux_init_seed':20261021,'current_inputs':'same accepted images/FK/language; targets only objective','A_frozen':True,'B_weight':0.,'C_weight':.05,'D_weight':.05,'C_teacher':'Step2 B','D_teacher':'Step2 C','not_equivalent_to_old_SparseHead':True,'common_training':common,'budget_this_run':0,'future_budget_BCD':100,'inference_equivalence_required_max_abs':0,'normalization':'existing corresponding train-only statistics; distinct values same algorithm','evaluation_limitations':'0 new independent states;20 historical exploratory;3 validation missing stove'})
print('CPU_AUDIT_PASS',len(rows),flush=True)
