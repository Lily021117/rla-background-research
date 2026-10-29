"""Freeze the complete design before extraction, model diagnostics or scoring."""
import sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.common import *
def main():
 inv=list(csv.DictReader((AE/'delivery/checkpoint_inventory.csv').open()))
 assets={};encoders={}
 for arm in ['C','D']:
  r=next(r for r in inv if r['arm']==arm and int(r['step'])==2000)
  assert sha(r['path'])==r['sha256'];encoders[arm]=r
  assert read(AE/arm/'completion.json')['successful_updates']==2000
  assets[r['path']]=r['sha256']
 for p in [AE/'delivery/report.md',AE/'initialization_audit.json',AE/'frozen_plan.json',AE/'data_manifest.json',GAP/'evaluation_protocol.json']:
  assets[str(p)]=sha(p)
 data=read(AE/'data_manifest.json');episodes=[e for e in data['episodes'] if e['task_key']=='stove']
 assert sum(e['split']=='train' for e in episodes)==20 and sum(e['split']=='validation' for e in episodes)==5
 reference=read(OLD/f'seed_{ORDERS[0]}/run_plan.json')
 physical=read(reference['inputs']['data_manifest'])['episodes']
 assert {e['episode'] for e in physical}=={e['episode'] for e in episodes}
 for e in episodes:
  p=next(x for x in physical if x['episode']==e['episode'])
  for k in ['root','split','valid_chunks','source_actions_sha256','source_states_sha256']:assert p[k]==e[k]
 save(ROOT/'data_manifest.json',dict(episodes=episodes,train_windows=sum(e['valid_chunks'] for e in episodes if e['split']=='train'),development_windows=sum(e['valid_chunks'] for e in episodes if e['split']=='validation'),role='reused exploratory development, not independent test'))
 save(ROOT/'encoder_inventory.json',encoders)
 for order in ORDERS:
  source=OLD/f'seed_{order}';root=ROOT/f'seed_{order}';plan=read(source/'run_plan.json')
  step0=source/'C/formal/step0000.pt'
  plan.update(common_step0=str(step0),common_step0_sha256=sha(step0),previous=str(ROOT),arms=ARMS,experiment='scratch_AE_downstream_weight_control',targets=str(ROOT/'targets'),paired_C_sample_trace=str(source/'C/formal/actual_sample_ids.jsonl'),target_source_by_arm={a:'C' if a=='C-fixed' else 'D' for a in ARMS},closed_loop_trials=None)
  plan['shared_protocol'].update(normalization='each new AE own train-only per(16,64) mean plus global raw scalar std; both appearances, all valid windows/horizons equally weighted',lambda_aux=LAMBDA)
  for key in ['target_center_definition','historical_center_source','original_common_step0','original_common_step0_sha256']:plan.pop(key,None)
  save(root/'run_plan.json',plan)
  index=read(plan['inputs']['B20_sampling_order']);assert len(index)==4800
  valid={e['episode']:e for e in episodes if e['split']=='train'}
  assert all(e in valid and 0<=t<valid[e]['valid_chunks'] for e,t in index)
  for p in [step0,source/'run_plan.json',Path(plan['paired_C_sample_trace']),Path(plan['inputs']['B20_sampling_order']),Path(plan['inputs']['data_manifest']),Path(plan['inputs']['action_proprio_statistics'])]:assets[str(p)]=sha(p)
 base=read(GAP/'evaluation_protocol.json');frozen=read(base['frozen_state_protocol']);assert len(frozen['new_development_states'])==20
 assets[base['frozen_state_protocol']]=sha(base['frozen_state_protocol'])
 for s in frozen['new_development_states']:
  assets[s['state_file']]=sha(s['state_file'])
  for c in s['conditions']:
   p=Path(s['state_file']).parent/(c['condition']+'.xml');assert sha(p)==c['XML_sha256'];assets[str(p)]=sha(p)
 models={f'{a}_order{o}_s600':str(ROOT/f'seed_{o}'/a/'formal/step0600.pt') for o in ORDERS for a in ARMS}
 for phase,conditions in [('clean',['clean']),('background',[f'new_background_{i}' for i in range(1,5)])]:
  ep=dict(base);ep.update(models=list(models),conditions=conditions,trials=8*20*len(conditions),scratch_policy_evaluation=True,checkpoint_selection='all eight final600; no selection',gate='at least one method Clean>=12/20 in BOTH orders',optimizer_updates=0,backward=0)
  root=ROOT/f'evaluation_{phase}';save(root/'evaluation_protocol.json',ep);save(root/'policy_registry.json',models);save(root/'run_plan.json',read(ROOT/f'seed_{ORDERS[0]}/run_plan.json'))
 protocol=dict(status='FROZEN',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),AE=encoders,orders=ORDERS,arms=ARMS,lambda0=LAMBDA,updates_each=600,total_policy_updates=4800,AE_updates=0,DINO_updates=0,checkpoints=[0,1,100,300,600],validation=[0,100,300,600],gradient_diagnostics=[0,300,600],gradient_samples='first complete historical effective training batch of each order; identical across arms/nodes,16 appearances, micro8 x accumulation2',gradient_scope='intersection of action/aux differentiable parameters, shared modules only; exclude action-only and auxiliary-only modules',lambda_formula={'C-fixed':'lambda0','D-fixed':'lambda0','D-low':'lambda0/2','D-decay':'lambda0*(1-k/599), k=successful updates before next step'},diagnostic_lambda_at_node='weight for next update, capped at k599 for final600',normalization='per-token/per-coordinate mean over train valid physical anchors x8 horizons x2 appearances, global RAW std from mean/std moments, not residual RMS',target_dtype='float16 stored, float32 normalization then official BF16 loss',Clean_rollouts=160,background_rollouts_max=640,gate='one method in BOTH orders Clean>=12/20',all_backgrounds_frozen=True,seed_scope='two orders, same model initialization/training RNG; not two full seeds',statistics='20000 bootstrap draws, physical-state clusters jointly retain all appearances/orders; no training-seed or task uncertainty',comparisons=[['D-fixed','C-fixed'],['D-low','D-fixed'],['D-decay','D-low'],['D-decay','D-fixed']],infrastructure='stop on OOM/CUDA/nonfinite/ECC change; preserve attempts; no automatic retry or device switching',sealed_read=False,teacher=False,keep=False,policy_future_reconstruction=False,policy_consistency=False)
 save(ROOT/'protocol.json',protocol);save(ROOT/'protected_inputs.json',assets)
 table(ROOT/'auxiliary_weight_schedule.csv',[dict(arm=a,successful_update=k+1,k=k,lambda_RLA=weight(a,k)) for a in ARMS for k in range(600)])
 assert weight('D-decay',0)==LAMBDA and weight('D-decay',599)==0
 print('FROZEN_PROTOCOL_AND_ASSETS_PASS',flush=True)
if __name__=='__main__':main()
