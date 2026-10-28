"""Additional audit and threshold-free recorded condition summaries (no inference)."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import read,save,sha,table
p=Path(sys.argv[1]);manifest=read(p/'paired_manifest.json');hashes=read(p/'input_hashes.json');checks=[];metrics=[];positions=[]
expected={'C':{20261101:'9aa823878aeb51ec02c94580cea94bc73591d47a10bfed3251f58029d75a11a8',20261102:'3e2cdcd514e8aa4f0d7851199b7c17a7079b54043098f4ea9035e754b7076222'},'shared':{20261101:'590a3ab0b14a5cdc94d201b827d2f007ec0cd2723810bf315afe3fa9f931ab23',20261102:'2f4ab1a06c65cb47a5f124a999d8496b5310358b5589f138e33e7a24c9883b9b'}}
source=Path(next(x for x in hashes if x.endswith('/reuse_sources.json'))).parent
protocol=read(source/'evaluation_protocol.json');frozen=Path(protocol['frozen_state_protocol']);states=read(frozen)['new_development_states'];extra={str(frozen):sha(frozen),str(source/'evaluation_protocol.json'):sha(source/'evaluation_protocol.json')}
for st in states:
    f=Path(st['state_file']);v=np.load(f,allow_pickle=False);assert hashlib.sha256(v.tobytes()).hexdigest()==st['state_sha256'];extra[str(f)]=sha(f)
for r in manifest:
    for label in ('C','shared'):
        assert r[label+'_checkpoint_sha256']==expected[label][r['order']]
        path=Path(r[label+'_trajectory']);res=read(path.with_name('result.json'));ph=read(path.with_name('phase_timeline.json'));z=dict(np.load(path,allow_pickle=False));a=z['executed_actions'];n=len(a);lift=res['first_lift_step'];post=ph[lift-1:] if lift is not None else []
        assert res['checkpoint_sha256']==r[label+'_checkpoint_sha256'] and res['seed']==r['state'] and res['condition']==r['condition']
        state=next(s for s in states if s['seed']==r['state']);assert state['state_sha256']==res['state_sha256']
        # Recover the scalar baseline from recorded maximum lift; does not simulate or call FK.
        initial_z=float(z['target_bowl_positions'][:,2].max()-res['max_lift_m']);ii=np.flatnonzero(z['target_bowl_positions'][:,2]-initial_z>=.03);rec=int(ii[0]+1) if len(ii) else None;assert rec==lift
        checks.append(dict(pair_id=r['pair_id'],model=label,checkpoint_inventory_match=True,result_and_index_match=True,initial_state_hash_match=True,recorded_lift_index_reproduced=True))
        m={k:r[k] for k in ('pair_id','order','state','condition','group')};m.update(model=label,success=res['success'],steps=n,first_lift=lift,postlift_steps=len(post),postlift_contact_outside_center_steps=sum(x['contact_condition'] and not x['xy_condition'] for x in post),postlift_center_without_contact_steps=sum(x['xy_condition'] and not x['contact_condition'] for x in post),first_contact_outside_center=next((x['step'] for x in post if x['contact_condition'] and not x['xy_condition'] and x['step']<n),None),terminal_contact=ph[-1]['contact_condition'],terminal_height=ph[-1]['height_condition'],terminal_center=ph[-1]['xy_condition'],terminal_goal=ph[-1]['goal'],first_close_chunk_position=next((i%8 for i in range(n) if a[i,6]>0),None))
        metrics.append(m)
        for pos in range(8):
            ids=[i for i in range(1,n) if i%8==pos];delta=abs(a[ids]-a[np.array(ids)-1]);positions.append(dict(pair_id=r['pair_id'],model=label,order=r['order'],group=r['group'],chunk_position=pos,n=len(ids),translation_command_jump=float(delta[:,:3].mean()),rotation_command_jump=float(delta[:,3:6].mean()),gripper_switch_fraction=float((delta[:,6]>0).mean())))
table(p/'identity_and_initial_state_audit.csv',checks);table(p/'recorded_condition_summary.csv',metrics);table(p/'within_vs_boundary_command_changes.csv',positions);save(p/'additional_source_hashes.json',extra)
save(p/'field_semantics.json',dict(executed_actions='7D OSC_POSE controller commands: xyz translation increments, three rotation-control components, gripper; not world displacement in metres or measured orientation error. Controller scaling not inferred from shape.',predicted_chunks='de-normalized action-space predictions,8x7 per inference; gripper raw0close/1open; deployed -sign(2*g-1), +1close/-1open; checked elementwise all400.',position_units='metres in simulator world; all after action; body and geometry bowl centers checked equal for all records.',states='92D serialized simulator vector; post-action; not decoded as arbitrary qpos offsets.',video='2.5fps at chunk inference, pre-action controls0,8,...; terminal result may occur after final video frame.',initial='frozen protocol arrays independently hashed; never replaced by first post-action row.',lift='existing3cm proxy, baseline reconstructed from max(body_z)-recorded max_lift solely to audit index; not grasp truth and not required for task success.',sources=['experiments/rla_lr_budget_sweep/extend_evaluation.py:63-109','experiments/official_libero_rla_consistency/stove_postlift/common.py:24-34','upstream/experiments/robot/libero/run_libero_eval.py:269-279'],pose_inference='No rotation-command difference is called actual angular error; no force, friction, slip, grasp stability or training coverage claim.'))
print(pd.DataFrame(metrics).groupby(['order','group','model'])[['terminal_contact','terminal_height','terminal_center']].sum().to_string())
