"""CPU-only field audit, no held-out image reads or model substitute."""
import os,json,csv,hashlib,subprocess
from pathlib import Path
import numpy as np
R=Path(os.environ['RLA_PROJECT_ROOT']);O=Path(os.environ['RUN_DIR'])
E=R/'experiments/rla_background_invariance';D=E/'step4_4_1_dataset_expansion/run_20261021_02'
W=R/'worktrees/rla-official-0fc6e97'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(open(p)))
def js(n,x):
    if (O/n).exists():
        assert json.load(open(O/n))==x, 'Refuse changed prior artifact'
        return
    with (O/n).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
sources=[W/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml',W/'rebuttal/exp2_vla_adapter/setup/setup_env.sh',W/'rebuttal/exp2_vla_adapter/vla_adapter_overlay/rla/loss.py',D/'sample_manifest.csv',D/'transition_manifest.csv']
js('source_provenance.json',{'official_commit':subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),'hashes':{str(p):sha(p) for p in sources},'previous_route':'no optimizer updates; all prior branch/results preserved'})
t_all=rows(D/'transition_manifest.csv');t=[x for x in t_all if x['valid']=='True'];s=rows(D/'sample_manifest.csv')
assert len(t)==220 and all(x['split'] in ['train','validation'] and x['valid']=='True' for x in t)
index={(x['episode'],int(x['source_state_id']),x['camera']):x for x in s}
full=[];ends=[];missing=[]
for x in t:
    e,a=x['episode'],int(x['source_start']); assert int(x['source_future'])==a+8
    ep=all((e,a+h,c) in index for h in [0,8] for c in ['agentview','robot0_eye_in_hand'])
    complete=all((e,a+h,c) in index for h in range(9) for c in ['agentview','robot0_eye_in_hand'])
    ends.append(ep);full.append(complete)
    if not complete: missing.append({'episode':e,'anchor':a,'missing_offsets':sorted({h for h in range(9) for c in ['agentview','robot0_eye_in_hand'] if (e,a+h,c) not in index})})
first=t[0];e=first['episode'];a=int(first['source_start']);arrays={}
for h in [0,8]:
    for camera in ['agentview','robot0_eye_in_hand']:
        sample=index[e,a+h,camera]
        assert all(sample[k]=='True' for k in ['geometry_check','segmentation_equal','depth_equal','finite'])
        assert sha(sample['array'])==sample['array_SHA256']
        with np.load(sample['array']) as z:
            arrays[f'{h}/{camera}']={k:z[k].copy() for k in ['camera_K','camera_c2w','state','action','raw_proprio','fk_proprio']}
arr=arrays['0/agentview']; diff=np.abs(arr['fk_proprio']-arr['raw_proprio'])
stats=[]
for key,z in arrays.items():
    stats.append({'state_camera':key,'K_shape':list(z['camera_K'].shape),'K_values':z['camera_K'].tolist(),'c2w_shape':list(z['camera_c2w'].shape),'finite':all(np.isfinite(v).all() for v in z.values())})
js('fixed_pair_manifest.json',{'episode':e,'anchor':a,'future':a+8,'samples':[index[e,a+h,c] for h in [0,8] for c in ['agentview','robot0_eye_in_hand']]})
js('data_interface_audit.json',{'status':'PARTIAL_NOT_LOSSLESS_OFFICIAL_INPUT','episodes_train':19,'episodes_validation':3,'transitions':220,'excluded_frozen_rows':len(t_all)-len(t),
    'dual_camera_h8_endpoints':sum(ends),'complete_h1_to_8_camera_sequences':sum(full),'missing':missing,
    'camera_evidence_fixed_pair':stats,'RGB_resolution':[128,128],'official_DINO_resolution':[224,224],
    'upscaling_is_not_lossless':True,'FK_minus_raw_max_xyz_m':float(diff[:3].max()),'FK_minus_raw_max_axisangle_rad':float(diff[3:6].max()),
    'FK_minus_raw_finger_max':float(diff[6:].max()),'proprio_semantics':'world EEF xyz + axisangle + two finger qpos; same field convention but not bit-identical to historical raw observed proprio',
    'state_hash_join':'FK full episode stream differs from RLDS raw state hash; needs own provenance/sidecar, no nearest-neighbor matching',
    'heldout_read':0,'policy_actions':{'raw':'OSC delta xyz,delta rotation3,gripper(-1open,+1close)','official_gripper':'1-clip(raw,0,1)','normalization':'BOUNDS_Q99 per stats mask; common fixed train stats required'},
    'complete_training_ready':False,'recovery':'HDF5 states/actions/XML exist for training demos; intermediate frames need fresh224 dualview geometry-validated replay; no substitution of h8 target'})
old=R/'artifacts/rla_policy_ood_and_data_readiness/sidecar_real_spatial_step750'
js('sidecar_alignment_audit.json',{'status':'PARTIAL','existing_real_RLDS_evidence':json.load(open(old/'vla_batch_acceptance.json')),
    'evidence_source_hashes':{n:sha(old/n) for n in ['vla_batch_acceptance.json','horizon_reference_acceptance.json','manifest.json']},
    'reuse_scope':'step750 pipeline evidence only; not officialstep40000 or new paired-sidecar acceptance',
    'variant_key_strategy':'separate clean/background official stores; explicit appearance selector required',
    'stock_join_collision':'sha1(raw states) identical between variants, so one unmodified global store cannot select both targets',
    'new_paired_sidecars_ready':False,'all220_missing_full_horizon_data':not any(full)})
print('DATA_AUDIT',len(t),sum(ends),sum(full),diff.tolist())
