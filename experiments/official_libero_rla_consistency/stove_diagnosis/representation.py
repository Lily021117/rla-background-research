"""CPU frozen-sidecar readout and same-state geometric-label verification.

Closed-form ridge is a diagnostic readout only. AE/policy weights are not loaded
or changed. Inputs are only the twenty training and five development episodes.
"""
import argparse,collections,os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import mujoco
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.linalg import solve
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);ae=root/'ae_run';es=read(root/'frozen_dataset_manifest.json')['episodes'];side=read(ae/'sidecar_provenance.json');lookup={(x['model'],x['episode'],x['appearance']):x for x in side};geom={};phases={};labelaudit=[]
for e in es:
 lab=np.load(Path(e['root'])/'trajectory_labels.npz');m=mujoco.MjModel.from_xml_path(str(root/'source_models'/e['source_demo']/'clean.xml'));d=mujoco.MjData(m);assert m.na==0 and lab['states'].shape[1]==1+m.nq+m.nv
 bid=m.body('akita_black_bowl_1_main').id;site=m.site('gripper0_grip_site').id;hand=m.body('robot0_right_hand').id;finger=[int(m.joint('gripper0_finger_joint'+str(i)).qposadr[0]) for i in (1,2)];positions=[];maxerr=0.
 for s,expected in zip(lab['states'],lab['proprio']):
  d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
  p=np.r_[d.site_xpos[site],Rotation.from_quat(np.roll(d.xquat[hand],-1)).as_rotvec(),d.qpos[finger]].astype(np.float32);maxerr=max(maxerr,float(np.max(abs(p-expected))));positions.append(np.r_[d.xpos[bid],d.qpos[finger],d.site_xpos[site]])
 assert maxerr<=1e-7,(e['episode'],maxerr);positions=np.array(positions);geom[e['episode']]=positions
 raw=lab['actions'][:,6];close=np.flatnonzero((raw[:-1]<0)&(raw[1:]>0))+1;first=int(close[0]) if len(close) else len(raw);phases[e['episode']]=['transport_or_placement' if x[2]-positions[0,2]>=.03 else 'closure_window' if abs(t-first)<=4 else 'approach' if t<first else 'after_close_no_lift' for t,x in enumerate(positions)]
 labelaudit.append(dict(episode=e['episode'],split=e['split'],frames=len(positions),state_dimension=lab['states'].shape[1],bowl_body='akita_black_bowl_1_main',gripper_joints='gripper0_finger_joint1,gripper0_finger_joint2',same_state_FK_max_abs_error=maxerr,XML_sha256=sha(root/'source_models'/e['source_demo']/'clean.xml'),labels='simulator-recorded-state derived bowl world xyz, finger qpos; external real-world accuracy not tested',task_joint_label='not applicable: task is bowl retrieval from static stove, not stove-knob manipulation'))
csvsave(a.out/'physical_label_availability.csv',labelaudit)
rec=csvread(ae/'reconstruction_metrics.csv');groups=collections.defaultdict(list)
for r in rec:
 r['phase']=phases[r['episode']][int(r['t'])];groups[r['model'],r['appearance'],r['camera'],r['h'],r['phase']].append(r)
csvsave(a.out/'AE_reconstruction_by_camera_horizon_phase.csv',[dict(model=k[0],appearance=k[1],camera=k[2],h=k[3],phase=k[4],n=len(rs),**{name:float(np.mean([float(r[name]) for r in rs])) for name in ('reconstruction_L1','reconstruction_MSE','copy_current_L1','copy_current_MSE')}) for k,rs in groups.items()])
dev=read(ae/'representation_sample_manifest.json')['samples'];stats=read(ae/'sidecars/C/clean/stats.json');offset=np.array(stats['mean']);std=np.array(stats['std']);scale=np.sqrt(np.mean(offset**2+std**2)-offset.mean()**2);distribution=[];pairs=[];variance=[]
def latent(arm,e,t,h,c):return np.load(lookup[arm,e,c]['file'],mmap_mode='r')[t,h-1].astype(np.float64).reshape(-1)
for arm in ('C','D'):
 z={c:np.stack([latent(arm,x['episode'],x['t'],x['h'],c) for x in dev]) for c in ('clean','background')};ids=np.array([x['episode'] for x in dev]);hs=np.array([x['h'] for x in dev]);different=(ids[:,None]!=ids[None,:])&(hs[:,None]==hs[None,:]);upper=np.triu(np.ones(different.shape,bool),1);different&=upper
 for normalization in ('raw','unit_L2','official_common_C_center'):
  normed={c:v if normalization=='raw' else v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12) if normalization=='unit_L2' else (v-offset.reshape(1,-1))/scale for c,v in z.items()};within=np.linalg.norm(normed['clean']-normed['background'],axis=1)
  for c,v in normed.items():
   centered=v-v.mean(0);sv=np.linalg.svd(centered,compute_uv=False);eig=sv**2;prob=eig/max(eig.sum(),1e-30);var=v.var(0);between=cdist(v,v)[different];nr=np.linalg.norm(v,axis=1);distribution.append(dict(model=arm,appearance=c,normalization=normalization,n=len(v),norm_mean=float(nr.mean()),norm_std=float(nr.std()),variance_mean=float(var.mean()),effective_rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))),participation_ratio=float(eig.sum()**2/max((eig**2).sum(),1e-30)),within_mean=float(within.mean()),between_same_horizon_different_episode_mean=float(between.mean()),within_between_ratio=float(within.mean()/max(between.mean(),1e-30)),between_p05=float(np.quantile(between,.05)),between_p50=float(np.median(between)),between_p95=float(np.quantile(between,.95))))
   if normalization=='raw':variance.extend(dict(model=arm,appearance=c,dimension=i,variance=float(x)) for i,x in enumerate(var))
  for i,x in enumerate(dev):pairs.append(dict(model=arm,normalization=normalization,**x,phase=phases[x['episode']][x['t']],distance=float(within[i])))
csvsave(a.out/'latent_distribution.csv',distribution);csvsave(a.out/'latent_dimension_variance.csv',variance);csvsave(a.out/'paired_latent_distance.csv',pairs)
# h=8, all episodes, ten deterministic anchors, training clean only. Center/zscore
# fitted independently per arm using identical procedure and training rows.
probe_samples=[dict(episode=e['episode'],split=e['split'],t=int(t),h=8) for e in es for t in np.linspace(0,e['valid_chunks']-1,10,dtype=int)];save(a.out/'probe_manifest.json',dict(samples=probe_samples,fit_appearance='clean',alpha_on_mean_squared_loss=1.,selection='no validation hyperparameter choice',feature_standardization='training per-dimension mean/std; floor1e-6',model='closed-form linear ridge; intercept via centering',limitations='small correlated windows; independent unit source episode, 20 train/5 development; motion-readout not control test'))
is_train=np.array([x['split']=='train' for x in probe_samples]);y=np.stack([geom[x['episode']][x['t']+8]-geom[x['episode']][x['t']] for x in probe_samples]);metrics=[];predictions=[]
for arm in ('C','D'):
 zc=np.stack([latent(arm,x['episode'],x['t'],8,'clean') for x in probe_samples]);mu=zc[is_train].mean(0);sd=np.maximum(zc[is_train].std(0),1e-6);X=(zc[is_train]-mu)/sd;ym=y[is_train].mean(0);Y=y[is_train]-ym;n=len(X);coef=solve(X@X.T+n*np.eye(n),Y,assume_a='pos');W=X.T@coef
 for appearance in ('clean','background'):
  z=np.stack([latent(arm,x['episode'],x['t'],8,appearance) for x in probe_samples]);pred=(z-mu)/sd@W+ym
  for i,x in enumerate(probe_samples):
   if x['split']=='train':continue
   predictions.append(dict(model=arm,appearance=appearance,**x,**{f'y{j}':float(y[i,j]) for j in range(8)},**{f'p{j}':float(pred[i,j]) for j in range(8)}))
  for label,dims in [('bowl_xyz',[0,1,2]),('finger_qpos',[3,4]),('eef_xyz_ancillary',[5,6,7])]:
   for episode in ['ALL']+[e['episode'] for e in es if e['split']=='validation']:
    ix=np.array([x['split']=='validation' and (episode=='ALL' or x['episode']==episode) for x in probe_samples]);truth=y[ix][:,dims];guess=pred[ix][:,dims];err=guess-truth;sst=((truth-truth.mean(0))**2).sum();baseline=ym[dims];metrics.append(dict(model=arm,appearance=appearance,label=label,episode=episode,n=int(ix.sum()),MAE_m=float(abs(err).mean()),RMSE_m=float(np.sqrt((err**2).mean())),R2_pooled=float(1-(err**2).sum()/sst) if sst>1e-16 else None,train_mean_baseline_MAE_m=float(abs(truth-baseline).mean()),zero_change_baseline_MAE_m=float(abs(truth).mean())))
csvsave(a.out/'motion_probe_metrics.csv',metrics);csvsave(a.out/'motion_probe_predictions.csv',predictions)
# Paired episode-level bootstrap, fixed five development episodes resampled as blocks.
rng=np.random.RandomState(20261024);episodes=[e['episode'] for e in es if e['split']=='validation'];idx=rng.randint(0,5,(10000,5));boot=[]
for c in ('clean','background'):
 for label in ('bowl_xyz','finger_qpos','eef_xyz_ancillary'):
  values={arm:np.array([next(r['MAE_m'] for r in metrics if r['model']==arm and r['appearance']==c and r['label']==label and r['episode']==ep) for ep in episodes]) for arm in ('C','D')};delta=values['D']-values['C'];boot.append(dict(appearance=c,label=label,D_minus_C_MAE_m=float(delta.mean()),episode_bootstrap95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist(),episodes=5))
save(a.out/'motion_probe_paired_uncertainty.json',boot);save(a.out/'representation_verification.json',dict(status='PASS_FROZEN_DIAGNOSTIC',AE_updates=0,policy_updates=0,backward=0,readout_fit='closed-form ridge only',external_motion_ground_truth=False,recorded_simulator_geometry_verified=True,sealed_read=False,common_sidecar_global_std=float(scale),reconstruction_source='existing exact same development samples; no new AE inference'))
print('REPRESENTATION DIAGNOSTIC COMPLETE',flush=True)
