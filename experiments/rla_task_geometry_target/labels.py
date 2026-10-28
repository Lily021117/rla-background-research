"""Training-only physical-window statistics; state setting + FK, never mj_step."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def build(manifest,out):
 out.mkdir(exist_ok=False);episodes=read(manifest)['episodes'];records=[];alltrain=[];protected={str(manifest):sha(manifest)}
 for e in episodes:
  root=Path(e['root']);path=root/'trajectory_labels.npz';lab=dict(np.load(path,allow_pickle=False));prov=read(root/'trajectory_provenance.json');xml=next(Path(p) for p in prov['source_hashes'] if p.endswith('/clean.xml'));assert sha(xml)==prov['source_hashes'][str(xml)]
  protected[str(path)]=sha(path);protected[str(xml)]=sha(xml);protected[str(root/'trajectory_provenance.json')]=sha(root/'trajectory_provenance.json')
  states=lab['states'];N=len(states);M=e['valid_chunks'];assert M==N-8 and hashlib.sha256(states.tobytes()).hexdigest()==e['source_states_sha256'];assert lab['valid_horizon_mask'][:M].all()
  np.testing.assert_allclose(np.diff(states[:,0]),.05,atol=1e-9,rtol=0)
  m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);assert m.na==0 and states.shape[1]==1+m.nq+m.nv
  bowl=m.body('akita_black_bowl_1_main').id;plate=m.body('plate_1_main').id;site=m.site('gripper0_grip_site').id
  positions=[];frames=[]
  for t,s in enumerate(states):
   mujoco.mj_resetData(m,d);d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
   np.testing.assert_array_equal(np.r_[d.time,d.qpos,d.qvel],s)
   np.testing.assert_allclose(d.site_xpos[site],lab['proprio'][t,:3],atol=6e-8,rtol=0)
   frame=root/'frames'/f'{t:04}.npz'
   with np.load(frame,allow_pickle=False) as f:
    for k in ('state','action','proprio'):np.testing.assert_array_equal(f[k],lab[{'state':'states','action':'actions','proprio':'proprio'}[k]][t])
    assert f['clean'].shape==f['background'].shape==(2,3,224,224)
   protected[str(frame)]=sha(frame)
   positions.append(np.stack([d.xpos[bowl],d.site_xpos[site],d.xpos[plate]]).copy())
  pos=np.array(positions);r=np.concatenate([pos[:,0]-pos[:,1],pos[:,0]-pos[:,2]],axis=1)
  future=np.arange(M)[:,None]+np.arange(1,9)[None,:];assert future.max()<N
  y=np.concatenate([np.broadcast_to(r[:M,None,:],(M,8,6)),r[future]-r[:M,None,:]],axis=-1)
  assert np.isfinite(y).all();np.savez_compressed(out/(e['episode']+'.npz'),raw=y,mask=lab['valid_horizon_mask'][:M],anchor=np.arange(M),future=future,positions=pos)
  if e['split']=='train':alltrain.append(y.reshape(-1,12))
  records.append(dict(episode=e['episode'],split=e['split'],windows=M,frames=N,source_states_sha256=e['source_states_sha256'],source_label_sha256=sha(path),xml_sha256=sha(xml),file=str(out/(e['episode']+'.npz')),file_sha256=sha(out/(e['episode']+'.npz')),labels_equal_across_appearance=True,state_not_advanced=True,source_demo=e['source_demo']))
  print('LABEL_PASS',e['episode'],M,flush=True)
 x=np.concatenate(alltrain);mean=x.mean(0);std=x.std(0);denom=np.maximum(std,.001)
 save(out/'statistics.json',dict(mean=mean.tolist(),raw_std=std.tolist(),denominator=denom.tolist(),floor_m=.001,floored_coordinates=np.flatnonzero(std<.001).tolist(),ddof=0,train_physical_horizon_samples=len(x),weighting='each eligible physical window once x all8 valid horizons; no appearance or exposure duplication',clip=False))
 save(out/'manifest.json',dict(status='PASS',episodes=records,definition='concat[r(t),r(t+h)-r(t)]',r='concat[bowl-eef,bowl-plate]',units='world metres',horizons=list(range(1,9)),horizon_evidence='ae_consistency/evaluate_and_extract.py and frozen sidecar_alignment_audit.json: current t to t+h; no tail used',bowl='akita_black_bowl_1_main body origin, established object geometry reference',plate='plate_1_main body origin',eef='gripper0_grip_site',policy_truth_inputs=False,AE_updates=0,optimizer_updates=0,environment_steps=0))
 save(out/'protected_inputs.json',protected)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.manifest,a.out)
