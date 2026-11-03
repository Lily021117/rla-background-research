"""CPU-only candidate metadata checks. No replay, render or learned forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,h5py,mujoco
from prepare import RUN,ROOT,sha,dump
from paired_renderer import PHYS

def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();src=RUN/'pilot_replay'/a.suite
 result=json.loads((src/'result.json').read_text());assert (src/'candidate_replay.npz').exists()
 original=ET.parse(src/'original_source.xml');runtime=ET.parse(src/'runtime_model.xml')
 known={(e.tag,e.get('name')):e for e in runtime.getroot().find('asset')};assets=[]
 for e in original.getroot().find('asset'):
  if not e.get('file'):continue
  r=known.get((e.tag,e.get('name')));assert r is not None and r.get('file')
  # Do not infer equality from basename: require full logical asset suffix.
  old=e.get('file');new=r.get('file');oldtail=old.split('/assets/',1)[-1];newtail=new.split('/assets/',1)[-1]
  oldtail=os.path.normpath(oldtail);newtail=os.path.normpath(newtail)
  assert oldtail==newtail,(oldtail,newtail)
  assets.append(dict(kind=e.tag,name=e.get('name'),logical_asset=oldtail,local_path=new,local_sha256=sha(new),historical_asset_hash='NOT_PUBLISHED'))
  e.set('file',new)
 patched=src/'source_xml_local_assets.xml';original.write(patched,encoding='unicode')
 sm=mujoco.MjModel.from_xml_path(str(patched));m=mujoco.MjModel.from_xml_path(str(src/'runtime_model.xml'))
 diffs=[]
 for field in PHYS:
  x,y=getattr(sm,field),getattr(m,field)
  diffs.append(dict(field=field,source_shape=list(x.shape),runtime_shape=list(y.shape),exact=bool(np.array_equal(x,y)),
   max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.shape==y.shape and x.size else None))
 dump(src/'asset_and_model_comparison.json',dict(assets=assets,fields=diffs,source_xml_remap='same named asset and full logical relative path; no guessed alternate asset',
  runtime_choice='Rebuild BDDL then set initial state, exactly the published OpenVLA regeneration code; original HDF5 XML is NOT the unpublished regenerated XML',
  original_xml_sha256=sha(src/'original_source.xml'),runtime_xml_sha256=sha(src/'runtime_model.xml'),source_with_local_assets_sha256=sha(patched),
  historical_asset_byte_equality='UNKNOWN; original asset bytes/hashes not published in HDF5'))
 arr=np.load(src/'candidate_replay.npz');n=len(arr['states']);cams=arr['camera_world'];pos=cams[:,:m.ncam*3].reshape(n,m.ncam,3);rot=cams[:,m.ncam*3:].reshape(n,m.ncam,3,3)
 camera_rows=[]
 with h5py.File(ROOT/'data/libero_converted'/a.suite/'traj_000010/metadata.h5') as h:
  for cam,stream in [('agentview','agentview_camera'),('robot0_eye_in_hand','wrist_camera')]:
   cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,cam);assert cid>=0
   f=128/np.tan(m.cam_fovy[cid]*np.pi/360);K=np.array([[-f,0,128],[0,f,128],[0,0,1.]])
   # MuJoCo camera axes -> conventional camera-to-world; mirror represented by negative fx.
   c2w=np.zeros((n,3,4));c2w[:,:,:3]=rot[:,cid]@np.diag([1.,-1.,-1.]);c2w[:,:,3]=pos[:,cid]
   oldk=h[stream+'_intrinsics'][:].reshape(n,3,3);olde=h[stream+'_extrinsics'][:].reshape(n,3,4)
   camera_rows.append(dict(name=cam,stream=stream,id=cid,resolution=[256,256],K=K.tolist(),K_vs_converted_max=float(np.max(abs(oldk-K))),
    c2w_translation_vs_converted_max_m=float(np.max(abs(c2w[:,:,3]-olde[:,:,3]))),
    c2w_rotation_matrix_vs_converted_max=float(np.max(abs(c2w[:,:,:3]-olde[:,:,:3]))),
    wrist_is_dynamic=cam=='robot0_eye_in_hand',pose_variation_m=float(np.max(np.ptp(pos[:,cid],axis=0)))))
 dump(src/'camera_and_time_audit.json',dict(cameras=camera_rows,candidate_frame_count=n,
  frame0_original_state_time=json.loads((src/'environment.json').read_text())['first_original_state_time'],
  frame0_candidate_time=float(arr['states'][0,0]),candidate_time_step_min=float(np.min(np.diff(arr['states'][:,0]))),
  candidate_time_step_max=float(np.max(np.diff(arr['states'][:,0]))),control_hz=20,source_action_indices=arr['source_action_indices'].tolist(),
  original_frame0_deleted=False,global_shift_applied=False,state_origin='REPLAY_CANDIDATE_NOT_ORIGINAL',
  image_transform='raw OpenGL bottom-up render rotated 180 degrees; official K negative fx convention retained',
  precision_note='K preserves official center W/2 convention; original/reference camera derives partly from cached observation, candidate from runtime cam_xpos/cam_xmat',
  uniform_candidate_time_does_not_prove_original_alignment=True,renderer_and_camera_validation_pass=result['numeric_gate'] and result['pixel_gate']))
 print(a.suite,'metadata audit complete',flush=True)
if __name__=='__main__':main()
