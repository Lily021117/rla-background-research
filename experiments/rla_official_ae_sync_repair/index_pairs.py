"""Index diagnostic arrays without generating images/features or new motion."""
from common import *
import cv2
import argparse
def main(suites):
 for suite in suites:
  out=RUN/'color_final'/suite;meta=np.load(out/'synchronized_metadata.npz');m=mujoco.MjModel.from_binary_path(str(out/'canonical_model.mjb'));cameras=[]
  for name in CAMS:
   cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,name);f=128/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
   cameras.append(dict(name=name,runtime_id=cid,resolution=[256,256],K=[[-float(f),0,128],[0,float(f),128],[0,0,1]],
    rotation_field='camera_rotations[:, runtime_id].reshape(T,3,3) is camera-to-world MuJoCo rotation',
    opencv_c2w_rule='rotation @ diag(1,-1,-1), translation=camera_positions[:,runtime_id]',
    image_rule='native top-down render, horizontal flip; negative fx accounts for mirror',dynamic=name=='robot0_eye_in_hand'))
  n=len(meta['states']);deltas=np.diff(meta['states'][:,0]);assert np.allclose(deltas,.05,atol=1e-10,rtol=0)
  rows=[]
  for i in range(n):
   rows.append(dict(suite=suite,trajectory_id='000010',split='train',frame=i,source_action_index=int(meta['source_action_indices'][i]),time=float(meta['states'][i,0]),
    state_sha256=hashlib.sha256(meta['states'][i].tobytes()).hexdigest(),action_sha256=hashlib.sha256(meta['actions'][i].tobytes()).hexdigest(),
    clean_key=f'paired_rgb.npz::clean[{i}]',h_key=f'paired_rgb.npz::expanded_h[{i}]',hs_key=f'paired_rgb.npz::expanded_hs[{i}]',
    camera_order='agentview|robot0_eye_in_hand',admission='DIAGNOSTIC_ONLY_ORIGINAL_B_FAILED'))
  csvout(out/'pairing_manifest.csv',rows)
  source=np.load(out/'paired_rgb.npz');masks=source['background_mask'];appearance=[]
  for variant in ['clean','expanded_h','expanded_hs']:
   rgb=source[variant]
   for t in range(n):
    for ci,cam in enumerate(CAMS):
     mask=masks[t,ci];a=rgb[t,ci].astype('float32')/255;hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);lum=a @ np.array([.2126,.7152,.0722])
     appearance.append(dict(frame=t,camera=cam,variant=variant,background_pixels=int(mask.sum()),
      mean_saturation=float(hsv[...,1][mask].mean()) if mask.any() else None,
      mean_value=float(hsv[...,2][mask].mean()) if mask.any() else None,
      mean_luminance=float(lum[mask].mean()) if mask.any() else None))
  csvout(out/'brightness_saturation.csv',appearance)
  capture=RUN/'capture'/suite/'synchronized.npz'
  if capture.exists():
   # Pure array comparison: no new render or state update.
   now=np.load(out/'paired_rgb.npz')['clean'];previous=np.load(capture)['rgb'];delta=np.abs(now.astype('int16')-previous.astype('int16'))
   dump(out/'fullstate_vs_legacy_render.json',dict(frames=n,exact=bool(np.array_equal(now,previous)),max_rgb_error=int(delta.max()),mean_rgb_error=float(delta.mean()),
    explanation='Full runtime integration snapshot vs previous legacy time/qpos/qvel with exact compiled model. Result only certifies these render outputs, not controller continuation.'))
  dump(out/'camera_time_metadata.json',dict(cameras=cameras,frames=n,frame_interval_s=.05,temporal_semantics='synchronized pre-next-action boundary; not relabelled as source cached observation time',
   future_rule='official train h=2..64 -> image indices t,t+h-1 with t+h<N; validation h=10..64; no frame repeat',
   train_possible_time_pairs=sum(max(0,n-h) for h in range(2,65)),old_candidate_state_source=str(OLD/'pilot_replay'/suite/'candidate_replay.npz'),
   source_actions_unchanged=True,original_videos_replaced=False,formal_loader_admission=False,
   model_sha256=sha(out/'canonical_model.mjb'),array_sha256=sha(out/'paired_rgb.npz'),metadata_sha256=sha(out/'synchronized_metadata.npz')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,nargs='+',default=SUITES);main(p.parse_args().suite)
