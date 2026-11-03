"""Export admitted rerenders without changing source files. CPU FK, no integration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,shutil
import numpy as np,h5py,cv2,mujoco
from PIL import Image,ImageDraw
from prepare import RUN,ROOT,sha,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();suite=a.suite
 src=RUN/'pilot_replay'/suite;paired=RUN/'paired_rendered'/suite
 assert json.loads((paired/'result.json').read_text())['restored_clean_vs_replay_gate']
 original=ROOT/'data/libero_converted'/suite/'traj_000010'
 replay=np.load(src/'candidate_replay.npz');states=replay['states'];n=len(states)
 model=mujoco.MjModel.from_xml_path(str(src/'runtime_model.xml'));data=mujoco.MjData(model)
 names=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']
 ids=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)] for name in names]
 assert all(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)>=0 for name in names)
 site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');assert site>=0
 qpos=[];eef=[];extr={};Ks={};images={label:np.load(paired/f'{label}.npz')['rgb'] for label in ['clean','background']}
 for s in states:
  data.time=s[0];data.qpos[:]=s[1:1+model.nq];data.qvel[:]=s[1+model.nq:];mujoco.mj_forward(model,data)
  qpos.append(data.qpos[ids].copy());quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site]);eef.append(np.r_[data.site_xpos[site],quat])
  for cam,stream in [('agentview','agentview_camera'),('robot0_eye_in_hand','wrist_camera')]:
   cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,cam)
   extr.setdefault(stream,[]).append(np.c_[data.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.]),data.cam_xpos[cid]])
   f=128/np.tan(model.cam_fovy[cid]*np.pi/360);Ks[stream]=np.array([[-f,0,128],[0,f,128],[0,0,1.]])
 qpos=np.array(qpos,dtype='float32');eef=np.array(eef,dtype='float32');metadata_changes=[]
 for label,rgb in images.items():
  dest=paired/'official_loader'/label/suite;dest.mkdir(parents=True,exist_ok=False);traj=dest/'traj_000010';traj.mkdir()
  shutil.copy2(original.parent/'metadata.json',dest/'metadata.json');shutil.copy2(original/'metadata.h5',traj/'metadata.h5')
  with h5py.File(traj/'metadata.h5','r+') as h:
   values={'qpos':qpos,'target_qpos':np.r_[qpos[1:],qpos[-1:]],'eef_pose':eef}
   for stream in Ks:
    values[stream+'_intrinsics']=np.tile(Ks[stream],(n,1,1,1)).astype('float32');values[stream+'_extrinsics']=np.array(extr[stream],dtype='float32')[:,None]
   for key,value in values.items():
    previous=h[key][:];assert value.shape==previous.shape;h[key][:]=value
    if label=='clean':metadata_changes.append(dict(field=key,max_abs_change=float(np.max(np.abs(value-previous)))))
   with h5py.File(original/'metadata.h5') as old:assert np.array_equal(h['action'][:],old['action'][:]) and h.attrs['task_description']==old.attrs['task_description']
  for ci,stream in enumerate(['agentview_camera','wrist_camera']):
   frames=traj/(stream+'_rgb');frames.mkdir()
   for t,im in enumerate(rgb[:,ci]):Image.fromarray(im).save(frames/f'{t:06d}.png')
  # Retain official all-white placeholders, explicitly NOT segmentation ground truth.
  for f in original.glob('*foreground_mask*'):shutil.copy2(f,traj/f.name)
  idx=json.loads((original.parent/'index.json').read_text());entry=idx['trajectories']['000010'];entry['path']=str(traj)
  dump(dest/'index.json',dict(trajectories={'000010':entry},trajectory_list=['000010']))
 caps=[cv2.VideoCapture(str(original/(s+'_rgb.mp4'))) for s in ['agentview_camera','wrist_camera']]
 writer=cv2.VideoWriter(str(paired/'full_original_clean_hue.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(768,512));assert writer.isOpened()
 fixed=sorted(set(np.linspace(0,n-1,7,dtype=int)));sheet=Image.new('RGB',(768,len(fixed)*538),'#222');draw=ImageDraw.Draw(sheet)
 for t in range(n):
  ori=[]
  for cap in caps:ok,bgr=cap.read();assert ok;ori.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
  panel=np.concatenate([np.concatenate([ori[c],images['clean'][t,c],images['background'][t,c]],1) for c in [0,1]],0)
  writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
  if t in fixed:
   y=fixed.index(t)*538;draw.text((2,y),f't={t}: Original | Rerender Clean | Rerender Hue; front / wrist',fill='white');sheet.paste(Image.fromarray(panel),(0,y+26))
 writer.release()
 for cap in caps:cap.release()
 sheet.save(paired/'fixed_original_clean_hue.png')
 dump(paired/'export_audit.json',dict(frames=n,state_restores=n,kinematic_forwards=n,simulation_steps=0,render_calls=0,
  metadata_changes=metadata_changes,actions_language_unchanged=True,proprio_and_camera='derived synchronized FK and per-frame actual camera; source metadata preserved untouched',
  placeholder_masks='copied official placeholders, never used for background selection',
  target_qpos_tail='official legacy achieved-qpos field repeats final qpos, not a future RGB/AE target; valid AE horizon excludes terminal extrapolation',
  stream_storage='lossless PNG frame directories supported by official datalib; comparison MP4 for viewing only',
  frames_per_camera=n,RGB_frames_two_appearances_two_views=n*4,AE_forwards=0,DINO_forwards=0,policy_forwards=0))
 print(suite,'exported',n,flush=True)
if __name__=='__main__':main()
