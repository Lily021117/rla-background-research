"""Inspect all 741 already saved states first; no physics integration."""
from common import *
import time
def main():
 for suite in SUITES:
  out=RUN/'offline'/suite;out.mkdir(parents=True,exist_ok=False);src=OLD/'pilot_replay'/suite
  a=np.load(src/'candidate_replay.npz');m=mujoco.MjModel.from_xml_path(str(src/'runtime_model.xml'));d=mujoco.MjData(m)
  r=mujoco.Renderer(m,256,256) if suite=='libero_object' else None;rows=[];start=time.time()
  try:
   for i,s in enumerate(a['states']):
    legacy(m,d,s)
    cam=np.r_[d.cam_xpos.ravel(),d.cam_xmat.ravel()]
    row=dict(frame=i,time=float(d.time),qpos_exact=bool(np.array_equal(d.qpos,s[1:1+m.nq])),qvel_exact=bool(np.array_equal(d.qvel,s[1+m.nq:])),
     body_delta_m=float(np.max(abs(d.xpos-a['body_positions'][i]))),camera_delta=float(np.max(abs(cam-a['camera_world'][i]))))
    if r:row.update(metrics(render(r,d),a['rgb'][i],'sync_vs_old_direct'))
    rows.append(row)
   csvout(out/'frames.csv',rows)
   dump(out/'result.json',dict(frames=len(rows),state_restores=len(rows),kinematic_forwards=len(rows),render_calls=2*len(rows) if r else 0,physics_steps=0,
    max_body_delta_m=max(x['body_delta_m'] for x in rows),max_camera_delta=max(x['camera_delta'] for x in rows),
    legacy_state_cannot_reproduce_unsynchronized_geometry=any(x['body_delta_m']>1e-12 for x in rows),seconds=time.time()-start))
   print(suite,rows[-1],flush=True)
  finally:
   if r:r.close()
if __name__=='__main__':main()
