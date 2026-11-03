"""Read-only pixel/metadata check on final saved pair, no render or model."""
import json,csv
import numpy as np,cv2,h5py
from prepare import RUN,ROOT,dump,csvout,sha
def main():
 p=RUN/'paired_rendered/libero_object';clean=np.load(p/'clean.npz')['rgb'];bg=np.load(p/'background.npz')['rgb'];rows=[]
 assert clean.shape==bg.shape==(143,2,256,256,3)
 for ci,name in enumerate(['agentview_camera','wrist_camera']):
  cap=cv2.VideoCapture(str(ROOT/'data/libero_converted/libero_object/traj_000010'/f'{name}_rgb.mp4'))
  for t in range(len(clean)):
   ok,im=cap.read();assert ok;orig=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);diff=np.abs(clean[t,ci].astype('int16')-orig.astype('int16'))
   rows.append(dict(camera=name,frame=t,mae=float(diff.mean()),p99=float(np.quantile(diff,.99))))
  assert not cap.read()[0];cap.release()
 csvout(p/'final_clean_original_pixel_check.csv',rows)
 with h5py.File(p/'official_loader/clean/libero_object/traj_000010/metadata.h5') as c,h5py.File(p/'official_loader/background/libero_object/traj_000010/metadata.h5') as b:
  assert set(c)==set(b)
  assert all(np.array_equal(c[k][:],b[k][:]) for k in c)
  assert dict(c.attrs)==dict(b.attrs)
 stats=json.loads((p/'loader_audit.json').read_text());assert stats['status']=='PASS'
 passed=all(r['mae']<=8 and r['p99']<=40 for r in rows)
 dump(p/'final_admission.json',dict(status='PASS_RECONSTRUCTED_PAIR' if passed else 'BLOCKED_FINAL_CLEAN_PIXELS',
  full_frame_original_vs_final_clean_check=passed,all_metadata_equal_between_appearances=True,paired_RGB_sha256={f:sha(p/f) for f in ['clean.npz','background.npz']},
  frames=143,pixel_rows=len(rows),full_original_scene_state_identity='not directly available; reconstructed trajectory passes frozen tolerance, not bitwise identity',
  foreground_pixel_equality_required=False,reason='legitimate reflection/shadow response allowed; foreground material references and physics unchanged',
  visuals_reviewed='AI fixed seven full-resolution triplets; Object original/replay all-frame contact sheets reviewed separately',
  known_limitation='walls only; wrist RGB effect zero. No source video overwritten.',simulation_steps=0,render_calls=0,model_forwards=0))
 assert passed
 print('PASS final pair original-correspondence + metadata',flush=True)
if __name__=='__main__':main()
