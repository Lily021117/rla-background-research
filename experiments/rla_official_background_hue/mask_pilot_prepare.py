"""Freeze selection before viewing; decode original videos for prompts/review."""
import argparse,csv,json,hashlib,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from audit import sha,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 rows=list(csv.DictReader(open(a.manifest))); chosen=[]
 for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
  seen=set()
  for r in sorted((r for r in rows if r['suite']==suite),key=lambda r:int(r['trajectory_id'])):
   if r['task'] in seen or r['admission']!='PASS_HEADERS':continue
   chosen.append(r);seen.add(r['task'])
   if len(seen)==2:break
 assert len(chosen)==8
 videos=[]
 for r in chosen:
  for cam in ['agentview_camera','wrist_camera']:
   path=Path(r['source'])/f'{cam}_rgb.mp4';n=int(r['frames'])
   videos.append(dict(id=f"{r['suite']}_{r['trajectory_id']}_{cam}",suite=r['suite'],trajectory_id=r['trajectory_id'],
    task=r['task'],camera=cam,path=str(path),sha256=sha(path),frames=n,
    fixed_review_frames=sorted(set(np.linspace(0,n-1,7,dtype=int).tolist()+[0,n//2,n-1]))))
 protocol=dict(selection_rule='Per suite numeric train trajectory order: accept first trajectory then first different task',
  source_manifest_sha256=sha(a.manifest),trajectories=chosen,videos=videos,
  approved_background=['table','wall'],foreground_exclusions=['robot','gripper','task objects','drawer','cabinet','doors','operation parts'],
  mask_rule=dict(logit_threshold=0,erosion_pixels_native=3,erosion_kernel='7x7 square',resize='nearest to224 then3x3 erosion',
   uncertain='exclude',area_empty_threshold=.001,area_excessive_threshold=.95,frame_area_jump_threshold=.15),
  augmentation=dict(delta_h_range=[-.25,.25],seed=20261102),
  review='fixed frame contact sheets + abnormal frames; AI visual inspection is not human ground truth; full videos saved, not all human-reviewed',
  counts=dict(max_trajectories=8,videos=16,AE_updates=0,policy_updates=0,DINO_updates=0,backward=0,rollouts=0))
 write_json(a.output/'selection_and_protocol.json',protocol)
 # Selection is persisted before opening a single frame.
 (a.output/'original_review').mkdir()
 for v in videos:
  cap=cv2.VideoCapture(v['path']); sheet=Image.new('RGB',(4*256,2*282),'#222222');dr=ImageDraw.Draw(sheet)
  for i,t in enumerate(v['fixed_review_frames']):
   cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=cap.read();assert ok
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert rgb.shape[:2]==(256,256)
   x=(i%4)*256;y=(i//4)*282;sheet.paste(Image.fromarray(rgb),(x,y+26));dr.text((x+3,y+3),f't={t} {v["camera"]}',fill='white')
  cap.release();sheet.save(a.output/'original_review'/f'{v["id"]}.png')
 print(json.dumps([{'id':v['id'],'frames':v['frames'],'task':v['task']} for v in videos],indent=2))

if __name__=='__main__':main()
