"""One logged correction pass after fixed v1 inspection; no image/HSV changes."""
import json,argparse
from pathlib import Path
from audit import write_json,sha
MID={
'libero_spatial_000010':(50,[[40,50],[150,70],[70,150]],[[200,200],[40,208],[213,208],[128,240],[245,15]]),
'libero_spatial_000011':(42,[[25,70],[25,170],[240,70]],[[135,60],[240,85],[150,210],[35,208],[213,208],[128,240]]),
'libero_object_000010':(71,[[80,60],[160,80],[210,110]],[[135,185],[245,145],[160,30],[35,208],[213,208],[128,240]]),
'libero_object_000011':(82,[[80,50],[160,50],[200,100]],[[120,185],[35,208],[213,208],[128,240]]),
'libero_goal_000010':(57,[[60,70],[160,140],[40,110]],[[230,40],[90,160],[160,40],[230,160],[35,208],[213,208],[128,240]]),
'libero_goal_000011':(40,[[20,30],[20,160],[245,200]],[[160,90],[35,208],[213,208],[128,240]]),
'libero_10_000010':(191,[[40,40],[180,50],[40,120]],[[180,180],[100,180],[35,208],[213,208],[128,240]]),
'libero_10_000011':(109,[[60,50],[220,70],[170,130]],[[100,140],[70,200],[35,208],[213,208],[128,240]])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 d=json.loads((a.run/'prompts_v1.json').read_text());log=[]
 for name,x in d.items():
  k=x['objects'][0]['keyframes'][0]
  if name.endswith('wrist_camera'):
   k['box']=[0,0,255,214]
   frame,pos,neg=MID[name.removesuffix('_wrist_camera')]
   x['objects'][0]['keyframes'].append(dict(frame=frame,positive=pos,negative=neg,box=[0,0,255,214],
    basis='Assistant inspection of pre-frozen middle original frame; tabletop only, object/robot exclusions; correction for loss of background tracking'))
  else:
   ymin=20 if name.startswith('libero_object') else 90
   if name.startswith('libero_10_000011'):ymin=65
   k['box']=[0,ymin,255,255]
  log.append(dict(video=name,reason='v1 first-frame fragmented/empty mask or later empty propagation; add semantic table bounding extent and wrist middle keyframe',
   original_prompt_sha256=sha(a.run/'prompts_v1.json'),evidence=f'v1/{name}/overlay_review.png',
   changes=x['objects'][0]['keyframes']))
 write_json(a.run/'prompts_v2.json',d);write_json(a.run/'correction_log_v2.json',log)
if __name__=='__main__':main()
