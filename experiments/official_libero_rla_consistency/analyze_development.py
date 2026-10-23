"""Offline analysis of the two existing trials; no new inference or rollout."""
import argparse,json,csv
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'development/offline_analysis';out.mkdir()
rows=json.loads((a.run/'episode_manifest.json').read_text());train_gripper=[]
for e in rows:
    if e['split']=='train':
        x=np.load(a.run/'episodes'/e['episode']/'trajectory_labels.npz')['actions']
        train_gripper.extend(x[:,6].tolist())
result=dict(train_raw_gripper_counts={str(v):int(np.sum(np.array(train_gripper)==v)) for v in np.unique(train_gripper)},sign_roundtrip={'raw_open_-1':'training1 -> deployment-1','raw_close_+1':'training0 -> deployment+1','prediction0.5':'official sign(2*p-1) = 0, not a close command'},trials={})
for condition in ('clean','background'):
    path=a.run/'development'/condition
    with np.load(path/'trajectory.npz') as x:
        g=x['predicted_chunks'][:,:,6].ravel();act=x['executed_actions'][:,6]
        result['trials'][condition]=dict(predicted_gripper_min=float(g.min()),predicted_gripper_max=float(g.max()),predicted_gripper_mean=float(g.mean()),fraction_exactly_half=float(np.mean(g==.5)),executed_open=int(np.sum(act==-1)),executed_close=int(np.sum(act==1)),executed_neutral=int(np.sum(act==0)))
    video=imageio.get_reader(path/'video.mp4');frames=[frame for frame in video];video.close();indices=sorted(set(min(i,len(frames)-1) for i in (0,6,12,18,24,len(frames)-1)))
    canvas=Image.new('RGB',(448*3,248*2),'white');draw=ImageDraw.Draw(canvas)
    for j,i in enumerate(indices):
        canvas.paste(Image.fromarray(frames[i]),((j%3)*448,(j//3)*248));draw.text(((j%3)*448,(j//3)*248+225),f'{condition}: video frame {i}, approximately control step {i*8}',fill='black')
    canvas.save(out/(condition+'_contact_sheet.png'))
with (out/'gripper_diagnosis.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
