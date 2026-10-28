"""Decode only the eight preselected video pairs; never render simulations."""
import json,sys
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);out=p/'video_review';out.mkdir(exist_ok=True);index=[]
for r in json.loads((p/'video_review_selection.json').read_text()):
    sheet=Image.new('RGB',(448*6,250*2),'white');draw=ImageDraw.Draw(sheet)
    for row,label in enumerate(['C','shared']):
        reader=imageio.get_reader(r[label+'_video']);frames=[frame for frame in reader];reader.close()
        ids=sorted(set(round(i*(len(frames)-1)/5) for i in range(6)))
        index.append(dict(pair_id=r['pair_id'],model=label,video=r[label+'_video'],frames=len(frames),review_frames=ids,control_step_before_action=[8*i for i in ids],note='fixed six uniformly spaced video frames; no precise contact inference'))
        for col,i in enumerate(ids):
            sheet.paste(Image.fromarray(frames[i]).resize((448,224)),(col*448,row*250));draw.text((col*448,row*250+224),f'{label} pre-action step {8*i}',fill='black')
    target=out/(r['pair_id']+'.jpg');assert not target.exists();sheet.save(target)
with (p/'video_review_index.json').open('x') as f:json.dump(index,f,indent=2)
