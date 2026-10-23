"""Contact sheets of the sole success-discordant development state."""
import argparse,json,os
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ['TMPDIR'])/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output-name',default='video_review');a=p.parse_args();r=json.loads((a.run/'per_trial_results.json').read_text());dest=a.run/a.output_name;dest.mkdir(exist_ok=False)
for c in ('clean','background'):
    ids=sorted(set(x['episode'] for x in r if x['appearance']==c));discord=[]
    for ep in ids:
        pair={x['arm']:x for x in r if x['episode']==ep and x['appearance']==c}
        if pair['B4']['success']!=pair['B20']['success']:discord.append(ep)
    for ep in discord:
        fig,axes=plt.subplots(2,4,figsize=(14,4.5))
        for row,arm in enumerate(('B4','B20')):
            t=next(x for x in r if x['episode']==ep and x['appearance']==c and x['arm']==arm)
            with imageio.get_reader(t['video']) as reader:frames=[frame for frame in reader]
            indices=[0,min(4,len(frames)-1),min(8,len(frames)-1),len(frames)-1]
            for col,i in enumerate(indices):axes[row,col].imshow(frames[i]);axes[row,col].set_title(f'{arm}: observation at control step {i*8}');axes[row,col].axis('off')
        fig.suptitle(f'{ep} / {c}: recorded video observations, no rerun');fig.tight_layout();fig.savefig(dest/f'{ep}_{c}.png',dpi=150);plt.close(fig)
