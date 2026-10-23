"""Learning curves from fixed logs only; no checkpoint selection."""
import argparse,csv,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ['TMPDIR'])/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();dest=a.run/'plots';dest.mkdir(exist_ok=False);fig,ax=plt.subplots(1,2,figsize=(10,4))
for arm in ('B4','B20'):
    rows=list(csv.DictReader((a.run/arm/'formal_B/training_metrics.csv').open()));loss=np.array([float(r['loss']) for r in rows]);ax[0].plot(np.arange(20,len(loss)+1),np.convolve(loss,np.ones(20)/20,'valid'),label=arm)
    rows=list(csv.DictReader((a.run/'validation_metrics.csv').open()))
    for c,style in [('clean','-'),('background','--')]:
        steps=[0,100,300,600];values=[]
        for s in steps:
            rs=[r for r in rows if r['arm']==arm and r['set']=='development' and int(r['step'])==s and r['appearance']==c]
            values.append(np.average([float(r['action_L1']) for r in rs],weights=[int(r['windows']) for r in rs]))
        ax[1].plot(steps,values,style,marker='o',label=arm+' '+c)
ax[0].set(title='Training BC loss (20-update moving mean)',xlabel='Successful updates',ylabel='Normalized action L1');ax[1].set(title='Fixed five-episode development',xlabel='Successful updates',ylabel='Normalized action L1')
for axis in ax:axis.legend();axis.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(dest/'training_and_development.png',dpi=160);fig.savefig(dest/'training_and_development.pdf');plt.close(fig)
