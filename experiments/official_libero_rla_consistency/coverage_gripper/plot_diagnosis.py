"""Scientific plots and grouped metrics from saved frozen predictions only."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ['TMPDIR'])/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();out=a.out/'plots';out.mkdir()
    fixed=json.loads((a.out/'checkpoint_diagnosis/fixed_sample_manifest.json').read_text())['samples'];entries=json.loads((a.previous/'episode_manifest.json').read_text());steps=(0,45,135,270);cache={s:np.load(a.out/'checkpoint_diagnosis'/f'predictions_step{s:04}.npz') for s in steps};metrics=[];events=[]
    for s in steps:
        for split in ('train','validation'):
            for appearance in ('clean','background'):
                ix=[i for i,x in enumerate(fixed) if x['split']==split and x['appearance']==appearance];pred=cache[s]['prediction'][ix];target=cache[s]['target'][ix]
                for h in ('all',*range(8)):
                    x=pred if h=='all' else pred[:,h:h+1];y=target if h=='all' else target[:,h:h+1];pg=x[:,:,6];tg=y[:,:,6];c=tg==0;o=tg==1
                    metrics.append(dict(checkpoint=s,split=split,appearance=appearance,chunk_position=h,windows=len(ix),translation_L1=float(abs(x[:,:,:3]-y[:,:,:3]).mean()),rotation_L1=float(abs(x[:,:,3:6]-y[:,:,3:6]).mean()),gripper_L1=float(abs(pg-tg).mean()),close_recall=float((pg[c]<.5).mean()),open_recall=float((pg[o]>.5).mean()),accuracy=float((((pg<.5)&c)|((pg>.5)&o)).mean()),neutral_fraction=float((pg==.5).mean()),prediction_std=float(pg.std())))
    for e in entries:
        lab=np.load(a.out/'label_traces'/f"{e['episode']}.npz");starts=lab['close_events']
        for appearance in ('clean','background'):
            ix=[i for i,x in enumerate(fixed) if x['episode']==e['episode'] and x['appearance']==appearance]
            for event in starts:
                for s in steps:
                    pr=cache[s]['prediction'][ix];tg=cache[s]['target'][ix]
                    for t in range(max(0,int(event)-8),min(len(ix),int(event)+9)):
                        events.append(dict(episode=e['episode'],split=e['split'],appearance=appearance,checkpoint=s,event=int(event),current_timestep=t,relative_to_event=t-int(event),target_h0=float(tg[t,0,6]),prediction_h0=float(pr[t,0,6]),target_h4=float(tg[t,4,6]),prediction_h4=float(pr[t,4,6]),target_h7=float(tg[t,7,6]),prediction_h7=float(pr[t,7,6])))
    for name,rows in [('checkpoint_summary.csv',metrics),('gripper_event_predictions.csv',events)]:
        with (a.out/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Fixed first training demo and fixed development demo, not selected by fit.
    for split in ('train','validation'):
        e=next(e for e in entries if e['split']==split);lab=np.load(a.out/'label_traces'/f"{e['episode']}.npz");fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
        for row,appearance in enumerate(('clean','background')):
            ix=[i for i,x in enumerate(fixed) if x['episode']==e['episode'] and x['appearance']==appearance];t=np.arange(len(ix));target=cache[270]['target'][ix]
            ax=axes[row,0];ax.plot(t,target[:,0,6],color='black',lw=2,label='label h0: 1=open, 0=close')
            for s in steps:ax.plot(t,cache[s]['prediction'][ix,0,6],label=f'step{s}',alpha=.85)
            ax=axes[row,1]
            for h,color in ((0,'tab:blue'),(4,'tab:orange'),(7,'tab:green')):
                ax.plot(t,cache[270]['prediction'][ix,h,6],color=color,label=f'prediction h{h}');ax.plot(t,target[:,h,6],color=color,ls=':',alpha=.7,label=f'label h{h}')
            for ax in axes[row]:
                ax.axhline(.5,color='gray',ls='--',lw=1,label='deployment threshold .5')
                for event in lab['close_events']:ax.axvline(event,color='purple',alpha=.25)
                ax.set_xlabel('Current recorded state timestep');ax.set_ylabel(f'{appearance}\ncontinuous gripper output');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
        axes[0,0].set_title('Checkpoint progression, current action');axes[0,1].set_title('Final checkpoint, chunk positions')
        fig.suptitle(f'{split}: {e["source_demo"]} — fixed demonstration observations');fig.tight_layout();fig.savefig(out/f'{split}_gripper_timeseries.png',dpi=150);fig.savefig(out/f'{split}_gripper_timeseries.pdf');plt.close(fig)
    current=[x for x in metrics if x['checkpoint']==270 and x['chunk_position']==0]
    print(json.dumps(current,indent=2))
if __name__=='__main__':main()
