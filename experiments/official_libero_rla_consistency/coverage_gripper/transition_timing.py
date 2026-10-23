"""First-close event timing by chunk slot, from saved predictions only."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ['TMPDIR'])/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
fixed=json.loads((a.out/'checkpoint_diagnosis/fixed_sample_manifest.json').read_text())['samples'];events=list(csv.DictReader((a.out/'gripper_events.csv').open()));rows=[]
for step in (0,45,135,270):
    data=np.load(a.out/'checkpoint_diagnosis'/f'predictions_step{step:04}.npz');pred=data['prediction'];target=data['target']
    for event in events:
        e=int(event['timestep'])
        for appearance in ('clean','background'):
            indices=[i for i,x in enumerate(fixed) if x['episode']==event['episode'] and x['appearance']==appearance];pr=pred[indices];gt=target[indices]
            for h in range(8):
                ideal=e-h;assert gt[ideal,h,6]==0
                ts=np.arange(max(0,e-8),min(len(pr),e+9));closed=ts[pr[ts,h,6]<.5];onset=int(closed[0]) if len(closed) else None
                rows.append(dict(checkpoint=step,episode=event['episode'],split=event['split'],appearance=appearance,event=e,chunk_position=h,ideal_current_timestep=ideal,prediction_at_event_action=float(pr[ideal,h,6]),correct_close_at_event_action=bool(pr[ideal,h,6]<.5),predicted_close_onset_current_timestep=onset,onset_error_vs_ideal_steps=onset-ideal if onset is not None else None))
with (a.out/'close_transition_timing.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for split in ('train','validation'):
    for h in range(8):
        r=[x for x in rows if x['checkpoint']==270 and x['split']==split and x['chunk_position']==h];offsets=[x['onset_error_vs_ideal_steps'] for x in r if x['onset_error_vs_ideal_steps'] is not None]
        summary.append(dict(split=split,chunk_position=h,event_appearance_pairs=len(r),correct=sum(x['correct_close_at_event_action'] for x in r),event_close_recall=float(np.mean([x['correct_close_at_event_action'] for x in r])),mean_onset_error_steps=float(np.mean(offsets)) if offsets else None,mean_onset_error_seconds=float(np.mean(offsets))*.05 if offsets else None))
with (a.out/'close_transition_summary.json').open('x') as f:json.dump(summary,f,indent=2)
final=np.load(a.out/'checkpoint_diagnosis/predictions_step0270.npz')['prediction'];fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,split in zip(axes,('train','validation')):
    ev=next(x for x in events if x['split']==split);e=int(ev['timestep']);ix=[i for i,x in enumerate(fixed) if x['episode']==ev['episode'] and x['appearance']=='clean'];ts=np.arange(e-8,e+5);mat=final[ix][ts,:,6].T
    im=ax.imshow(mat,aspect='auto',origin='lower',extent=[-8.5,4.5,-.5,7.5],vmin=0,vmax=1,cmap='coolwarm');ax.plot(-np.arange(8),np.arange(8),'k--',label='label first-close boundary')
    ax.set_xlabel('Current timestep relative to first close event');ax.set_ylabel('Action chunk position');ax.set_title(f'{split}: {ev["episode"]}\nstep270, clean');ax.legend(fontsize=8)
fig.colorbar(im,ax=axes.ravel().tolist(),label='Gripper prediction: 1=open, 0=close',shrink=.8);fig.savefig(a.out/'plots/close_event_chunk_timing.png',dpi=150,bbox_inches='tight');plt.close(fig)
print(json.dumps([x for x in summary if x['chunk_position'] in (0,7)],indent=2))
