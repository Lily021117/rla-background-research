import argparse,collections
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();groups=collections.defaultdict(list)
for x in csv.DictReader((a.out/'stage_timeline_local_only.csv').open()):groups[x['model'],x['seed'],x['appearance']].append(x)
rows=[]
for (model,seed,appearance),allrows in groups.items():
 rs=[x for x in allrows if x['relative_to_lift'] and int(x['relative_to_lift'])>=0];xy=[x for x in rs if x['xy_condition']=='True'];contact=[x for x in rs if x['contact_condition']=='True'];opened=[x for x in rs if x['close_command']=='False'];lastxy=int(xy[-1]['step']) if xy else None;firstcontact=int(contact[0]['step']) if contact else None;firstopen=int(opened[0]['step']) if opened else None
 rows.append(dict(model=model,seed=seed,appearance=appearance,first_xy_step=int(xy[0]['step']) if xy else None,last_xy_step=lastxy,first_plate_contact_step=firstcontact,first_open_step=firstopen,contact_and_xy_steps=sum(x['xy_condition']=='True' and x['contact_condition']=='True' for x in rs),min_bowl_plate_height_during_xy_m=min(float(x['vertical_bowl_plate_m']) for x in xy) if xy else None,contact_minus_last_xy=None if firstcontact is None or lastxy is None else firstcontact-lastxy,open_minus_last_xy=None if firstopen is None or lastxy is None else firstopen-lastxy))
csvsave(a.out/'condition_sequence.csv',rows)
# Uniform first frozen state, event-aligned curves; no outcome-based selection.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,1,figsize=(9,9),sharex=True)
for arm in ('C','D-center'):
 rs=[x for x in groups[arm,str(states()[0]['seed']),'clean'] if x['relative_to_lift'] and int(x['relative_to_lift'])>=0];t=[int(x['relative_to_lift']) for x in rs]
 for ax,key in zip(axes,['xy_distance_m','vertical_bowl_plate_m','contact_condition','close_command']):ax.plot(t,[float(x[key]) if key.endswith('_m') else int(x[key]=='True') for x in rs],label=arm);ax.set_ylabel(key);ax.legend()
axes[0].axhline(.03,color='gray',ls='--');axes[-1].set_xlabel('control steps after first lift');fig.tight_layout();fig.savefig(a.out/'event_aligned_first_state.png');plt.close(fig)
