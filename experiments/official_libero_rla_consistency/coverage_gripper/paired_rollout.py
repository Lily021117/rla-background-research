"""State-paired frozen B4/B20 development execution, no retries."""
import json,csv,hashlib
from pathlib import Path
from types import SimpleNamespace
from single_task_rollout import execute as rollout

def execute(g):
    base=g['a'].run.parent;protocol=json.loads((base/'evaluation_protocol.json').read_text());results=[]
    trainable={k:{n for n,p in m.named_parameters() if p.requires_grad} for k,m in g['mods'].items()};geometry_cache={}
    for i,trial in enumerate(protocol['trials']):
        arm=trial['arm'];dest=base/'closed_loop'/f'{i:02}_{arm}_{trial["source_demo"]}_{trial["appearance"]}';dest.mkdir(parents=True,exist_ok=False)
        def save(n,x):
            with (dest/n).open('x') as f:json.dump(x,f,indent=2)
        local=dict(g);local.update(a=SimpleNamespace(run=base/arm,reference=g['a'].reference),o=dest,save=save,trial_override=dict(trial,split='validation'),trial_conditions=[trial['appearance']],paired_geometry_cache=geometry_cache)
        # Each call strictly reloads the matching final model; no model RNG is shared across trials.
        for k,m in g['mods'].items():
            for n,p in m.named_parameters():p.requires_grad_(n in trainable[k])
        rollout(local)
        row=json.loads((dest/trial['appearance']/'result.json').read_text());row.update(arm=arm,trial=i,source_demo=trial['source_demo'],video=str(dest/trial['appearance']/'video.mp4'),trajectory=str(dest/trial['appearance']/'trajectory.npz'));results.append(row)
        with (base/'per_trial_results.json').open('w') as f:json.dump(results,f,indent=2)
        assert row['status']=='COMPLETE','invalid trial: frozen stop rule, no automatic retry'
    keys=sorted(set().union(*(r.keys() for r in results)))
    with (base/'per_trial_results.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(results)
