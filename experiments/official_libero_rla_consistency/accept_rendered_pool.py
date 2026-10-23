"""Freeze the completed render manifest after all per-episode invariants pass."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.run/'episode_manifest.json').read_text()); accepted=[]
for e in rows:
    root=a.run/'episodes'/e['episode']; p=json.loads((root/'trajectory_provenance.json').read_text()); camera=json.loads((root/'camera_geometry_audit.json').read_text())
    assert p['frames']==e['frames'] and p['split']==e['split'] and camera['status']=='PASS_RASTER_PROJECTION'
    metrics=list(csv.DictReader((root/'regional_sequence_metrics.csv').open()))
    assert len(metrics)==e['frames']*2
    assert all(x['seg_equal']=='True' and x['depth_equal']=='True' for x in metrics)
    zero=[x for x in metrics if int(x['changed_pixels'])==0]
    assert not zero, 'Background visibility gap: retain files and stop admission'
    files=[]
    for t in range(e['frames']):
        path=root/'frames'/f'{t:04}.npz'
        with np.load(path) as d:
            for k in ('clean','background'):assert d[k].shape==(2,3,224,224) and d[k].dtype==np.uint8
            assert all(np.isfinite(d[k]).all() for k in d.files)
        files.append(dict(timestep=t,path=str(path.relative_to(a.run)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    accepted.append(dict(**{k:v for k,v in e.items() if k!='render_status'},render_status='PASS',files=files))
with (a.run/'accepted_dataset_manifest.json').open('x') as f:json.dump(dict(status='PASS_RENDERED_POOL',episodes=accepted,frames=sum(e['frames'] for e in rows),camera_pairs=sum(e['frames'] for e in rows)*2,rgb_images=sum(e['frames'] for e in rows)*4,physical_chunks=sum(e['complete_h8_chunks'] for e in rows),heldout_used=False),f,indent=2)
print('All five admitted episodes passed rendering and file checks')
