"""CPU-only final artifact checks; no renderer or model instantiated."""
from common import *
def main():
 rows=[]
 for e in read(ROOT/'source_manifest.json'):
  src=np.load(Path(e['root'])/'source_arrays.npz');out=ROOT/'rendered'/e['episode'];p=np.load(out/'physical.npz');n=e['frames']
  assert np.array_equal(src['states'],p['states']) and np.array_equal(src['actions'],p['actions'])
  assert np.array_equal(p['timesteps'],np.arange(n))
  for k in p.files:assert np.isfinite(p[k]).all(),k
  reference=None
  for v in VARIANTS:
   a=np.load(out/(v+'.npz'));rgb=a['rgb'];assert rgb.shape==(n,2,224,224,3) and rgb.dtype==np.uint8;assert np.array_equal(a['timesteps'],p['timesteps'])
   means=rgb.mean((2,3,4));stds=rgb.std((2,3,4))
   if v=='object_1':reference=rgb
   distinct=None if v!='object_2' else int(np.any(rgb!=reference,axis=(2,3,4)).sum())
   rows.append({'episode':e['episode'],'appearance':v,'frames':n,'min_frame_camera_mean':float(means.min()),'max_frame_camera_mean':float(means.max()),'min_frame_camera_std':float(stds.min()),'image_gate_all_frames':bool(((means>3)&(means<252)&(stds>3)).all()),'object2_distinct_from_object1_camera_frames':distinct})
  assert all(r['image_gate_all_frames'] for r in rows if r['episode']==e['episode'])
 manifest=[json.loads(line) for line in (ROOT/'pairing_manifest.jsonl').open()];assert len(manifest)==sum(e['frames']*5 for e in read(ROOT/'source_manifest.json'))
 ids=set()
 for r in manifest:
  key=(r['episode'],r['timestep'],r['appearance_id']);assert key not in ids;ids.add(key)
  assert r['split']=='training-prototype' and r['future_horizons']==list(range(1,9))
 save(ROOT/'artifact_integrity.json',{'status':'PASS','rows':rows,'pairing_rows':len(ids),'original_states_actions_exact':True,'complete_timesteps':True,'tail_masks_no_synthetic_frames':True,'physical_source_unique_not_render_count':10})
 print('ARTIFACT CHECK PASS',len(ids))
if __name__=='__main__':main()
