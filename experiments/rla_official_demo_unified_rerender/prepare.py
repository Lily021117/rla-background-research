"""Freeze protocol and inspect local official HDF5 sources, without render/replay."""
from common import *
import h5py, argparse, xml.etree.ElementTree as ET

def freeze():
 p=RUN/'protocol.json';assert not p.exists()
 source=OLD/'delivery/source_mapping.csv';rows=list(csv.DictReader(open(source)))
 assert len(rows)==1693 and sum(r['split']=='train' for r in rows)==1653
 dump(p,dict(name='官方示范统一重渲染派生版',parent_commit='b581d02405f70e6b6f7992d80f44b0dadb0662b7',official_rla_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',libero_commit='8f1084e3132a39270c3a13ebe37270a43ece2a01',source_manifest=str(source),source_manifest_sha256=sha(source),train=1653,val=40,
  old_conclusion='PARTIAL; original-video correspondence failed; unchanged',
  source_join='Unique exact FULL filtered action array + exact task language/path + independently nearest complete robot joint trajectory among all demos in source HDF5; compare normalized-progress paths ONLY as identity evidence, never resample output states. Require runner-up distance at least twice winner and winner RMSE <=0.1rad. Ambiguity stays unresolved. Record all evidence and source hashes; not proof of byte-identical unpublished intermediate HDF5.',
  static_source='Official HDF5 states at original indices, before corresponding source actions. Source RGB/obs are post-action and NOT reused as synchronized observations.',
  pilots=[dict(suite=s,trajectory_id='000010') for s in SUITES],
  acceptance=dict(state_model_layout='exact 1+nq+nv; na=0; XML named joint order and finite values; monotonic source timestamps',restore_geometry_atol=1e-12,restore_rgb_max=0,paired_model='all compiled model arrays except tex_data exact bytes',paired_physics='same time/qpos/qvel and synchronized body/camera/site arrays',visible_weak_threshold_uint8=1.,original_video_threshold='NONE; descriptive comparison only'),
  camera=dict(resolution=[256,256],names=CAMS,orientation='native MuJoCo top-down render followed by horizontal mirror; negative fx; c2w per actual source model and frame'),
  timing='Keep filtered source indices and real source times. Official h is retained-frame count, future index t+h-1, t+h<N; actual elapsed seconds supplied separately, never assert fixed .05s across filtered gaps.',
  augmentation=dict(delta_h=.2,saturation='S + .35*(1-S)',V='unchanged before RGB quantization',background_whitelist=WHITELIST,shared_across='all frames and both cameras',texture_alias='fail closed if foreground shares texture; no hidden foreground recolor',lights_reflection_geometry='unchanged'),
  batch_gate='Four original-state pilot sequences pass self-restore, geometry/time and visual checks, then source-confirmed entries only; split conflicts withheld; resumable; minimum 100GiB remaining disk',
  visual_rule='7 equally spaced frames each pilot + full sequence automated checks, full videos retained; AI review explicitly distinguished from human review',
  learned_model_forwards=0,updates=0,backward=0,policy_rollout=0,source_action_replay=0))

def inventory():
 rows=list(csv.DictReader(open(OLD/'delivery/source_mapping.csv')))
 # Only local published-source directories, not Cosmos, derived-state caches or sealed data.
 roots=[OLD/'source_hdf5',ROOT/'data/libero_original_minimal',ROOT/'experiments/rla_new_task_baseline/run_20261028_02/downloads',ROOT/'experiments/rla_official_demo_unified_rerender/run_20261103_01/downloads',RUN/'downloads']
 candidates={}
 for root in roots:
  if root.exists():
   for p in root.rglob('*.hdf5'):candidates.setdefault((p.parent.name,p.name),[]).append(p)
 filelist=[];sources={};details=[];rejected=[]
 planpath=RUN/'download_plan.json';expected={x['filename']:x for x in json.loads(planpath.read_text())['files']} if planpath.exists() else {}
 for key,paths in sorted(candidates.items()):
  if key[0] not in SUITES:continue
  hashes={str(p):sha(p) for p in paths}
  entry=expected.get('/'.join(key))
  if entry:
   good=[]
   for p in paths:
    if hashes[str(p)]==entry['sha256'] and p.stat().st_size==entry['bytes']:good.append(p)
    else:rejected.append(dict(path=str(p),expected_sha256=entry['sha256'],actual_sha256=hashes[str(p)],expected_bytes=entry['bytes'],actual_bytes=p.stat().st_size,status='QUARANTINED_NOT_READ_AS_HDF5'))
   paths=good;hashes={str(p):hashes[str(p)] for p in paths}
   if not paths:continue
  assert len(set(hashes.values()))==1,('Different local source copies',key,hashes)
  p=paths[0];sources[key]=p
  filelist.append(dict(suite=key[0],filename=key[1],path=str(p),sha256=hashes[str(p)],bytes=p.stat().st_size,copies=list(hashes),source='existing official published HDF5; previous download provenance retained',revision='f13aa24a3da8c43c7225569f28c562979fa0e35a' if key[0]=='libero_10' else '97773100c1474cd0d686ebd173cc0e4fd5442466'))
 dump(RUN/'source_files.json',filelist)
 dump(RUN/'rejected_source_files.json',rejected)
 cache={}
 for row in rows:
  key=(row['suite'],Path(row['rlds_episode_file_path']).name);p=sources.get(key)
  item=dict(suite=row['suite'],trajectory_id=row['trajectory_id'],split=row['split'],task=row['task'],frames=int(row['frames']),rlds_index=row['rlds_enumeration_index'],rlds_mapping_status=row['rlds_mapping_status'],rlds_path=row['rlds_episode_file_path'],source_file=str(p) if p else '',source_sha256='',demo='',source_status='MISSING_OFFICIAL_HDF5',filtered_indices='',evidence='',split_conflict=False,render_status='NOT_RUN')
  if p:
   if key not in cache:
    with h5py.File(p) as f:
     lang=json.loads(f['data'].attrs['problem_info'])['language_instruction'];demos=[]
     for name,d in f['data'].items():
      actions=d['actions'][:];ids=filtered(actions);j=d['obs/joint_states'][:][ids]
      demos.append(dict(name=name,ids=ids,actions=actions[ids].astype(np.float32),joints=j,xml=str(d.attrs['model_file']),raw_len=len(actions)))
    cache[key]=(lang,demos)
   lang,demos=cache[key];n=np.load(OLD/'rlds_mapping_v2'/row['suite']/f"numeric_{row['trajectory_id']}.npz")
   matches=[d for d in demos if d['actions'].shape==n['action'].shape and np.array_equal(d['actions'],n['action'])]
   distances=[]
   for d in demos:
    ix=np.rint(np.linspace(0,len(d['joints'])-1,len(n['joints']))).astype(int)
    distances.append((float(np.sqrt(np.mean((d['joints'][ix]-n['joints'])**2))),d['name']))
   distances.sort();margin=distances[1][0]/max(distances[0][0],1e-15)
   language_ok=lang==row['task'];ok=len(matches)==1 and language_ok and distances[0][1]==matches[0]['name'] and margin>=2 and distances[0][0]<=.1
   evidence=dict(suite=row['suite'],trajectory_id=row['trajectory_id'],full_action_candidates=[d['name'] for d in matches],language_exact=language_ok,joint_distances=distances,nearest_ratio=margin,identity_only_progress_resampling=True,formal_state_resampling=False,decision='MULTI_EVIDENCE_JOIN' if ok else 'UNRESOLVED',source_frame_phase='source joint obs post-action; RLDS regenerated obs; distances support identity not exact state equality')
   ep=RUN/'mapping_evidence'/row['suite']/f"{row['trajectory_id']}.json";dump(ep,evidence);item.update(source_sha256=next(x['sha256'] for x in filelist if x['path']==str(p)),source_status=evidence['decision'],evidence=str(ep))
   if len(matches)==1:
    d=matches[0];item.update(demo=d['name'],filtered_indices=json.dumps(d['ids'].tolist()))
  details.append(item)
 groups={}
 for x in details:
  if x['source_status']=='MULTI_EVIDENCE_JOIN':groups.setdefault((x['source_sha256'],x['demo']),[]).append(x)
 conflicts=[]
 for key,g in groups.items():
  if len({x['split'] for x in g})>1:
   conflicts.append(dict(source_sha256=key[0],demo=key[1],rows=[{k:x[k] for k in ['suite','trajectory_id','split']} for x in g]))
   for x in g:x['split_conflict']=True;x['render_status']='WITHHELD_SPLIT_CONFLICT'
 dump(RUN/'split_conflicts.json',conflicts);csvout(RUN/'source_manifest.csv',details)
 summary=[]
 for s in SUITES:
  for split in ['train','val']:
   g=[x for x in details if x['suite']==s and x['split']==split]
   summary.append(dict(suite=s,split=split,requested=len(g),source_file_available=sum(bool(x['source_file']) for x in g),joined=sum(x['source_status']=='MULTI_EVIDENCE_JOIN' for x in g),unresolved=sum(x['source_status']=='UNRESOLVED' for x in g),missing=sum(x['source_status']=='MISSING_OFFICIAL_HDF5' for x in g),split_conflicts=sum(x['split_conflict'] for x in g)))
 csvout(RUN/'source_coverage.csv',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
 if a.freeze:freeze()
 inventory()
