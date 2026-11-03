"""Uniform review of all 1693 entries. No threshold tuning or trajectory replay.

v2 corrects the interpretation of v1 diagnostics, NOT its numerical thresholds.
Joint-path agreement is corroboration, not a necessary condition for true ancestry.
Failing it is insufficient independent evidence, NOT proof of a different source.
No additional entries are admitted just because their actions uniquely match.
"""
from common import *
from complete_sources import PREVIOUS
import h5py, xml.etree.ElementTree as ET, shutil

def main():
 rows=list(csv.DictReader(open(RUN/'source_manifest.csv')))
 prior={(x['suite'],x['trajectory_id']):x for x in csv.DictReader(open(PREVIOUS/'source_manifest.csv'))}
 assert len(rows)==1693 and sum(x['split']=='train' for x in rows)==1653
 audits=[];phases=[];diff=[]
 for row in rows:
  old=prior[row['suite'],row['trajectory_id']]
  assert row['split']==old['split'] and row['task']==old['task'] and row['rlds_index']==old['rlds_index']
  a=dict(suite=row['suite'],trajectory_id=row['trajectory_id'],split=row['split'],task=row['task'],old_status=old['source_status'],new_status=row['source_status'],candidate_demo=row['demo'],full_action_candidate_count='',language_exact='',task_path_exact='',joint_nearest='',joint_second='',joint_nearest_agrees='',nearest_rmse='',second_rmse='',nearest_ratio='',candidate_rmse='',old_numeric_failed_conditions='',source_initial_state_sha256='',rlds_first_joint_sha256='',raw_initial_vs_regenerated_first_joint_rmse='',initial_phase_comparable=False,decision_reason='SOURCE_FILE_MISSING_OR_UNVERIFIED',rule_version='v2_interpretation_same_conservative_admission')
  if row['source_file']:
   e=json.loads(Path(row['evidence']).read_text());d=e['joint_distances'];matches=e['full_action_candidates'];path_ok=Path(row['source_file']).name==Path(row['rlds_path']).name and Path(row['source_file']).parent.name==row['suite']
   failed=[]
   if len(matches)!=1:failed.append('FULL_ACTION_CANDIDATES_NOT_UNIQUE')
   if not e['language_exact']:failed.append('LANGUAGE_MISMATCH')
   if not path_ok:failed.append('TASK_SOURCE_PATH_MISMATCH')
   if row['demo'] and d[0][1]!=row['demo']:failed.append('JOINT_NEAREST_NOT_ACTION_CANDIDATE')
   if e['nearest_ratio']<2:failed.append('JOINT_RATIO_LT_2')
   if d[0][0]>.1:failed.append('JOINT_NEAREST_RMSE_GT_0_1')
   a.update(full_action_candidate_count=len(matches),language_exact=e['language_exact'],task_path_exact=path_ok,joint_nearest=d[0][1],joint_second=d[1][1],joint_nearest_agrees=d[0][1]==row['demo'],nearest_rmse=d[0][0],second_rmse=d[1][0],nearest_ratio=e['nearest_ratio'],candidate_rmse=next((v for v,n in d if n==row['demo']),''),old_numeric_failed_conditions=';'.join(failed))
   if row['demo']:
    with h5py.File(row['source_file']) as f:
     source=f['data'][row['demo']];raw=source['states'][:];actions=source['actions'][:];post=source['obs/joint_states'][:];ids=filtered(actions)
     xml=ET.fromstring(str(source.attrs['model_file']));joints=[n for n in xml.find('worldbody').iter() if n.tag in ['joint','freejoint']]
     assert [n.get('name') for n in joints[:7]]==['robot0_joint'+str(i) for i in range(1,8)]
     assert all(n.tag=='joint' and n.get('type','hinge')=='hinge' for n in joints[:7])
     pre=raw[:,1:8]
     numeric=np.load(OLD/'rlds_mapping_v2'/row['suite']/f"numeric_{row['trajectory_id']}.npz")
     assert np.array_equal(actions[ids].astype(np.float32),numeric['action'])
     a.update(source_initial_state_sha256=digest(raw[0]),rlds_first_joint_sha256=digest(numeric['joints'][0]),raw_initial_vs_regenerated_first_joint_rmse=float(np.sqrt(np.mean((pre[0]-numeric['joints'][0])**2))))
     phases.append(dict(suite=row['suite'],trajectory_id=row['trajectory_id'],source_demo=row['demo'],source_first_time_s=float(raw[0,0]),first_retained_source_index=int(ids[0]),source_postobs_vs_same_prestate_rmse=float(np.sqrt(np.mean((post-pre)**2))),source_postobs_vs_next_prestate_rmse=float(np.sqrt(np.mean((post[:-1]-pre[1:])**2))),filtered_postobs_vs_rlds_preobs_rmse=float(np.sqrt(np.mean((post[ids]-numeric['joints'])**2))),filtered_prestate_vs_rlds_preobs_rmse=float(np.sqrt(np.mean((pre[ids]-numeric['joints'])**2))),note='Different integrations; next-state comparison is diagnostic, not an invented exact alignment. Initial RLDS joints follow ten settling actions; no settling is replayed here.'))
   if not failed:
    a['decision_reason']='UNIQUE_FULL_ACTIONS_TASK_PATH_LANGUAGE_WITH_CORROBORATING_ROBOT_PATH_NOT_EXACT_INTERMEDIATE_STATE_PROOF'
   else:
    a['decision_reason']='INSUFFICIENT_INDEPENDENT_LINEAGE_EVIDENCE; JOINT_PATH_DIAGNOSTIC_NOT_IDENTITY_DISPROOF' if len(matches)==1 and e['language_exact'] and path_ok else 'AMBIGUOUS_ACTION_OR_EXPLICIT_METADATA_CONFLICT'
   if not path_ok:
    row['source_status']='UNRESOLVED';a['new_status']='UNRESOLVED'
  audits.append(a)
  diff.append(dict(suite=row['suite'],trajectory_id=row['trajectory_id'],old_status=old['source_status'],new_status=row['source_status'],old_demo=old['demo'],new_demo=row['demo'],old_source_sha256=old['source_sha256'],new_source_sha256=row['source_sha256'],old_split=old['split'],new_split=row['split'],decision_reason=a['decision_reason']))
 csvout(RUN/'lineage_audit.csv',audits);csvout(RUN/'phase_audit.csv',phases);csvout(RUN/'source_rule_diff.csv',diff)
 csvout(RUN/'previous_210_audit.csv',[a for a in audits if a['old_status']=='UNRESOLVED'])
 csvout(RUN/'source_manifest.csv',rows)
 # Copy protocol state; historical files remain unchanged. Known numeric evidence is not deleted.
 dump(RUN/'lineage_rule_v2.json',dict(version='v2_interpretation_same_conservative_admission',uniform_scope=1693,admission_thresholds_changed=False,old_ratio=2,old_nearest_rmse=.1,reason='Original HDF5 obs are post-action; regenerated RLDS obs are pre-action after ten settling actions and a different integration with no-ops removed. Comparing normalized-progress trajectories is approximate corroboration, not a necessary identity criterion.',missing='Released intermediate regenerated HDF5 or a per-episode RLDS-to-demo map plus generation initial-state metadata. Original source initial states alone cannot prove the RLDS episode initial state.',no_new_promotion_from_unique_actions_alone=True,initial_state_comparison='Recorded with explicit phase mismatch; not admitted as a same-time equality test.'))

def reuse():
 rows=list(csv.DictReader(open(RUN/'source_manifest.csv')));old={(x['suite'],x['trajectory_id']):x for x in csv.DictReader(open(PREVIOUS/'source_manifest.csv'))};records=[]
 for x in rows:
  before=old[x['suite'],x['trajectory_id']];p=PREVIOUS/'trajectories'/x['suite']/f"traj_{x['trajectory_id']}"
  if not (p/'result.json').exists():continue
  same=all(x[k]==before[k] for k in ['source_sha256','demo','filtered_indices','split','source_status','split_conflict'])
  record=dict(suite=x['suite'],trajectory_id=x['trajectory_id'],status='WITHHELD_CHANGED_LINEAGE',original=str(p),source_rule_unchanged=same)
  if same and x['source_status']=='MULTI_EVIDENCE_JOIN' and x['split_conflict']=='False':
   result=json.loads((p/'result.json').read_text());assert json.loads((p/'loader_audit.json').read_text())['status']=='PASS'
   for filename,key in [('clean_model.mjb','clean_model_sha256'),('background_model.mjb','background_model_sha256'),('state_cache.npz','state_cache_sha256')]:
    # Locate cache by its recorded name instead of guessing alternative contents.
    f=p/filename
    if filename=='state_cache.npz' and not f.exists():
     choices=list(p.glob('*.npz'));assert len(choices)==1;f=choices[0]
    assert sha(f)==result[key],str(f)
   cache=np.load(p/'source_state_geometry.npz');ids=np.array(json.loads(x['filtered_indices']),dtype=int)
   assert np.array_equal(ids,cache['source_indices'])
   with h5py.File(x['source_file']) as source:
    demo=source['data'][x['demo']]
    assert np.array_equal(demo['states'][:][ids],cache['states'])
    assert np.array_equal(demo['actions'][:][ids],cache['actions'])
    assert str(demo.attrs['model_file'])==(p/'original_source.xml').read_text()
   ct=json.loads((p/'camera_time.json').read_text())
   for app in ['clean','background']:
    with h5py.File(PREVIOUS/'official_loader'/app/x['suite']/p.name/'metadata.h5') as h:
     for field,expected in [('action',cache['actions'].astype('float32')),('qpos',cache['qpos'].astype('float32')),('eef_pose',cache['eef_pose'].astype('float32')),('source_frame_index',ids),('source_time_s',cache['states'][:,0])]:
      assert np.array_equal(h[field][:],expected),(str(p),app,field)
     for camera in ct['cameras']:
      cid=camera['model_id'];stream=camera['stream']
      extr=np.concatenate([cache['camera_rotations'][:,cid].reshape(-1,3,3)@np.diag([1.,-1.,-1.]),cache['camera_positions'][:,cid,:,None]],axis=2)[:,None].astype('float32')
      assert np.array_equal(h[stream+'_extrinsics'][:],extr)
      assert np.array_equal(h[stream+'_intrinsics'][:],np.tile(np.array(camera['K']),(len(ids),1,1,1)).astype('float32'))
   rgb_count=0
   for app in ['clean','background']:
    for stream in STREAMS:
     imgs=list((p/app/(stream+'_rgb')).glob('*.png'));assert len(imgs)==int(x['frames'])
     for image in imgs:
      with image.open('rb') as f:
       assert f.read(8)==b'\x89PNG\r\n\x1a\n';f.seek(-12,2);assert f.read()==b'\x00\x00\x00\x00IEND\xaeB`\x82'
     rgb_count+=len(imgs)
   dest=RUN/'trajectories'/x['suite']/p.name;dest.mkdir(parents=True,exist_ok=False)
   for child in p.iterdir():
    if child.name=='loader_audit.json':shutil.copy2(child,dest/child.name)
    else:(dest/child.name).symlink_to(child,target_is_directory=child.is_dir())
   for app in ['clean','background']:
    d=RUN/'official_loader'/app/x['suite'];d.mkdir(parents=True,exist_ok=True)
    src=PREVIOUS/'official_loader'/app/x['suite'];(d/p.name).symlink_to(src/p.name,target_is_directory=True)
    if not (d/'metadata.json').exists():shutil.copy2(src/'metadata.json',d/'metadata.json')
   record.update(status='REUSED_VERIFIED',rgb_files=rgb_count,result_sha256=sha(p/'result.json'),validation='Recorded model and state-cache hashes; all PNG signatures/end markers and frame counts; prior full-sequence numerical audit. PNG bytes have no historical per-file digest, so not claimed byte-hash reverified.')
  records.append(record);dump(RUN/'reuse_audit.json',records)
 print('reused',sum(x['status']=='REUSED_VERIFIED' for x in records),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');a=p.parse_args()
 if a.reuse:reuse()
 else:main()
