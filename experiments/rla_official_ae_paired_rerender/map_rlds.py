"""Actual TFDS enumeration + full numeric and original-RGB evidence, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import argparse,csv,json,io,time,hashlib
from pathlib import Path
import numpy as np,h5py,cv2,decord
from PIL import Image
from prepare import ROOT,RUN,sha,dump,csvout
def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args()
 import tensorflow as tf,tensorflow_datasets as tfds
 tf.config.set_visible_devices([],'GPU');tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.threading.set_intra_op_parallelism_threads(2)
 root=ROOT/'data/libero' if a.suite=='libero_spatial' else RUN/'source_rlds'
 src=root/f'{a.suite}_no_noops/1.0.0';b=tfds.builder_from_directory(str(src))
 rows=[r for r in csv.DictReader(open(RUN/'source_mapping.csv')) if r['suite']==a.suite];byid={int(r['trajectory_id']):r for r in rows}
 assert b.info.splits['train'].num_examples==len(rows)
 dec={'steps':{'observation':{'image':tfds.decode.SkipDecoding(),'wrist_image':tfds.decode.SkipDecoding()}}}
 ds=b.as_dataset(split='train',shuffle_files=False,decoders=dec);options=tf.data.Options();options.threading.private_threadpool_size=2;ds=ds.with_options(options)
 out=RUN/'rlds_mapping_v2'/a.suite;out.mkdir(parents=True,exist_ok=True);results=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else [];started=time.time()
 previous={r['enumeration_index']:r for r in results};resumed_records=len(results)
 assert len(previous)==len(results)
 def signature(action,qpos,eef):
  return hashlib.sha256(b''.join(np.ascontiguousarray(x,dtype=np.float32).tobytes() for x in [action,qpos,eef])).hexdigest()
 signatures={}
 for row in rows:
  with h5py.File(Path(row['source'])/'metadata.h5') as h:
   key=signature(h['action'][:],h['qpos'][:],h['eef_pose'][:,:3])
  signatures.setdefault(key,[]).append(row)
 for idx,e in enumerate(ds):
  t0=time.time();steps=next(iter(e['steps'].batch(10000)));obs=steps['observation'];n=len(steps['action'])
  fp=e['episode_metadata']['file_path'].numpy().decode();lang=steps['language_instruction'][0].numpy().decode()
  action=steps['action'].numpy();state=obs['state'].numpy();joints=obs['joint_state'].numpy()
  candidates=signatures.get(signature(action,np.concatenate([joints,state[:,6:8]],1),state[:,:3]),[])
  if len(candidates)!=1:
   result=dict(suite=a.suite,enumeration_index=idx,rlds_match='NUMERIC_JOIN_AMBIGUOUS_OR_MISSING',candidate_ids=[r['trajectory_id'] for r in candidates],episode_file_path=fp,frames=n)
   results.append(result);dump(out/'progress.json',results);continue
  row=candidates[0];tid=int(row['trajectory_id']);path=Path(row['source'])
  if idx in previous:
   old=previous[idx]
   assert old['trajectory_id']==row['trajectory_id'] and old['episode_file_path']==fp and old['rlds_match']=='PASS', 'Resume prefix changed'
   cached=np.load(out/f'numeric_{tid:06d}.npz')
   assert np.array_equal(cached['action'],action) and np.array_equal(cached['state'],state) and np.array_equal(cached['joints'],joints)
   continue
  with h5py.File(path/'metadata.h5') as h:
   lengths=n==len(h['action']);numeric=dict(action_max=float(np.max(abs(action-h['action'][:]))) if lengths else None,
    joints_max=float(np.max(abs(np.concatenate([joints,state[:,6:8]],1)-h['qpos'][:]))) if lengths else None,
    eef_position_max=float(np.max(abs(state[:,:3]-h['eef_pose'][:,:3]))) if lengths else None)
  frames=list(range(n)) if tid==10 else sorted(set(np.linspace(0,n-1,5,dtype=int).tolist()))
  image_checks={};pilot_images={}
  for stream,key in [('agentview_camera','image'),('wrist_camera','wrist_image')]:
   v=decord.VideoReader(str(path/f'{stream}_rgb.mp4'),num_threads=1);mae=[];maximum=[]
   for t in frames:
    # Exact TFDS Image decoder, not PIL's different JPEG upsampling/DCT.
    raw=tf.io.decode_image(obs[key][t],channels=3).numpy()
    actual=v[t].asnumpy();diff=np.abs(raw.astype('int16')-actual.astype('int16'));mae.append(float(diff.mean()));maximum.append(int(diff.max()))
    if tid==10:pilot_images.setdefault(key,[]).append(raw)
   image_checks[stream]=dict(frames=frames,max_abs=max(maximum),mean_abs=float(np.mean(mae)),all_frames=tid==10,video_sha256=sha(path/f'{stream}_rgb.mp4'))
  passed=lengths and lang==row['task'] and all(v is not None and v<1e-6 for v in numeric.values()) and all(x['max_abs']<=3 and x['mean_abs']<=1 for x in image_checks.values())
  result=dict(suite=a.suite,trajectory_id=row['trajectory_id'],split=row['split'],enumeration_index=idx,episode_file_path=fp,
   hdf5_basename=fp.rsplit('/',1)[-1],hdf5_demo_id='NOT_IN_FILE_PATH',task=lang,frames=n,
   rlds_match='PASS' if passed else 'MISMATCH',numeric=numeric,images=image_checks,
   raw_state_semantics='8D eef axis-angle gripper ONLY; not full MuJoCo state',seconds=time.time()-t0)
  results.append(result)
  if tid==10:
   np.savez_compressed(out/'pilot_rlds.npz',action=action,state=state,joints=joints,**{k:np.stack(v) for k,v in pilot_images.items()})
  # Persist numeric signatures for eventual HDF5 join, not policy training input.
  np.savez_compressed(out/f'numeric_{tid:06d}.npz',action=action,state=state,joints=joints)
  dump(out/'progress.json',results)
  print(json.dumps({k:result[k] for k in ['suite','trajectory_id','rlds_match','hdf5_basename']},ensure_ascii=False),flush=True)
 dump(out/'complete.json',dict(tfds_version=tfds.__version__,tensorflow_version=tf.__version__,shuffle_files=False,seconds=time.time()-started,resumed_records=resumed_records,resume_prefix_rechecked=True,
  records=len(results),passed=sum(x['rlds_match']=='PASS' for x in results),results=results))
if __name__=='__main__':main()
