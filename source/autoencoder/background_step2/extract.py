from .common import *
def main():
 import torch,decord,h5py,xml.etree.ElementTree as ET
 from utils.dino import DINOv3FeatureExtractor
 write(OUT/'cache_gpu_gate.json',gpu_gate())
 dino=DINOv3FeatureExtractor(model_name=str(DINO),use_compile=False).cuda().eval().requires_grad_(False)
 assert 'dinov3' in dino.config.model_type and dino.embed_dim==1024
 before=model_hash(dino)
 manifests=read(OUT/'split_manifest.json')['splits'];caches=[];records={s:[] for s in ('train','validation')}
 for split in records:
  for e in manifests[split]:
   c=read(e['manifest_path']);entry=c['entry'];geo=np.load(e['geometry_path'])
   assert sha(e['manifest_path'])==e['manifest_sha256'] and sha(e['geometry_path'])==e['geometry_sha256']
   with h5py.File(e['source_hdf5'],'r') as f:
    d=f['data'][e['source_demo']]
    assert np.array_equal(d['states'][()][geo['state_indices']],geo['states'])
    assert np.array_equal(d['actions'][()],geo['actions'])
   assert max(c['variant_geometry_max_abs_errors']['background'].values())<=1e-9
   x=ET.parse(Path(e['manifest_path']).parent/'clean.xml').getroot();y=ET.parse(Path(e['manifest_path']).parent/'background.xml').getroot()
   assert len(list(x.iter()))==len(list(y.iter()))
   for a,b in zip(x.iter(),y.iter()):
    assert a.tag==b.tag
    if a.attrib!=b.attrib:
     assert a.tag=='texture' and a.get('name') in ('tex-table','tex-wall');a.attrib.clear();a.attrib.update(b.attrib)
   assert ET.tostring(x)==ET.tostring(y)
   ids=sorted({w[k] for w in e['windows'] for k in ('output_t','output_future')});look={v:i for i,v in enumerate(ids)}
   videoids=c.get('video_source_output_indices',list(range(entry['frames'])))
   assert set(ids)<=set(videoids) and len(set(videoids))==len(videoids)
   for ci,camera in enumerate(('agentview','wrist')):
    paths={}
    mask_evidence='Raw replay RGB full-frame support, explicitly same whole-image Step1 protocol; no fabricated segmentation file.'
    if entry.get('converted_metadata'):
     mp=Path(entry['converted_metadata']).parent/(('agentview_camera' if ci==0 else 'wrist_camera')+'_foreground_mask.mp4')
     vr=decord.VideoReader(str(mp),width=512,height=512,num_threads=1)
     assert (vr.get_batch(ids).asnumpy()>0).all()
     mask_evidence='Existing converted foreground-mask stream verified positive on all selected frames: '+str(mp)
    images={}
    for vi,v in enumerate(('clean','background')):
     p=Path(c['videos'][v][ci]);assert sha(p)==c['video_sha256'][v][ci]
     vr=decord.VideoReader(str(p),width=512,height=512,num_threads=1);assert len(vr)==len(videoids)
     dest=OUT/'features'/f'{e["episode_id"]}_{camera}_{v}.npy';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
     mmap=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float16,shape=(len(ids),1024,1024))
     for i,source in enumerate(ids):
      rgb=vr[videoids.index(source)].asnumpy()[::-1,::-1].copy()
      if i==0:images[v]=rgb
      ten=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().cuda()/255
      with torch.inference_mode():_,token=dino(ten,return_spatial_grid=False)
      assert token.shape==(1,1024,1024) and torch.isfinite(token).all()
      mmap[i]=token[0].float().cpu().numpy();del ten,token
     mmap.flush();del mmap
     a=np.load(dest,mmap_mode='r')
     assert all(np.isfinite(a[i]).all() for i in range(len(ids)))
     meta=dict(split=split,episode_id=e['episode_id'],camera=camera,appearance=v,path=str(dest),shape=list(a.shape),dtype=str(a.dtype),sha256=sha(dest),
      source_video=str(p),source_video_sha256=c['video_sha256'][v][ci],source_output_ids=ids,video_output_ids=videoids,mask_evidence=mask_evidence,
      geometry_sha256=e['geometry_sha256'],preprocessing='decord512/rotate180/RGB255/original DINOv3 mean,std; fp16 extraction; raw cached patch features')
     write(dest.with_suffix('.json'),meta);caches.append(meta);paths[v]=str(dest)
    assert np.abs(images['clean'].astype(float)-images['background']).mean()>0
    for w in e['windows']:
     assert geo['state_indices'][w['output_t']]==w['state_t'] and geo['state_indices'][w['output_future']]==w['state_future']
     records[split].append(dict(split=split,episode_id=e['episode_id'],task=e['task'],camera=camera,**w,cache=paths,current_cache_index=look[w['output_t']],future_cache_index=look[w['output_future']]))
    health();append(OUT/'resource_monitor.jsonl',dict(stage='extract',episode_id=e['episode_id'],camera=camera,peak_gpu_bytes=torch.cuda.max_memory_allocated(),**resources()))
    print('CACHED',split,e['episode_id'],camera,flush=True)
 for split,rs in records.items():write(OUT/('training_records.json' if split=='train' else 'validation_records.json'),rs)
 assert len(records['train'])==380 and len(records['validation'])==80
 assert before==model_hash(dino)
 write(OUT/'feature_cache_acceptance.json',dict(status='PASS',files=len(caches),train_view_pairs=380,validation_view_pairs=80,test_read_for_extraction=False,
  cache_inventory=caches,dino_before=before,dino_after=before,all_finite=True,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
 # Balanced camera sequence per update: same 4 units and both cameras for B/C.
 rng=np.random.default_rng(20261020);schedule=[]
 for i in range(200):
  indices=[]
  for camera in ('agentview','wrist'):
   allowed=[j for j,r in enumerate(records['train']) if r['camera']==camera]
   indices.extend(int(j) for j in rng.choice(allowed,size=2,replace=False))
  schedule.append(indices)
 write(OUT/'sampling_sequence.json',dict(seed=20261020,steps=schedule,unique_sequence_for_B_and_C=True,physical_identity_keys='episode_id/output_t/output_future/camera'))
if __name__=='__main__':main()
