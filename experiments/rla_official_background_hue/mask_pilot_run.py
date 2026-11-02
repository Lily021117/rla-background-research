"""Frozen SAM2.1 video inference; original RGB untouched, all frames retained."""
import argparse,json,os,time,subprocess,gc,traceback,collections
from pathlib import Path
import numpy as np,cv2,torch,decord
from PIL import Image,ImageDraw
from audit import sha,write_json,write_csv
from paired_hue import hue_rgb,hue_parameter,array_hash

GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
def resource():
 raw=subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-x'],text=True)
 import xml.etree.ElementTree as ET
 e=ET.fromstring(raw).find('gpu');free=int(e.findtext('fb_memory_usage/free').split()[0])
 counts={f'{period}/{c.tag}':int(c.text) for period in ['volatile','aggregate'] for c in e.find(f'ecc_errors/{period}') if c.text.isdigit()}
 assert 'volatile/sram_uncorrectable_parity' in counts
 return dict(xml=raw,free_mib=free,ecc_counts=counts)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--sam-code',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--prompts',default='prompts_v1.json');p.add_argument('--revision',default='v1');p.add_argument('--only',default='')
 a=p.parse_args();out=a.run/a.revision;out.mkdir(exist_ok=False)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
 baseline=resource();assert baseline['free_mib']>=26624
 write_json(out/'resource_before.json',baseline)
 protocol=json.loads((a.run/'selection_and_protocol.json').read_text());prompts=json.loads((a.run/a.prompts).read_text())
 selected=[v for v in protocol['videos'] if not a.only or v['id'] in a.only.split(',')]
 import sys;sys.path.insert(0,str(a.sam_code))
 from sam2.build_sam import build_sam2_video_predictor
 torch.manual_seed(20261102);np.random.seed(20261102);torch.set_num_threads(4)
 start=time.time(); counters=collections.Counter();completed=[]
 cfg='configs/sam2.1/sam2.1_hiera_b+.yaml'
 identity=dict(code_commit=subprocess.check_output(['git','-C',str(a.sam_code),'rev-parse','HEAD'],text=True).strip(),
   checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),config=cfg,
   config_sha256=sha(a.sam_code/'sam2'/cfg),torch=torch.__version__,gpu_uuid=GPU,
   compile=False,autocast='bfloat16',official_extra_postprocessing=False,offload_video_to_cpu=True,
   masks_source='reviewed_model_segmentation_PENDING_REVIEW',prompt_sha256=sha(a.run/a.prompts),
   selection_sha256=sha(a.run/'selection_and_protocol.json'))
 write_json(out/'model_identity.json',identity)
 try:
  model=build_sam2_video_predictor(cfg,str(a.checkpoint),device='cuda',apply_postprocessing=False)
  model.eval();model.requires_grad_(False)
  for name in ['image_encoder','sam_mask_decoder','memory_encoder']:
   getattr(model,name).register_forward_hook(lambda mod,args,result,n=name:counters.update([n]))
  for v in selected:
   assert sha(v['path'])==v['sha256'];before=resource()
   assert before['ecc_counts']==baseline['ecc_counts']
   vd=out/v['id'];vd.mkdir();tick=time.time();count_before=dict(counters)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=model.init_state(v['path'],offload_video_to_cpu=True,offload_state_to_cpu=False)
    assert state['num_frames']==v['frames']
    for obj in prompts[v['id']]['objects']:
     for k in obj['keyframes']:
      pts=np.array(k['positive']+k['negative'],np.float32);labels=np.array([1]*len(k['positive'])+[0]*len(k['negative']),np.int32)
      model.add_new_points_or_box(state,k['frame'],obj['object_id'],points=pts,labels=labels)
    raw=np.zeros((v['frames'],256,256),np.uint8); seen=[]
    for t,ids,logits in model.propagate_in_video(state,start_frame_idx=0):
     assert logits.shape[-2:]==(256,256)
     raw[t]=(logits[:,0]>0).any(0).cpu().numpy().astype(np.uint8);seen.append(t)
    assert seen==list(range(v['frames']))
   torch.cuda.synchronize();infer_seconds=time.time()-tick
   np.savez_compressed(vd/'raw_model_masks.npz',masks=raw)
   masks=np.stack([cv2.erode(m,np.ones((7,7),np.uint8)) for m in raw])
   np.savez_compressed(vd/'conservative_masks_native.npz',masks=masks)
   decord.bridge.set_bridge('native');vr=decord.VideoReader(v['path'],num_threads=1)
   writer=cv2.VideoWriter(str(vd/'overlay_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(512,512));assert writer.isOpened()
   sheet=Image.new('RGB',(4*256,len(v['fixed_review_frames'])*282),'#222222');dr=ImageDraw.Draw(sheet)
   records=[];mask224=[];framehashes=[];last=None
   # One frozen hue for this trajectory's visualization, shared between cameras.
   delta,key=hue_parameter(20261102,'train',{'suite':v['suite'],'trajectory':v['trajectory_id'],'visualization':'full_video'},0)
   for t in range(v['frames']):
    rgb=vr[t].asnumpy();assert rgb.shape==(256,256,3)
    m=masks[t].astype(bool);aug=hue_rgb(rgb,m,delta)
    assert np.array_equal(rgb[~m],aug[~m])
    overlay=rgb.copy();overlay[m]=(rgb[m]*.55+np.array([0,255,0])*.45).astype(np.uint8)
    diff=np.abs(aug.astype(np.int16)-rgb.astype(np.int16)).astype(np.uint8)
    tile=np.concatenate([np.concatenate([rgb,overlay],1),np.concatenate([aug,np.clip(diff.astype(float)*4,0,255).astype(np.uint8)],1)],0)
    writer.write(cv2.cvtColor(tile,cv2.COLOR_RGB2BGR))
    if t in v['fixed_review_frames']:
     y=v['fixed_review_frames'].index(t)*282
     for j,im in enumerate([rgb,np.repeat((m*255).astype(np.uint8)[...,None],3,-1),aug,np.clip(diff.astype(float)*4,0,255).astype(np.uint8)]):
      sheet.paste(Image.fromarray(im),(j*256,y+26));dr.text((j*256+3,y+3),f't={t} '+['ORIGINAL','ESTIMATED BG','HUE','DIFF x4'][j],fill='white')
    hsv=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2HSV);area=float(m.mean())
    jump=0 if last is None else float(abs(area-last.mean()))
    union=0 if last is None else np.logical_or(m,last).sum()
    iou=None if last is None or union==0 else float(np.logical_and(m,last).sum()/union)
    flags=[]
    if area<.001:flags.append('empty_or_near_empty')
    if area>.95:flags.append('near_full')
    if jump>.15:flags.append('area_jump')
    change=float(diff[m].mean()/255) if m.any() else 0
    records.append(dict(frame=t,coverage=area,raw_coverage=float(raw[t].mean()),mean_rgb_change=change,
      low_saturation_fraction=float((hsv[...,1][m]<.05).mean()) if m.any() else None,
      weak_change=change<1/255,adjacent_mask_iou=iou,area_jump=jump,flags=';'.join(flags)))
    r224=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_LINEAR)
    m224=cv2.erode(cv2.resize(m.astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST),np.ones((3,3),np.uint8))
    mask224.append(m224);framehashes.append({'frame':t,'rgb_native_hash':array_hash(rgb),'rgb_224_hash':array_hash(r224)})
    last=m
   writer.release();sheet.save(vd/'fixed_review.png');np.savez_compressed(vd/'masks_224.npz',masks=np.stack(mask224))
   write_csv(vd/'frame_metrics.csv',records);write_json(vd/'frame_identity.json',framehashes)
   write_json(vd/'abnormal_frames.json',[x for x in records if x['flags']])
   del state,vr;gc.collect();torch.cuda.empty_cache()
   item=dict(id=v['id'],frames=v['frames'],inference_seconds=infer_seconds,total_seconds=time.time()-tick,
    forward_counts={k:counters[k]-count_before.get(k,0) for k in counters},mean_coverage=float(np.mean([x['coverage'] for x in records])),
    weak_color_frames=sum(x['weak_change'] for x in records),automatic_anomaly_frames=sum(bool(x['flags']) for x in records),
    delta_h=delta,augmentation_key=key,source_video_sha256=v['sha256'],mask_sha256=sha(vd/'masks_224.npz'),
    status='PENDING_SEMANTIC_REVIEW',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
   write_json(vd/'summary.json',item);completed.append(item);write_json(out/'progress.json',completed)
   after=resource();assert after['ecc_counts']==baseline['ecc_counts']
   print(json.dumps(item),flush=True)
  write_json(out/'completion.json',dict(videos=len(completed),frames=sum(x['frames'] for x in completed),
    forward_counts=dict(counters),seconds=time.time()-start,AE_updates=0,DINO_updates=0,policy_updates=0,backward=0,rollouts=0))
 except BaseException as e:
  write_json(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed,forward_counts=dict(counters)));raise
 finally:
  write_json(out/'resource_after.json',resource())

if __name__=='__main__':main()
