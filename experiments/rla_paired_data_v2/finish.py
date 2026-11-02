"""CPU packaging; run only after all full renders have completed."""
import csv,subprocess,sys
from PIL import Image,ImageDraw
from common import *

def main():
 eps=read(ROOT/'source_manifest.json');results=[];pairs=[]
 for e in eps:
  folder=ROOT/'rendered'/e['episode'];result=read(folder/'result.json');results.append(result);physical=np.load(folder/'physical.npz')
  save(folder/'paired_sequence_manifest.json',{'episode':e['episode'],'source_id':e['source_id'],'frames':e['frames'],'conditions':VARIANTS,'cameras':CAMS,'resolution':224,'reflection':1,'MSAA':4,'valid_full_chunks':e['frames']-8,'tail':'missing future frames have false validity, never clamped or fabricated','storage_format':'appearance_episode_npz_NHWC_v2','rgb_pattern':'{appearance}.npz:rgb[timestep]','physical':'physical.npz','identity':['source_id','timestep','appearance_id','paired_v2_revision2']})
  for t in range(e['frames']):
   for v in VARIANTS:
    pairs.append({'source_id':e['source_id'],'episode':e['episode'],'timestep':t,'appearance_id':v,'version':'paired_v2_revision2','split':'training-prototype','rgb_file':str(folder/(v+'.npz')),'rgb_index':t,'physical_file':str(folder/'physical.npz'),'future_horizons':list(range(1,9)),'future_valid_mask':[t+h<e['frames'] for h in range(1,9)],'camera_order':CAMS,'source_time':float(physical['states'][t,0])})
 with (ROOT/'pairing_manifest.jsonl').open('x') as f:
  for p in pairs:f.write(json.dumps(p)+'\n')
 viz=ROOT/'visualizations';viz.mkdir(exist_ok=False)
 import imageio.v2 as imageio
 for task in ['stove','drawer']:
  e=next(e for e in eps if e['task']==task);folder=ROOT/'rendered'/e['episode'];arrays=[np.load(folder/(v+'.npz'))['rgb'] for v in VARIANTS]
  idx=np.unique(np.linspace(0,e['frames']-1,min(40,e['frames'])).astype(int))
  with imageio.get_writer(str(viz/(task+'_comparison.mp4')),fps=5,macro_block_size=1) as w:
   for t in idx:
    canvas=Image.new('RGB',(1120,488),'white');draw=ImageDraw.Draw(canvas)
    for j,v in enumerate(VARIANTS):
     for ci,c in enumerate(CAMS):
      canvas.paste(Image.fromarray(arrays[j][t,ci]),(j*224,ci*244+20));draw.text((j*224,ci*244),v+' t='+str(t),fill='black')
    w.append_data(np.asarray(canvas))
  # Fixed temporal midpoint, not picked by visual strength.
  t=e['frames']//2;canvas=Image.new('RGB',(1120,488),'white');draw=ImageDraw.Draw(canvas)
  for j,v in enumerate(VARIANTS):
   for ci,c in enumerate(CAMS):canvas.paste(Image.fromarray(arrays[j][t,ci]),(j*224,ci*244+20));draw.text((j*224,ci*244),v+' '+c,fill='black')
  canvas.save(viz/(task+'_midpoint.png'))
 prior=ROOT.parent
 samples=[read(p) for root in [prior,ROOT] for p in (root/'sample').glob('*/result.json')]
 counts={'independent_source_demonstrations':len(eps),'new_to_training_pool':sum(e['identity']=='new_to_training_pool' for e in eps),'reused_old_training':sum(e['identity']=='reused_training' for e in eps),'physical_frames':sum(e['frames'] for e in eps),'complete_render_versions':len(eps)*5,'full_rgb_images':sum(r['rgb_images'] for r in results),'state_restorations':sum(r['state_restorations'] for r in results+samples),'render_calls':sum(r['render_calls'] for r in results+samples),'sample_rgb_images_including_rejected_revision':sum(r['rgb_images'] for r in samples),'source_action_open_loop_replays':0,'model_forwards':0,'backward':0,'optimizer_updates':0,'AE_updates':0,'DINO_updates':0,'policy_rollouts':0,'simulation_steps':0}
 save(ROOT/'validation_results.json',{'status':'PASS_PROTOTYPE_WITH_LIMITATIONS','counts':counts,'complete_sources':results,'sample_attempts':samples,'sample_review':read(ROOT/'sample_review.json'),'limits':['Not an open-loop reproduction validation: no dynamics integration performed.','Geometry depth/segmentation checked at five fixed moments/task; compiled physics and FK/cameras checked at every complete rendered state.','Additional unexamined candidates are metadata potential, not certified admitted data.','No downstream quality or control claims.']})
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
