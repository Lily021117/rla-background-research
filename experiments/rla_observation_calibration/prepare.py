from common import *
import torch
from PIL import Image,ImageDraw
from transforms import apply,intrinsic,noise_seed
def main():
 torch.set_num_threads(1)
 protocol=dict(name='Frozen C-low observation degradation development calibration',base_commit='39645b9e0ff7a379157947d01684a7e645f75083',conditions=CONDITIONS,combo=COMBO,max_steps={'stove':220,'drawer':300},seeds=SEEDS,max_new_valid=660,max_with_clean_fallback=780,clean_reuse=120,updates=0,backward=0,
  selection=dict(target='0.4 <= equally weighted three-seed rate < 0.6',priority=['blur','noise','crop'],within_family='lowest strength',combo_gate='all nine conditions admitted and each >= 0.6; only blur2_noise007',combo_candidate='same 0.4 <= rate < 0.6',review='only one selected candidate per task; no reselection after review'),
  image=dict(position='after original resize/crop/to_tensor at 224x224 [0,1], before both backbone normalizations; once per camera',blur='reflect padding, kernel 2*ceil(3*sigma)+1',noise='float32 PCG64 local generator; SHA256-derived 128-bit seed; no model seed/strength in key; clip [0,1]',crop='floor(224*ratio), offset floor((224-size)/2), tensor bilinear antialias=True; half-pixel K affine; preserve negative fx',clean='exact existing processor geometry; no added reencoding'),
  image_admission=dict(sample='first numeric train demo per task; floor linspace(0,n-1,5)',exclude=['wrong channel/shape/nonfinite/empty images','transformation bug or global severe saturation obscuring manipulation','crop removes previously visible task-critical content from BOTH cameras at any sampled time','target/gripper/manipulation region becomes unrecognizable in BOTH cameras at a previously observable sample'],manual='all conditions inspected before any scoring; no success-based exclusions; no replacement'),
  statistics=dict(bootstrap=20000,seed=20261102,unit='physical state preserving all policies/conditions',calibration_and_review_separate=True,uncertainty_excludes=['full training seeds','AE','noise implementation','task']),
  failures=dict(task_failure='valid, never retry',infrastructure='preserve attempt, stop; no auto hardware switch',no_future_expansion=True))
 save(BASE/'calibration_protocol.json',protocol)
 models=read(OLD/'checkpoint_inventory.json')
 for m in models:
  assert sha(m['path'])==m['sha256'];assert sha(m['statistics'])==m['statistics_sha256']
 save(BASE/'checkpoint_inventory.json',models)
 states=read(OLD/'state_manifest.json');split=[]
 for task in TASK_NAMES:
  ss=[s for s in states if s['task']==task]
  for s in ss:
   assert digest(np.load(s['state_file']))==s['state_sha256'];assert sha(s['xmls']['clean'])==s['xml_sha256']['clean']
   s['sort_key']=hashlib.sha256(f'rla_observation_calibration_v1:{TASK_NAMES[task]}:{s["state_sha256"]}'.encode()).hexdigest()
   s['xmls']={'clean':s['xmls']['clean']};s['xml_sha256']={'clean':s['xml_sha256']['clean']}
  ss.sort(key=lambda s:s['sort_key'])
  for i,s in enumerate(ss):s['subset']='calibration' if i<10 else 'review';s['rank']=i;split.append(s)
 save(BASE/'state_manifest.json',split);save(BASE/'clean_reuse.json',read(OLD/'clean_reuse.json'))
 refs={'stove':P/'experiments/official_libero_rla_consistency/run_20261023_10/stove','drawer':P/'experiments/rla_new_task_baseline/run_20261028_02/dataset_v2'}
 records=[];samples=[]
 out=BASE/'image_acceptance';out.mkdir()
 for task,root in refs.items():
  manifest=read(root/'frozen_dataset_manifest.json');episode=sorted([e for e in manifest['episodes'] if e['split']=='train'],key=lambda e:int(e['source_demo'].split('_')[-1]))[0]
  times=np.linspace(0,episode['frames']-1,5).astype(int)
  for t in times:
   f=Path(episode['root'])/'frames'/f'{t:04}.npz';a=np.load(f);sample=dict(task=task,episode=episode['episode'],split='train',t=int(t),path=str(f),sha256=sha(f));samples.append(sample)
   sheet=Image.new('RGB',(224*11,2*250),'white');draw=ImageDraw.Draw(sheet)
   for ci,cam in enumerate(CAMS):
    x=torch.from_numpy(a['clean'][ci].copy()).float()/255
    for j,c in enumerate(['clean']+CONDITIONS+[COMBO]):
     before=(torch.get_rng_state().clone(),np.random.get_state());y=apply(x,c,TASK_NAMES[task],digest(a['state']),int(t),cam)
     assert torch.equal(before[0],torch.get_rng_state());assert all(np.array_equal(u,v) for u,v in zip(before[1],np.random.get_state()))
     if c=='clean':assert torch.equal(x,y)
     assert torch.equal(y,apply(x,c,TASK_NAMES[task],digest(a['state']),int(t),cam))
     rgb=np.round(y.permute(1,2,0).numpy()*255).astype(np.uint8);sheet.paste(Image.fromarray(rgb),(j*224,ci*250+26));draw.text((j*224+3,ci*250+4),c+' '+cam,fill='black')
     K,A=intrinsic(a['K'][ci],c);assert K[0,0]<0
     records.append(dict(**sample,camera=cam,condition=c,mean_abs_rgb_change=float((x-y).abs().mean()),clip_fraction=float(((y==0)|(y==1)).float().mean()),std=float(y.std()),K_original=a['K'][ci].tolist(),K_transformed=K.tolist(),pixel_affine=A.tolist(),noise_seed=str(noise_seed(TASK_NAMES[task],digest(a['state']),int(t),cam))))
   sheet.save(out/f'{task}_{t:04}.png')
 save(out/'samples.json',samples);save(out/'numeric_checks.json',records)
 save(out/'implementation_checks.json',dict(no_global_rng_consumption=True,noise_repeatable=True,clean_pixel_identity=True,finite_unit_interval=True,negative_fx_preserved=True,policy_forwards=0))
 print('PREPARED: manual image review and runtime inference equivalence still required')
if __name__=='__main__':main()
