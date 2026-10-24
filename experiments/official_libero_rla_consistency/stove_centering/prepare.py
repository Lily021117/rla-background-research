"""Freeze the stove center-only experiment before any new model outputs."""
import argparse, json, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'stove_diagnosis'))
from common import read,save,sha,csvsave
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
R=Path('/data/lyc/projects/rla-wm');S=R/'experiments/official_libero_rla_consistency/run_20261023_10/stove'
a.out.mkdir(parents=True,exist_ok=False)
plan=read(S/'policy_run/run_plan.json');old_plan=read(S/'policy_run/run_plan.json')
plan.update(experiment='stove_target_center_only',arms=['D-old','D-center'],closed_loop_trials=120,historical_source=str(S),diagnostic_steps=[0,1,100,300,600])
plan['target_center_definition']={'D-old':'(z_D - mu_C) / s_C','D-center':'(z_D - mu_D) / s_C','scale':'unchanged official float32 scalar s_C','mean_source':'existing FP32-extractor training-only moments, independently checked against stored FP16 sidecars','no_auto_compensation':True}
plan['shared_protocol']['normalization']='same C scale; arm-specific C versus D training mean only'
plan['shared_protocol']['C_D_only_target_source_differs']=False
plan['shared_protocol']['center_only_difference']=True
plan['recovery']='full model/optimizer/scheduler/RNG/sample position; stop on CUDA/ECC/nonfinite; preserve attempts, no silent retry'
save(a.out/'run_plan.json',plan)
save(a.out/'historical_run_plan.json',old_plan)
manifest=read(S/'frozen_dataset_manifest.json');entries=manifest['episodes'] if isinstance(manifest,dict) else manifest
prov=read(S/'ae_run/sidecar_provenance.json')
statsC=read(S/'ae_run/sidecars/D/clean/stats.json');statsD=read(S/'ae_run/sidecars/D/clean/raw_target_statistics.json')
assert statsC==read(S/'ae_run/sidecars/D/background/stats.json')
assert statsD==read(S/'ae_run/sidecars/D/background/raw_target_statistics.json')
muC=np.asarray(statsC['mean'],np.float32);stdC=np.asarray(statsC['std'],np.float32);muD=np.asarray(statsD['mean'],np.float32)
scale=np.float32(np.sqrt((muC**2+stdC**2).mean()-float(muC.mean())**2))
total=np.zeros((16,64),np.float64);count=0;rows=[];max_offset=max_invariant=0.;max_value=0.
hashes={}
for x in prov:
 if x['model']!='D' or x['split']!='train':continue
 path=Path(x['file']);assert sha(path)==x['sha256'];hashes[str(path)]=x['sha256']
 z=np.load(path,mmap_mode='r');mask=np.load(x['mask_file']);N=x['valid_full_chunks'];assert mask[:N].all()
 for start in range(0,N,16):
  raw=z[start:min(start+16,N)].astype(np.float32);assert np.isfinite(raw).all()
  v=raw.reshape(-1,16,64);total+=v.astype(np.float64).sum(0);count+=len(v)
  old=(raw-muC)/scale;new=(raw-muD)/scale;offset=(muC-muD)/scale
  max_offset=max(max_offset,float(abs((new-old)-offset).max()))
  if len(raw)>1:max_invariant=max(max_invariant,float(abs((new[1:]-new[:-1])-(old[1:]-old[:-1])).max()))
  max_value=max(max_value,float(abs(old).max()),float(abs(new).max()))
 rows.append(dict(episode=x['episode'],appearance=x['appearance'],split=x['split'],valid_windows=N,statistic_rows=N*8,file=str(path),sha256=x['sha256']))
assert len(rows)==40 and count==statsD['count']==statsC['count']==40880
# Exact algebra; float32 subtraction/reduction introduces bounded last-bit rounding.
tol=8*np.finfo(np.float32).eps*max(1,max_value)
assert max_offset<=tol and max_invariant<=tol
quant_error=float(abs(total/count-np.asarray(statsD['mean'])).max())
assert quant_error<1e-3,'Stored FP16 moments inconsistent with authoritative pre-storage moments'
save(a.out/'normalization.json',dict(mu_C=muC.tolist(),mu_D=muD.tolist(),s_C=float(scale),statistics_C=statsC,statistics_D=statsD,statistic_count=count,weighting='equal cumulative horizons1..8 of complete windows only; all20 training episodes, both appearances; no dev; no per-episode reweighting',source_precision='FP32 encoder output summed in FP64 before FP16 sidecar storage',stored_FP16_recomputed_mean_max_difference=quant_error))
save(a.out/'target_center_audit.json',dict(status='PASS',fixed_offset_max_absolute_error=max_offset,between_sample_difference_max_error=max_invariant,float32_tolerance=tol,tolerance_definition='8 float32 eps * max(1, maximum normalized absolute target); frozen before model outputs',mask='only original full-valid windows; unchanged8 horizons',mu_D_source='verified existing train-only FP32 extractor moment, not dev or a new fit to quantized cache',raw_sidecars_unchanged=True))
csvsave(a.out/'mean_training_sources.csv',rows)
for path in [S/'frozen_dataset_manifest.json',Path(plan['common_step0']),Path(plan['inputs']['action_proprio_statistics']),Path(plan['inputs']['B20_sampling_order']),S/'policy_run/evaluation_protocol.json',S/'ae_run/development_protocol/evaluation_protocol.json']:
 hashes[str(path)]=sha(path)
for app in ('clean','background'):
 for name in ('stats.json','raw_target_statistics.json','manifest.json','libero_spatial/keys.json'):
  path=S/'ae_run/sidecars/D'/app/name;hashes[str(path)]=sha(path)
for arm in ('C','D'):
 path=S/'policy_run'/arm/'isolated_smoke/step0600.pt';hashes[str(path)]=sha(path)
save(a.out/'protected_input_hashes.json',hashes)
protocol=read(S/'policy_run/evaluation_protocol.json');protocol.update(models=['D-old','D-center'],trials=120,primary_comparison='D-center minus D-old',historical_C='reference only; no rerun',retry_rule='stop and preserve infrastructure error; only documented infrastructure retries, never task-failure reruns')
save(a.out/'evaluation_protocol.json',protocol)
save(a.out/'policy_registry.json',{arm:str(a.out/arm/'isolated_smoke/step0600.pt') for arm in plan['arms']})
(a.out/'preflight6/isolated_smoke').mkdir(parents=True)
save(a.out/'preflight6/isolated_smoke/preflight_acceptance.json',read(S/'policy_run/preflight6/isolated_smoke/preflight_acceptance.json'))
save(a.out/'reused_preflight_provenance.json',dict(source=str(S/'policy_run/preflight6/isolated_smoke/preflight_acceptance.json'),reason='same model/batch/precision/activation checkpointing; new center checked separately, no optimizer update in preflight'))
fixed=read(R/'experiments/official_libero_rla_consistency/stove_diagnosis/run_20261024_02/fixed_sample_manifest.json');save(a.out/'fixed_sample_manifest.json',fixed)
save(a.out/'source_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),status=subprocess.check_output(['git','status','--short'],text=True),official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950'))
print(json.dumps(dict(output=str(a.out),count=count,s_C=float(scale),offset_check=max_offset,FP16_mean_check=quant_error)),flush=True)
