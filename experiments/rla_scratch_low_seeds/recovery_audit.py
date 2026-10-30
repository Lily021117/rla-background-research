import sys,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_low_seeds.common import *
import torch
def main():
 p=ROOT/'seed_31002/D-low/formal/step0001.pt';ck=torch.load(p,map_location='cpu');plan=read(ROOT/'seed_31002/run_plan.json')
 assert ck['global_step']==1 and ck['next_physical_sample']==8 and ck['auxiliary_schedule_counter']==1
 assert ck['configuration']==plan and ck['arm']=='D-low'
 assert ck['scheduler']['last_epoch']==1 and ck['AMP_scaler'] is None and ck['dtype']=='bfloat16'
 steps=sorted({int(v['step']) for v in ck['optimizer']['state'].values()});assert steps==[1]
 assert all(torch.isfinite(v).all() for m in ck['parameters'].values() for v in m.values())
 for k in ['torch_rng','cuda_rng','python_rng','numpy_rng']:assert k in ck
 for seed,arm in [(31001,'C-low'),(31001,'D-low'),(31002,'C-low')]:assert read(ROOT/f'seed_{seed}/{arm}/formal/training_acceptance.json')['successful_updates']==600
 before=subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True);time.sleep(15)
 after=subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True)
 def counts(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ['Correctable','Uncorrectable','SRAM SM','Threshold','Repair Pending','Unrepairable'])]
 assert counts(before)==counts(after)
 free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=26624
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 save(ROOT/'recovery_gpu0_01/checkpoint_audit.json',dict(PASS=True,path=str(p),sha256=sha(p),global_step=1,optimizer_steps=steps,scheduler_last_epoch=1,all_RNG_present=True,next_physical_sample=8,AMP_scaler=None,precision='BF16 no scaler',configuration_exact=True,parameters_finite=True,original_successful_computations=50,replay_expected=49,remaining_new_effective_updates=1750,remaining_update_computations=1799,GPU=GPU,free_MiB=free,ECC_before=before,ECC_after=after,observed_counts_stable=True,correctable_volatile_baseline=2,correctable_aggregate_baseline=10,uncorrectable_baseline=0,limitation='stability observation not proof of repaired hardware; user authorized conditional GPU0 continuation; any new increment stops',no_other_process_terminated=True))
 print('FULL_RECOVERY_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
