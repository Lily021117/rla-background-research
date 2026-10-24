"""Append-only recovery/source receipt, no historical output modification."""
import argparse,json,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'recovery_20261024';out.mkdir(exist_ok=False)
files=[]
for folder in ('ae_consistency','multitask'):
 for source in (Path('experiments/official_libero_rla_consistency')/folder).glob('*'):
  if source.suffix not in ('.py','.sh'):continue
  content=source.read_bytes();dest=out/'source'/folder/source.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content);files.append(dict(file=str(source),sha256=hashlib.sha256(content).hexdigest()))
for index in (0,1,3):
 (out/f'gpu{index}_ecc_start.txt').write_text(subprocess.check_output(['nvidia-smi','-i',str(index),'-q','-d','ECC'],text=True))
record=dict(source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_files=files,authorized_devices=[0,1],excluded_devices=[2,3],reason='user requested available-GPU recovery; GPU3 had previous new ECC increment, not reused',recovery='preserve previous44 logged updates per stove AE, restore full step0 state, no changed method or budget',cache_wrapper_deviation='drawer cache Python completed PASS and persisted acceptance; launch shell edited while active produced trailing parse error after completion. Current launch script passes bash-n. No cache was accepted based only on shell exit; its full acceptance receipt is checked.',planned_AE_effective_updates=2400,planned_policy_updates=3600,planned_new_valid_trials=360,heldout_read=False)
(out/'startup_receipt.json').write_text(json.dumps(record,indent=2))
