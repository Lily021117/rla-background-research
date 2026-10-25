"""Explicitly authorized GPU0 SERIAL resume. Stop on any failure, no retry."""
import argparse,json,time,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;src=Path(__file__).parent
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',RLA_AUTHORIZED_GPU_UUID='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',RLA_POLICY_REGISTRY='policy_registry_resumed.json')
with (o/'recovery_authorization.json').open('x') as f:json.dump(dict(user_authorization='允许 GPU 0 串行恢复',GPU=env['RLA_AUTHORIZED_GPU_UUID'],serial=True,restore_step=300,remaining_updates_each=300,changed_training_hyperparameters=False,retained_original_attempts=True,validation300_repeated=True,optimizer_replay=0),f,indent=2)
with (o/'policy_registry_resumed.json').open('x') as f:json.dump({arm:str(o/arm/'resume_gpu0/step0600.pt') for arm in ('C-low','D-center-low')},f,indent=2)
f=o/'resume-precheck_gpu0_C_process.json'
while not f.exists():time.sleep(10)
assert json.loads(f.read_text())['returncode']==0
def launch(stage,attempt):
 subprocess.run(['/usr/bin/python3',str(src/'launch.py'),'--out',str(o),'--stage',stage,'--attempt',attempt],env=env,check=True)
for arm in ('C-low','D-center-low'):
 ck=o/arm/'isolated_smoke/step0300.pt'
 if arm=='D-center-low':
  env['RLA_RESUME_PRECHECK']=str(ck);launch('resume-precheck','gpu0_D')
 env['POLICY_RESUME']=str(ck);launch('train-'+arm,'resume_gpu0')
 assert json.loads((o/arm/'resume_gpu0/training_acceptance.json').read_text())['successful_updates']==600
env.pop('POLICY_RESUME',None);env.pop('RLA_RESUME_PRECHECK',None)
launch('evaluate','gpu0');launch('diagnostics','gpu0')
print('GPU0_SERIAL_TRAINING_AND_EVALUATION_COMPLETE',flush=True)
