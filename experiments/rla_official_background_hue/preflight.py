"""Fail-closed admission CLI. No model creation/forward/training side effects."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text()); reasons=[]
    if c['paired']['mask_registry'] is None: reasons.append('No verified original-RGB background-mask registry')
    if c['initialization']['shared_step0_sha256'] is None: reasons.append('Common untrained step0 not yet constructed/hashed')
    if c['loss']['lambda_cons'] is None: reasons.append('Train-only finite calibration not yet authorized/executed')
    if not c['launch_safety']['formal_training_enabled']: reasons.append(c['launch_safety']['reason'])
    print(json.dumps({'method':c['method'],'ready':not reasons,'blockers':reasons,
        'model_forwards':0,'updates':0},indent=2))
    raise SystemExit(2 if reasons else 0)

if __name__=='__main__':main()
