"""Join an already-running arm; never repeats an optimizer update."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arm-root',type=Path,required=True);a=p.parse_args();started=time.monotonic();root=a.arm_root/'isolated_smoke'
while True:
 receipt=root/'training_acceptance.json'
 if receipt.exists():
  x=json.loads(receipt.read_text());assert x['successful_updates']==600 and x['strict_reload'] and (root/'step0600.pt').is_file();print('JOIN COMPLETED FIXED600 WITHOUT NEW UPDATES',a.arm_root,flush=True);break
 active=False
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:args=(proc/'cmdline').read_bytes().decode().split('\0')
  except (OSError,UnicodeDecodeError):continue
  if '--run' in args and args[args.index('--run')+1]==str(a.arm_root) and 'ae_consistency.policy_training' in args:active=True;break
 assert active,'Existing incomplete arm has no live training process; preserve it and stop, no automatic restart'
 assert time.monotonic()-started<4*3600,'Fixed-budget training join exceeded4 hours'
 print('WAIT FOR EXISTING ARM, NO DUPLICATE UPDATES',a.arm_root,flush=True);time.sleep(20)
