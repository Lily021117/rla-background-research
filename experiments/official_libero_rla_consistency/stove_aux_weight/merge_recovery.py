"""Derived metric views; never overwrite interrupted attempts."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
audit=[]
for arm in ('C-low','D-center-low'):
 old=a.out/arm/'isolated_smoke';new=a.out/arm/'resume_gpu0';dest=a.out/'combined_metrics'/arm;dest.mkdir(parents=True,exist_ok=False)
 for name in ('training_metrics.csv','validation_metrics.csv'):
  first=list(csv.DictReader((old/name).open()));second=list(csv.DictReader((new/name).open()))
  if name.startswith('validation'):first=[r for r in first if int(r['step'])<300]
  rows=first+second
  if name.startswith('training'):assert [int(r['step']) for r in rows]==list(range(1,601))
  with (dest/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  audit.append(dict(arm=arm,file=name,rows=len(rows),sources=[str(old/name),str(new/name)],excluded='incomplete original validation300 only' if name.startswith('validation') else None))
with (a.out/'metric_merge_provenance.json').open('x') as f:json.dump(audit,f,indent=2)
