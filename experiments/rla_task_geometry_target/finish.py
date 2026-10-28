"""Bounded completion monitor: analysis/publication only, never launches training.

Stops if the recorded runner exits before all frozen trials complete. No retry.
"""
import argparse,csv,json,os,re,subprocess,sys,time
from pathlib import Path
from labels import read,save
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def main(out,pid):
 while Path(f'/proc/{pid}').exists():
  if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
  time.sleep(30)
 evaluation=out/'evaluation_process.json'
 if not evaluation.exists() or read(evaluation)['returncode']!=0:
  save(out/'completion_monitor.json',dict(status='STOPPED_INCOMPLETE',reason='runner ended before successful400-trial evaluation; no automated retry or extra training',optimizer_updates_by_monitor=0));return
 subprocess.run([sys.executable,str(HERE/'analyze.py'),'--out',str(out)],check=True)
 # Append full tables to the human-readable report; do not select winners or extend budgets.
 counts=list(csv.DictReader((out/'success_summary.csv').open()));stats=read(out/'paired_statistics.json')['results']
 addition=['','## Paired numerical comparisons (percentage points)']
 for r in stats:
  if 'four_background_mean' in r['comparison'] or r['comparison'].endswith(' clean'):
   addition.append(f"- {r['comparison']}: {r['difference']*100:+.2f}; clustered95%CI [{r['CI95'][0]*100:+.2f}, {r['CI95'][1]*100:+.2f}].")
 addition+=['','## Decision boundary','These are exploratory development comparisons. No independent test or complete training-seed uncertainty is represented. An improvement over action-only without improving C is not an RLA-method advance. Single-order gains are order-dependent. No gains or harm warrants archiving this fixed target, without automatically expanding labels/losses. No follow-up is launched.']
 report=(out/'report.md').read_text()+'\n'.join(addition)+'\n'
 # A separate report avoids overwriting the initial machine-generated report.
 with (out/'final_report.md').open('x') as f:f.write(report)
 dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
 names=['final_report.md','protocol.json','success_summary.csv','paired_flips.csv','paired_statistics.json','checkpoint_inventory.csv','training_metrics.csv','validation_metrics.csv','delivery_verification.json']
 for name in names:
  text=(out/name).read_text();text=re.sub(r'/data/[^/]+/projects/rla-wm/','${PROJECT}/',text);text=re.sub(r'/data/[^/]+/','${DATA}/',text);text=re.sub(r'GPU-[0-9a-f-]{36}','${GPU_UUID}',text)
  assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',text)
  with (dest/name).open('x') as f:f.write(text)
 for name in ['statistics.json']:
  save(dest/('geometry_'+name),read(out/'labels'/name))
 for seed in [20261101,20261102]:save(dest/f'preflight_{seed}.json',read(out/f'seed_{seed}/preflight/formal/preflight_acceptance.json'))
 def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
 branch='experiments/rla-task-geometry-target';assert git('branch','--show-current')==branch
 git('add','--',str(dest.relative_to(REPO)));assert all(x.startswith(str(dest.relative_to(REPO))+'/') for x in git('diff','--cached','--name-only').splitlines())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report fixed geometry-target training and paired development evaluation')
 auth=['-c','credential.helper=','-c','credential.helper=!'+os.environ['RLA_GH_BINARY']+' auth git-credential']
 git(*auth,'push','origin','HEAD:refs/heads/'+branch);commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch).split()[0]==commit
 save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',branch=branch,commit=commit,repository='https://github.com/Lily021117/rla-background-research'))
 save(out/'completion_monitor.json',dict(status='COMPLETE',no_new_training_launched=True,optimizer_updates_by_monitor=0,report='final_report.md'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--runner-pid',type=int,required=True);a=p.parse_args();main(a.out,a.runner_pid)
