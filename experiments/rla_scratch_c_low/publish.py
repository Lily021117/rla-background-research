"""Add only new source and sanitized small summaries; historical delivery untouched."""
import sys,os,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_c_low.common import *
BRANCH='experiments/rla-scratch-policy-aux'
def git(*a,env=None):return subprocess.check_output(['git',*a],cwd=CODE,env=env,text=True).strip()
def main(stage):
 assert stage in ['ready','final'] and git('branch','--show-current')==BRANCH
 dest=HERE/'delivery'/ROOT.name/stage;dest.mkdir(parents=True,exist_ok=True)
 files=[ROOT/'protocol.json',ROOT/'protected_inputs.json',ROOT/'auxiliary_weight_schedule.csv']
 for order in ORDERS:files += [ROOT/f'seed_{order}/run_plan.json',ROOT/f'seed_{order}/expected_initialization.json']
 files += [ROOT/'evaluation/evaluation_protocol.json']
 if stage=='final':files += [p for p in (ROOT/'delivery').iterdir() if p.suffix in ['.json','.csv','.md'] and p.name not in ['per_trial_results.json','per_trial_results.csv','validation_metrics.csv']]
 for p in files:
  assert p.stat().st_size<10_000_000
  text=p.read_text().replace(str(P),'<PROJECT_ROOT>').replace('/data/lyc/','<DATA_ROOT>/');text=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',text)
  assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',text)
  q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert q.read_text()==text
  else:
   with q.open('x') as f:f.write(text)
 rel=str(HERE.relative_to(CODE));git('add','--',rel);staged=git('diff','--cached','--name-only').splitlines()
 assert staged and all(x.startswith(rel+'/') for x in staged)
 for name in staged:
  p=CODE/name;assert p.suffix in ['.py','.md','.json','.csv'] and p.stat().st_size<10_000_000
  assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',p.read_text())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m',f'Append fixed C-low matched completion: {stage}')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
 auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 git(*auth,'push','origin','HEAD:refs/heads/'+BRANCH,env=env);commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]==commit
 save(ROOT/f'github_sync_{stage}.json',dict(status='PUSHED_VERIFIED',commit=commit,branch=BRANCH,url='https://github.com/Lily021117/rla-background-research/tree/'+BRANCH))
if __name__=='__main__':main(sys.argv[1])
