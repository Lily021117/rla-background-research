"""Whitelist small sanitized documents only; never raw labels, checkpoints or rollouts."""
import os,re,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.common import *
BRANCH='experiments/rla-scratch-policy-aux'
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CODE,env=env,text=True).strip()
def main(stage):
 assert stage in ['ready','final'] and git('branch','--show-current')==BRANCH
 dest=HERE/'delivery'/ROOT.name/stage;dest.mkdir(parents=True,exist_ok=False)
 files=[ROOT/p for p in ['protocol.json','encoder_inventory.json','data_manifest.json','auxiliary_weight_schedule.csv','extraction_configuration.json','target_manifest.json','target_acceptance.json','protected_inputs.json']]
 for a in ['C','D']:files += [ROOT/'targets'/a/'statistics.json',ROOT/'targets'/a/'distribution.json']
 for order in ORDERS:
  files += [ROOT/f'seed_{order}/run_plan.json']
  p=ROOT/f'seed_{order}/preflight/formal/preflight_acceptance.json'
  if p.exists():files.append(p)
 if stage=='ready':files.append(ROOT/'preflight_report.md')
 else:files += [p for p in (ROOT/'delivery').iterdir() if p.suffix in ['.md','.json','.csv'] and p.name not in ['per_trial_results.json','per_trial_results.csv','validation_metrics.csv']]+[ROOT/'clean_gate.json']
 for p in files:
  assert p.stat().st_size<15_000_000
  text=p.read_text().replace(str(P),'<PROJECT_ROOT>').replace('/data/lyc/','<DATA_ROOT>/')
  text=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',text)
  assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',text)
  target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('x') as f:f.write(text)
 rel=str(HERE.relative_to(CODE));git('add','--',rel)
 staged=git('diff','--cached','--name-only').splitlines();assert staged and all(p.startswith(rel+'/') for p in staged)
 for p in staged:
  assert Path(p).suffix not in ['.pt','.pth','.npy','.npz','.h5','.hdf5','.mp4','.jpg','.png','.safetensors']
  assert (CODE/p).stat().st_size<15_000_000
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m',f'Scratch AE downstream policy weight control: {stage}')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
 auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 # Existing remote ancestry was previously public/authorized. Inspect newly uploaded objects too.
 objects=git('rev-list','--objects','HEAD','--not','--remotes').splitlines()
 for line in objects:
  oid=line.split()[0];size=int(git('cat-file','-s',oid));assert size<15_000_000,('large new object',oid,size)
  if len(line.split(' ',1))>1:
   name=line.split(' ',1)[1];assert not re.search(r'(^|/)(\.env[^/]*|id_rsa|id_ed25519|credentials|token)(/|$)',name)
 git(*auth,'push','origin','HEAD:refs/heads/'+BRANCH,env=env)
 commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+BRANCH,env=env).split()[0]==commit
 save(ROOT/f'github_sync_{stage}.json',dict(status='PUSHED_VERIFIED',branch=BRANCH,commit=commit,url='https://github.com/Lily021117/rla-background-research/tree/'+BRANCH,scope='source, frozen configuration, sanitized small audits/results only; no models/features/rawdata/videos'))
if __name__=='__main__':main(sys.argv[1])
