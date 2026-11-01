"""Publish allowlisted text/summary artifacts only; never dataset/checkpoint caches."""
import sys,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_drawer_baseline_gap.common import *
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CODE,env=env,text=True).strip()
def main():
 stage=sys.argv[1];dest=HERE/('frozen_protocol' if stage=='protocol' else 'delivery');dest.mkdir(exist_ok=False)
 sources=[ROOT/'protocol.json',ROOT/'encoder_inventory.json',ROOT/'data_source_audit.json',ROOT/'historical_result_archive.json',ROOT/'policy_execution_plan.json'] if stage=='protocol' else list((ROOT/'delivery').glob('*'))
 if stage=='protocol':sources += [ROOT/f'seed_{s}/planned_run_plan.json' for s in SEEDS]
 for p in sources:
  name=(p.parent.name+'_'+p.name) if p.name=='planned_run_plan.json' else p.name
  if p.suffix=='.png':(dest/name).write_bytes(p.read_bytes());continue
  assert p.suffix in ['.json','.csv','.md'];text=p.read_text().replace('/data/lyc/projects/rla-wm','<PROJECT_ROOT>').replace('/data/lyc/','<DATA_ROOT>/')
  assert not any(x in text.lower() for x in ['ghp_','github_pat_','begin openssh private'])
  (dest/name).write_text(text)
 git('add','experiments/rla_drawer_baseline_gap')
 names=git('diff','--cached','--name-only').splitlines();assert names and all(n.startswith('experiments/rla_drawer_baseline_gap/') for n in names)
 for n in names:assert (CODE/n).stat().st_size<20*1024**2 and Path(n).suffix in ['.py','.md','.json','.csv','.png']
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m',f'Drawer baseline capability and appearance gap: {stage}')
 env=os.environ.copy()
 for k in ['HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy']:env.pop(k,None)
 env['GH_CONFIG_DIR']='/data/lyc/tools/github-cli/config'
 auth=['-c','http.proxy=','-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 branch='experiments/rla-drawer-baseline-gap';git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env)
 head=git('rev-parse','HEAD');remote=git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0];assert remote==head
 save(ROOT/('protocol_commit.json' if stage=='protocol' else 'github_final.json'),dict(commit=head,branch=branch,remote_verified=True,repository='https://github.com/Lily021117/rla-background-research'))
 print('PUSHED_VERIFIED',head,flush=True)
if __name__=='__main__':main()
