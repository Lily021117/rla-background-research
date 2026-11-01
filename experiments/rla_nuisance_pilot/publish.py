import sys,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CODE,env=env,text=True,timeout=120).strip()
def main():
 stage=sys.argv[1];dest=HERE/('frozen_protocol' if stage=='ready' else 'delivery');dest.mkdir(exist_ok=False)
 names=['protocol.json','state_selection.json','checkpoint_inventory.json','conditions.json','candidate_pool.json','candidate_technical_checks.json','selected_state_static_acceptance.json','encoder_inventory.json','normalization_inventory.json','offline_anchors.json','state_protocol.json']
 sources=[ROOT/n for n in names] if stage=='ready' else list((ROOT/'delivery').glob('*'))
 for p in sources:
  assert p.suffix in ['.json','.csv','.md'];text=p.read_text().replace('/data/lyc/projects/rla-wm','<PROJECT_ROOT>').replace('/data/lyc/','<DATA_ROOT>/');assert not any(x in text for x in ['ghp_','github_pat_','BEGIN OPENSSH PRIVATE']);(dest/p.name).write_text(text)
 git('add','experiments/rla_nuisance_pilot');names=git('diff','--cached','--name-only').splitlines();assert names and all(n.startswith('experiments/rla_nuisance_pilot/') for n in names)
 for n in names:assert (CODE/n).stat().st_size<25*1024**2 and Path(n).suffix in ['.py','.md','.json','.csv']
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Frozen C/D nuisance pilot: '+stage)
 head=git('rev-parse','HEAD');env=os.environ.copy()
 for k in ['HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy']:env.pop(k,None)
 env['GH_CONFIG_DIR']='/data/lyc/tools/github-cli/config';auth=['-c','http.version=HTTP/1.1','-c','http.proxy=','-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential'];branch='experiments/rla-frozen-nuisance-pilot'
 git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env);remote=git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0];assert head==remote
 save(ROOT/f'github_{stage}.json',dict(commit=head,remote_verified=True,branch=branch))
 if stage=='ready':save(ROOT/'scoring_gate.json',dict(committed_before_scoring=True,remote_verified=True,commit=head,protocol_sha256=sha(ROOT/'protocol.json'),states_sha256=sha(ROOT/'state_protocol.json'),conditions_sha256=sha(ROOT/'conditions.json')))
 print('REMOTE_VERIFIED',head,flush=True)
if __name__=='__main__':main()
