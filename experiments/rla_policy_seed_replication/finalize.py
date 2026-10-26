"""Bounded publication after verified completion; no data/weights or unrelated changes."""
import argparse,json,time,subprocess,os,re,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args();o=a.out;here=Path(__file__).parent;repo=here.parents[1]
while not (o/'report.md').exists() or not (o/'analysis.log').exists() or 'REPLICATION_ANALYSIS_COMPLETE' not in (o/'analysis.log').read_text():time.sleep(30)
v=json.loads((o/'delivery_verification.json').read_text());assert v['policy_successful_updates']==2400 and v['valid_new_rollouts']==240
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=repo,env=env,text=True).strip()
assert git('branch','--show-current')=='experiments/rla-policy-seed-replication';assert git('rev-parse','HEAD')==a.source_commit
subprocess.run(['python',str(here/'publish.py'),'--out',str(o)],cwd=repo,check=True)
delivery=here/'delivery'/o.name;relative=str(delivery.relative_to(repo))
git('add','--',relative)
staged=git('diff','--cached','--name-only').splitlines();assert staged and all(x.startswith(relative+'/') for x in staged)
for path in staged:
 b=subprocess.check_output(['git','show',':'+path],cwd=repo);assert len(b)<5_000_000
 assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|(?m:^-----BEGIN [A-Z ]*PRIVATE KEY-----$)',b)
 assert not path.endswith(('.pt','.pth','.npy','.npz','.mp4','.env'))
git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report fixed-AE paired policy-seed replication with complete recovery accounting')
env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
credential=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
git(*credential,'push','origin','HEAD:refs/heads/experiments/rla-policy-seed-replication',env=env)
commit=git('rev-parse','HEAD');remote=git(*credential,'ls-remote','origin','refs/heads/experiments/rla-policy-seed-replication',env=env).split()[0];assert commit==remote
with (o/'github_sync_report.json').open('x') as f:json.dump(dict(status='PUSHED_VERIFIED',repository='https://github.com/Lily021117/rla-background-research',visibility='public; user authorized sanitized publication',branch='experiments/rla-policy-seed-replication',commit=commit,remote_commit=remote,scope='source, configuration, sanitized report and small metrics',excluded=['weights','raw data','videos','trajectories','sidecars','credentials']),f,indent=2)
print('PUSHED_VERIFIED',commit,flush=True)
