"""Explicit allowlist of text metadata only. Never publishes state/image payloads."""
import argparse,json,os,subprocess
from pathlib import Path

def publish(root,stage):
 code=Path(__file__).resolve().parents[2];dest=code/'reports/frozen1800_generated_states'/stage;dest.mkdir(parents=True,exist_ok=False)
 project=os.environ['RLA_PROJECT'];libero=os.environ['LIBERO_ROOT']
 def sanitized(text):
  for a,b in [(project,'<PROJECT_ROOT>'),(libero,'<LIBERO_ROOT>'),('/data/lyc/projects','<PROJECTS_ROOT>'),('/data/lyc','<DATA_ROOT>')]:text=text.replace(a,b)
  return text
 files=['source_content_audit.json','sealed_content_access.json','frozen_inputs.json','protocol_revision_notes.md','phase_aware_dedup.json','pre_scoring_acceptance.json']
 files += ['generation_protocol.json','route_b_v2/generation_protocol.json','route_b_v3/generation_protocol.json','route_b_v3/generation_content_reads.json','route_b_v3/candidate_results.json','route_b_v3/state_protocol.json']
 files += ['evaluation/evaluation_protocol.json','evaluation/checkpoint_inventory_frozen.json']
 if stage=='complete':files += [str(p.relative_to(root)) for p in (root/'delivery').iterdir() if p.suffix in ('.json','.csv','.md')]+['process.json','resource_release.json']
 for name in files:
  p=root/name
  if not p.exists():continue
  target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);text=sanitized(p.read_text())
  assert len(text)<4_000_000 and '/data/lyc' not in text
  target.write_text(text)
 subprocess.run(['git','add','experiments/frozen1800_state_readiness',str(dest.relative_to(code))],cwd=code,check=True)
 subprocess.run(['git','diff','--cached','--check'],cwd=code,check=True)
 subprocess.run(['git','-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Frozen1800 generated-state validation: '+stage],cwd=code,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=code,text=True).strip()
 env=os.environ.copy()
 for k in ('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy'):env.pop(k,None)
 gh=os.environ['GH_BIN'];cmd=['git','-c','http.proxy=','-c','credential.helper=','-c','credential.helper=!'+gh+' auth git-credential']
 subprocess.run(cmd+['push','origin','HEAD:refs/heads/'+branch],cwd=code,env=env,check=True)
 remote=subprocess.check_output(cmd+['ls-remote','origin','refs/heads/'+branch],cwd=code,env=env,text=True).split()[0];assert remote==commit
 record=dict(stage=stage,branch=branch,commit=commit,remote_commit=remote,scope='allowlisted sanitized code and text summaries only')
 with (root/('github_sync_'+stage+'.json')).open('x') as f:json.dump(record,f,indent=2)
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['ready','complete'],required=True);a=p.parse_args();publish(a.root,a.stage)
