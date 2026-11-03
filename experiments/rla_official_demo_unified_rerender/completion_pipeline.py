"""Bounded source completion -> uniform review -> reuse -> incremental rendering."""
from common import *
from complete_sources import PREVIOUS
from pipeline import run,CPU,IO,CODE,SCRIPTS
import time,shutil,subprocess

def sync():
 paths=['experiments/rla_official_demo_unified_rerender','experiments/rla_official_demo_source_completion']
 subprocess.run(['git','add',*paths],cwd=CODE,check=True)
 subprocess.run(['git','diff','--cached','--check'],cwd=CODE,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=CODE).returncode:
  subprocess.run(['git','-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Complete bounded source download and uniform lineage audit with incremental render coverage'],cwd=CODE,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
 env=os.environ.copy();env['GH_CONFIG_DIR']='/data/lyc/tools/github-cli/config'
 git=['git','-c','http.version=HTTP/1.1','-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 branch='experiments/rla-official-demo-source-completion'
 try:
  subprocess.run(git+['push','origin',f'HEAD:refs/heads/{branch}'],cwd=CODE,env=env,timeout=90,check=True)
  remote=subprocess.check_output(git+['ls-remote','origin',f'refs/heads/{branch}'],cwd=CODE,env=env,timeout=90,text=True).split()[0];assert remote==commit
  dump(RUN/'final_sync.json',dict(commit=commit,remote=remote,verified=True))
 except Exception as e:dump(RUN/'final_sync.json',dict(commit=commit,verified=False,error=repr(e)))

def main():
 dump(RUN/'pipeline_runtime.json',dict(pid=os.getpid(),status='WAITING_BOUNDED_DOWNLOAD',started_unix=time.time(),no_model_commands=True))
 deadline=time.time()+7*3600
 while not (RUN/'download_complete.json').exists():
  if time.time()>deadline:raise RuntimeError('Download deadline; no extra attempts authorized')
  time.sleep(15)
 assert json.loads((RUN/'download_complete.json').read_text())['files']==40
 archive=RUN/'initial_28_file_review';archive.mkdir(exist_ok=False)
 for name in ['source_manifest.csv','lineage_audit.csv','previous_210_audit.csv','phase_audit.csv','source_rule_diff.csv']:
  p=RUN/name
  if p.exists():shutil.copy2(p,archive/name)
 run(CPU,'prepare.py')
 run(CPU,'lineage_review.py')
 run(CPU,'lineage_review.py',['--reuse'])
 run(CPU,'completion_report.py')
 # Source identity and bounded protocol are committed before incremental rendering.
 sync()
 try:run(CPU,'batch.py')
 finally:run(CPU,'completion_report.py')
 try:run(IO,'loader_check.py')
 finally:run(CPU,'completion_report.py')
 sync()
 dump(RUN/'pipeline_complete.json',dict(status='BOUNDED_DATA_WORK_COMPLETE_WITH_EXPLICIT_GAPS',time_unix=time.time(),model_forwards=0,updates=0,backward=0,policy_rollouts=0,action_replay=0))
 dump(RUN/'resource_release.json',dict(cpu_render_workers=0,loader_workers=0,gpu_allocations=0,time_unix=time.time()))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(RUN/'pipeline_failure.json',dict(error=repr(e),time_unix=time.time(),auto_restart=False));raise
