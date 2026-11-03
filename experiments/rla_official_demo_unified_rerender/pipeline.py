"""CPU-only bounded continuation after pilot admission. No training commands."""
from common import *
import subprocess, shutil, time, argparse
CODE=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).parent
CPU='/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python'
IO='/data/lyc/projects/rla-wm/.venv/bin/python'
def run(python,script,args=()):
 log=RUN/'pipeline_logs'/f'{script}_{int(time.time())}.log';log.parent.mkdir(exist_ok=True)
 with open(log,'x') as f:r=subprocess.run([python,'-u',str(SCRIPTS/script),*args],cwd=CODE,stdout=f,stderr=subprocess.STDOUT)
 dump(RUN/'pipeline_stage.json',dict(script=script,exit_code=r.returncode,log=str(log),time_unix=time.time()))
 if r.returncode:raise RuntimeError(f'{script} failed: {r.returncode}; see {log}')
def main():
 assert json.loads((RUN/'pilot_admission.json').read_text())['all_four_pass']
 dump(RUN/'pipeline_runtime.json',dict(pid=os.getpid(),started_unix=time.time(),status='WAITING_FOR_FINITE_DOWNLOAD',no_model_commands=True))
 # Await the already authorized finite 40-file list, not a new download scope.
 deadline=time.time()+4*3600
 while not (RUN/'download_complete.json').exists():
  if time.time()>deadline:raise RuntimeError('Four-hour download wait bound reached; do not freeze moving source list')
  time.sleep(15)
 archive=RUN/'initial_local_inventory';archive.mkdir(exist_ok=False)
 for name in ['source_manifest.csv','source_coverage.csv','source_files.json','split_conflicts.json']:
  shutil.copy2(RUN/name,archive/name)
 run(CPU,'prepare.py')
 run(CPU,'report.py')
 run(CPU,'batch.py')
 try:run(IO,'loader_check.py')
 finally:run(CPU,'report.py')
 dump(RUN/'pipeline_complete.json',dict(status='DATA_SCOPE_FINISHED_WITH_REPORTED_EXCLUSIONS',time_unix=time.time(),gpu_use=False,all_models_forwards_updates_backward_rollouts=0))
 # Never stage raw run artifacts; only dedicated code and sanitized delivery.
 subprocess.run(['git','add','experiments/rla_official_demo_unified_rerender'],cwd=CODE,check=True)
 subprocess.run(['git','diff','--cached','--check'],cwd=CODE,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=CODE).returncode:
  subprocess.run(['git','-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Complete confirmed official-demo paired rerender subset and disclose unresolved lineage'],cwd=CODE,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
 env=os.environ.copy();env['GH_CONFIG_DIR']='/data/lyc/tools/github-cli/config'
 git=['git','-c','http.version=HTTP/1.1','-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 branch='experiments/rla-official-demo-unified-rerender'
 try:
  subprocess.run(git+['push','origin',f'HEAD:refs/heads/{branch}'],cwd=CODE,env=env,timeout=90,check=True)
  remote=subprocess.check_output(git+['ls-remote','origin',f'refs/heads/{branch}'],cwd=CODE,env=env,timeout=90,text=True).split()[0];assert remote==commit
  dump(RUN/'final_sync.json',dict(commit=commit,remote=remote,verified=True))
 except Exception as e:dump(RUN/'final_sync.json',dict(commit=commit,verified=False,error=repr(e)))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(RUN/'pipeline_failure.json',dict(error=repr(e),time_unix=time.time(),no_auto_training=True));raise
