"""CPU postprocessing only, after the complete frozen rollout ledger is accepted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
src = Path(__file__).parent
env = os.environ.copy()
cache = '/data/lyc/tmp/rla_background_invariance/cache'
env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
           HF_HOME=cache+'/huggingface', TORCH_HOME=cache+'/torch',
           XDG_CACHE_HOME=cache+'/xdg', TMPDIR=cache+'/tmp',
           MPLCONFIGDIR=cache+'/matplotlib', OPENBLAS_NUM_THREADS='4',
           OMP_NUM_THREADS='4', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa',
           LP_NUM_THREADS='2', PYTHONPATH='/data/lyc/projects/LIBERO-plus',
           LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero')

def accepted(name):
    path = a.out/name
    while not path.exists():
        time.sleep(5)
    result = json.loads(path.read_text())
    assert result['returncode'] == 0, result

def run(name):
    with (a.out/(name+'.log')).open('x') as log:
        result = subprocess.run([
            '/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python', '-u',
            str(src/(name+'.py')), '--out', str(a.out)],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (name, result.returncode)
    print(name, 'COMPLETE', flush=True)

accepted('evaluate_gpu0_process.json')
run('analyze')
accepted('diagnostics_gpu0_process.json')
run('finalize')
print('CPU_POSTPROCESSING_COMPLETE', flush=True)
