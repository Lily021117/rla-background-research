"""Persistent, non-reserving GPU0 gate for the frozen-encoder preparation stage.

No scheduler reservation is implied. Training remains gated by its own interface
acceptance; this process never starts unvalidated training or another GPU.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

UUID = 'GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
MIN_MIB = 26624


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--attempt', default='gpu0_queue')
    a = p.parse_args()
    q = a.run / a.attempt
    q.mkdir(exist_ok=True)
    lock = (q / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    script = Path(__file__).with_name('extract_frozen.py')
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    baseline = None

    def status(state, **kw):
        row = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   pid=os.getpid(), state=state, gpu_uuid=UUID,
                   minimum_free_MiB=MIN_MIB, stage='frozen_encoder_sidecars', **kw)
        with (q / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        tmp = q / 'status.tmp'
        tmp.write_text(json.dumps(row, indent=2))
        tmp.replace(q / 'status.json')
        print(json.dumps(row), flush=True)

    try:
        stable = 0
        while True:
            identity = subprocess.check_output([
                'nvidia-smi', '-i', UUID,
                '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
            assert identity[1].strip() == UUID
            free = int(identity[2].strip())
            ecc = subprocess.check_output(['nvidia-smi', '-i', UUID, '-q', '-d', 'ECC'], text=True)
            counters = [s.strip() for s in ecc.splitlines() if ':' in s and any(
                k in s for k in ('Correctable', 'Uncorrectable', 'SRAM SM', 'Threshold Exceeded'))]
            assert all(int(s.rsplit(':', 1)[1]) == 0 for s in counters if 'Uncorrectable' in s)
            assert all(s.rsplit(':', 1)[1].strip() == 'No' for s in counters if 'Threshold Exceeded' in s)
            if baseline is None:
                baseline = counters
            assert counters == baseline, 'ECC counters changed while waiting'
            manifest = a.run / 'dataset_v2/frozen_dataset_manifest.json'
            ready = manifest.exists() and json.loads(manifest.read_text())['status'] == 'PASS'
            stable = stable + 1 if free >= MIN_MIB and ready else 0
            status('WAITING_GPU' if ready else 'WAITING_DATA_ADMISSION',
                   free_MiB=free, stable_checks=stable, ecc=counters)
            if stable >= 3:
                break
            time.sleep(30)
        assert hashlib.sha256(script.read_bytes()).hexdigest() == digest, 'Queued entry changed; review required'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=UUID, RLA_GPU_UUID=UUID, RLA_MIN_FREE_GIB='26',
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        cache = a.run / 'cache'
        for variable, folder in {'HF_HOME':'huggingface', 'HF_HUB_CACHE':'huggingface/hub',
                                 'TORCH_HOME':'torch', 'XDG_CACHE_HOME':'xdg', 'TMPDIR':'tmp',
                                 'CUDA_CACHE_PATH':'cuda', 'TRITON_CACHE_DIR':'triton',
                                 'MPLCONFIGDIR':'matplotlib'}.items():
            path = cache / folder
            path.mkdir(parents=True, exist_ok=True)
            env[variable] = str(path)
        command = [a.python, '-u', str(script), '--project', str(a.project),
                   '--dataset', str(a.run / 'dataset_v2'), '--out', str(a.run / 'frozen_sidecars')]
        status('RUNNING', command=command, entry_sha256=digest)
        with (q / 'extract.log').open('x') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status('BLOCKED_STAGE_FAILURE', returncode=result.returncode)
            return
        status('PREPARATION_COMPLETE', policy_updates=0,
               next_gate='Policy initialization/full-batch acceptance required before training')
    except Exception as exc:
        status('BLOCKED', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
