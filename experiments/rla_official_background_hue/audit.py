"""Read-only metadata/video-header coverage audit; writes only a new output directory."""
import argparse, csv, hashlib, json
from pathlib import Path
from collections import defaultdict
import h5py
import cv2
import numpy as np

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CAMS = ['agentview_camera', 'wrist_camera']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def write_csv(path, rows):
    with open(path, 'w') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    tasks=json.loads((a.root/'camera_params.json').read_text())['tasks']
    rows=[]; evidence=[]; missing=[]
    for suite in SUITES:
        root=a.root/suite; index=json.loads((root/'index.json').read_text())['trajectories']
        evidence.append({'path':str(root/'index.json'),'sha256':sha(root/'index.json')})
        disk={x.name.removeprefix('traj_') for x in root.glob('traj_*') if x.is_dir()}
        missing.append({'suite':suite,'indexed_missing_directory':sorted(set(index)-disk),'unindexed_directory':sorted(disk-set(index))})
        for tid in sorted(index,key=int):
            d=root/f'traj_{tid}'; errors=[]; task='UNMAPPED'; n=0; has_state=False
            try:
                with h5py.File(d/'metadata.h5') as f:
                    task=str(f.attrs.get('task_description','UNMAPPED')); n=len(f['action'])
                    has_state='states' in f or 'sim_state' in f
                    for k in ('qpos','target_qpos','eef_pose',*[f'{c}_{s}' for c in CAMS for s in ['intrinsics','extrinsics']]):
                        if k not in f or len(f[k])!=n: errors.append(f'metadata_length:{k}')
                evidence.append({'path':str(d/'metadata.h5'),'sha256':sha(d/'metadata.h5')})
            except Exception as e: errors.append(f'metadata:{type(e).__name__}:{e}')
            counts=[]; bytes_=0
            for c in CAMS:
                v=d/f'{c}_rgb.mp4'; cap=cv2.VideoCapture(str(v))
                count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); counts.append(count); cap.release()
                if count!=n or not count: errors.append(f'video_length:{c}:{count}')
                if v.exists(): bytes_+=v.stat().st_size
            split='val' if int(tid)<10 else 'train'; low=10 if split=='val' else 2
            if n<5: errors.append('min_num_frames')
            starts=max(0,n-low)
            if starts==0: errors.append('no_official_valid_starts')
            # Official reserves one final frame; horizon is span length, gap=h-1.
            pairs=sum(max(0,n-h) for h in range(low,65))
            mapping=tasks.get(task,{})
            rows.append(dict(suite=suite,task=task,task_bddl=mapping.get('bddl_file','UNMAPPED'),
                trajectory_id=tid,split=split,frames=n,agentview_frames=counts[0],wrist_frames=counts[1],
                valid_starts=starts if not errors else 0,possible_time_pairs=pairs if not errors else 0,
                admission='PASS_HEADERS' if not errors else 'FILTERED',reason=';'.join(errors),
                original_demo_mapping='UNKNOWN',full_sim_states_available=has_state,
                verified_background_masks=0,rgb_bytes=bytes_,source=str(d)))
    write_csv(a.output/'trajectory_manifest.csv',rows)
    for split in ('train','val'): write_csv(a.output/f'{split}_manifest.csv',[r for r in rows if r['split']==split])
    summary=[]
    for suite in SUITES:
        for task in sorted({r['task'] for r in rows if r['suite']==suite}):
            for split in ('train','val'):
                rs=[r for r in rows if r['suite']==suite and r['task']==task and r['split']==split]
                summary.append(dict(suite=suite,task=task,split=split,trajectories=len(rs),
                    admitted=sum(r['admission']=='PASS_HEADERS' for r in rs),
                    frames=sum(r['frames'] for r in rs),valid_starts=sum(r['valid_starts'] for r in rs),
                    possible_time_pairs=sum(r['possible_time_pairs'] for r in rs),verified_mask_trajectories=0))
    write_csv(a.output/'coverage_by_task.csv',summary)
    # Freeze BEFORE inspecting RGB: first admitted numeric trajectory/task;
    # three uniform starts with train h=2 / val h=10 (official feasible spans).
    selected=[]
    for suite in SUITES:
        for task in sorted({r['task'] for r in rows if r['suite']==suite}):
            candidates=[r for r in rows if r['suite']==suite and r['task']==task and r['split']=='train' and r['admission']=='PASS_HEADERS']
            if not candidates: continue
            r=candidates[0]
            for t in np.linspace(0,r['valid_starts']-1,3,dtype=int):
                selected.append({'suite':suite,'task':task,'trajectory_id':r['trajectory_id'],
                    'frames':[int(t),int(t)+1],'horizon':2,'cameras':CAMS,'split':'train','source':r['source']})
    write_json(a.output/'fixed_visualization_manifest.json',selected)
    write_json(a.output/'index_discrepancies.json',missing)
    write_json(a.output/'evidence_manifest.json',evidence)
    totals={s:{k:sum(r[k] for r in rows if r['suite']==s) for k in ('frames','valid_starts','possible_time_pairs','rgb_bytes')} for s in SUITES}
    write_json(a.output/'totals.json',totals)
    print(json.dumps({'trajectories':len(rows),'filtered':sum(r['admission']!='PASS_HEADERS' for r in rows),'visual_pairs':len(selected),'totals':totals},indent=2))

if __name__=='__main__': main()
