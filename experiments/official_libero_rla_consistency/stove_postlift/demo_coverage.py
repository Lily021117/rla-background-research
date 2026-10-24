import argparse,collections,mujoco
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();manifest=read(S/'frozen_dataset_manifest.json');order=read(S/'policy_run/sampling_order.json');counts=collections.Counter((e,t) for e,t in order[:4800]);rows=[];samples=[]
for e in manifest['episodes']:
 root=Path(e['root']);lab=np.load(root/'trajectory_labels.npz');xml=S/'source_models'/e['source_demo']/'clean.xml';m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);bid=m.body('akita_black_bowl_1_main').id;heights=[]
 for state in lab['states']:
  d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d);heights.append(float(d.xpos[bid,2]))
 lifted=np.flatnonzero(np.array(heights)-heights[0]>=.03);first=int(lifted[0]) if len(lifted) else None;N=e['valid_chunks'];post=max(N-first,0) if first is not None else 0;exposure=sum(counts[e['episode'],t] for t in range(N));post_exposure=sum(counts[e['episode'],t] for t in range(first or N,N))
 rows.append(dict(episode=e['episode'],split=e['split'],frames=len(heights),valid_chunks=N,first_lift=first,postlift_windows=post,postlift_fraction=post/N,sampled_physical_windows=exposure,sampled_postlift=post_exposure,sampled_postlift_fraction=post_exposure/exposure if exposure else None,source_xml_sha256=sha(xml)))
 if first is not None:
  for t in sorted(set(min(v,N-1) for v in (first,first+8,first+24))):
   for appearance in ('clean','background'):samples.append(dict(source='demo',episode=e['episode'],split=e['split'],t=t,appearance=appearance,root=str(root)))
csvsave(a.out/'demo_phase_coverage.csv',rows);save(a.out/'fixed_demo_samples.json',samples)
save(a.out/'coverage_verification.json',dict(episodes=len(rows),train_episodes=sum(x['split']=='train' for x in rows),actual_sampling_entries=min(len(order),4800),train_postlift_exposures=sum(x['sampled_postlift'] for x in rows),threshold_source='previous frozen .03m lift proxy, not official success',new_rollouts=0))
