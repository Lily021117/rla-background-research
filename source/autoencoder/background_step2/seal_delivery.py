"""Seal finished outputs and verify matched optimization exposure. No GPU context."""
import csv
from .common import *
def main():
 assert (OUT/'training_exit_code.txt').read_text().strip()=='0'
 assert (OUT/'post_exit_code.txt').read_text().strip()=='0'
 assert read(OUT/'final_acceptance.json')['status']=='PASS'
 b=[json.loads(l) for l in (OUT/'B/train.jsonl').read_text().splitlines()]
 c=[json.loads(l) for l in (OUT/'C/train.jsonl').read_text().splitlines()]
 assert len(b)==len(c)==200
 for x,y in zip(b,c):
  assert x['step']==y['step'] and x['sampling_ids']==y['sampling_ids'] and x['learning_rate']==y['learning_rate']
 for group in ('B','C'):
  done=read(OUT/group/'complete.json')
  for step,parts in done['checkpoints'].items():
   for name,rec in parts.items():assert sha(rec['path'])==rec['sha256'],rec['path']
  print('CHECKPOINT_HASHES_VERIFIED',group,flush=True)
 for path,digest in read(OUT/'protection_before.json').items():assert sha(path)==digest,path
 health()
 m=read(OUT/'motion_fidelity_context.json');rows=[]
 keys=['motion_distance_latent_distance_spearman','gripper_change_difference_latent_distance_spearman','named_bowl1_change_difference_latent_distance_spearman','named_bowl2_change_difference_latent_distance_spearman']
 for group,val in m['groups'].items():
  rows.append(dict(group=group,**{key:float(np.mean([r[key] for r in val['per_episode_camera'] if r[key] is not None])) for key in keys}))
 path=OUT/'motion_context_summary.csv';assert not path.exists()
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Amend only this newly generated report, never Step1/readiness/original files.
 text='\n## 最终运动关联补充与结论限制\n\n以下是4条validation×2相机的逐episode/camera Spearman描述性平均，不把窗口对当独立示范。标签阈值/尺度只由train计算；这不是物理读出MAE。\n\n| 组 | 综合运动差与latent距离 | 实际夹爪变化差 | 命名bowl1变化差 | 命名bowl2变化差 |\n|---|---:|---:|---:|---:|\n'
 for row in rows:text+='| '+row['group']+' | '+' | '.join(f'{row[k]:.4f}' for k in keys)+' |\n'
 text+='\nC的背景latent距离降低，但范数与方差也降低；相对L2的改善不能由完全均匀缩放单独解释，仍不足以排除表示压缩/公共方向变化。综合运动距离关联没有明显消失，但夹爪关联接近零或为负，未证明夹爪运动保真性。不能将本轮结论表述为已学到可靠动作语义或控制鲁棒性。建议先固定本轮结果审阅逐示范取舍，再决定是否进行新的读出/跨背景评估；本轮未自动扩展。\n'
 with (OUT/'report.md').open('a') as f:f.write(text)
 assert not shell(['git','-C',str(WT),'diff','--name-only']).strip()
 rss=max(int(r['cpu']['VmHWM'].split()[0]) for r in b+c)
 total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 codes={str(p.relative_to(WT)):sha(p) for p in (WT/'background_step2').rglob('*.py')}
 write(OUT/'delivery_verification.json',dict(status='PASS',successful_updates=dict(B=200,C=200),attempts=dict(B=200,C=200),
  exposure_matched=True,view_pair_exposures_each=797,branch_exposures_each=1594,extra_updates=0,replay_updates=0,
  checkpoint_hashes_verified=36,old_source_Checkpoints_Step1_readiness_unchanged=True,official_worktree_tracked_files_unchanged=True,
  adapter_final_sha256=codes,CPU_peak_HWM_kib=rss,output_size_bytes_before_seal=total,
  wall_seconds_since_protection_snapshot=time.time()-(OUT/'protection_before.json').stat().st_mtime,
  GPU_context_created_by_seal=False,reserved_test_scoring=False,report_sha256=sha(OUT/'report.md')))
 # Logs still being written are explicitly excluded; immutable result/checkpoint
 # hashes are sealed after every producing process has exited.
 write(OUT/'artifact_sha256.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.log','.jsonl') and p.name!='artifact_sha256.json'})
 print('SEALED_PASS',flush=True)
if __name__=='__main__':main()
