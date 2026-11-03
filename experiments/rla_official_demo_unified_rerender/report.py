"""CPU-only manifest/metric delivery; never promotes missing results to passes."""
from common import *
import shutil, time, subprocess
def main():
 out=RUN/'delivery';out.mkdir(exist_ok=True);public=Path(__file__).parent/'delivery';public.mkdir(exist_ok=True)
 rows=list(csv.DictReader(open(RUN/'source_manifest.csv')));summaries=[];videos=[];pilots=[];colors=[];counts={k:0 for k in ['state_restores','kinematic_forwards','rgb_renders','segmentation_renders','depth_renders','physics_steps','action_replay']}
 failures=[];completed=0;totalframes=0
 for row in rows:
  p=RUN/'trajectories'/row['suite']/f"traj_{row['trajectory_id']}";result=p/'result.json';audit=p/'loader_audit.json'
  row['restored_states']=0;row['paired_frames']=0;row['loader_status']='NOT_TESTED';row['output_path']=str(p) if p.exists() else ''
  if result.exists():
   r=json.loads(result.read_text());row['render_status']=r['status'];row['restored_states']=r['frames'];row['paired_frames']=r['frames'];completed+=1;totalframes+=r['frames']
   for k in counts:counts[k]+=r['counts'].get(k,0)
   if audit.exists():row['loader_status']=json.loads(audit.read_text())['status']
   cc=list(csv.DictReader(open(p/'background_metrics.csv')))
   for cam in CAMS:
    c=[x for x in cc if x['camera']==cam];visible=[x for x in c if int(x['background_pixels'])>0]
    colors.append(dict(suite=row['suite'],trajectory_id=row['trajectory_id'],split=row['split'],camera=cam,frames=len(c),background_fraction_mean=float(np.mean([float(x['background_fraction']) for x in c])),background_rgb_mae=float(np.mean([float(x['background_rgb_mae']) for x in visible])) if visible else None,weak_frames=sum(x['visible_but_weak']=='True' for x in c),invisible_frames=len(c)-len(visible),luminance_abs_mean=float(np.mean([float(x['luminance_absolute_mean']) for x in c]))))
   if row['trajectory_id']=='000010':
    a=json.loads((p/'static_state_audit.json').read_text());pilots.append(dict(suite=row['suite'],source_demo=row['demo'],source_sha256=row['source_sha256'],frames=r['frames'],source_start_s=a['source_state_time_start'],source_end_s=a['source_state_time_end'],filtered_gaps=a['filtered_gap_count'],self_restore='PASS',pair_geometry='PASS',loader=row['loader_status'],visual='see pilot_admission.json',seconds=r['seconds']))
    videos.append(dict(suite=row['suite'],path=str(p/'full_comparison.mp4'),sha256=sha(p/'full_comparison.mp4'),frames=r['frames'],view_layout='agentview top; wrist bottom; old video | new Clean | segmentation overlay | H+S',display_fps=10,time_warning='display fps not physical sampling rate; read source_time_s'))
  elif (p/'failure.json').exists():
   failure=json.loads((p/'failure.json').read_text());row['render_status']='FAILED';failures.append(dict(suite=row['suite'],trajectory_id=row['trajectory_id'],**failure))
   for k in counts:counts[k]+=failure['counts'].get(k,0)
  elif p.exists():row['render_status']='IN_PROGRESS_OR_INCOMPLETE'
 # Include preserved failed full attempts, not as additional independent data.
 repeated_frames=0
 for p in (RUN/'failed_attempts').glob('*/*/result.json'):
  r=json.loads(p.read_text());repeated_frames+=r['frames']
  for k in counts:counts[k]+=r['counts'].get(k,0)
 counts.update(completed_unique_trajectories=completed,completed_source_frames=totalframes,paired_rgb_frames=4*totalframes,repeated_render_source_frames=repeated_frames,AE_forwards=0,DINO_forwards=0,policy_forwards=0,updates=0,backward=0,policy_rollouts=0,download_wire_bytes='not instrumented')
 for suite in SUITES:
  for split in ['train','val']:
   g=[x for x in rows if x['suite']==suite and x['split']==split]
   summaries.append(dict(suite=suite,split=split,author_requested=len(g),rlds_mapped=len(g),hdf5_multi_evidence_join=sum(x['source_status']=='MULTI_EVIDENCE_JOIN' for x in g),source_ambiguous=sum(x['source_status']=='UNRESOLVED' for x in g),source_missing=sum(x['source_status']=='MISSING_OFFICIAL_HDF5' for x in g),split_conflicts=sum(x['split_conflict']=='True' for x in g),paired_complete=sum(int(x['paired_frames'])>0 for x in g),paired_source_frames=sum(int(x['paired_frames']) for x in g),loader_pass=sum(x['loader_status']=='PASS' for x in g)))
 csvout(out/'full_status.csv',rows);csvout(out/'coverage.csv',summaries)
 if pilots:csvout(out/'pilot_summary.csv',pilots)
 if colors:csvout(out/'background_summary.csv',colors)
 if videos:csvout(out/'video_index.csv',videos)
 dump(out/'counts.json',counts);dump(out/'batch_failures.json',failures)
 batchdone=(RUN/'batch_complete.json').exists();status='PARTIAL_DATA_BUILD_COMPLETE' if batchdone else 'IN_PROGRESS_NOT_FINAL'
 evidence=[RUN/'protocol.json',RUN/'source_manifest.csv',RUN/'source_files.json',RUN/'source_coverage.csv',RUN/'split_conflicts.json',Path(__file__),OLD/'libero_official/scripts/create_dataset.py']
 for name in ['protocol.json','source_files.json','source_coverage.csv','split_conflicts.json','download_plan.json','download_complete.json','rejected_source_files.json','pilot_admission.json','resource_estimate.json','visual_state_repair_record.json','batch_frozen_manifest.csv','batch_complete.json','loader_audit.json']:
  p=RUN/name
  if p.exists():evidence.append(p);shutil.copy2(p,out/name)
 for p in (RUN/'trajectories').glob('*/*/result.json'):
  evidence.extend([p,p.parent/'static_state_audit.json',p.parent/'appearance.json',p.parent/'frame_mapping.csv',p.parent/'camera_time.json'])
 dump(out/'evidence_manifest.json',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in evidence])
 lines=['|套件|划分|原清单|HDF5联合证据对应|歧义|文件缺失/未核验|配对完成|loader通过|','|---|---|---:|---:|---:|---:|---:|---:|']
 for x in summaries:lines.append(f"|{x['suite']}|{x['split']}|{x['author_requested']}|{x['hdf5_multi_evidence_join']}|{x['source_ambiguous']}|{x['source_missing']}|{x['paired_complete']}|{x['loader_pass']}|")
 text=f'''# 官方示范统一重渲染派生版

状态：**{status}**。原始追溯范围始终为 **1653训练＋40验证**；当前完成 **{completed} 条、{totalframes} 个源状态时刻**的双相机配对生成。不能把本表的 RLDS 对应数量称为原始 demo 全量确认，也不能把四小样通过称为全量通过。本轮不具备自动启动训练的授权。

## 新旧协议边界

父实验 `b581d02405f70e6b6f7992d80f44b0dadb0662b7` 保持 PARTIAL / 原视频对应失败。新协议直接恢复官方 HDF5 `states`，不使用旧重放候选状态；不再要求与作者旧视频逐像素/逐时刻精确对应。旧视频只作描述性旁列，不是新准入门槛。历史文件未改写。

{chr(10).join(lines)}

## 来源证据与未完成项

全量逐条 `full_status.csv` 记录 RLA、RLDS、HDF5、demo、过滤原始索引、联合证据和歧义。依据完整过滤动作逐元素一致、语言/来源路径和全长机器人关节路径的独立比较，不只看单个动作哈希。关节路径按进度对齐仅是身份辅助证据，不用于输出状态插值。所有50条源 demo 的候选排序留本地 `mapping_evidence`。没有获得未发布 regenerated HDF5 的字节证据；联合对应证据的边界如实保留。

来源文件按固定官方 HF revision 和 LFS SHA-256 核验；不合格下载隔离，不用其内容。未映射条目、缺失文件或划分冲突不补数、不重新划分。现有子集未发现冲突不等于未核验部分也无冲突。来源/覆盖未完成时，不能把子集当作原全量直接开训。

## 原始状态与画面

固定官方 LIBERO `scripts/create_dataset.py` 先执行 action 后取 obs，而最终输出 `states[valid_index]` 是对应动作前状态。本派生版从该状态重建同步 RGB、FK proprio 与实际相机；不复制源 RGB/proprio 的行相位。第一帧保留真实过滤索引和源时间，Goal 原索引从1开始是记录在案的 noop 过滤，不是为了对齐视频删帧。

每个 demo 使用其 `model_file`；只把完整逻辑资产相对路径映射到固定官方本地资产。记录 XML、资产和编译模型哈希。原始历史资产字节未公开，不能宣称历史渲染环境完全相同。状态维度、XML关节顺序、有限性、单调时间及静态渲染相关字段受检。模型没有 mocap/plugin 未存动态状态时才通过本实现。控制器历史不作为静态渲染自动阻塞，也不宣称能无缝动力学续跑。

四小样每帧同时检查 MJB 重载 RGB完全一致、状态不推进、两外观几何/相机一致；固定7时刻检查深度一致。AI查看了每条固定7时刻×双视角对比图，不是人工逐帧看完整视频。完整视频另存，视觉抽查范围见 `pilot_admission.json`。

LIBERO-10 第一遍发现按钮开启但炉圈显示关闭：源 XML 不包含逐帧Python视觉状态更新。已按固定官方 FlatStove `button qpos>=0.5` 与 site alpha 规则修复，两路同样更新；这不是改光照或物理参数。第一遍383帧完整记录保留于 `failed_attempts`，纠正渲染单列计算，不冒充新增轨迹。未通过视觉门控之前未启动批量。

## 背景颜色与时序接口

固定 ΔH=0.20，S′=S+0.35(1−S)，不主动修改V；轨迹所有时刻和双相机共用。只改桌面/墙面/地面白名单纹理字节；任务物体、机器人、篮子、可操作柜体排除。先确认每个纹理全部引用者属于背景；前景共享则失败关闭。所有其他编译数组在副本创建时逐字节相同；每帧确定性视觉状态也在两路一致。光源、反射、阴影、碰撞与动力学未改。背景反射可改变前景RGB，不以“前景RGB不变”假冒物理一致性。

颜色、亮度、可见背景面积、弱变化/不可见帧见 `background_summary.csv` 和逐帧 `background_metrics.csv`。绿色外观是一次冻结的工程参数，不是通过策略筛选的扰动。

逐帧 `frame_mapping.csv` 和 `camera_time.json` 保留真实时间及原始索引；官方h仍是保留帧计数（未来索引t+h−1、t+h<N），不是固定物理时长。`PairedInterface` 同时返回真实间隔，不跨episode、不补末帧。`root_poses` 仅保留官方恒等占位兼容字段，非已测机器人基座；它不参与本轮FK与相机计算。两路 `metadata.h5` 的实际相机/FK与动作已一致核验。

后续 Base用(Clean,Clean)，Aug/Cons用(Clean,Background)，共享来源、划分、时间采样、相机和Clean处理。本轮只运行固定官方数据loader，无模型初始化、DINO缓存或训练。

## 计数、资源与可复用性

累计完成数据的物理时刻 {totalframes}，保存配对双相机RGB {4*totalframes} 张；失败技术版本重复渲染 {repeated_frames} 个源时刻，不作为新数据。实际状态恢复 {counts['state_restores']}；运动学forward {counts['kinematic_forwards']}（不是学习模型前向）；RGB渲染 {counts['rgb_renders']}；分割 {counts['segmentation_renders']}；深度 {counts['depth_renders']}。运行中的未完成计数不并入已完成结果，最终以批量记录核对。

源动作重放=0，物理积分=0；AE/DINO/策略前向=0，更新=0，backward=0，策略闭环=0。CPU OSMesa，不申请GPU；估计和实时记录见 `resource_estimate.json` / `batch_runtime.json`。下载量区分唯一校验文件字节与未计量网络重传，详见下载记录。

数据有效性不等于方法有效；本轮不产生控制收益结论。完成确认子集不意味着1693条身份和划分已全部解决；明确缺口以后续来源资料补齐，而非换示范或改开训范围。
'''
 (out/'report.md').write_text(text)
 # Public artifacts exclude raw states/images/models, per-sample action arrays and videos.
 for p in out.iterdir():
  if p.suffix not in ['.json','.csv','.md']:continue
  content=p.read_text().replace(str(ROOT),'<PROJECT>').replace('/data/lyc/projects/vla-adapter-ref','<ENV_PROJECT>')
  (public/p.name).write_text(content)
 dump(RUN/'delivery_status.json',dict(status=status,completed=completed,frames=totalframes,time_unix=time.time(),public=str(public)))
 print(status,completed,totalframes,flush=True)
if __name__=='__main__':main()
