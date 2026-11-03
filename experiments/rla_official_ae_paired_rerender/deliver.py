"""CPU summary of saved evidence only; no environment/model construction."""
import csv,json,shutil,sys,hashlib,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from prepare import RUN,ROOT,OLD,OFFICIAL,sha,dump,csvout
sys.path.insert(0,str(Path(__file__).parents[2]))
from experiments.rla_official_background_hue.paired_hue import hue_parameter
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
def read(p):return json.loads(Path(p).read_text())
def main():
 out=RUN/'delivery';out.mkdir(exist_ok=True);source=list(csv.DictReader(open(RUN/'source_mapping.csv')))
 if (out/'report.md').exists() and not (RUN/'delivery_draft_before_final_pixel_gate').exists():
  shutil.copytree(out,RUN/'delivery_draft_before_final_pixel_gate')
 final_pair=read(RUN/'paired_rendered/libero_object/final_admission.json')['status']=='PASS_RECONSTRUCTED_PAIR'
 maps={};pilots={};validations=[];videos=[];evidence=[];counts=defaultdict(int)
 for suite in SUITES:
  complete=read(RUN/'rlds_mapping_v2'/suite/'complete.json');assert complete['records']==complete['passed']
  records=complete['results'];assert len(set(r['trajectory_id'] for r in records))==len(records)
  maps.update({(suite,r['trajectory_id']):r for r in records})
  p=RUN/'pilot_replay'/suite;r=read(p/'result.json');frames=list(csv.DictReader(open(p/'frame_alignment.csv')));pilots[suite]=r
  for k,v in r['counters'].items():counts[k]+=v
  counts['replay_trajectories']+=1;counts['candidate_frames']+=r['frames']
  v=dict(suite=suite,trajectory_id='000010',demo_candidate=r['source_candidate'],frames=r['frames'],numeric_gate=r['numeric_gate'],pixel_gate=r['pixel_gate'],
   max_joint_error=r['max_joint_error'],max_eef_error_m=r['max_eef_error'],front_max_mae=max(float(f['front_mae']) for f in frames),wrist_max_mae=max(float(f['wrist_mae']) for f in frames),
   front_max_p99=max(float(f['front_p99']) for f in frames),wrist_max_p99=max(float(f['wrist_p99']) for f in frames),source_state_origin='replay candidate',
   original_saved_states_acquired=0,rolling_replay_visual_pass=suite=='libero_object',clean_admitted=suite=='libero_object' and final_pair,paired_admitted=suite=='libero_object' and final_pair)
  validations.append(v)
  videos.append(dict(suite=suite,kind='original_vs_candidate_full_sequence',path=str(p/'full_clean_comparison.mp4'),sha256=sha(p/'full_clean_comparison.mp4'),frames=r['frames'],camera_layout='front pair top, wrist pair bottom',display_fps=10,physical_control_hz=20,status=r['status']))
  for name in ['result.json','environment.json','source_model_identity.json','camera_and_time_audit.json','asset_and_model_comparison.json','frame_alignment.csv']:
   evidence.append(p/name)
  evidence.extend([p/'candidate_replay.npz',p/'original_source.xml',p/'runtime_model.xml'])
  dest=out/'pilot_evidence'/suite;dest.mkdir(parents=True,exist_ok=True)
  # Public summaries omit full raw XML, simulator state arrays and private upstream env paths.
  for name in ['result.json','environment.json','camera_and_time_audit.json','frame_alignment.csv']:
   shutil.copy2(p/name,dest/name)
  shutil.copy2(RUN/'paired_prepare'/suite/'background_whitelist.json',dest/'background_whitelist.json')
  audit=read(p/'asset_and_model_comparison.json');dump(dest/'asset_and_model_comparison.json',{k:v for k,v in audit.items() if k not in ['original_xml_private_paths']})
 rows=[]
 for r in source:
  s=r['suite'];tid=r['trajectory_id'];m=maps[(s,tid)];pilot=tid=='000010';generated=pilot and s=='libero_object';admitted=generated and final_pair
  dh,key=hue_parameter(20261103 if r['split']=='train' else 20261104,r['split'],dict(suite=s,trajectory_id=tid,data_version='official_ae_rerender_v1'),0)
  rows.append(dict(suite=s,split=r['split'],trajectory_id=tid,task=r['task'],frames=int(r['frames']),valid_starts=int(r['valid_starts']),possible_time_pairs=int(r['possible_time_pairs']),
   source=r['source'],metadata_sha256=r['metadata_sha256'],rlds_enumeration_index=m['enumeration_index'],rlds_mapping_status=m['rlds_match'],
   rlds_episode_file_path='regenerated--no_noops/'+s+'/'+m['hdf5_basename'],upstream_absolute_file_path_sha256=hashlib.sha256(m['episode_file_path'].encode()).hexdigest(),
   rlds_full_numeric_sequence_verified=True,rgb_frame_checks_per_camera=len(m['images']['agentview_camera']['frames']),rgb_all_frames_checked=pilot,
   agentview_sha256=m['images']['agentview_camera']['video_sha256'],wrist_sha256=m['images']['wrist_camera']['video_sha256'],
   hdf5_demo_candidate=pilots[s]['source_candidate'] if pilot else 'UNKNOWN',hdf5_demo_status='ADMITTED_RECONSTRUCTION_UNDER_FROZEN_TOLERANCE' if admitted else ('UNIQUE_ACTION_CANDIDATE_NOT_CONFIRMED' if pilot else 'NOT_JOINED_TO_DEMO'),
   state_status='REPLAY_CANDIDATES_'+str(pilots[s]['frames']) if pilot else 'NOT_RECOVERED',original_saved_state_available=False,
   clean_status='PASS_NUMERIC_PIXEL_AI_SEQUENCE_REVIEW' if admitted else ('BLOCKED_FINAL_RESTORED_CLEAN_PIXELS' if generated else (pilots[s]['status'] if pilot else 'NOT_TESTED')),
   paired_status='PASS_PAIR_AND_OFFICIAL_LOADER' if admitted else ('GENERATED_CANDIDATE_NOT_ADMITTED' if generated else 'BLOCKED_NOT_GENERATED'),
   delta_h=dh,hue_seed=20261103 if r['split']=='train' else 20261104,hue_key=key,augmentation_generated=generated,paired_admitted=admitted,
   frame_rule='converted t = RLDS step t; HDF5 mapping unknown unless pilot frame_alignment.csv',
   time_pair_rule='train h=2..64; val h=10..64; t+h<N; RGB frames [t,t+h-1]'))
 assert len(rows)==1693 and sum(r['split']=='train' for r in rows)==1653
 csvout(out/'source_mapping.csv',rows)
 coverage=[]
 for suite in SUITES:
  for split in ['train','val']:
   rs=[r for r in rows if r['suite']==suite and r['split']==split]
   coverage.append(dict(suite=suite,split=split,trajectories=len(rs),frames=sum(r['frames'] for r in rs),
    valid_starts=sum(r['valid_starts'] for r in rs),possible_time_pairs=sum(r['possible_time_pairs'] for r in rs),rlds_sources_matched=len(rs),
    hdf5_demo_candidates=sum(r['hdf5_demo_candidate']!='UNKNOWN' for r in rs),candidate_state_sequences=sum(r['state_status']!='NOT_RECOVERED' for r in rs),
    original_state_sequences=0,clean_correspondence_pass=sum(r['paired_admitted'] for r in rs),paired_sequences=sum(r['paired_admitted'] for r in rs),candidate_pairs_generated=sum(r['augmentation_generated'] for r in rs)))
 csvout(out/'coverage_by_suite_split.csv',coverage);csvout(out/'pilot_validation.csv',validations)
 p=RUN/'paired_rendered/libero_object';paired=read(p/'result.json');export=read(p/'export_audit.json');loader=read(p/'loader_audit.json');assert loader['status']=='PASS'
 assert read(p/'final_admission.json')['status'] in ['PASS_RECONSTRUCTED_PAIR','BLOCKED_FINAL_CLEAN_PIXELS']
 for k in ['state_restores','render_calls','simulation_steps']:counts[k]+=paired[k]+export[k]
 counts['pair_and_export_integration_steps']=counts.pop('simulation_steps')
 counts['kinematic_forwards_for_export']=export['kinematic_forwards'];counts['paired_sequences']=1;counts['paired_physical_frames']=143;counts['paired_rgb_frames']=572
 counts['paired_sequences_generated_candidate']=counts.pop('paired_sequences');counts['paired_sequences_admitted']=int(final_pair)
 for k in ['AE_forwards','DINO_forwards','policy_forwards','AE_updates','DINO_updates','policy_updates','backward','policy_closed_loops','SAM_forwards']:counts[k]=0
 counts['rolling_replay_reproduction_failed_sequences']=3;counts['restored_clean_final_gate_failed_sequences']=int(not final_pair);counts['gpu_jobs']=0
 dump(out/'execution_counts.json',dict(counts));csvout(out/'pairing_manifest.csv',list(csv.DictReader(open(p/'pairing_manifest.csv'))))
 for name in ['result.json','export_audit.json','loader_audit.json','same_state_render_checks.csv','final_admission.json','final_clean_original_pixel_check.csv']:
  shutil.copy2(p/name,out/('paired_'+name));evidence.append(p/name)
 evidence.extend([p/'clean.npz',p/'background.npz',p/'camera_poses.npz',p/'pairing_manifest.csv'])
 shutil.copy2(RUN/'pilot_replay/libero_object/visual_admission.json',out/'object_visual_admission.json')
 videos.append(dict(suite='libero_object',kind='original_clean_hue_full_sequence',path=str(p/'full_original_clean_hue.mp4'),sha256=sha(p/'full_original_clean_hue.mp4'),frames=143,camera_layout='front original|clean|hue above wrist original|clean|hue',display_fps=20,physical_control_hz=20,status=read(p/'final_admission.json')['status']))
 csvout(out/'video_index.csv',videos)
 download=read(RUN/'rlds_download_complete.json');hf=read(RUN/'hdf5_download_complete.json');newbytes=download['bytes']+hf['total_bytes']
 for name in ['frozen_protocol.json','initial_evidence.json','sam_pause.json','rlds_download_complete.json','hdf5_download_complete.json','hdf5_download_plan.json']:
  shutil.copy2(RUN/name,out/name);evidence.append(RUN/name)
 oldloader=OLD/'loader_validation.json';evidence.append(oldloader)
 dump(out/'fair_comparison_interface.json',dict(status='DATA_INCOMPLETE_DO_NOT_TRAIN',scope_trajectories=1693,train=1653,val=40,
  base='same admitted rerender Clean as augmented arms; never compare untouched original videos against only-augmented rerenders without disclosing renderer confound',
  aug='Clean + hue of same restored full state; independent RNG fixed per trajectory',aug_cons='same data as aug; future loss choice not executed',
  geometry='per-frame actual c2w stored under official extrinsics/w2c convention and negative fx; wrist never static',
  sampling='official uniform suite/trajectory/start and feasible horizon; original IDs/splits, no final-frame fabrication or cross-episode pairs',
  validation='fixed independent hue seed 20261104; zero val augmented trajectories generated',
  prohibition='one passing prototype does not authorize subset AE training or a different split'))
 errors=[dict(stage='initial RLDS join',issue='Initially assumed enumeration==converted ID and PIL JPEG decode; first 32 numeric joins passed but exact image criteria failed and later enumeration diverged.',fix='Whole-sequence numeric signature join + actual TFDS TensorFlow decoder; final 1693 joins verified. Initial files retained in rlds_mapping/.'),
  dict(stage='source mapping engineering',issue='Missing Path import in one attempt',fix='Import fixed; no simulator/model computation in failed attempt.'),
  dict(stage='network',issue='Official HF truncated transfers / proxy TLS errors; initial RLDS and HDF5 commands failed.',fix='Pinned same-origin resumable retries, verify final sizes/SHA-256; no mirror substitution. Cache and error JSON retained.'),
  dict(stage='CPU job interruption',issue='One resumed RLDS download and two Object mapping attempts ended with exit143; cause not established.',fix='Verified files reused; Object mapping resumes persisted record prefix after rechecking actions/state/joints and identities. No physics replay repeated.'),
  dict(stage='official loader smoke',issue='Test initially used field extrinsics; official output field is w2c. One manually chosen end handle included the final frame reserved by official sampler.',fix='Use actual w2c; freeze legal boundary start=N-65,h=64; prior boundary audit preserved. No original or rendered RGB changed.' )]
 errors.append(dict(stage='final restored-Clean direct gate',issue='Continuous Object replay passed, but final state-restored Clean vs original fails p99<=40 at 12 wrist frames (84-92,96,99,139), max56. Pair vs rolling replay tolerance alone is not transitive.',fix='No rerender retry, threshold relaxation or frame removal. Candidate data retained but final admission revoked. Earlier delivery draft archived separately; final public report corrected.'))
 dump(out/'incidents.json',errors)
 obs=list(csv.DictReader(open(p/'same_state_render_checks.csv')));color={cam:dict(mean_rgb_abs_change=float(np.mean([float(x[cam+'_color_change']) for x in obs])),zero_change_frames=sum(float(x[cam+'_color_change'])==0 for x in obs)) for cam in ['front','wrist']}
 dump(out/'color_effect.json',dict(per_view=color,interpretation='Object arena has no table_visual; whitelist changes six walls only. Floor excluded. Wrist wall region invisible; no compensation with saturation/brightness or new geom.'))
 projected_seconds=paired['seconds']/143*sum(r['frames'] for r in rows)
 replay_seconds=sum(r['seconds'] for r in pilots.values())/sum(r['frames'] for r in pilots.values())*sum(r['frames'] for r in rows)
 dump(out/'resource_estimate.json',dict(new_verified_source_bytes=newbytes,network_actual_wire_bytes='UNKNOWN: retry/partial transfer overhead not instrumented',
  rlds_new_verified_bytes=download['bytes'],hdf5_new_verified_bytes=hf['total_bytes'],git_checkout_disk_bytes=sum(f.stat().st_size for f in (RUN/'libero_official').rglob('*') if f.is_file()),
  disk_free_bytes=shutil.disk_usage(RUN).free,total_published_frames=sum(r['frames'] for r in rows),uncompressed_two_appearances_two_views_RGB_bytes=sum(r['frames'] for r in rows)*256*256*3*4,
  hypothetical_single_worker_replay_hours=replay_seconds/3600,hypothetical_single_worker_paired_render_hours=projected_seconds/3600,
  estimate_caveat='Only if source admission succeeds; OSMesa CPU pilot measurements, not guaranteed throughput/ETA; no batch started. PNG, compressed caches, review videos and download archives require extra space.'))
 # Only referenced files, no broad historic weight hashing.
 evidence.extend([ROOT/'AGENTS.md',ROOT/'worktrees/rla-official-0fc6e97/src/datasets/trajectory_dataset.py',ROOT/'worktrees/rla-official-0fc6e97/datalib/dataset.py',
  Path('/data/lyc/projects/vla-adapter-ref/experiments/robot/libero/regenerate_libero_dataset.py')])
 evidence.extend(Path(x['path']) for x in read(RUN/'initial_evidence.json'))
 evidence.extend(RUN/'rlds_mapping_v2'/s/'complete.json' for s in SUITES)
 evidence.extend(Path(__file__).parent.glob('*.py'))
 dump(out/'evidence_manifest.json',[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(set(evidence))])
 def table(data,keys):return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(str(r[k]) for k in keys)+' |' for r in data])
 report=f'''# 官方 LIBERO AE 配对背景重渲染：来源恢复与准入报告

## 结论：PARTIAL；全量准入未通过，不启动训练

保留原1653训练＋40验证、40任务、273465帧。**1693/1693转换轨迹已认证到源RLDS episode；4条固定小样取得唯一原HDF5动作候选并完整重放；1条完成候选配对渲染和官方loader读取，但最终恢复状态后的Clean直对原视频失败。最终准入0/1693。** 没有把动作相同、同名任务或同帧数单独当作示范认证。没有将候选重放状态冒充作者保存的完整状态。

Object/000010（milk→basket）143帧为本轮唯一已生成的候选配对原型，**不是合格原样本增强**：连续重放阶段通过，但最终重新恢复state、FK后渲染的腕图有12帧超过预设p99≤40门槛，最大56；最大MAE仍仅4.205/255。不能因为“接近”或A≈B、B≈C就宣称A≈C通过。Spatial图像门槛未过；Goal、10有数值/图像实质偏离。四条未全部通过，因此全量批处理为**NOT_STARTED_BLOCKED_BY_PILOT_ADMISSION**，不是因磁盘或GPU不足。

## 固定代码与真实数据链

- RLA `0fc6e97212442ebccffa06f1d42f5481e5f22950`；公开包 `xyzhang368/RLA-WM@5fc55b89ba14e85c1d8ac064fd8a6eb11b5329f8`。复用既有转换包与哈希证据，不重新下载10GB归档。
- RLDS `openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`：Spatial复用本地，其他三套单独下载并核验。TFDS4.9.3 / TensorFlow2.15.1 CPU解码。
- 原HDF5来自 `yifengzhu-hf/LIBERO-datasets`：Spatial/Object/Goal固定`97773100c1474cd0d686ebd173cc0e4fd5442466`；该版本没有LIBERO10，事前查元数据后将唯一所需10任务文件固定到`f13aa24a3da8c43c7225569f28c562979fa0e35a`。只下载四个任务文件；未用Cosmos/旧单任务派生样本替代。
- 运行环境：官方LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`，robosuite1.4.1、MuJoCo3.3.2、OSMesa CPU。无LIBERO-Plus、无环境版本搜索。发布数据未附确切再生成环境/资产字节哈希，因此这是有验证门槛的候选复现，不是声称完全相同的作者环境。
- RLA README/16x64配置/转换器/下载器/DATA.md已核验，哈希见evidence_manifest。DATA.md仍有“未上传”旧说明，以固定HF实际文件为准。当前下载只用官方HF，没有启用旧脚本的镜像回退。

链路：原始公开HDF5 → OpenVLA regenerated--no_noops（该中间文件/状态未取得）→ RLDS → 作者转换视频。RLDS file_path实际给出任务HDF5路径，**不含demo编号**；RLDS observation.state为8维机器人EEF/夹爪，不是全场景MuJoCo state。

## 全量映射与划分

{table(coverage,['suite','split','trajectories','frames','rlds_sources_matched','hdf5_demo_candidates','candidate_state_sequences','original_state_sequences','clean_correspondence_pass','paired_sequences'])}

逐条清单：`source_mapping.csv`。验证ID仍是每套件0…9，不是每任务10条。原盘点的有效起点/时间对统计原样核对；没有更换为约2000条原示范、libero90或历史60条子集。

全1693条以完整action、joint+gripper、EEF位置数组签名找唯一候选，再比对语言及双视角像素。普通轨迹每相机固定5帧，四pilot每帧比对。所有最终映射通过冻结像素/数值门槛；这不是普通轨迹全帧几何认证。直接TFDS枚举与转换编号**不总相同**，例如Spatial枚举32对应转换54；转换器分worker读取TFDS切片，不能把直接枚举号当转换ID。使用真实TFDS图像解码器后验证，不能拿PIL解码差异误判身份。

每条映射保留RLDS枚举索引、源任务文件、数值/图像覆盖、原视频hash、metadata hash。公开表将上游私人根路径脱敏，只保留逻辑任务路径及原字符串hash。4条以外尚未连接到具体demo_N，不把知道任务文件称为已找到完整源示范。

## 时间、XML和状态：不是简单逐行贴状态

按公开OpenVLA再生成脚本：env.reset → 设置原states[0] → 10个零运动/开夹爪等待 → 按“运动范数<1e-4且夹爪未变化”过滤noop → 保存动作前obs → 执行动作。只保留成功重放的源生成规则已记录；**本轮不再按重放结果筛选或替换四条样本**。

该脚本首个保存state直接复制等待前原states[0]，但首个RGB/机器人obs来自等待后；所以不能拿再生成states首行直接配首图。本次四个原initial时间均0.25s，重放候选首时刻0.75s；Goal第一个保留动作原索引为1，其余为0。没有删首帧，没有整体平移序列，没有插值。候选时间步0.05s，真正来源身份仍靠上述多模态检查。

原HDF5 XML逐条保留，源/运行joint名称与顺序核对；原资产的完整逻辑相对路径映射到已固定本地资源，保存实际字节哈希，源历史字节是否同一仍未知。公开生成器按BDDL建环境并设置初态，不加载原demo XML；因此本次原XML与运行XML同时保存并比较，而不是随意把一个XML贴到另一份数据。body初始位置等存在差异，不能仅据state维度宣称模型相同。详见每条`asset_and_model_comparison.json`，不要把未使用相机差异直接说成失败原因。

## 四条完整小样结果

{table(validations,['suite','demo_candidate','frames','numeric_gate','pixel_gate','max_joint_error','max_eef_error_m','front_max_mae','wrist_max_mae','clean_admitted'])}

门槛在小样前冻结：逐帧最大关节绝对误差≤0.002、EEF位置分量误差≤0.002m；逐帧每视角RGB MAE≤8、p99≤40（0…255）；还需视觉检查事件、遮挡、物体与相机对应。阈值不是所有渲染环境的通用定律；Spatial失败可能包含渲染/纹理敏感性，但本轮没有看结果后放宽它。Goal及10的机器人/腕相机误差还提供了像素以外的阻断证据，不能只归为视频编码。

自动检查覆盖四条全部741帧。AI视觉：Object全部143帧六张逐帧缩略联系表＋7个固定高分辨率双视角对照；Spatial/Goal/10分别固定7/8/7帧高分辨率抽查。不是人工观看了全部视频，也不是分割/物体位姿真值。Object接近、夹持、搬运、篮中放置及干扰物罐子倾倒时序未见明显错位；Goal柜体/腕相机可见差异，10在中段操作后姿态分化。原视频与重放候选的完整对比视频四套件均保存在`video_index.csv`指向位置；失败样本没有伪造增强视频。

## 背景颜色实现与唯一已完成原型

背景按具名geom白名单选择，不用SAM；`table_visual`与六个wall视觉geom。只深复制所引用material和texture并改背景geom引用；其余前景共用引用不变。HSV只改H，S/V保持（uint8量化另有误差），纹理坐标/空间图案、反射、阴影、灯光不关闭不重写。所有原模型物理数组、网格、相机、灯光与改色编译模型逐元素比较通过。每条固定一个由独立SHA256/PCG64产生的delta_h，训练seed20261103、验证20261104；同轨迹所有时刻/两相机共享，完整参数见清单。

Object实际没有table_visual，只命中六面墙。没有把floor/机器人/商品/篮子自动加入白名单。墙纹理低饱和比例约86.7%；实际前视平均RGB变化{color['front']['mean_rgb_abs_change']:.6f}/255，腕相机{color['wrist']['mean_rgb_abs_change']:.6f}/255（{color['wrist']['zero_change_frames']}/143帧零变化，墙不可见）。这是**可追溯但较弱的墙面增强**，不声称对象任务有明显全视角改色，也没有自动加饱和度、改光照或扩白名单以追求效果。

对同一恢复state分别在物理相同的两个编译模型中渲染：286次状态设置、572次单相机RGB render、0次积分。每次render前后time/qpos/qvel完全不变，两路cam/body世界几何完全相同。Clean重渲染与候选视频比较通过，**但最终直接与作者原视频比较失败**（12个腕帧，见paired_final_clean_original_pixel_check.csv），不能用两级容差相加豁免。提示需要核查缓存obs与保存state/重新FK的相位，尚未确认它就是差异的唯一来源。实际c2w逐帧计算，wrist不静态；图像沿用OpenGL旋转180°、负fx约定。后台接口独立使用同步FK生成qpos/EEF，来源动作和语言原样保留，原metadata不覆盖；所有派生字段差值见export_audit。

已用固定官方TrajectoryDataset读取这条配对的合法h=2/10/64窗口，两时刻×两相机×3×224×224，时间索引、qpos、target_qpos、K与官方名为w2c的c2w字段逐元素一致。正式采样窗口保留官方尾部规则t+h<N，不跨episode、不用末帧拼未来；旧target_qpos末行重复仅为官方既有机器人字段，非新增有效AE未来目标。原视频只读；导出PNG流无损，比较MP4仅用于查看。保留的全白foreground流明确是官方占位，未用作背景真值。

AE-Base/Aug/Aug-Cons以后必须共享同一套**验收通过的派生Clean**、源ID、划分、时间采样和同步几何，不让只有增强组承担重渲染变化。接口见`fair_comparison_interface.json`。目前候选生成1条，最终训练准入0/1653、验证0/40，**不具备正式训练条件**。

## 具体阻塞与最小补齐方案

1. **episode身份**：全RLDS已找回，但其metadata无demo_N；仅四候选做过HDF5全动作join，Object连续重放阶段接近，但最终恢复后原视频对应仍未过关。
2. **生成时完整状态**：公开RLDS只有机器人信息；作者使用的regenerated--no_noops中间HDF5及其正确首图时刻state尚未取得。最小补齐是源再生成文件的demo-ID映射、逐帧完整场景state、生成XML/资产标识和obs采样时刻（首帧需另核实）。
3. **环境/资产和重放**：若拿不到上述状态，需要生成RLDS时的确切LIBERO/robosuite/MuJoCo/资产与reset初始化历史，才能在原四小样上复现；本轮仅测试一个固定官方环境，无无限版本搜索。当前不能断言漂移统一由哪一版本/物理因素造成。
4. **时间映射**：已建立公开代码的候选生成规则及原动作过滤索引，不意味着未通过轨迹取得真实图像时刻。不能用重新切帧或别的示范补数。

这不再是“缺少mask”。原视频、原HDF5、所有失败图表及候选状态均保留。暂停SAM，未创建任何AE训练任务；也没有因单条通过而偷偷缩小训练范围。

## 下载、资源、失败和批量预算

新增完整校验的数据文件合计 **{newbytes:,} bytes（{newbytes/2**30:.3f} GiB）**：三套RLDS {download['bytes']:,} B，四HDF5 {hf['total_bytes']:,} B。这是唯一文件净下载量；网络中断/断点重试的线缆传输总量未计量，不能冒称精确。另有官方git checkout，磁盘量与证据见resource_estimate，未把本地复用Spatial算新增下载。

全量两外观×两相机256²原始RGB约{sum(r['frames'] for r in rows)*256*256*3*4/2**30:.1f} GiB，另加PNG/缓存、状态、原视频及检查视频。按本轮CPU候选重放/配对速度外推，串行约{replay_seconds/3600:.1f}h重放＋{projected_seconds/3600:.1f}h配对渲染；仅条件性容量规划，不是已放行ETA。磁盘充足，但来源/对应门槛阻断批量。支持RLDS已验记录断点续作；已验配对目录拒绝覆盖，批量执行尚未启动/验收。

工程故障完整列于incidents：官方网络中断，Object CPU映射exit143后按持久化前缀核验继续，最初错误JPEG解码/枚举假设修复；一次loader字段名和一次边界检查脚本修正。未重放物理轨迹来刷通过，没有删除失败。

## 实际计数与交付

{table([dict(metric=k,value=v) for k,v in counts.items()],['metric','value'])}

4条源动作开环重放，共741个原动作＋40个稳定等待控制步，不是策略闭环，也不报告策略成功率。恢复741个候选frame state，不是741个已认证原始state；最终完整准入0。候选配对143物理帧/572张双外观双视角RGB，全部保留为未准入。FK是MuJoCo几何计算，不是AE/DINO/策略模型前向。状态设置包括导出同步FK；render_calls包括env隐式观测和显式相机渲染，不能仅按保存视频帧数算。

本地：`{RUN}`。`delivery/source_mapping.csv`、`pilot_validation.csv`、`coverage_by_suite_split.csv`、`pairing_manifest.csv`、`video_index.csv`、`evidence_manifest.json`及各pilot证据可直接检查。完整视频/状态/下载数据只留本地。全部模型前向、更新、backward和策略闭环=0，GPU任务=0，CPU进程结束后释放本任务资源。代码/脱敏表及报告提交新分支，远端核验另见sync记录。本轮完成即停止，不自动训练或扩环境搜索。
'''
 (out/'report.md').write_text(report)
 batch=dict(status='BLOCKED_BY_FIXED_PILOTS',required_pilot_passes=4,actual=int(final_pair),candidate_pairs_generated=1,bulk_trajectories_started=0,source_scope=1693,clean_original_correspondence_failed=[s+'/000010' for s in SUITES if s!='libero_object' or not final_pair],disk_sufficient=True)
 dump(out/'batch_status.json',batch)
 target=Path(__file__).parent/'delivery';target.mkdir(exist_ok=True)
 for f in out.rglob('*'):
  if f.is_file() and f.suffix in ['.json','.csv','.md']:
   dest=target/f.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True)
   if f.suffix=='.csv':dest.write_text(f.read_text())  # LF for Git; raw local CSV evidence retained
   else:shutil.copy2(f,dest)
 print(json.dumps(dict(mapped=len(rows),paired_candidates_generated=1,paired_admitted=int(final_pair),counts=dict(counts),new_bytes=newbytes),indent=2))
if __name__=='__main__':main()
