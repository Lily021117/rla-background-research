"""CPU arithmetic and evidence packaging only; never starts simulation."""
import csv,json,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
ROOT=Path('/data/lyc/projects/rla-wm');OLD=ROOT/'experiments/rla_official_ae_paired_rerender/run_20261103_01';RUN=ROOT/'experiments/rla_official_ae_sync_repair/run_20261103_01'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def writecsv(p,r):
 with open(p,'w') as f:w=csv.DictWriter(f,fieldnames=list(r[0]),lineterminator='\n');w.writeheader();w.writerows(r)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 out=RUN/'delivery';out.mkdir(exist_ok=False);evidence=set();summary=[];colors=[];capture=[];video=[];counts=dict(download_bytes=0,source_actions=0,wait_steps=0,physics_substeps=0,state_restores=0,rgb_render_calls=0,segmentation_render_calls=0,explicit_diagnostic_forward_calls=0)
 for s in SUITES:
  result=read(RUN/'color_final'/s/'result.json');f=rows(RUN/'color_final'/s/'frame_correspondence.csv');c=rows(RUN/'color_final'/s/'frame_color_metrics.csv');assert len(f)==result['frames']
  previous=read(OLD/'pilot_replay'/s/'result.json');summary.append(dict(suite=s,frames=len(f),A_self_restore=result['A_self_restore'],B_original_gate=result['B_original_numeric_pixel'],
   joint_max=max(float(x['B_joint_error']) for x in f),eef_max_m=max(float(x['B_eef_error_m']) for x in f),
   front_max_mae=max(float(x['B_front_mae']) for x in f),wrist_max_mae=max(float(x['B_wrist_mae']) for x in f),
   front_max_p99=max(float(x['B_front_p99']) for x in f),wrist_max_p99=max(float(x['B_wrist_p99']) for x in f),
   failed_frames=sum(not(float(x['B_joint_error'])<=.002 and float(x['B_eef_error_m'])<=.002 and all(float(x[f'B_{cam}_mae'])<=8 and float(x[f'B_{cam}_p99'])<=40 for cam in ['front','wrist'])) for x in f),
   previous_numeric_gate=previous['numeric_gate'],previous_pixel_gate=previous['pixel_gate'],admitted_original_augmentation=False))
  for mode in ['expanded_h','expanded_hs']:
   for cam in ['agentview','robot0_eye_in_hand']:
    rr=[x for x in c if x['mode']==mode and x['camera']==cam];vis=[x for x in rr if int(x['background_pixels'])>0]
    colors.append(dict(suite=s,mode=mode,camera=cam,frames=len(rr),background_visible_frames=len(vis),
     mean_background_fraction=float(np.mean([float(x['background_fraction']) for x in rr])),
     mean_old_background_fraction=float(np.mean([int(x['old_whitelist_pixels'])/65536 for x in rr])),
     mean_bg_rgb_mae=float(np.mean([float(x['background_rgb_mae']) for x in vis])) if vis else None,
     weak_visible_frames=sum(x['visible_but_weak']=='True' for x in rr),full_rgb_mae=float(np.mean([float(x['full_rgb_mae']) for x in rr])),
     signed_luminance_change=float(np.mean([float(x['luminance_signed_mean']) for x in rr])),abs_luminance_change=float(np.mean([float(x['luminance_abs_mean']) for x in rr]))))
  cp=RUN/'capture'/s;cr=read(cp/('result.json' if (cp/'result.json').exists() else 'failure.json'));cc=cr['counts']
  counts['source_actions']+=cc['source_actions'];counts['wait_steps']+=cc['wait_steps'];counts['physics_substeps']+=cc['physics_substeps'];counts['state_restores']+=cc['state_restores'];counts['rgb_render_calls']+=cc['env_render_calls']+cc['diagnostic_rgb_renders'];counts['explicit_diagnostic_forward_calls']+=cc['copy_forwards']
  if s=='libero_object':
   counts['wait_steps']+=1 # old instrumentation raised after physical step, before successful return
   capture.append(dict(suite=s,frames=0,status='FAILED_AFTER_ONE_WAIT_NO_RETRY',main_previous_exact='UNAVAILABLE',renderer_exact='UNAVAILABLE',A_geometry_max='UNAVAILABLE',obs_time_lag_s='UNAVAILABLE',geometry_time_lag_s='UNAVAILABLE',cached_obs_joint_max='UNAVAILABLE',sensor_geometry_recovery_rgb_max='UNAVAILABLE'))
  else:
   cf=rows(cp/'frame_results.csv');capture.append(dict(suite=s,frames=len(cf),status='FULL_METRICS_ARCHIVE_PARTIAL' if (cp/'failure.json').exists() else cr['status'],
    main_previous_exact=all(float(x['main_previous_state_delta'])==0 for x in cf),renderer_exact=all(float(x[f'renderer_equivalence_{cam}_max'])==0 for x in cf for cam in ['front','wrist']),
    A_geometry_max=max(float(x['A_geometry_delta']) for x in cf),obs_time_lag_s=sorted(set(round(float(x['boundary_time'])-float(x['obs_acquisition_time']),6) for x in cf)),geometry_time_lag_s=sorted(set(round(float(x['boundary_time'])-float(x['obs_geometry_time']),6) for x in cf)),
    cached_obs_joint_max=max(float(x['obs_joint_error']) for x in cf),sensor_geometry_recovery_rgb_max=max(float(x[f'obs_geometry_recovery_{cam}_max']) for x in cf for cam in ['front','wrist'])))
  off=read(RUN/'offline'/s/'result.json');counts['state_restores']+=off['state_restores'];counts['rgb_render_calls']+=off['render_calls'];counts['explicit_diagnostic_forward_calls']+=off['kinematic_forwards']
  for directory in ['color','color_final']:
   path=RUN/directory/s/'result.json'
   if path.exists():
    r=read(path)['counts'];counts['state_restores']+=r['state_restores'];counts['rgb_render_calls']+=r['rgb_renders'];counts['segmentation_render_calls']+=r['segmentation_renders'];counts['explicit_diagnostic_forward_calls']+=r['kinematic_forwards']
  snap=RUN/'model_snapshot'/s/'result.json'
  if snap.exists():counts['rgb_render_calls']+=read(snap)['counts']['render_calls'];assert read(snap)['counts']['physics_substeps']==0
  for directory in ['offline','capture','color','color_v2','color_final','model_snapshot']:
   for p in (RUN/directory/s).glob('*'):
    if p.suffix in ['.json','.csv']:evidence.add(p)
  video.append(dict(suite=s,frames=len(f),control_hz=20,video_fps=10,note='visualization playback half-speed; video index is not physical time',
   paired_video=str(RUN/'color_final'/s/'full_comparison.mp4'),fixed_images=str(RUN/'color_final'/s/'fixed_comparison.jpg'),
   paired_video_sha256=sha(RUN/'color_final'/s/'full_comparison.mp4'),fixed_images_sha256=sha(RUN/'color_final'/s/'fixed_comparison.jpg'),
   four_path_video=str(cp/'four_paths_full.mp4') if (cp/'four_paths_full.mp4').exists() else 'MISSING_OBJECT_CAPTURE_FAILURE'))
 assert counts['source_actions']<=741 and counts['wait_steps']<=40
 counts.update(render_calls=counts['rgb_render_calls']+counts['segmentation_render_calls'],source_replay_attempts=4,complete_source_replays=3,
  offline_old_candidate_frames=741,final_diagnostic_paired_frames=741,final_diagnostic_rgb_frames=741*2*3,admitted_author_pairs=0,
  AE_forwards=0,DINO_forwards=0,policy_forwards=0,AE_updates=0,DINO_updates=0,policy_updates=0,backward=0,policy_rollouts=0,GPU_processes=0,
  implicit_environment_forward_calls='not separately instrumented; explicit diagnostic forward calls listed separately',retries_with_source_actions=0)
 writecsv(out/'pilot_summary.csv',summary);writecsv(out/'color_summary.csv',colors);writecsv(out/'runtime_capture_summary.csv',capture);writecsv(out/'video_index.csv',video);dump(out/'execution_counts.json',counts)
 allrows=rows(OLD/'delivery/source_mapping.csv')
 for r in allrows:
  pilot=r['trajectory_id']=='000010'
  r.update(sync_repair_A='PASS' if pilot else 'NOT_RUN',sync_repair_B='FAIL' if pilot else 'NOT_RUN',sync_diagnostic_frames=r['frames'] if pilot else '0',repair_augmented_diagnostic=pilot,repair_author_admitted=False,
   repair_status='BLOCKED_ORIGINAL_CORRESPONDENCE' if pilot else 'NOT_STARTED_SYSTEMIC_PILOT_GATE')
 writecsv(out/'full_status.csv',allrows);coverage=[]
 for s in SUITES:
  for split in ['train','val']:
   rr=[r for r in allrows if r['suite']==s and r['split']==split]
   coverage.append(dict(suite=s,split=split,trajectories=len(rr),frames=sum(int(r['frames']) for r in rr),rlds_mapped=sum(r['rlds_mapping_status']=='PASS' for r in rr),
    candidate_hdf5=1 if split=='train' else 0,restored_candidate_trajectories=1 if split=='train' else 0,original_correspondence_passed=0,diagnostic_pairs=1 if split=='train' else 0,admitted_augmented_trajectories=0))
 writecsv(out/'coverage.csv',coverage)
 assert sum(r['trajectories'] for r in coverage)==1693 and sum(r['trajectories'] for r in coverage if r['split']=='train')==1653
 dump(out/'batch_status.json',dict(status='STOP_SYSTEMIC_ORIGINAL_CORRESPONDENCE_FAILURE',admitted=0,scope=1693,bulk_started=0,alternative_source_substitution=False))
 incidents=[dict(id='object_observable_hook',error='__modality__ missing',impact='1 wait physically executed,0 source actions; returned-observation recapture missing',resolution='hook fixed; Object not replayed again; offline143 retained'),
  dict(id='spatial_optional_archive',error='duplicate sensor_geometry keyword in NPZ',impact='101 source actions complete; all CSV/time/video/sync state saved; optional raw multi-path archive missing',resolution='keyword fixed for remaining pilots; no replay'),
  dict(id='XML_roundtrip',error='XML recompile not exact model snapshot',impact='rounding plus recalculated some inertias/masses; no original-correspondence claim',resolution='MJB snapshots, texture-only isolated model copies; old XML color143frames preserved'),
  dict(id='object_model_copy_check',error='pre-existing mesh_polynormal NaNs compare unequal numerically',impact='preflight only,0 render/action',resolution='exact array-byte comparison; no tolerance relaxed')]
 dump(out/'incidents.json',incidents)
 for p in [RUN/'frozen_protocol.json',RUN/'visual_review.json',OLD/'frozen_protocol.json',OLD/'delivery/report.md',OLD/'delivery/source_mapping.csv']:
  if p.exists():evidence.add(p)
 for p in Path(__file__).parent.glob('*.py'):evidence.add(p)
 evidence.add(Path('/data/lyc/projects/vla-adapter-ref/.venv-cpu/lib/python3.10/site-packages/robosuite/utils/observables.py'))
 evidence.add(Path('/data/lyc/projects/vla-adapter-ref/.venv-cpu/lib/python3.10/site-packages/robosuite/environments/base.py'))
 dump(out/'evidence_manifest.json',[dict(path=str(p),sha256=sha(p)) for p in sorted(evidence)])
 # Evidence-backed report; no claim of bulk admission or exact original state recovery.
 lines=['# 官方 AE 数据：同步修复与背景可见性检查','',
  '**结论：修复版四条完整轨迹的自身保存／恢复（A）通过；作者原视频对应（B）仍为0/4。停止批量扩展，作者范围仍为1653训练＋40验证，正式增强准入0/1693。**',
  '', '本轮只做CPU状态、渲染与数据审核；没有训练、模型推理、SAM或策略闭环。全部旧分支、失败和原始视频保留。',
  '', '## 1. 同步问题：确认了什么', '',
  '1. 先检查旧741帧：相同time/qpos/qvel经forward后，body/camera变换不再等于旧缓存。这是状态与衍生几何未同步，不是把RGB解码差异当作时间差异。',
  '2. 追加运行时捕获仅完成Spatial/Goal/10三条。它们的主重放状态与旧候选逐项一致，统一参数后native和环境直接渲染逐像素一致。缓存obs在边界前2ms采样，其几何来自边界前4ms；该观测不是边界时刻的同步观测。没有删除、移位或插值。Object的同类运行时证据缺失，不能把其他套件结果冒充其实际记录。',
  '3. 完整积分状态用mjSTATE_INTEGRATION保存，包含ctrl/mocap/外力/warmstart等；图像与FK、body和camera统一由副本forward得到。主回放不受诊断修改。旧741帧仅保留legacy状态，不冒称恢复了丢失的控制器/观测缓存。',
  '运行时另按传感器实际几何采样时刻保存诊断状态；Goal/10用MJB恢复该时刻可复现缓存obs（数值见runtime_capture_summary.csv），但没有将其替换主导出或整体移位。缓存obs的机器人状态对作者RLDS仍有偏差，所以不能靠改用缓存相位认证原来源。',
  '4. get_xml再编译不等于原运行时模型：除了微米级位置舍入，还会重算部分质量与惯量。因此精确恢复使用MJB；新颜色路径复制编译模型，仅修改白名单纹理字节，其他模型数组逐字节一致。该物理差异不能单独被认定为此前连续重放漂移的原因：连续重放没有使用这次XML重编译模型。',
  '5. Object/Spatial仅重建同一固定环境的模型以保存MJB：XML哈希与上一轮相同，无动作、无等待。Goal/10直接保存运行时MJB。A是本轮同步Clean的可重现性，不是作者状态身份认证。',
  '', '## 2. 四条完整轨迹', '', '|套件|帧|A|B|失败帧|最大关节差|最大EEF位置差(m)|front/wrist最大p99|', '|---|---:|---|---|---:|---:|---:|---|']
 for x in summary:lines.append(f"|{x['suite']}|{x['frames']}|通过|未通过|{x['failed_frames']}|{x['joint_max']:.8f}|{x['eef_max_m']:.8f}|{x['front_max_p99']:.0f}/{x['wrist_max_p99']:.0f}|")
 lines += ['', '旧阈值未改变：逐帧双相机MAE≤8、p99≤40；关节≤0.002、EEF分量≤0.002m。详见 `pilot_summary.csv` 与各 `color_final/*/frame_correspondence.csv`。Goal/10同步后仍有机器人状态漂移，不能归因于编码差异。',
  '', 'Object曾经连续渲染像素门槛通过，但旧保存后重渲染失败；本轮不把连续、恢复、作者原片三种比较作传递推断。Spatial运行时XML恢复有小误差，后续以精确编译模型的完整离线A检查修正；不隐去第一次失败。',
  '', '## 3. 背景实际覆盖与颜色', '',
  '显式白名单：table_visual、命名的wall_*_visual以及floor。floor是实际静态背景平面，Object不存在table_visual。机器人、篮子、任务对象、可操作柜体均不纳入。使用实际GEOM类型分割，SITE不作为实体。逐帧所有可见geom与面积保存在visible_geoms.json，未列入白名单的对象仍可核查。',
  '旧纯H及其弱变化结果不变；新增 expanded_h 使用旧每轨迹色相、扩展地面白名单；expanded_hs 使用预先固定ΔH=0.20、S′=S+0.35(1−S)，HSV的V不主动修改。纹理结构保留、没有反射或阴影关闭。渲染后的亮度会变化，不能宣称全部亮度严格不变。',
  '同一轨迹所有帧与双相机使用同一参数；没有策略评分。两路状态、body/camera几何相同，渲染不推进时间。背景可见但区域平均RGB变化<1（0—255尺度，相当于归一化的1/255）记为弱变化；完全不可见单列，不能据此判失败。',
  '', '|套件|视角|可见帧|平均背景占比|旧白名单占比|H区内MAE|H+S区内MAE|H+S可见但弱帧|', '|---|---|---:|---:|---:|---:|---:|---:|']
 for s in SUITES:
  for cam in ['agentview','robot0_eye_in_hand']:
   h=next(x for x in colors if x['suite']==s and x['camera']==cam and x['mode']=='expanded_h');v=next(x for x in colors if x['suite']==s and x['camera']==cam and x['mode']=='expanded_hs')
   fmt=lambda n:'不可见' if n is None else f'{n:.3f}'
   lines.append(f"|{s}|{cam}|{v['background_visible_frames']}/{v['frames']}|{v['mean_background_fraction']:.3%}|{v['mean_old_background_fraction']:.3%}|{fmt(h['mean_bg_rgb_mae'])}|{fmt(v['mean_bg_rgb_mae'])}|{v['weak_visible_frames']}|")
 lines += ['', '区外RGB变化也记录，但不是自动失败：未改前景材质的前提下，反射、边界混合可以发生合法变化。物理参数和纹理引用检查比“前景RGB全不变”更适合本轮。数值汇总见color_summary.csv，颜色/亮度逐帧记录完整保留。',
  '', '## 4. 证据范围及工程失败', '',
  '自动检查覆盖四条完整741帧、两相机；AI视觉只检查预先固定7帧/轨迹的对照图，完整视频供后续查看，未声称人类逐帧审阅。具体记录见visual_review.json。',
  'Object运行时记录包装缺少__modality__，在第一个等待动作物理完成后报错：0源动作、1等待。为守预算未重启Object。Spatial完成101源动作后的附加NPZ打包键重名，CSV、时间记录、视频和同步状态已保存，不重放补图。两项均保留原failure.json。',
  '模型常量检查遇到源模型已有的mesh_polynormal NaN：修正为逐字节比较，确认复制前后内容完全相同，而不是放宽数值容差。未据这些NaN猜测任务失败。',
  '', '## 5. 全量状态与停止决定', '',
  '1693条RLA→RLDS映射沿用已核验结果；这不等于1693条HDF5 demo或全场景状态已认证。四条只有唯一动作候选来源及候选状态，原视频全程对应仍未通过。其余1689条未启动重渲染，不能换其他示范补齐。全量逐条状态见full_status.csv，按套件/划分计数见coverage.csv。',
  '本轮不再搜索环境版本。缺失的是生成regenerated--no_noops时的逐episode身份、同步完整状态/观测采样时刻、运行时模型与编译/资产信息，而不是背景掩码。原始HDF5或8维RLDS机器人状态不能替代这些资料。',
  '', '### 后续二选一（本轮均不执行）', '',
  '① 向发布者索取生成RLDS的中间HDF5及demo映射、逐帧同步状态/观测时序、完整运行时模型或精确构建配置、资产版本与过滤/稳定等待记录。代价是外部等待、额外数据传输与一次对齐验收；无法预估获得时间。优点是有机会保留作者原视频样本语义。拿到资料前不批准全量。',
  '② 另立“同源官方示范统一重建Clean/增强”协议。代价是显式数据版本改变、重新确定可追溯源示范及划分映射、重建Clean与所有增强、重新计算几何/时间及特征缓存验收；不能称为作者视频的精确增强，也不能保证无证据时保留原1693条一一身份。需要用户明确选择，不在本轮自动替换。',
  '', '## 6. 实际计算与资源', '',f'新增源动作{counts["source_actions"]}，等待{counts["wait_steps"]}；未超过741+40上限。4次捕获尝试，3条完整源动作重放；没有动作重试。旧741帧离线使用，不计新增采集。',
  f'本轮恢复{counts["state_restores"]}次；RGB渲染{counts["rgb_render_calls"]}次、分割渲染{counts["segmentation_render_calls"]}次（合计{counts["render_calls"]}）。计入环境隐式图像渲染及保留的失败/中间版本。下载0字节。',
  '最终诊断集：741物理帧×2相机×3图像版本（Clean/H/H+S）=4446 RGB图像，不是新增物理示范；正式作者增强准入0。',
  'AE/DINO/策略前向=0，所有训练更新=0，backward=0，策略闭环=0。mj_forward是仿真运动学/动力学同步，不是神经网络前向。仅CPU OSMesa，无GPU占用。资源释放以交付resource_release.json为准。',
  '', '## 7. 可检查产物', '',f'- 本地完整对比视频与图片索引：{out}/video_index.csv',f'- 四套件诊断数据：{RUN}/color_final/',f'- 同步捕获、缺失及原失败：{RUN}/capture/',
  '- Git仅提交源码、协议、CSV/JSON统计与脱敏报告；MJB、NPZ、XML、视频和图片留本地。']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 shutil.copy2(RUN/'frozen_protocol.json',out/'frozen_protocol.json')
 if (RUN/'visual_review.json').exists():shutil.copy2(RUN/'visual_review.json',out/'visual_review.json')
 for s in SUITES:
  target=out/'pilot_evidence'/s;target.mkdir(parents=True)
  for section in ['offline','capture','color_final','model_snapshot']:
   for filename in ['result.json','failure.json','frame_results.csv','frame_correspondence.csv','frame_color_metrics.csv','frames.csv','model_serialization.json','render_options.json','camera_time_metadata.json','appearance_configuration.json','pairing_manifest.csv','brightness_saturation.csv','fullstate_vs_legacy_render.json']:
    path=RUN/section/s/filename
    if path.exists():shutil.copy2(path,target/f'{section}_{filename}')
 public=Path(__file__).parent/'delivery';public.mkdir(exist_ok=False)
 for p in out.iterdir():
  if p.suffix in ['.json','.csv','.md']:shutil.copy2(p,public/p.name)
 shutil.copytree(out/'pilot_evidence',public/'pilot_evidence')
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
