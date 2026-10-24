import argparse,collections,subprocess
from scipy.spatial.transform import Rotation
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def csvread(p):return list(csv.DictReader(Path(p).open()))
stage=csvread(o/'stage_summary.csv');events=csvread(o/'release_events.csv');handoff=read(o/'handoff_acceptance.json');forward=read(o/'forward_acceptance.json');assert handoff['status']=='PASS_RESTORE' and len(handoff['prefixes'])==10 and len(handoff['continuations'])==20
summ=[]
for arm in ('C','D-center'):
 for appearance in ('clean','seen_background','unseen_background'):
  rs=[x for x in stage if x['model']==arm and x['appearance']==appearance];es=[x for x in events if x['model']==arm and x['appearance']==appearance and x['first_postlift_open_step']]
  summ.append(dict(model=arm,appearance=appearance,trials=len(rs),original_successes=sum(x['recorded_success']=='True' for x in rs),terminal_xy_pass=sum(x['final_xy']=='True' for x in rs),terminal_contact_pass=sum(x['final_contact']=='True' for x in rs),terminal_height_pass=sum(x['final_height']=='True' for x in rs),terminal_xy_median_m=float(np.median([float(x['final_xy_m']) for x in rs])),postlift_open_trials=len(es),open_outside_goal=sum(x['first_open_outside_goal']=='True' for x in es),first_open_xy_median_m=float(np.median([float(x['first_open_xy_m']) for x in es])) if es else None))
csvsave(o/'stage_condition_summary.csv',summ)
hs=[]
for prefix in ('C','D-center'):
 for continuation in ('C','D-center'):
  rs=[x for x in handoff['continuations'] if x['prefix']==prefix and x['continuation']==continuation];hs.append(dict(prefix=prefix,continuation=continuation,successes=sum(x['success'] for x in rs),trials=len(rs),diagnostic_not_formal=True))
csvsave(o/'handoff_summary.csv',hs)
metrics_rows=csvread(o/'same_input_action_metrics.csv');groups=collections.defaultdict(list)
for x in metrics_rows:groups[x['source'],x['split'],x['appearance'],x['horizon']].append(x)
out=[]
for (source,split,appearance,h),xs in groups.items():
 r=dict(source=source,split=split,appearance=appearance,horizon=h,samples=len(xs))
 for label,dims in [('translation',range(3)),('rotation',range(3,6)),('gripper',[6])]:
  r[label+'_mean_absolute_C_D_difference']=float(np.mean([abs(float(x['dim'+str(i)+'_output_difference'])) for x in xs for i in dims]))
  if source=='demo':
   for arm in ('C','D-center'):r[arm+'_'+label+'_MAE']=float(np.mean([float(x[arm+'_dim'+str(i)+'_absolute_error']) for x in xs for i in dims]))
 out.append(r)
csvsave(o/'same_input_action_summary.csv',out)
demo_table=[]
for split in ('train','validation'):
 for h in ('1','8'):
  xs=[x for x in metrics_rows if x['source']=='demo' and x['split']==split and x['horizon']==h]
  for arm in ('C','D-center'):
   vals=[float(np.mean([float(x[arm+'_dim'+str(i)+'_absolute_error']) for x in xs for i in dims])) for dims in (range(3),range(3,6),[6])];demo_table.append(f"| {split} | {h} | {arm} | {vals[0]:.5f} | {vals[1]:.5f} | {vals[2]:.5f} |")
demo_text='\n'.join(demo_table)
pr=csvread(o/'prefix_geometry_metrics.csv');geometry=[]
for seed in sorted({x['seed'] for x in pr}):
 c=next(x for x in pr if x['seed']==seed and x['prefix']=='C');d=next(x for x in pr if x['seed']==seed and x['prefix']=='D-center');rel=[]
 for x in (c,d):
  q=Rotation.from_quat(np.roll([float(x['ee_quat_'+str(i)]) for i in range(4)],-1));v=np.array([float(x['bowl_'+str(i)])-float(x['ee_'+str(i)]) for i in range(3)]);rel.append(q.inv().apply(v))
 rc,rd=[Rotation.from_rotvec([float(x['relative_orientation_'+k]) for k in 'xyz']) for x in (c,d)]
 geometry.append(dict(seed=seed,bowl_relative_EE_position_difference_m=float(np.linalg.norm(rel[1]-rel[0])),relative_rotation_difference_rad=float((rc.inv()*rd).magnitude())))
csvsave(o/'prefix_relative_geometry_comparison.csv',geometry)
aligned=csvread(o/'lift_aligned_state_metrics.csv');angles=[]
for arm in ('C','D-center'):
 for lag in ('0','8','16','32','64'):
  xs=[x for x in aligned if x['model']==arm and x['lag']==lag]
  angles.append(dict(model=arm,steps_after_lift=lag,available_trials=len(xs),bowl_tilt_median_rad=float(np.median([float(x['bowl_tilt_radians']) for x in xs])),xy_distance_median_m=float(np.median([float(x['xy_m']) for x in xs]))))
csvsave(o/'event_aligned_pose_summary.csv',angles)
protected=read(o/'protected_inputs.json');assert all(sha(p)==h for p,h in protected.items());sv=read(o/'source_version.json');assert all(sha(p)==sv['policy_hashes'][k] for k,p in read(o/'policy_registry.json').items())
csvsave(o/'video_index.csv',[dict(seed=x['seed'],prefix=x['prefix'],continuation=x['continuation'],video=x['video'],trajectory=x['trajectory'],diagnostic=True) for x in handoff['continuations']])
history=read(o/'history_verification.json');coverage=read(o/'coverage_verification.json');gates=handoff['gates'];gp=read(o/'frozen_forward_process.json');assert gp['returncode']==0
save(o/'delivery_verification.json',dict(status='PASS_DIAGNOSTIC_NOT_METHOD',AE_optimizer_updates=0,policy_optimizer_updates=0,backward=0,existing_trials_analyzed=120,new_prefix_collections=10,diagnostic_continuations=20,new_formal_trials=0,restore_gates_passed=len(gates),max_restore_state_error=max(x['state_max_error'] for x in gates),max_restore_action_error=max(x['action_max_error'] for x in gates),historical_files_unchanged=True,protected_small_files_checked=len(protected),final_policy_hashes_rechecked=True,sealed_read=False,GPU_process_exited=True,process_seconds=gp['seconds'],source_commit_before_run=sv['commit']))
for name in ('method_definitions.md','official_config_diff.md'):
 with (o/name).open('x') as f:f.write(Path(__file__).with_name(name).read_text())
table='\n'.join(f"| {x['prefix']} | {x['continuation']} | {x['successes']}/{x['trials']} |" for x in hs)
ctable='\n'.join(f"| {x['model']} | {x['appearance']} | {x['original_successes']}/{x['trials']} | {x['terminal_xy_pass']}/20 | {x['terminal_contact_pass']}/20 | {x['terminal_height_pass']}/20 | {x['terminal_xy_median_m']:.4f} |" for x in summ)
text=f'''# 炉灶抬升后失败定位

## 结论

本轮恢复与诊断流程通过，不是方法有效性通过。任务是将炉灶上的目标黑碗放到盘子上，不是操作炉灶开关。

**已确认的直接失败条件：D-center全部60条历史轨迹终点都不满足盘子水平中心3cm范围要求。** 首次抬升后发出张开命令也全部发生在该范围之外，中位水平距离约5.51cm；终点中位距离约6.12cm。这定位了放置位置与松爪协调缺口，不单独证明松爪是唯一根因。

更精确的顺序是：D-center全部60条都曾水平居中，但当时碗仍悬在盘子上方，没有盘碗接触。各轨迹在居中期间最低中心高度差的中位数约13.58cm（范围7.45–17.02cm）。随后在下降时横向偏离；首次盘碗接触比最后一次居中晚13–32步（中位21步），首次张开比最后居中晚28–59步（中位35步）。全部60条中“居中且接触”的步数为0。**并非只差松爪：下降与水平位置保持先失配，松爪发生在已经偏离的位置。** 这来自保存状态重建，不能直接推断某一网络模块是唯一根因。

## 历史全部120条轨迹

| 模型 | 外观 | 原记录成功 | 终点水平条件 | 接触条件 | 高度条件 | 终点水平距离中位m |
|---|---|---:|---:|---:|---:|---:|
{ctable}

官方goal为On(akita_black_bowl_1,plate_1)，由高度、接触、水平<0.03m三条件合取；没有独立松爪或保持时长要求。C有59/60原记录成功，通常在张开前触发成功而结束，所以不能把其较少松爪直接解读为更好的释放动作。

离线重建有{history['terminal_disagreements']}条终点判定与原记录不同，均保留。保存的积分状态重新mj_forward与原最后子步的缓存接触/位姿不是同一采样相位；不改写历史成功率。D-center的水平条件缺口远大于该相位差，全部60条均存在。

逐步物体/末端位置、朝向及接触见本地`stage_timeline_local_only.csv`；以首次抬升对齐的0/8/16/32/64步见`lift_aligned_state_metrics.csv`，其中后期样本数会因C早成功结束而减少，不能做未注明的幸存样本比较。接触丢失/高度下降只是代理，不将其自动称为掉落；下降也可能是正常放置。

D-center有60/60次重建的手指接触丢失事件，其中57次第一次丢失在首次张开当步或之后，3次先于张开；时间差中位+2步、范围−14至+5步。前3次可能涉及短暂接触变化或滑动，状态记录不足以将其精确判为掉落。所有D-center此前均已达到抬升，主要失败不是最初抓取失败。姿态倾角/到盘距离按抬升相对时间汇总在`event_aligned_pose_summary.csv`，姿态不是此任务的独立成功条件。

## 有界接手诊断

固定前5个Clean development种子2026102500…2026102504。新增10次前缀采集，每个前缀在首次达到历史3cm抬升阈值后的第一个chunk边界保存；同一前缀两种接手使用完全相同剩余预算，不保留原策略未执行chunk。没有根据结果筛选状态。

| 前缀 | 接手 | 成功次数 |
|---|---|---:|
{table}

10个原策略继续验收均与历史后半段动作、完整flattened仿真状态和终点判定一致，最大状态/动作误差均0。快照保留MuJoCo integration状态（包括控制输入和warmstart）、控制器目标/缓存、夹爪current_action、robot buffers、观测缓存和RNG。推理是单当前观测、无跨chunk KV/history状态；chunk边界队列为空。快照与视频只存本地。

两种前缀并非完全相同的抓持构型：碗相对末端坐标的位置差中位{np.median([x['bowl_relative_EE_position_difference_m'] for x in geometry]):.4f}m，相对朝向差中位{np.median([x['relative_rotation_difference_rad'] for x in geometry]):.4f}rad，详见`prefix_relative_geometry_comparison.csv`。二者切换时刻按各自首次抬升后的chunk边界确定，不是强行相同帧号；这些差异不能单独确定哪一个造成失败。

C前缀多数第64步交接，D-center前缀多数第72步交接；同一前缀两种接手剩余预算相同，不同前缀预算不同。交叉接手的成功均发生在总步数122–139之间，远早于220步上限，但这不消除不同到达状态与时间的全部混杂。

**解释：** C接手D-center前缀仍能完成，不能把这些前缀状态称为物理不可恢复。D-center接手C前缀有成功也有失败，因此不能称它完全没有后半段能力；证据支持其后半段控制对到达状态更敏感，且前缀状态与后续策略存在交互。仅5个状态、一次固定切换时点，不足以区分抓持相对位姿、物体朝向、到达位置和释放策略各自的唯一贡献。接手试验不并入正式成功率。

## 数据与相同输入比较

20训练/5development示范；两外观均参与C/D训练。实际4800个物理窗口曝光中有{coverage['train_postlift_exposures']}个属于首次抬升后，占{coverage['train_postlift_exposures']/4800:.2%}。因此不能解释为“完全没有后半段监督”；数量存在不代表分布覆盖或学习充分。

固定取每条允许示范首次抬升、+8、+24帧（截到最后有效chunk且去重），以及全部历史闭环首次抬升后的chunk边界、+16、+40步。两模型读取完全相同双图、FK proprio、语言。完整逐维/逐跨度输出和示范误差见`same_input_action_metrics.csv`，汇总见`same_input_action_summary.csv`。闭环观测无示范真值，只比较输出差，不以C输出为答案。

动作度量在部署反归一化后的控制命令坐标中：平移/旋转仍为控制器delta命令，不能当成米或弧度真实位移；夹爪0闭/1开。示范前6维按共同Q99界限裁剪后比较，明确有损裁剪，不要求反变换完全可逆。三个量纲分开报告。

后半段固定示范观测的当前/最远预测步误差（Clean/Background合并，均为同一批输入）：

| 划分 | chunk位置1…8 | 模型 | 平移命令MAE | 旋转命令MAE | 夹爪MAE |
|---|---:|---|---:|---:|---:|
{demo_text}

已有development日志按同一抬升分段重新汇总到`historical_validation_phase_curves.csv`；训练日志仅有混合batch损失，不能恢复逐窗口阶段误差。未新增backward或补训练checkpoint。

这些固定示范观测上D-center的平移、夹爪误差更低，而旋转误差更高；所以不能以“平均动作误差更差”概括其闭环失败，也不能由此直接认定旋转分支是根因。闭环经历的状态、误差累积和多维动作协调仍有影响。全量示范分阶段曲线与本次固定子集应分别解读，不用本次离线误差选择checkpoint。

## 唯一优先后续干预（本轮未执行）

建议做一次**只抑制居中后下降阶段的水平动作**的D-center有界诊断：原D-center作对照；实验组在首次满足官方水平<3cm条件后、首次盘碗接触前，仅将控制器世界坐标x/y增量置0，保留模型的z、旋转、夹爪输出和其余推理流程。状态与220步预算提前固定，不改阈值、不选成功样本。此为使用仿真物体状态触发的oracle诊断，不是可部署策略或训练改进。目的检验下降时横向漂移是否是当前结果的直接瓶颈；旋转/抓持相对位姿仍可能使碗移动，因此失败不自动排除其作用。只做这一因素，不同时延迟松爪、改LR或加loss；需下一轮明确授权，本轮未执行。

## 保护与资源

AE更新0，策略更新0，backward0；新增前缀10、诊断续跑20，正式评分新增0，封存集未读取。历史小文件与实际两组checkpoint哈希复核通过。本轮GPU1进程正常结束；ECC沿用风险基线、无新增错误。所有新缓存和产物在数据盘。代码与脱敏汇总同步当前私有分支，原始轨迹、视频、快照、权重不上传。
'''
with (o/'report.md').open('x') as f:f.write(text)
print('FINALIZED',hs,flush=True)
