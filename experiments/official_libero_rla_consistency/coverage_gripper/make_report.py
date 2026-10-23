"""Assemble evidence-backed delivery and a data-coverage-only next experiment."""
import argparse,csv,json,hashlib,subprocess,re
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def csvread(p):return list(csv.DictReader(p.open()))
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def text(p,x):
    with p.open('x') as f:f.write(x)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--code',type=Path,required=True);a=p.parse_args();o=a.out
    labelroot=o/'label_audit_v2' if (o/'label_audit_v2').exists() else o
    availability=read(o/'availability_summary.json');data=read(o/'frozen_dataset_manifest.json');labels=csvread(o/'gripper_label_coverage.csv');events=csvread(o/'gripper_events.csv');summary=csvread(o/'checkpoint_summary.csv');timing=read(o/'close_transition_summary.json');rollout=read(labelroot/'rollout_quantitative_analysis.json');response=read(o/'gripper_execution_response.json');infer=read(o/'checkpoint_diagnosis/acceptance.json');deploy=read(o/'checkpoint_diagnosis/deployment_path_comparison.json');loader=read(o/'expanded_loader_audit.json')
    assert availability['available']==43 and data['training_episodes']==20 and data['development_episodes']==5 and loader['episodes']==25
    assert infer['optimizer_updates']==infer['backward']==infer['new_closed_loop']==0
    assert all(x['max_control_action_difference']==0 for x in deploy)
    step270=[r for r in summary if r['checkpoint']=='270' and r['chunk_position']=='0'];event_rows=csvread(o/'close_transition_timing.csv');old_events=[r for r in event_rows if r['checkpoint']=='270' and r['chunk_position']=='0'];future=[r for r in event_rows if r['checkpoint']=='270' and r['chunk_position']=='7']
    expanded=[]
    for e in data['episodes']:
        lab=np.load(Path(e['root'])/'trajectory_labels.npz');g=lab['actions'][:,6];ev=np.flatnonzero((g[:-1]<0)&(g[1:]>0))+1
        expanded.append(dict(episode=e['episode'],split=e['split'],source_demo=e['source_demo'],frames=len(g),close_events=len(ev),close_event_indices=json.dumps(ev.tolist()),open_count=int(np.sum(g==-1)),close_count=int(np.sum(g==1)),other_command_count=int(np.sum((g!=-1)&(g!=1))),new_export=not e['export_reused']))
    with (o/'expanded_gripper_label_summary.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(expanded[0]));w.writeheader();w.writerows(expanded)
    # Concrete line references refer to the actual executed files, not similarly named copies.
    upstream=a.reference/'vla-official';script=a.code/'experiments/official_libero_rla_consistency/single_task_training.py'
    def loc(path,needle,label):
        lines=path.read_text().splitlines();n=next(i for i,x in enumerate(lines,1) if needle in x);return f'`{label}:{n}`'
    refs=dict(convert=loc(script,'actions[:,6]', 'single_task_training.py') if 'actions[:,6]' in script.read_text() else loc(script,'actions[:,-1]','single_task_training.py'),norm=loc(upstream/'prismatic/vla/datasets/rlds/utils/data_utils.py','def normalize_action_and_proprio','upstream@23fa0c9c/prismatic/vla/datasets/rlds/utils/data_utils.py'),transform=loc(upstream/'prismatic/vla/datasets/rlds/oxe/transforms.py','def libero_dataset_transform','upstream@23fa0c9c/prismatic/vla/datasets/rlds/oxe/transforms.py'),batch=loc(upstream/'prismatic/vla/datasets/datasets.py','class RLDSBatchTransform','upstream@23fa0c9c/prismatic/vla/datasets/datasets.py'),collate=loc(upstream/'prismatic/util/data_utils.py','class PaddedCollatorForActionPrediction','upstream@23fa0c9c/prismatic/util/data_utils.py'),loss=loc(upstream/'vla-scripts/finetune.py','loss = torch.nn.L1Loss()(predicted_actions','upstream@23fa0c9c/vla-scripts/finetune.py'),unnorm=loc(upstream/'prismatic/extern/hf/modeling_prismatic.py','def _unnormalize_actions','upstream@23fa0c9c/prismatic/extern/hf/modeling_prismatic.py'),process=loc(upstream/'experiments/robot/libero/run_libero_eval.py','def process_action','upstream@23fa0c9c/experiments/robot/libero/run_libero_eval.py'),threshold=loc(upstream/'experiments/robot/robot_utils.py','def normalize_gripper_action','upstream@23fa0c9c/experiments/robot/robot_utils.py'))
    pipeline=f'''# 夹爪完整链路审计

结论：没有发现开/闭符号、动作字段、normalization或部署阈值接入错误。发现的数据范围限制是旧split/cache入口只覆盖4+1，已通过显式新manifest与原始XML导出修复。

|阶段|实际定义与范围|执行证据|
|---|---|---|
|原始动作|HDF5 `actions[:,6]`，检查的旧5条为-1=open、+1=close；与frame/action缓存逐元素相等|`gripper_label_coverage.csv`；不是两指qpos|
|dataset transform|`1-clip(raw,0,1)`；因此1=open、0=close；连续区间有损clip但本数据二值端点可往返|本地{refs['convert']}；官方`libero_dataset_transform` {refs['transform']}；全5条数值相等|
|normalization|BOUNDS_Q99，action.mask=[T,T,T,T,T,T,F]；夹爪不做Q99缩放，min!=max故不触发unused-dimension归零|{refs['norm']}；`batch_pipeline_audit.json`|
|batch与loss|8×7连续动作，完整有效chunk；所有夹爪维度进入mean L1，维度占比1/7，无额外gripper权重|{refs['batch']}；{refs['collate']}；{refs['loss']}|
|policy output|连续回归数值，无sigmoid；允许略超[0,1]，不能把数值0.7当已校准概率|所有checkpoint分布见`checkpoint_gripper_metrics.csv`|
|unnormalization|同一mask最后维False，夹爪保持原值；仅前6维按旧4条train统计解码|{refs['unnorm']}|
|后处理|`command=-sign(2*p-1)`：p<0.5→+1 close；p>0.5→-1 open；p==0.5→0 neutral|{refs['process']}；{refs['threshold']}|
|仿真入口|ControlEnv.step→SingleArm.control→Manipulator.grip_action→PandaGripper.format_action→actuator ctrlrange映射|robosuite1.4.1 `robots/manipulator.py:16`、`models/grippers/panda_gripper.py:43`|

PandaGripper按sign(command)更新内部current_action，speed=0.01；+1为关闭、-1为打开。两个手指qpos是proprio的最后两维，来源和形状都与动作第7维不同，未作为动作标签。

16个固定观测检查（step0/270、train/dev、起点/闭合事件、Clean/Background）：官方loss路径与实际predict_action入口解码后的动作最大差0，重复推理差0；没有未来图像或示范动作作为部署条件。历史两条rollout的440条执行命令与记录预测经官方后处理后逐元素相等。

Background的10条关闭命令均未被后处理翻成打开；记录的手指分离量在close命令步平均减少2.61mm。完整actuator ctrl和控制器current_action没有保存，不能声称重建了控制器每个内部子步。没有发现接口错误需要立刻重训；0.5 tie是官方中性行为，保持原阈值。

动作chunk位置0..7指a_t..a_{{t+7}}，不等同于RLA累计未来状态h1..8。当前B仍沿用完整未来状态筛选以保持既定协议，因此每条最后一个动作标签未进入任何有效chunk；旧5条被遗漏的均是open，闭合事件和close动作未被遗漏。尾部release的当前观测窗口较少，详见phase_window_coverage.csv，未悄悄改变mask。
'''
    text(o/'gripper_pipeline_audit.md',pipeline)
    fail=['# 已有两条闭环失败分析','','仅分析既有220+220控制步、视频和保存状态，本轮策略闭环=0。模型输出与实际执行命令全量配对一致。','', '|条件|close/open/neutral|chunks|最近专家闭合相对位置差|该步姿态差|手指-目标接触步|','|---|---|---|---|---|---|']
    for r in rollout:fail.append(f"|{r['appearance']}|{r['close_commands']}/{r['open_commands']}/{r['neutral_commands']}|{r['chunks']}|{r['nearest_expert_close_relative_position_m']*100:.2f}cm|{r['orientation_error_at_that_step_rad']:.3f}rad|{r['finger_target_contact_steps']}|")
    fail+=['','两条均28次query，完整chunk执行8步，最后chunk只执行4步以遵守220步上限。平移/旋转饱和比例均0；按末端每步运动<0.1mm的描述性定义，Clean为0，Background为1/220；没有长期停滞证据。这不能单独排除所有局部振荡。','',
'两条均接近并接触了BDDL指定的akita_black_bowl_1；接触不等于抓取位姿命中。Clean与专家闭合相对位置最近仍差约4.0cm；Background最近差约2.4cm且姿态差约1.23rad。没有证据证明已经在正确抓取位姿下只缺一个关闭命令。','',
'Background的10条close分成6段（4段单步、1段2步、1段4步），持续性弱。最后4步关闭时，手指分离量减少约19.8mm，说明执行器对关闭命令有响应。Clean主要输出open而不是close被执行器丢弃。','',
'固定视频抽帧显示接近饼干盒上目标后偏离；这只是视觉观察。保存的物理状态支持真实接触和位姿偏差，不能据此唯一归因为gripper head、视觉定位、动力学或数据覆盖。','',
'综合模式：C有证据（示范观测能闭合，实际闭环观测多输出open）；A也有局部证据（训练集的未来chunk首次闭合时机未学准）。B不受当前唯一development示范支持；D没有发现（命令路径一致、关闭时手指有响应）。当前样本仅4条训练、1条development及2条旧rollout，不能做唯一因果归因。','',
'CPU日志v1的停滞统计曾使用错误np.diff轴；已在本轮label_audit_v2/中修正为时间轴，v1中间文件保留且不用于结论。该问题属于本轮诊断计算，不涉及历史策略训练或执行代码。']
    text(o/'rollout_failure_analysis.md','\n'.join(fail)+'\n')
    # One controlled experiment: training demonstration coverage, shared stats.
    normsha=sha(o/'normalization_statistics_expanded_train_only.json');common=dict(status='PREPARED_NOT_AUTHORIZED_TO_RUN_THIS_TURN',experiment='B4_vs_B20_coverage_only',seed=20261023,initialization='verified Prismatic + exact formal_B/step0000 trainable parameters; never step270',successful_updates=270,microbatch=8,accumulation=2,effective_appearance_batch=16,physical_window_exposures=2160,appearance_observation_exposures=4320,BC_loss='official mean action L1',gripper_weight=1,gripper_threshold=.5,action_chunk=8,execution_chunk=8,lr=.0002,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,Math_SDPA=True,STOP='existing inference-aligned repair',checkpoint_steps=[0,45,90,135,180,225,270],checkpoint_selection='fixed step270',shared_normalization='train20-only BOUNDS_Q99, identical for both arms',shared_normalization_sha256=normsha,development_episodes=5,background='frozen tex-table/tex-wall reflection1 MSAA4',future_closed_loop_plan='after separate authorization: all5 fixed dev source state0, clean/background, seed0,220steps, no retries; 10 trials per B arm',AE_C_D=False)
    for arm,count in [('B4',4),('B20',20)]:
        allowed=[e['episode'] for e in data['episodes'] if e['split']=='train' and (arm=='B20' or e['export_reused'])];assert len(allowed)==count
        write(o/f'next_training_config_{arm}.json',dict(**common,arm=arm,train_episodes=count,train_episode_ids=allowed,equivalent_window_epochs=2160/(431 if arm=='B4' else data['train_chunks'])))
    nextplan='''# 下一轮唯一优先实验：扩大独立示范覆盖

建议一个受控B4/B20实验，唯一组间主变量是用于优化的训练示范数4→20。从共同的官方Prismatic/原正式step0重新开始；不续训step270。LR、模型结构、BC loss、夹爪权重、0.5阈值、chunk8、配对外观规则、seed、有效batch和270成功更新预算保持一致；不启动C/D。

为什么先选数据：已经实际补齐16条train和4条development；训练闭合事件从5个增加到22个；当前示范上的闭合已学会，仍有未来chunk边界误差和on-policy位姿偏离。此证据支持优先检验独立轨迹覆盖，而非直接加gripper loss。数据扩展不保证解决这些问题。

共同归一化使用20条train拟合的BOUNDS_Q99，两个新B组完全相同，development不参与。旧4条统计在新增train上的动作ry约10.6%、proprio轴角z约16.0%的值需要裁剪（示范等权均值），不宜默默沿用并把裁剪当数据问题。B4也共享这份统计，所以应如实记录其额外统计曝光；该设计隔离优化数据覆盖，不声称B4完全未接触新增训练数据的信息。历史B270使用旧统计，只作历史参照，不能直接当严格数据量消融对照。

两新组各2160物理窗口曝光、4320外观观测，B4约5.01窗口epoch，B20约1.14；相同更新预算下比较多样性与重复，不把重复称为新示范。新的5条development从源episode隔离，原dev34保留；新dev4仅做数据质量检查，尚未用于本轮模型选择。

冻结验收：共同step0参数/RNG/固定batch输出一致；25条真实数据loader及时间检查通过；成功更新、checkpoint保存恢复、finite、ECC记录齐全。预定离线指标同时包括xyz/rotation/gripper误差、当前动作闭合召回、首次闭合事件各chunk位置召回、未来h7切换误差；不要只看总体L1。

若获下一轮授权，再在固定final270上执行5个development源state0各Clean/Background一次，220步上限，chunk8与成功判定不变。先分别检查接近/抓取姿态和关闭持续性，再看任务成功。不依据中途结果延长270步，也不叠加降低LR、夹爪加权或修改阈值。若两组训练观测上的未来切换仍差，再另行设计一个优化诊断，而不是本轮自动追加。

现有数据已足够本次20+5准备，无需重新采集；另外18条合法完整候选保留。正式多任务、独立测试和多seed仍未准备完成。
'''
    text(o/'next_B_experiment.md',nextplan)
    report=['# 训练数据覆盖与夹爪失败诊断','','结论：数据不足来自旧脚本选取范围；已实际扩充并冻结20条train+5条development。没有发现夹爪标签/符号/部署命令错误。当前动作闭合已学会，但未来chunk闭合时机仍未学准，闭环接近位姿也存在偏差。此为定位证据，不是唯一根因。','',
'## 1. 数据覆盖的真实原因','',
'源任务cookie-box共有50条示范。按旧seal和释放记录，仅2条仍封存；另5条属于Step1冻结test，均只核对身份与schema元数据，不读取其数据内容。43条允许开发，其中38条此前只因旧split/cache入口没遍历而未使用；没有以缺RLDS身份哈希排除。上一轮把4+1当作导出范围过于保守，现已修复。','',
'43条均有完整states/actions、双相机字段与原始model_file XML；逐条state/action有限性、长度、20Hz时序及资产模型预检通过。时间/字段/资产/长度预检排除0。选定25条完整配对回放的几何、segmentation、depth、FK检查均通过；其余18条是固定20+5预算后的候选，不是被判坏数据，未对它们宣称完整渲染验收。保护身份内容未读取；仅metadata不等于其数据已验收。','',
'新manifest入口从原始HDF5/XML生成派生ID，资产仅做路径重定位，背景仅修改tex-table/tex-wall文件引用；不依赖旧geometry缓存。原4train/1dev角色不变，新增加16train+4dev，按照预先固定的SHA256(seed|source_demo)顺序选择，不用成功率或模型得分。源rewards/dones作为来源记录保留，不用它筛选。','',
f"冻结数据：train{data['train_frames']}帧/{data['train_chunks']}窗口，development{data['development_frames']}帧/{data['development_chunks']}窗口；共{data['total_frames']}物理帧、{data['total_h8_chunks']}完整h8窗口、{data['total_frames']*2}相机配对、{data['total_frames']*4}张RGB。实际新增2004帧/8016张RGB，其余复用既有5条。20条train有22个open→close事件，5条dev有6个。每条示范及所有外观/窗口保持同一split。",'',
'数据保留原反射=1、MSAA=4，允许纹理引起的间接RGB响应；不是严格前景RGB固定。每帧K/逐帧wrist外参、state/proprio/action/camera映射与原始来源均保存。raw末端派生传感器刷新相位仍未知，训练输入继续使用同一保存状态的机器人关节FK，不声称恢复原RLDS像素或raw proprio。','',
'## 2. 夹爪链路与监督覆盖','',
'原始-1=open/+1=close → 官方clip+invert得到1=open/0=close → Q99最后维mask=False → 无sigmoid连续回归 → 不反归一化夹爪维 → -sign(2*p-1) → 仿真+1=close/-1=open，等于0.5时neutral。与官方transform逐元素一致；真实batch动作标签等于源actions的正确chunk，没有把两指qpos当动作。全部有效8×7动作参与BC L1。源码位置及数值范围见gripper_pipeline_audit.md。','',
'旧4条train共463帧，open255/close208；5个闭合事件全部包含在8个合法chunk位置，每事件实际得到80次动作槽监督（含两外观），当前观测曝光10次。没有闭合事件被padding或mask删除；每条最后一个被省略的动作均为open。独立示范仍只有4条、闭合事件只有5个，重复80次不能增加独立事件。末尾release当前观测窗口偏少，详细阶段代理覆盖见phase_window_coverage.csv，未把command/lift代理当确认抓取真值。','',
'## 3. 固定观测模型诊断','',
'4份checkpoint：step0/45/135/270，按预先固定的全512物理窗口×2外观推理；原来的train-only normalization不变。eval/inference_mode，全部参数冻结，Math SDPA。没有optimizer、反向传播或额外策略rollout。','',
'最终checkpoint当前动作位置0的结果（窗口等权、窗口相关，不能当独立统计样本）：','', '|集合|外观|平移L1（归一化）|旋转L1（归一化）|夹爪L1|闭合召回|开口召回|','|---|---|---|---|---|---|---|']
    for r in step270:report.append(f"|{r['split']}|{r['appearance']}|{float(r['translation_L1']):.4f}|{float(r['rotation_L1']):.4f}|{float(r['gripper_L1']):.4f}|{100*float(r['close_recall']):.2f}%|{100*float(r['open_recall']):.2f}%|")
    report+=['',
'step270在所有5个train闭合事件与1个dev闭合事件的当前位置上均输出close，两种外观均正确（10/10和2/2事件-外观对）；阈值切换一般提前1–2步。因此“完全不会闭合”不受示范观测证据支持。训练集总体close召回约91%，不代表所有时刻无误。','',
'关键局部欠拟合：最后一个chunk位置7在必须提前7步预测首次关闭的样本上为train 0/10、dev 0/2正确。平均预测边界比理想未来边界晚4.7/5.0步（0.235/0.250秒）。这来自固定示范离线轨迹，不是直接测得的闭环触发延迟。稳定闭合长区间的低L1会掩盖切换边界误差。图见plots/train_gripper_timeseries.png、validation_gripper_timeseries.png与close_event_chunk_timing.png。','',
'16个部署入口对照中解码动作差0、夹爪阈值分歧0；同一输入重复推理差0。旧2条rollout共440条命令全量与预测后处理一致。没有D类证据：close不是被接口翻成open。','',
'## 4. 闭环证据的边界','',
'沿用原视频与保存状态，未重跑。Clean无关闭；Background仅10个关闭，分6个短段。两者执行chunk与220步限额符合冻结规则；没有持续动作饱和或长时间末端停滞证据。记录的关闭步手指确实朝关闭方向变化，最后4步缩小约19.8mm。','',
'两者接近/接触正确的目标碗，但最近专家闭合相对位置分别仍差3.97cm、2.44cm，对应姿态差0.171rad、1.230rad。不能宣称在完全正确抓取位姿下只缺夹爪关闭，也不能把没有闭合单独当head故障。完整controller/actuator内部状态未保存，不能复原每个控制子步。详见rollout_failure_analysis.md。','',
'综合定位：当前观测支持C；未来chunk边界支持A的局部版本；当前唯一dev示范不支持B；没有发现D。数据覆盖不足、未见状态下的接近/姿态偏差与未来切换欠拟合是有证据的候选，不是已经分离的唯一因果原因。','',
'## 5. 下一轮只改变什么','',
'只检验独立训练示范覆盖4→20，建议新的B4/B20共同step0受控对照，各270更新、同LR/seed/loss/阈值/chunk/曝光预算，保留原BC目标。两组共享从20条train拟合的同一统计并记录额外统计曝光；这样不把历史旧统计造成的额外裁剪混成数据量收益。旧B270只作历史参照。具体预注册草案见next_B_experiment.md与next_training_config_B4/B20.json。','',
'现有合法数据足够20+5，不需要新采集。本轮未执行下一轮训练，也未启动AE-C/AE-D或策略C/D。','',
'## 6. 保护、资源和交付','',
f"本轮optimizer updates=0，backward=0，新增策略闭环=0。GPU0仅冻结推理，峰值allocated {infer['peak_allocated']/1024**3:.2f}GiB，reserved {infer['peak_reserved']/1024**3:.2f}GiB，ECC无新增。数据生成是CPU保存状态重渲染，不执行action replay，不推进物理时间。全部输出/缓存在数据盘，旧checkpoint、数据与历史报告未覆盖。",'',
'metadata封存排除表、逐episode可用性、来源/帧身份与内容hash、分区完整性、25条真实loader的50个配对batch验收均保存。未使用新dev的策略表现选择数据或超参数。新dev仍是development，不能称全新独立测试。',
'本轮代码/脱敏汇总推送同一私有研究分支；实际GitHub commit和远端核验见github_sync_receipt.json。权重、完整图像、原始轨迹、逐样本预测和凭证不上传。']
    text(o/'report.md','\n'.join(report)+'\n')
    # Verify only permitted inputs; never hash the shared HDF5 container.
    protected={str(a.previous/'report.md'):'3ac43cb5ee3b77d21bab2f9547e7aa1889aae3deaf4023bf5cf26d736b2a2ce5'}
    inventory=csvread(a.previous/'checkpoint_inventory.csv')
    for row in inventory:
        if int(row['step']) in (0,45,135,270):protected[str(a.previous/row['path'])]=row['sha256']
    assert all(sha(Path(k))==v for k,v in protected.items())
    write(o/'delivery_verification.json',dict(status='PASS_AUDIT_AND_DATA_EXPORT_NOT_UNIQUE_CAUSAL_PROOF',optimizer_updates=0,backward=0,new_policy_closed_loops=0,new_independent_episodes=20,train_episodes=20,development_episodes=5,source_data_candidates=50,protected_episodes_skipped=7,all_source_hash_checks_pass=True,checked_protected_artifact_hashes=protected,shared_HDF5_container_hash_performed=False,gripper_interface_error_found=False,current_grip_learned=True,future_chunk_close_timing_learned=False,next_primary_variable='independent training episode coverage',AE_C_D_started=False,ECC_increment=False,report_sha256=sha(o/'report.md'),frozen_manifest_sha256=sha(o/'frozen_dataset_manifest.json')))
    publish=Path(__file__).resolve().parent/'delivery';publish.mkdir(exist_ok=False)
    for name in ('report.md','gripper_pipeline_audit.md','rollout_failure_analysis.md','next_B_experiment.md','checkpoint_summary.csv','close_transition_summary.json','availability_summary.json','next_training_config_B4.json','next_training_config_B20.json'):
        value=(o/name).read_text();value=value.replace(str(a.previous),'${PREVIOUS_RUN}').replace(str(a.reference),'${REFERENCE_RUN}').replace(str(o),'${RUN_DIR}');value=re.sub(r'/(?:data|root|opt)/[^\s\"\']+','${LOCAL_PATH}',value);text(publish/name,value)
    print('REPORTS_AND_PROTECTION_PASS')
if __name__=='__main__':main()
