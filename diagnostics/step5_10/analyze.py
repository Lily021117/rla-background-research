"""Episode-aware summaries of the frozen 2x2x2 intervention."""
from common import *
import itertools,math,collections
assert read(O/'inference_acceptance.json')['status']=='PASS'
r=csvread(O/'per_position_metrics.csv');manifest=read(O/'fixed_sample_manifest.json');samples=manifest['samples'];inv=read(O/'component_inventory.json');up=csvread(O/'parameter_update_statistics.csv')
metrics=['normalized_L1','normalized_drift_L1','normalized_translation_L1','normalized_rotation_L1','normalized_gripper_L1','decoded_translation_command_L1','decoded_rotation_command_L1','gripper_command_L1','gripper_command_matches','controller_translation_delta_L1_m','controller_rotation_delta_L1_rad','normalized_first6_outside_unit_ratio','gripper_outside_0_1','controller_first6_clipping_ratio']
def numeric(v):return float(v=='True') if v in ['True','False'] else float(v)
def means(rows):return {k:float(np.mean([numeric(x[k]) for x in rows])) for k in metrics}
demos=[];summary=[];tasks=[];positions=[]
for bits in manifest['group_order']:
 for ep in sorted({x['episode'] for x in r}):
  rows=[x for x in r if x['combination']==bits and x['episode']==ep]
  demos.append({'combination':bits,'episode':ep,'task':rows[0]['task'],'split':rows[0]['split'],'appearance_instances':2,'physical_windows':1,**means(rows)})
 for split in ['train','validation','all']:
  dd=[d for d in demos if d['combination']==bits and (split=='all' or d['split']==split)]
  summary.append({'combination':bits,'VLM_step':int(bits[0]),'head_step':int(bits[1]),'proprio_step':int(bits[2]),'split':split,'demonstrations':len(dd),'physical_windows':len(dd),'appearance_instances':len(dd)*2,**{k:float(np.mean([d[k] for d in dd])) for k in metrics}})
  for pos in range(8):
   rows=[x for x in r if x['combination']==bits and int(x['chunk_position'])==pos and (split=='all' or x['split']==split)]
   positions.append({'combination':bits,'split':split,'chunk_position':pos,**means(rows)})
 for task in sorted({x['task'] for x in r}):
  for split in ['train','validation']:
   rows=[x for x in r if x['combination']==bits and x['task']==task and x['split']==split]
   if rows:tasks.append({'combination':bits,'task':task,'split':split,'demonstrations':len({x['episode'] for x in rows}),**means(rows)})
cs('component_swap_metrics.csv',summary);cs('per_demo_metrics.csv',demos);cs('task_metrics.csv',tasks);cs('chunk_position_summary.csv',positions)
lookup={(x['combination'],x['split']):x for x in summary};effects=[]
for split in ['train','validation','all']:
 for k,component in enumerate(['VLM','action_head','proprio']):
  for bits in manifest['group_order']:
   if bits[k]!='0':continue
   other=bits[:k]+'1'+bits[k+1:];a=lookup[bits,split];b=lookup[other,split]
   effects.append({'component_changed_0_to_1':component,'from':bits,'to':other,'split':split,'delta_error_normalized_L1':b['normalized_L1']-a['normalized_L1'],'delta_drift_L1':b['normalized_drift_L1']-a['normalized_drift_L1']})
cs('conditional_component_effects.csv',effects)
inter=[]
for split in ['train','validation','all']:
 for pair in [(0,1),(0,2),(1,2)]:
  b1=''.join('1' if i==pair[0] else '0' for i in range(3));b2=''.join('1' if i==pair[1] else '0' for i in range(3));both=''.join('1' if i in pair else '0' for i in range(3))
  inter.append({'split':split,'components':str(pair),'interaction_error_L1_at_other_step0':lookup[both,split]['normalized_L1']-lookup[b1,split]['normalized_L1']-lookup[b2,split]['normalized_L1']+lookup['000',split]['normalized_L1']})
cs('component_interactions.csv',inter)
updates=[]
for c in ['VLM','action_head','proprio','auxiliary_fixed']:
 rows=[x for x in up if x['component']==c];ref=sum(float(x['parameter_norm_step0'])**2 for x in rows)**.5;delta=sum(float(x['update_norm'])**2 for x in rows)**.5
 updates.append({'component':c,'total_parameter_count':inv['components'][c]['parameter_count_unique'],'historically_trainable_count':inv['components'][c]['historically_trainable_parameter_count'],'saved_parameter_count':sum(int(x['numel']) for x in rows),'changed_tensors':sum(int(x['changed_coordinates'])>0 for x in rows),'changed_coordinates':sum(int(x['changed_coordinates']) for x in rows),'saved_reference_norm':ref,'update_norm':delta,'relative_saved_reference_norm':delta/ref if ref else None,'full_reference_norm':inv['components'][c]['full_step0_parameter_norm'],'relative_full_component_norm':delta/inv['components'][c]['full_step0_parameter_norm'],'max_absolute_update':max(float(x['max_absolute_update']) for x in rows),'zero_initialized_tensors':sum(x['zero_initialization']=='True' for x in rows),'changed_zero_initialized_tensors':sum(x['zero_initialization']=='True' and int(x['changed_coordinates'])>0 for x in rows)})
cs('component_update_summary.csv',updates)
z=np.load(O/'fixed_outputs.npz');differences={}
for bits in manifest['group_order']:
 drift=abs(z['combo_'+bits]-z['combo_000']);differences[bits]={'translation_drift':float(drift[:,:,:3].mean()),'rotation_drift':float(drift[:,:,3:6].mean()),'gripper_drift':float(drift[:,:,6].mean()),'max_abs':float(drift.max())}
prior=csvread(P/'fixed_input_action_probe.csv');cross=[]
for bits,arm,step in [('000','A',0),('111','B',1)]:
 for i,s in enumerate(samples):
  old=next(x for x in prior if x['arm']==arm and int(x['step'])==step and int(x['sample_id'])==s['id']);ref=np.array(json.loads(old['first_deploy_action']));cross.append({'combination':bits,'sample_id':s['id'],'first_action_max_abs_vs_Step59':float(abs(z['combo_'+bits][i,0]-ref).max())})
cs('cross_run_reference_check.csv',cross)
js('analysis_summary.json',{'metric_0155_definition':{'space':'published normalized action coordinates, float32 diagnostic','samples':14,'independent_demonstrations':7,'train_demos':4,'validation_demos':3,'two_current_cameras_per_input':True,'chunk_positions':list(range(8)),'dimensions':7,'denominator':14*8*7,'padding':0,'mask':'all8 actual actions valid; no action padding','reduction':'mean absolute difference over appearance instances x8x7; equivalent demo-equal because exactly2 appearances per demo','stochastic_inference':False,'native_training_phase_noise_enabled_in_this_diagnostic':False,'seed':20261021,'baseline_error':lookup['000','all']['normalized_L1'],'step1_error':lookup['111','all']['normalized_L1'],'step1_drift':lookup['111','all']['normalized_drift_L1']},'component_drift_by_dimension_group':differences,'gripper_target_min_max':[float(z['targets'][:,:,6].min()),float(z['targets'][:,:,6].max())],'max_cross_GPU_prior_first_action_delta':max(x['first_action_max_abs_vs_Step59'] for x in cross),'component_order':['VLM','action_head','proprio'],'decoded_units':{'decoded_translation_command_L1':'dimensionless OSC command','decoded_rotation_command_L1':'dimensionless OSC command','controller_translation_delta_L1_m':'m of commanded delta after controller input clipping/scaling, not measured robot displacement','controller_rotation_delta_L1_rad':'rad per rotation-vector command component after clipping/scaling, not geodesic orientation error','gripper_command_L1':'absolute difference in binary{-1,+1},0 at exact threshold; not finger displacement'},'all_saturation_ratios_are_descriptive':True,'combinations_do_not_prove_longterm_causality':True})
table='\n'.join(f"|{bits}|{lookup[bits,'train']['normalized_L1']:.6f}|{lookup[bits,'train']['normalized_drift_L1']:.6f}|{lookup[bits,'validation']['normalized_L1']:.6f}|{lookup[bits,'validation']['normalized_drift_L1']:.6f}|" for bits in manifest['group_order'])
u_table='\n'.join(f"|{v['component']}|{v['total_parameter_count']:,}|{v['historically_trainable_count']:,}|{v['changed_coordinates']:,}|{v['update_norm']:.6g}|{v['relative_saved_reference_norm']:.6g}|{v['max_absolute_update']:.6g}|" for v in updates)
print('SWAP_METRICS\n'+table,flush=True)
# Recommendation follows the observed selective substitution; no training is invoked.
h=lookup['010','validation']['normalized_L1']-lookup['000','validation']['normalized_L1'];v=lookup['100','validation']['normalized_L1']-lookup['000','validation']['normalized_L1'];p=lookup['001','validation']['normalized_L1']-lookup['000','validation']['normalized_L1']
assert h>0 and h>v and h>p,'report template requires evidence-backed dominant head effect; inspect unexpected results'
js('next_short_training_config.json',{'status':'PROPOSED_NOT_EXECUTED','unique_between_arm_change':'native action-head LR multiplied by0.1','control_LR':{'VLM':1e-4,'native_action_head':1e-4,'proprio':1e-4,'auxiliary':1e-4},'experimental_LR':{'VLM':1e-4,'native_action_head':1e-5,'proprio':1e-4,'auxiliary':1e-4},'same_explicit_optimizer_groups_in_both':True,'initialization':'same validated B step0 model parameters with new empty AdamW state; never B step1','successful_updates_each':10,'checkpoint_steps':[0,1,5,10],'seed':20261021,'data_order':'first10 rows of original frozen100-row sampling_order.json','microbatch':2,'accumulation':2,'effective_batch':4,'loss':'BC action L1 only; same unified auxiliary head retained with weight0','common_STOP':'append STOP2 to training input_ids and labels, attention_mask1; both arms same; preflight exact training/deployment equivalence in eval','dtype_backend':'unchanged BF16 native/FP32 LoRA, Math SDPA','optimizer':'same AdamW betas(.9,.999),eps1e-8,weight_decay0; constantLR, no clipping','native_training_phase_perturbation':'unchanged original Training-mode0.02 noise, same RNG rule','extra_losses':[],'all_other_data_model_augmentation_seeds_trainables_equal':True,'main_checkpoint':10,'no_validation_checkpoint_selection':True,'historical_noSTOP_run_not_strict_control':True,'not_authorization_to_train':True})
report=f'''# Step 5.10 B首步组件置换诊断

**PASS：8种冻结置换、来源与保护验收完成。主要敏感模块为原生action head。** 它的step1权重不仅引起动作漂移，也在本固定验证样本上提高示范动作误差。该结论是离线模块定位，未证明100步训练后的长期闭环失败机制。optimizer updates=0、backward=0、闭环=0、保留集新增评分=0。

## 1. 0.155到底是什么

Step5.9的0.155257是归一化动作坐标中，B-step1与共同step0/A的平均绝对输出差。14输入=4训练示范与3验证示范各一物理h8窗口的clean/background两版本；每个输入包含当前双相机，而不是把相机当独立样本。对14×8×7=784个标量等权平均，包含chunk位置0–7，全部为真实动作，无padding或无效步。前6维经公开Q01/Q99映射至[-1,1]，夹爪为[0,1]；它不是米、弧度、闭环成功率或物体位移。

同输入下：step0示范动作L1={lookup['000','all']['normalized_L1']:.6f}；step1={lookup['111','all']['normalized_L1']:.6f}；step1对step0漂移={lookup['111','all']['normalized_drift_L1']:.6f}。因此这一批样本确实存在误差恶化，不能仅用漂移本身代替误差证据。它不是整个190窗口训练集的平均；这4个固定训练代表窗口也不等于首步实际使用的4个窗口。

## 2. 来源、STOP与A等价

使用同一Step5.7 B运行的`B/step0000.pt`和`B/step0001.pt`。step0参数逐元素等于共同`common_step0.pt`；同sources/config SHA、同normalization。首次成功更新日志global_step=1，无跳过/重放；step0 Adam状态为空、step1已有状态step均1，scheduler0→1。完整路径、SHA及来源在checkpoint_provenance.json。没有新训替代权重。

先冻结样本与组合顺序，再评分。每组合严格核对VLM保存子集键名，native head/proprio均strict=True；原始基础VLM冻结权重全程不变。完整step0与加载公开初始化的原生A在本轮14个完整8×7输出上逐元素相等。统一STOP2后，每组合训练前向与部署输出max_abs=0，首样本重复推理max_abs=0。eval/inference_mode、固定seed20261021、Math SDPA，Flash/efficient均关闭。没有Training阶段的随机query扰动，故这些差异不含该训练噪声。

## 3. 组件覆盖与合法性

VLM组包括视觉backbone、视觉投影、语言模型、LoRA、action_queries、词嵌入及其归一化；只有保存的LoRA/action_queries在0/1间变化，其他基础参数固定。action head组为native Pro trunk和最终动作投影，不含新增辅助层；proprio组为原projector。完整参数/前缀、历史可训练标记、buffer与绑定别名见component_inventory.json。

VLM词嵌入与lm_head base weight为同组内绑定参数，保持同一个对象；不存在跨组件参数或storage绑定，没有遗漏历史trainable参数。72个VLM buffer、24个head buffer归属明确；非持久RoPE等buffer由相同配置构造，不含差异权重或未解释恢复。辅助层固定step0；将其输出置零不改变action，max_abs=0。故三个改变组互斥且可分别加载。

|组件|全部参数量|历史可训练量|实际变化坐标数|更新L2|更新/保存参考L2|最大绝对更新|
|---|---:|---:|---:|---:|---:|---:|
{u_table}

VLM更新/参考比只针对保存的可训练子集；额外的完整组件norm与比值另见component_update_summary.csv。近零/全零LoRA-B、bias等单独标记：不将除零或极小分母的比率当作更新失控证据。changed_coordinates指张量标量坐标，不等于改变的参数张量个数；两种统计均保存。

## 4. 8种组合主要结果

三个bit依次为VLM、action head、proprio来源；0=step0，1=step1。每个示范先平均chunk位置及两外观，再在各split示范等权平均。训练只有4示范、验证只有3示范，每条只选1窗口，不构成完整泛化评估。

|组合|训练动作L1|训练漂移L1|验证动作L1|验证漂移L1|
|---|---:|---:|---:|---:|
{table}

单独替换head(010)的验证L1从0.152682升至0.252129；三条验证示范均变差。单独替换VLM(100)训练误差下降，验证小幅升至0.157106（1条改善、2条变差）；单独替换proprio(001)变化很小。所有包含step1 head的组合均显著高于不含step1 head的对应组合。这里“显著”指明显的数值差，未进行统计显著性声明。

完整111验证0.270636，较单独head还差约0.01851；VLM+head在proprio固定0时的二阶误差交互约+0.01414，提示组合敏感性。**恶化不是只在拼接时才出现**：完整合法step1本身也更差。但混合参数的模块匹配关系可能不同，置换不能单独推断长期因果。

## 5. 物理量分组、任务与chunk

验证111对比000：平移标准化L1约0.185225→0.338335；旋转0.168510→0.234123；夹爪0.007568→0.177083。不是仅夹爪一个维度造成总体恶化。所有三条验证示范恶化；任务/示范表保存cookie-box、ramekin、drawer逐项，验证缺stove。训练4示范中，单独head对ramekin略改善，对其余3条变差；完整111对4条均变差。

部署解码后的OSC增量命令分开报告，不混合不同量纲。控制器clip/scale后的验证平移命令每分量MAE约7.69→14.27 mm，旋转命令每分量约0.01398→0.01863 rad。这是**命令空间误差**，不是实际机器人位移或姿态测量误差。解码后的原始translation/rotation command仍为无量纲，另列字段。比较标签经过既有Q99有损裁剪，不要求回到原始动作完全可逆。

这14窗口的夹爪标签均为open=1，所有组合经过0.5阈值后仍给相同open命令；所以夹爪连续回归变差不等于这批样本的离散控制符号出错，也完全不足以评估真实抓取/释放时序。NaN/Inf为0；饱和、越界与控制器clipping比例见component_swap_metrics.csv，chunk0–7见chunk_position_summary.csv；没有按位置或任务筛选结果。

## 6. 首步优化审计与实现错误判断

读取真实optimizer字典、配置和日志：同一个AdamW参数组，lr1e-4，betas(.9,.999)，eps1e-8，weight_decay0；constant scheduler首步没有warmup或LR跳变。microbatch2×累积2，有效batch4；无gradient clipping。首次更新前状态空，未从退化权重恢复。native BF16参数的Adam moments也为BF16，LoRA为FP32；这个数值条件需要保留，但本轮不能由dtype直接断言错误。

原日志首步shared-backbone grad norm1.06484、action-head7.53754、aux0；完整首步梯度张量未保存，不补backward。参数数量、LR大小或grad norm本身都不是根因证据。本轮依据实际置换发现head敏感，而非依据参数量猜测。未定位新的实现错误。此前已确认的STOP问题在本轮两路径统一，无法解释这里仍存在的step1恶化。

## 7. 下一步唯一推荐

提出**仅将native action head参数组LR降为原来的1/10**的10次成功更新对照；VLM、proprio、auxiliary的LR及全部loss/data/seed/trainability保持原样，详见next_experiment_proposal.md与next_short_training_config.json。两组从共同step0重新开始、均采用统一STOP；历史无STOP训练不是严格同协议对照。保留step0/1/5/10，最终固定step10，不按验证选择模型。本轮未执行此训练。

## 8. 资源、保护与结论范围

GPU1不满足26GiB门槛，使用事先告知并核验的已授权GPU0；UUID/ECC基线与其他进程记录保存。仅冻结推理，无模型保存，ECC无增量。历史文件保护结果见delivery_verification.json，参数在每组合推理前后hash不变，置换仅在内存中进行。所有新缓存和结果在数据盘；没有读保留集产生新分数，没有闭环。

本轮能确定：head-step1在这些固定输入上引起主要动作变化，并提高验证示范动作误差；proprio贡献很小，VLM有较小且分任务不同的影响。不能确定：head LR降低一定恢复控制成功率、100步退化完全由head首步引起、或某一层是唯一根因。模块置换、7条代表示范、单训练seed和缺stove验证均限制外推。
'''
with (O/'report.md').open('x') as f:f.write(report)
proposal='''# 下一步：仅降低native action-head LR的10更新对照（未执行）

唯一主要组间改变：native Pro action head LR从1e-4降至1e-5。理由：仅加载head-step1就将固定验证动作L1从0.15268升至0.25213，三条验证示范均变差；单独VLM影响较小，proprio影响极小。没有定位新的实现错误；此方案验证优化敏感性，不预设降低LR可恢复闭环。

对照组：BC-only unified架构，各组LR均1e-4。实验组：同样BC-only，只有native head LR=1e-5。VLM/LoRA/action queries、proprio和保留的auxiliary head均1e-4；辅助权重均0，不新增物理、蒸馏或其他loss。可训练范围不变，不冻结backbone。两组都显式使用相同参数组划分，避免组结构不同。

从已核验共同step0参数开始，创建同样空AdamW状态，不能载入step1 optimizer。betas(.9,.999)、eps1e-8、weight_decay0、constant scheduler、BF16 native/FP32 LoRA、Math SDPA均不变；不加gradient clipping。microbatch2、累积2、有效batch4，原相同sampling_order前10行，seed20261021与原逐microbatch RNG规则相同，初始化与新增head相同。clean/background数据、语言/FK、归一化和背景协议不变。

两组训练loader共同补末尾STOP2、标签STOP与attention mask。共同实现该协议并通过eval模式训练forward/deploy动作max_abs=0后，才开始计数。历史Step5.7没有这一STOP，因此不能拿历史10步或100步直接当严格对照；必须重跑同STOP对照组。Training阶段原生0.02扰动保留，不能同时调整。

预算各10个成功optimizer updates；任何预检更新计入。保存step0/1/5/10并严格重载，主结果固定step10，不选最佳，不延长预算。记录每组动作/辅助0损失、参数与grad norm、dtype与moment状态、各任务曝光、峰值内存/ECC、跳过和恢复。若需技术smoke，避免额外未计数更新。

观察：同一固定诊断样本的误差/漂移和chunk/分量；原固定30验证窗口按3条示范等权、clean/background分别报告。夹爪诊断必须包括已有训练/验证中的真实开合窗口，先固定清单，不能依结果更换；当前14个代表窗口全为open，不能据它们判断抓取时序。不得用保留集调参。

进入后续闭环验证的条件：两组共同协议、初始化/数据顺序/RNG及10次成功预算匹配；STOP/eval等价、有限值/重载/保护/ECC均通过；离线分项与逐示范结果无未解释严重恶化。若降低head LR仍恶化，应交付并停止。只有在完整审查后另行冻结配对任务/状态和成功判据，才建议开展小规模闭环，不能由离线L1宣布控制收益。两组最终checkpoint均固定step10，验证不用来挑checkpoint或追加步数。

本轮仅方案，不执行任何训练或闭环。
'''
with (O/'next_experiment_proposal.md').open('x') as f:f.write(proposal)
print('ANALYSIS_AND_REPORT_PASS',flush=True)
