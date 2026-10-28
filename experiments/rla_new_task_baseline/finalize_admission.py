"""Publish a blocked data-admission result, not a completed training experiment."""
import argparse,csv,json,subprocess
from pathlib import Path
from audit_source import sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();r=a.out
 source=json.loads((r/'source_audit/candidate_split.json').read_text());summary=json.loads((r/'source_audit/source_summary.json').read_text())
 ae=a.project/'experiments/official_libero_rla_consistency/run_20261023_01/downloads/extracted/runs/16x64_libero/20260725_00-39-52/ckpts'
 expected={'encoder_step0090000.pt':'a22efe9311f1588675f19c83f5a2af3afb1dd221e59c8170fb64ced60037086c','decoder_step0090000.pt':'916952b1da241492db095d959ce850eed2f77693b380e9f8b9756b1c7612ff1d'}
 models=[]
 for name,digest in expected.items():
  actual=sha(ae/name);assert actual==digest
  models.append(dict(file=str(ae/name),sha256=actual,step=90000,historical_strict_load_reused=True,new_forward=False))
 save(r/'encoder_inventory.json',dict(status='AVAILABLE_HASH_MATCHED',models=models,architecture='official joint dual-view Plucker 16x64',frozen=True,training_source='official LIBERO converted trajectories; precise task exclusion not established',unseen_task_generalization_claim=False))
 plan=dict(status='BLOCKED_BEFORE_DATA_ADMISSION',task='libero_goal/open_the_middle_drawer_of_the_cabinet',parent_commit='0d8a04c25ebc07b821f52647e020decbc30ec103',methods=['BC-standard','BC-matched','C-frozen-RLA'],sampling_orders=[20261101,20261102],updates_per_group=600,total_planned_updates=3600,lambda_aux=.22408536585365854,AE_updates=0,teacher=False,keep=False,initialization='untrained common Prismatic policy origin; no task-trained checkpoint',target='frozen original step90000, appearance-specific cumulative targets; train-only new-task statistics',sample_indices='NOT_GENERATED: no accepted physical windows',normalization='NOT_FITTED: training admission blocked',development_states='NOT_GENERATED: no frozen eligible data/interface yet',backgrounds='NOT_FROZEN: no model scoring performed',clean_trials_planned=120,background_trials_max=480,gate='some one method Clean >=12/20 in BOTH orders; otherwise stop without background scoring',max_steps=300,control_freq=20,execution_chunk=8,max_steps_source='official VLA-Adapter run_libero_eval.py TASK_MAX_STEPS LIBERO_GOAL; not stove220',actual_policy_updates=0,actual_clean_trials=0,actual_background_trials=0)
 save(r/'requested_protocol.json',plan)
 evidence=[Path(source['source']),a.data_root/'datasets/LIBERO-Cosmos-Policy/README.md',a.data_root/'datasets/LIBERO-Cosmos-Policy/success_only/libero_goal_regen/libero_goal_metainfo.json',a.data_root/'projects/cosmos-policy/cosmos_policy/experiments/robot/libero/regenerate_libero_dataset.py',a.data_root/'projects/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot/meta/info.json',a.project.parent/'LIBERO-plus/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl']
 save(r/'source_inventory.json',[dict(path=str(x),sha256=sha(x)) for x in evidence])
 assert sha(Path(source['source']))==source['source_sha256']
 report='''# 新操作任务基线筛查：数据准入 BLOCKED

本轮没有启动训练或策略闭环。任务已找到，但现有源数据尚不能可靠接入被冻结的同状态配对/相机几何管线。不能据此判断基线学不会，也不能判断没有外观缺口。没有换任务、下载替代数据或扩大协议。

## 任务与来源

任务固定为 `libero_goal/open_the_middle_drawer_of_the_cabinet`。本地BDDL语言为 `Open the middle layer of the drawer`，goal 为 `(Open wooden_cabinet_1_middle_region)`，不是打开抽屉取碗放盘。本地 `WoodenCabinet.is_open` 采用关节 qpos < -0.14 的 Open 条件（本地 articulated_objects.py:181–188，base_predicates.py:101–103）。正式评估预算如获数据放行应沿用 LIBERO-Goal 300 控制步，而非炉灶220步。

官方定义：[LIBERO任务BDDL](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl)。本地BDDL SHA256见 source_summary.json。

仅找到一份该任务含仿真states的本地HDF5：`<DATA_ROOT>/datasets/LIBERO-Cosmos-Policy/success_only/libero_goal_regen/open_the_middle_drawer_of_the_cabinet_demo.hdf5`。它有42条成功再生成示范，不是原始LIBERO带XML轨迹。按数字episode ID冻结前20条训练候选、随后5条development候选；其余17条仅检查字段元数据，没有换用策略表现更好的候选。名单仍是候选，不标记已验收。

## 已确认的阻塞和可恢复部分

1. 42条示范均无 `model_file` XML，HDF5全局/episode也没有环境版本、模型布局或相机标定来源。辅助metainfo只有success与initial_state，不提供XML或源环境配置。
2. 选定25条的第一个状态间隔均不是20Hz。以demo_0为例，state时间为0.25、0.80、0.85秒。再生成代码先稳定等待10次dummy动作，却在首项保存等待前的 `orig_states[0]`；首项图像/观测来自等待后。代码证据：`regenerate_libero_dataset.py` 的稳定等待、`if states == []` 和执行动作前记录观测分支。首帧图像与state[0]不能直接配对，第一组累计horizon也不能冒称8个等间隔真实转移。
3. 第一项之后的states保持20Hz，这是可能恢复的片段，不代表整份数据没有价值。明确排除首项可以处理时间异常，但**单靠删首帧不能证明缺失的源模型、相机和版本一致**，本轮没有悄悄执行该替代方案。
4. 已做无物理推进的恢复候选检查：当前本地任务模型维度与79维state相容；demo_0在t=0、1、10可设置，均非成功。源JPEG与当前模型渲染不逐像素相等；原始记录有JPEG、引擎与传感器刷新差异，不能用这个差异直接断言几何错误，也不能仅以外观相似认证源模型。检查表和并排图保存在source_audit，仅作恢复证据，不放行训练。
5. `all_episodes`对应任务文件只有两路JPEG、9维proprio和actions，没有完整states/XML；FastWAM本地LeRobot版本的schema只有机器人状态/动作/视频，没有完整仿真state/XML。均不能独立补足缺口。没有读取其他任务轨迹或封存初态。

## 编码器与未执行项

官方原始step90000 encoder/decoder存在，SHA256分别为 `a22efe9311f1588675f19c83f5a2af3afb1dd221e59c8170fb64ced60037086c`、`916952b1da241492db095d959ce850eed2f77693b380e9f8b9756b1c7612ff1d`，匹配历史严格加载记录。保持双视角16×64、Plucker接口，未加载炉灶AE。其公开训练来源是LIBERO converted；尚不能排除包含本任务，不能宣称未见任务泛化。

未生成配对序列/sidecar、归一化、训练索引或新策略权重；未评分Clean，因而也未触发基于Clean能力的背景放行/停止判定。这里是更早的数据准入停止，不是Clean门槛失败。

## 继续所需的最小证据

优先提供此任务原始HDF5（逐轨迹模型XML、完整states/actions、双视角观测及生成代码版本），或该再生成数据对应的可追溯源模型/环境版本和状态布局。补齐后可核验候选模型与实际源记录，显式排除不一致首帧并登记派生样本映射，再重新审查20+5候选。不能仅凭模型尺寸相同绕过此验收。本轮不擅自下载整个benchmark、采集新示范或改任务。

## 实际执行计数

- 策略optimizer updates=0；重放更新=0；AE/fencoder/DINO updates=0。
- 模型forward=0；backward=0；Clean闭环=0；Background闭环=0；无效闭环=0。
- CPU候选环境reset=2；保存状态设置=6（首次只读预检3次、持久化审计3次）；物理step=0；两相机静态渲染6张。
- 历史HDF5首尾哈希一致；旧代码、checkpoint和结果未修改。新分支源于指定提交，不混入主目录未提交改动。没有GPU训练进程或本任务GPU资源需要释放。

本轮结论为数据/接口准备BLOCKED，非方法负结果。按用户要求停止所有依赖训练与评估步骤。
'''
 (r/'report.md').write_text(report)
 save(r/'delivery_verification.json',dict(status='BLOCKED_DATA_ADMISSION',policy_updates=0,actual_update_computations=0,replayed_updates=0,AE_updates=0,fencoder_updates=0,DINO_updates=0,model_forwards=0,backwards=0,clean_rollouts=0,background_rollouts=0,attempted_policy_rollouts=0,invalid_policy_rollouts=0,environment_resets=2,state_sets=6,physics_steps=0,static_renders=6,sealed_contents_read=False,source_hdf5_hash_unchanged=True,historical_writes=False,gpu_processes_launched=0,capability_gate='NOT_REACHED'))
 delivery=Path(__file__).parent/'delivery/run_20261028_01';delivery.mkdir(parents=True,exist_ok=False)
 def sanitize(x):
  if isinstance(x,str):return x.replace(str(a.project),'<PROJECT_ROOT>').replace(str(a.data_root),'<DATA_ROOT>')
  if isinstance(x,list):return [sanitize(y) for y in x]
  if isinstance(x,dict):return {sanitize(k):sanitize(v) for k,v in x.items()}
  return x
 for pth in [r/'report.md',r/'requested_protocol.json',r/'encoder_inventory.json',r/'source_inventory.json',r/'delivery_verification.json',*[(r/'source_audit'/n) for n in ['candidate_split.json','source_summary.json','trajectory_availability.csv','recovery_candidate_audit.json']]]:
  target=delivery/pth.name
  if pth.suffix=='.json':save(target,sanitize(json.loads(pth.read_text())))
  else:target.write_text(sanitize(pth.read_text()))
if __name__=='__main__':main()
