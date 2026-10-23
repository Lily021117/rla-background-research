"""Verify completed training/rollouts, then write evidence-bounded delivery."""
import argparse,json,csv,hashlib,subprocess,re
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--publish',type=Path,required=True);a=p.parse_args();root=a.run;plan=json.loads((root/'run_plan.json').read_text());prev=Path(plan['previous']);registry=json.loads((root/'policy_registry.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def save(name,obj):(root/name).write_text(json.dumps(obj,indent=2))
inventory=[];initials={};actual={}
for arm in ('C','D'):
 folder=Path(registry[arm]).parent;accept=json.loads((folder/'training_acceptance.json').read_text());assert accept['successful_updates']==600
 execution=json.loads((folder/'execution_configuration.json').read_text());encoder=Path(json.loads((prev/'active_models.json').read_text())[arm])/'encoder_step0600.pt'
 save(f'training_config_{arm}.json',dict(arm=arm,formal_training_scope='OFFICIAL_PAIRED_RLA_POLICY',entry='baseline_update_smoke.py --experiment-entry ae_consistency.policy_training',legacy_directory_name='isolated_smoke is only the reused bootstrap directory label; formal600 updates are verified by checkpoint scope and counters',common_step0=plan['common_step0'],common_step0_sha256=plan['common_step0_sha256'],shared=plan['shared_protocol'],actual_execution=execution,encoder=str(encoder),encoder_sha256=sha(encoder),sidecar_roots={c:str(prev/'sidecars'/arm/c) for c in ('clean','background')},inputs=plan['inputs'],memory_override='execution_resource_update.json'))
 metrics=list(csv.DictReader((folder/'training_metrics.csv').open()));assert [int(r['step']) for r in metrics]==list(range(1,601));assert all(np.isfinite(float(r[k])) for r in metrics for k in ('action_loss','auxiliary_loss','total_loss','gradient_vla','gradient_head','gradient_aux_head'))
 ck=torch.load(registry[arm],map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['scheduler']['last_epoch']==600
 assert ck['AMP_scaler'] is None and ck['activation_checkpointing']
 assert ck['optimizer']['state'] and all(int(v['step'])==600 for v in ck['optimizer']['state'].values())
 inventory.append(dict(model=arm,path=registry[arm],sha256=sha(registry[arm]),global_step=600,bytes=Path(registry[arm]).stat().st_size));del ck
 initials[arm]=torch.load(folder/'step0000.pt',map_location='cpu');actual[arm]=accept
for k in initials['C']['parameters']:
 assert initials['C']['parameters'][k].keys()==initials['D']['parameters'][k].keys()
 for n,x in initials['C']['parameters'][k].items():assert torch.equal(x,initials['D']['parameters'][k][n])
assert torch.equal(initials['C']['torch_rng'],initials['D']['torch_rng']) and all(torch.equal(x,y) for x,y in zip(initials['C']['cuda_rng'],initials['D']['cuda_rng']))
assert initials['C']['python_rng']==initials['D']['python_rng'];assert initials['C']['numpy_rng'][0]==initials['D']['numpy_rng'][0] and np.array_equal(initials['C']['numpy_rng'][1],initials['D']['numpy_rng'][1]) and initials['C']['numpy_rng'][2:]==initials['D']['numpy_rng'][2:];del initials
old=list(csv.DictReader((root/'C/isolated_smoke/training_metrics.csv').open()));new=list(csv.DictReader((Path(registry['C']).parent/'training_metrics.csv').open()));assert len(old)==1
replay={k:abs(float(old[0][k])-float(new[0][k])) for k in ('action_loss','auxiliary_loss','weighted_auxiliary','total_loss','gradient_vla','gradient_head','gradient_proprio','gradient_aux_head')}
save('C_policy_replay_audit.json',dict(replayed_successful_updates=1,C_effective_updates=600,C_computed_successes=601,first_update_log_absolute_differences=replay,maximum_difference=max(replay.values()),allocator_and_GPU_change='memory/runtime only, GPU0 to authorized GPU1; original1step preserved',no_hyperparameter_changes=True))
prior=json.loads((root/'initial_inventory.json').read_text());protected={path:sha(path)==digest for path,digest in prior['sha256'].items()};assert all(protected.values())
summary=json.loads((root/'success_rate_summary.json').read_text());assert summary['valid_trials']==180
rows=json.loads((root/'per_trial_results.json').read_text());assert all(Path(r['video']).is_file() and Path(r['trajectory']).is_file() for r in rows)
inventory.append(dict(model='B20',path=registry['B20'],sha256=sha(registry['B20']),global_step=600,bytes=Path(registry['B20']).stat().st_size))
with (root/'checkpoint_inventory.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
records={(r['model'],r['condition']):r for r in summary['summary']};inter=json.loads((root/'interaction_statistics.json').read_text())
lines=['# 官方 LIBERO VLA-RLA C/D训练及统一development闭环','', '状态：训练与评估流程完成；不等于方法有效性通过。研究定义：基于LIBERO的自定义背景材质干预协议，不是官方LIBERO-Plus。','', '## 固定条件与训练','', '20条训练/5条development，来源与episode隔离不变；同一B20采样清单，每个物理窗口Clean/Background各一次，microbatch8、累积2、有效16外观，BF16、Math SDPA、AdamW LR2e-4、WD.01，600成功更新；600之后scheduler降LR，没有额外低LR更新。C/D均从原始共同step0开始，不从B20 final开始。官方RLA结构相同，唯一算法差别是冻结AE-C/AE-D target；两组共同lambda_aux=0.22408536585365854及C训练集RLA中心化统计，无按development调权。','', 'B20与C/D使用相同观测外观及训练曝光规则。B20没有RLA查询/辅助头，C−B20代表完整RLA接入增量，不是等架构比较。C/D加入非reentrant activation checkpointing（Qwen/视觉/官方RLA头），保留RNG；另用expandable allocator和更新间释放空缓存，未改microbatch。B20未使用这些内存优化，浮点重算及额外RLA随机查询是可比性限制。','', f"C/D参数及CPU/CUDA/Python RNG初始状态核验；两组各600有效更新。C在GPU0第2步前向OOM，已计算1步未持久化；在授权GPU1从共同step0重放，实际601步。D实际600步。C首步重放日志最大差：{max(replay.values()):.9g}。原失败目录/日志保留。",'', '初次完整batch预检只在backward后预留optimizer内存，未覆盖下一步常驻状态，故并不充分；随后增加两个连续有效batch、前向期间常驻Adam moment等量存储的隔离预检，0更新，通过后才继续。GPU0上C一次OOM后未改LR/loss/数据；新进程GPU1运行。GPU1已有SRAM parity=1/SM=1风险记录，重新授权不是维修证明。','', '## AE-D恢复复核','', '恢复了匹配Encoder/Decoder、FP32 master、AdamW moments/step、CPU/CUDA/Python/NumPy RNG、采样位置和有效计数。AE固定LR，无scheduler；固定loss scale256，无动态AMP scaler。80步重放loss及梯度日志逐项一致，未见路径变化证据；AE-D有效600、实际680，AE-C有效/实际600。详见AE_D_recovery_audit.json；计时不包括全部排队/初始化/故障退出时间。','', '## 统一闭环：20个development物理状态','', '|模型|Clean|已见背景|未见背景|','|---|---:|---:|---:|']
for m in ('B20','C','D'):lines.append('|'+m+'|'+'|'.join(f"{records[m,c]['successes']}/20 ({records[m,c]['success_rate']:.0%})" for c in ('clean','seen_background','unseen_warm_teal'))+'|')
lines+=['','180有效闭环，0无效；同一状态三种外观只算一个物理状态。固定state/condition/model顺序，220控制步、chunk8、完整reset到1e-8；tex-table/tex-wall、反射1、MSAA4、固定已见/未见纹理、相同语言/FK proprio/action后处理。未读取封存测试集，没有运行旧5状态的额外回归闭环。','', '评估首次启动在trial前将状态数组内容SHA与NPY文件SHA误比较，已依据冻结生成代码修正为sha(state.tobytes())，未修改状态或协议；此时0次rollout，原日志和空目录保留，修正后在新目录执行。详见evaluation_pretrial_incident.json。','', '## 配对比较与不确定性','', '|比较|条件|成功率差|状态聚类bootstrap95%CI|胜/负翻转|','|---|---|---:|---|---|']
for r in summary['paired_comparisons']:
 if 'success_gain' in r:lines.append(f"|{r['comparison']}|{r['condition']}|{100*r['difference']:+.1f}pp|[{100*r['state_cluster_bootstrap95'][0]:+.1f},{100*r['state_cluster_bootstrap95'][1]:+.1f}]pp|{r['success_gain']}/{r['success_loss']}|")
lines+=['','跨三外观等权汇总（仍为20个物理状态聚类，而非60个独立样本）：','']
for r in summary['paired_comparisons']:
 if r['condition']=='mean_across_three_appearances':lines.append(f"- {r['comparison']}: {100*r['difference']:+.1f}pp，95%CI [{100*r['state_cluster_bootstrap95'][0]:+.1f},{100*r['state_cluster_bootstrap95'][1]:+.1f}]pp。")
lines+=['','背景交互（背景−Clean之差的差）：','']
for r in inter['interactions']:
 if r['comparison']=='D-C':lines.append(f"- {r['background']}: {100*r['interaction']:+.1f}pp，95%CI [{100*r['state_cluster_bootstrap95'][0]:+.1f},{100*r['state_cluster_bootstrap95'][1]:+.1f}]pp")
lines+=['','区间以20个物理状态为重采样单位，保留三模型/三外观的相关性；绝对SR同时提供Wilson区间。若全成功，经验bootstrap会退化，不能解释为真实成功率确定为100%。例如20/20的Wilson95%为约83.9%–100%。区间/多项比较均为探索性，未作多重比较校正。单任务、单训练seed、20个development状态，无独立最终测试，均限制外推。','', '## 离线验证与失败解释','', '固定5条development示范的最终动作L1（各示范等权、归一化空间）：','', '|模型|Clean|Background|','|---|---:|---:|']
validation=list(csv.DictReader((root/'validation_summary.csv').open()))
for m in ('B20','C','D'):
 values=[np.mean([float(v['action_L1']) for v in validation if v['model']==m and int(v['step'])==600 and v['appearance']==c]) for c in ('clean','background')];lines.append(f'|{m}|{values[0]:.6f}|{values[1]:.6f}|')
lines+=['','C离线L1略优于D，但闭环D只多1个成功；不能据离线L1选checkpoint或将其替代成功率。平移/旋转/夹爪和按示范数据见validation_summary.csv及原逐窗口日志。','', '前轮AE-D归一化跨外观距离改善，但latent norm/方差/有效秩明显下降，重建约增加1.8%；不是单一尺度改善，也不能据此证明运动信息保留。共同aux权重下，前轮D的辅助backbone梯度约为C的1.40倍；target整体差异/尺度与梯度强度是D−C解释的限制。不能仅根据策略成功/失败推断表示坍塌。','', 'failure_stage_summary.csv按统一、事前固定规则使用目标物体最大抬升和末端到目标距离作描述性阶段代理；不是接触/抓取物理真值。全部8个任务失败（B20为7、C为1）均运行至220步；均达到目标中心10cm以内，但未出现高于初始目标位置3cm的抬升，属于接近后抓取/接触阶段代理，而不是已确认的夹爪head错误。','', '按每个模型/外观的最小失败seed统一选取4组视频面板（failure_evidence/manifest.json）：可见机械臂接近后对位/接触、目标物体在桌面附近移动，未完成稳定搬运到目标。这里只是视觉观察，不是独立接触真值，无法唯一判定夹爪、朝向或路径哪项是根因。不得将初始高度参照下未超过3cm直接等同于从未发生任何小幅抬升。D无失败，不人为挑选成功解释机制。','', '## 成本、保护与决策','']
for arm in ('C','D'):lines.append(f"- {arm}: 600有效更新，记录训练/验收计时{actual[arm]['seconds']:.1f}s，峰值allocated {actual[arm]['peak_allocated']/1024**3:.2f}GiB。")
scores=[r['success_rate'] for r in summary['summary']]
lines+=['','结论：D相对C只修复已见背景1个失败；已见背景交互+5pp、CI[0,+15]pp，未见背景交互0且两组均20/20。未观察到Clean代价，但不能证明跨外观一致性的稳定额外收益。C/D明显天花板，尤其D为60/60，当前协议对D−C的区分能力不足。','', 'D−B20的跨外观聚类汇总区间在本次数据上高于0，是整体RLA方法相对action-only的探索性正向证据；这不等于一致性增量已获证明，更不能替代多训练seed/多任务/独立状态检验。C−B20跨外观汇总区间含0；D−C也含0。下一步优先预先扩大任务和独立合法状态的区分度，再做匹配多seed验证；不在本轮搜索让B失败而D成功的新背景或追加训练。','', '本轮未更新AE；策略有效更新C600/D600，C额外重放1；另有一次未完成更新的OOM前向，不计成功预算。新增有效闭环180；没有读取封存集。历史保护哈希检查通过。全部新产物位于数据盘。既有bootstrap目录名isolated_smoke不代表只做smoke：最终checkpoint scope为OFFICIAL_PAIRED_RLA_POLICY、global_step=600，完整optimizer/scheduler/RNG均核验。源码/配置/脱敏汇总同步私有研究分支，不上传权重、图像、sidecar、视频、原始轨迹或凭证。']
(root/'report.md').write_text('\n'.join(lines)+'\n')
save('delivery_verification.json',dict(status='PASS_PROCESS_NOT_METHOD',C_effective_updates=600,C_computed_successes=601,D_effective_updates=600,D_computed_successes=600,preflight_optimizer_updates=0,valid_rollouts=180,invalid_rollouts=0,physical_development_states=20,AE_updates_this_run=0,sealed_read=False,protected_hashes_unchanged=protected,common_C_D_initial_parameters_and_RNG=True,final600_only=True,video_trajectory_indices_complete=True))
a.publish.mkdir(exist_ok=False)
names=['report.md','run_plan.json','training_config_C.json','training_config_D.json','memory_and_execution_freeze.json','execution_resource_update.json','AE_D_recovery_audit.json','policy_memory_incident.json','evaluation_pretrial_incident.json','C_policy_replay_audit.json','checkpoint_inventory.csv','success_rate_summary.json','interaction_statistics.json','paired_state_results.csv','failure_stage_summary.csv','validation_summary.csv','delivery_verification.json']
def clean(s):return re.sub(r'GPU-[0-9a-f-]{36}','${GPU_UUID}',re.sub(r'/data/[^\s"\n,;`]+','${LOCAL_PATH}',s))
for n in names:(a.publish/n).write_text(clean((root/n).read_text()))
for arm in ('C','D'):
 folder=Path(registry[arm]).parent
 for n in ('execution_configuration.json','training_metrics.csv','training_acceptance.json'):(a.publish/f'{arm}_{n}').write_text(clean((folder/n).read_text()))
print('DELIVERY PASS',flush=True)
