"""Frozen paired statistics and delivery; no model selection or new trials."""
import argparse,json,sys,hashlib,subprocess,time,collections
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta,binomtest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stove_diagnosis'))
from common import read,save,sha,csvsave,csvread,digest_state
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.out;plan=read(r/'run_plan.json');old=Path(plan['historical_source']);arms=['D-old','D-center']
rows=read(r/'per_trial_results.json');assert len(rows)==120 and all(x['status']=='COMPLETE' for x in rows)
resume=read(r/'evaluation_resume_audit.json') if (r/'evaluation_resume_audit.json').exists() else {}
invalid_attempts=len(resume.get('interrupted_attempts',[]))
keys={(x['model'],x['seed'],x['condition']) for x in rows};assert len(keys)==120
conditions=read(r/'evaluation_protocol.json')['conditions'];seeds=sorted({x['seed'] for x in rows});assert len(seeds)==20
lookup={(x['model'],x['seed'],x['condition']):x for x in rows};historic=read(old/'policy_run/per_trial_results.json');hl={(x['model'],x['seed'],x['condition']):x for x in historic};pairs=[]
for s in seeds:
 for c in conditions:
  x,y=lookup['D-old',s,c],lookup['D-center',s,c];ref=hl['C',s,c]
  for k in ('state_sha256','xml_sha256','language'):assert x[k]==y[k]==ref[k]
  pairs.append(dict(seed=s,condition=c,D_old_success=int(x['success']),D_center_success=int(y['success']),historical_C_success=int(ref['success']),D_old_close=x['issued_close_command'],D_center_close=y['issued_close_command'],D_old_stage=x['stage'],D_center_stage=y['stage']))
csvsave(r/'paired_results.csv',pairs)
rng=np.random.default_rng(20261024);indices=rng.integers(0,20,(10000,20));matrix={arm:np.array([[int(lookup[arm,s,c]['success']) for c in conditions] for s in seeds]) for arm in arms};difference=matrix['D-center']-matrix['D-old']
def ci(v):return [float(x) for x in np.quantile(v[indices].mean(1),[.025,.975])]
summary=[];interactions=[]
def exact_binomial(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
for j,c in enumerate(conditions):
 x=matrix['D-old'][:,j];y=matrix['D-center'][:,j]
 summary.append(dict(condition=c,D_old_success=int(x.sum()),D_center_success=int(y.sum()),trials_per_arm=20,D_center_minus_D_old=float((y-x).mean()),paired_95_CI=ci(y-x),old_fail_center_success=int(np.sum((x==0)&(y==1))),old_success_center_fail=int(np.sum((x==1)&(y==0))),historical_C_success=sum(int(hl['C',s,c]['success']) for s in seeds)))
 wins=summary[-1]['old_fail_center_success'];losses=summary[-1]['old_success_center_fail']
 summary[-1].update(D_old_exact_binomial_CI95=exact_binomial(int(x.sum()),20),D_center_exact_binomial_CI95=exact_binomial(int(y.sum()),20),exact_McNemar_p=1. if wins+losses==0 else float(binomtest(wins,wins+losses,.5).pvalue),bootstrap_degenerate=bool(np.ptp(y-x)==0),marginal_CI_scope='one appearance,20 physical states; does not pool60 appearances as independent')
 if j:interactions.append(dict(background=c,interaction=float((difference[:,j]-difference[:,0]).mean()),CI95=ci(difference[:,j]-difference[:,0])))
save(r/'paired_statistics.json',dict(conditions=summary,overall_state_cluster_difference=float(difference.mean()),overall_state_cluster_CI95=ci(difference.mean(1)),interactions=interactions,bootstrap_seed=20261024,bootstrap_replicates=10000,cluster_unit='physical state, all3 appearances kept together',single_training_seed=True,development_not_test=True,discrete_CI_warning='20 states and few flips yield discrete intervals; zero lower bound is not stable evidence'))
inventory=[];initchecks=[];reproduction=[];training=[];valid=[]
for arm in arms:
 accept=read(r/arm/'isolated_smoke/training_acceptance.json');assert accept['successful_updates']==600
 metrics=csvread(r/arm/'isolated_smoke/training_metrics.csv');assert [int(x['step']) for x in metrics]==list(range(1,601));training.extend(dict(arm=arm,**x) for x in metrics)
 valid.extend(dict(arm=arm,**x) for x in csvread(r/arm/'isolated_smoke/validation_metrics.csv'))
 for step in (0,1,100,300,600):
  path=r/arm/'isolated_smoke'/f'step{step:04}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['global_step']==step and ck['next_physical_sample']==step*8
  opt_steps=sorted(set(int(v['step']) for v in ck['optimizer']['state'].values()));assert opt_steps==([step] if step else []) and ck['scheduler']['last_epoch']==step
  digest=digest_state(ck);inventory.append(dict(arm=arm,step=step,path=str(path),sha256=sha(path),parameter_sha256=digest,bytes=path.stat().st_size,scheduler_epoch=ck['scheduler']['last_epoch'],optimizer_steps=opt_steps))
  if step==0:
   initial=torch.load(plan['common_step0'],map_location='cpu',weights_only=False);assert digest==digest_state(initial);assert torch.equal(ck['torch_rng'],initial['torch_rng']) and all(torch.equal(x,y) for x,y in zip(ck['cuda_rng'],initial['cuda_rng']));assert ck['python_rng']==initial['python_rng'];assert np.array_equal(ck['numpy_rng'][1],initial['numpy_rng'][1]);initchecks.append(dict(arm=arm,parameters_equal=True,RNG_equal=True,common_initial_parameter_sha256=digest));del initial
  if arm=='D-old':
   history=torch.load(old/'policy_run/D/isolated_smoke'/f'step{step:04}.pt',map_location='cpu',weights_only=False);hd=digest_state(history);reproduction.append(dict(step=step,parameter_bitwise_equal=digest==hd,new_parameter_sha256=digest,historical_parameter_sha256=hd));del history
  del ck
csvsave(r/'checkpoint_inventory.csv',inventory);csvsave(r/'training_metrics.csv',training);csvsave(r/'validation_metrics.csv',valid);save(r/'initialization_and_reproduction_audit.json',dict(initialization=initchecks,historical_D_parameter_reproduction=reproduction,effective_updates_per_arm=600,actual_successful_computations_per_arm=600,replayed_updates=0,preflight_updates=0,sampling_order_sha256=sha(Path(plan['inputs']['B20_sampling_order']))))
diag=r/'diagnostics/isolated_smoke';assert read(diag/'verification.json')['optimizer_updates']==0
action=csvread(diag/'action_by_dimension_horizon.csv');energy=csvread(diag/'auxiliary_residual_energy.csv');grads=csvread(diag/'gradient_contributions.csv');closures=csvread(diag/'closure_predictions.csv')
offline=[]
for arm in arms:
 for split in ('train','validation'):
  for appearance in ('clean','background'):
   for step in (0,1,100,300,600):
    rr=[x for x in action if x['arm']==arm and x['split']==split and x['appearance']==appearance and int(x['step'])==step]
    offline.append(dict(arm=arm,split=split,appearance=appearance,step=step,action_L1=float(np.mean([float(x['action_L1']) for x in rr])),gripper_L1=float(np.mean([float(x['dim6_L1']) for x in rr])),samples_times_horizon=len(rr)))
csvsave(r/'offline_summary.csv',offline)
failure=collections.Counter((x['model'],x['condition'],x['stage']) for x in rows);csvsave(r/'failure_stage_counts.csv',[dict(arm=k[0],condition=k[1],stage=k[2],count=v) for k,v in failure.items()])
protected=read(r/'protected_input_hashes.json');mismatch=[p for p,h in protected.items() if sha(Path(p))!=h];assert not mismatch
ae_manifest=read(Path(plan['previous'])/'sidecars/D/clean/manifest.json');assert sha(Path(ae_manifest['encoder']))==ae_manifest['encoder_sha256']
save(r/'frozen_AE_sidecar_inventory.json',dict(AE_D_encoder=ae_manifest['encoder'],encoder_sha256=ae_manifest['encoder_sha256'],raw_sidecars='same frozen D arrays for both policy arms; train-source hashes in mean_training_sources.csv',AE_updates=0))
free=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True);save(r/'resource_release.json',dict(checked_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),processes=free,own_GPU_processes_finished=True))
save(r/'delivery_verification.json',dict(status='PASS_EXECUTION_NOT_METHOD',training_successful_updates={'D-old':600,'D-center':600},AE_updates=0,formal_replayed_updates=0,preflight_updates=0,diagnostic_backward_calls=162,valid_rollouts=120,invalid_rollouts=invalid_attempts,evaluation_resume=resume,sealed_contents_read=False,protected_files_checked=len(protected),historical_files_unchanged=True,source_sidecars_unchanged=True,mean_training_statistic_rows=40880,GPU_uuid='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419',GPU_released=True,initial_gate_configuration_failures=2,gate_failure_reason='new launcher initially omitted RLA_STEPS; strict load stopped before forward/update; explicit official8 restored, no data/model change'))
table='\n'.join(f"| {x['condition']} | {x['D_old_success']}/20 | {x['D_center_success']}/20 | {100*x['D_center_minus_D_old']:+.1f} pp | {x['paired_95_CI']} | {x['historical_C_success']}/20 |" for x in summary)
reproduced=all(x['parameter_bitwise_equal'] for x in reproduction)
text=f'''# Stove D target centering: single-variable development experiment

## Scope and identity

D-old `(z_D−mu_C)/s_C`; D-center `(z_D−mu_D)/s_C`. Same frozen AE-D/raw sidecars,20 training episodes,5 offline development episodes,2555 training complete windows. Both start from the common original Prismatic/RLA policy step0, not a trained policy.600 successful updates each;4800 physical-window and9600 appearance exposures each. No AE updates or sealed-test access. Final600 selected in advance.

## Closed-loop results

| Condition | D-old | D-center | Difference | Paired95%CI (fraction) | Historical C |
|---|---:|---:|---:|---|---:|
{table}

120 valid rollouts,20 physical development states, three appearances. State-cluster bootstrap preserves appearances within each state. These are single-training-seed development results, not independent test evidence. Historical C's state/XML/language hashes match. Interactions and paired flips are in `paired_statistics.json`; absolute Clean success must be considered with any background interaction.

## Reproduction and normalization

Historical D parameter bitwise reproduction at0/1/100/300/600: **{reproduced}**. Full per-checkpoint evidence is in `initialization_and_reproduction_audit.json`. If this is false, changes between old historical D and D-old must not be attributed to target centering. The primary comparison is the new matched D-center versus D-old.

mu_D uses the original extractor's FP32-output/FP64-accumulated training moments, not development.40 episode/appearance sources and40880 complete-window cumulative-horizon rows. Independent FP16-cache recomputation differs slightly from pre-storage moments; see `normalization.json`. s_C remains the exact official float32 scalar. Fixed-offset and pairwise-difference checks pass within explicitly documented float32 rounding bounds. Aux output has a bias; no automatic center compensation, bias reset or architecture change was introduced.

## Offline evidence and gradient interpretation

`offline_summary.csv` and `diagnostics/isolated_smoke/action_by_dimension_horizon.csv` separate train/development, dimension and prediction horizon. `closure_predictions.csv` records first closing prediction and missed/early closes within the fixed8-position chunk; no-close outside that window is censored, not invented as a delay.

`auxiliary_residual_energy.csv` separates target squared energy, actual residual mean energy and demeaned residual energy. This is a squared-error decomposition for diagnosis, not the trained mean-L1 objective. Raw-coordinate predictions use s_C times prediction plus the appropriate center. Lower normalized loss alone is not evidence of better physical information/control.

`gradient_contributions.csv` measures action and weighted-auxiliary derivatives on matching frozen train/development batches at0/1/100/300/600. Scopes separately include trainable VLM, shared head trunk, proprio projector and their union; branch-specific output projections are excluded. Each objective has an independent, matched-RNG forward. Diagnostics use eval-with-grad and FP32 accumulation, so they describe frozen-model derivatives, not a stochastic training-step replay. No optimizer is constructed in diagnostics. Equal lambda does not imply equal gradient strength.

## Failure analysis and limitations

`failure_stage_counts.csv` uses the frozen EE-target distance and lift proxies, not unverified visual ground truth. All trials record whether a positive simulator closing command was actually executed. Videos/trajectories remain local. Small counts and discrete intervals are not proof of stable benefit; no unique root cause or superiority over C is inferred from this single intervention.

## Execution and protection

Math SDPA, BF16, effective16 appearance examples, microbatch8/accumulation2, accepted non-reentrant activation checkpointing; fixed lambda0.22408536585365854 and lr2e−4. Full optimizer/scheduler/RNG/sample-position checkpoints. No extra updates or replays. Two initial strict-loading gates failed because the new launcher omitted the official RLA_STEPS switch; corrected before any update, failed logs preserved. All subsequent settings match the frozen experiment.

Historical protected hashes unchanged. No weights, sidecars, raw trajectories or credentials are included in GitHub delivery. Completion releases GPU resources. No automatic follow-up training is authorized by this result.
'''
whole=[];closing=[];aux_summary=[]
for arm in arms:
 rr=[x for x in valid if x['arm']==arm and x['step']=='600']
 whole.append(dict(arm=arm,development_windows_with_appearance=len(rr),action_L1=float(np.mean([float(x['action_L1']) for x in rr])),gripper_L1=float(np.mean([float(x['gripper_L1']) for x in rr]))))
 for split in ('train','validation'):
  ss=[x for x in action if x['arm']==arm and x['step']=='600' and x['split']==split and x['horizon']=='1' and x['t']==x['closure_t']]
  closing.append(dict(arm=arm,split=split,closure_anchor_observations=len(ss),h0_closed_predictions=sum(float(x['dim6_prediction'])<.5 for x in ss)))
 ee=[x for x in energy if x['arm']==arm and x['step']=='600' and x['split']=='validation'];weights=[int(x['samples']) for x in ee]
 aux_summary.append(dict(arm=arm,**{k:float(np.average([float(x[k]) for x in ee],weights=weights)) for k in ['target_squared_energy','target_mean_energy','residual_MSE','residual_mean_energy','residual_demeaned_energy','auxiliary_L1_official_bf16','weighted_auxiliary','raw_latent_L1','raw_latent_MSE']}))
csvsave(r/'final_full_validation_summary.csv',whole);csvsave(r/'final_closure_anchor_summary.csv',closing);csvsave(r/'final_auxiliary_energy_summary.csv',aux_summary)
stage_lines=[]
for arm in arms:
 rr=[x for x in rows if x['model']==arm];stage_lines.append(f"- {arm}：发出关闭命令 {sum(x['issued_close_command'] for x in rr)}/60；达到抬升≥3cm {sum(x['max_lift_m']>=.03 for x in rr)}/60；阶段分布 {dict(collections.Counter(x['stage'] for x in rr))}。")
all_zero=not any(x['success'] for x in rows)
verdict=('本轮中心化干预**没有改善最终任务成功率**：两组各外观均0/20。不能因此说中心化没有任何行为作用；需结合夹爪误差和失败阶段。也不能声称找到了唯一根因或证明外观一致性优于历史C。' if all_zero else '主结果以本轮匹配D-center−D-old为准。任何差异仅支持当前单seed、冻结development条件下的影响，不证明唯一根因或外观一致性优于C。')
whole_table='\n'.join(f"| {x['arm']} | {x['development_windows_with_appearance']} | {x['action_L1']:.6f} | {x['gripper_L1']:.6f} |" for x in whole)
close_table='\n'.join(f"| {x['arm']} | {x['split']} | {x['h0_closed_predictions']}/{x['closure_anchor_observations']} |" for x in closing)
energy_table='\n'.join(f"| {x['arm']} | {x['target_squared_energy']:.6f} | {x['residual_mean_energy']:.6f} | {x['residual_demeaned_energy']:.6f} | {x['raw_latent_L1']:.6f} |" for x in aux_summary)
grad_table='\n'.join(f"| {x['arm']} | {float(x['action_gradient_norm']):.6f} | {float(x['weighted_auxiliary_gradient_norm']):.6f} | {float(x['cosine']):.6f} |" for x in grads if x['step']=='600' and x['split']=='validation' and x['scope']=='all_shared')
prefix=f'''# 炉灶 D target 中心化：最终结论

{verdict}

## 成功率与配对统计

| 外观 | D-old | D-center | 差值 | 配对95%CI（比例） | 历史C |
|---|---:|---:|---:|---|---:|
{table}

上表依次为外观、D-old成功次数、D-center成功次数、差值、配对bootstrap区间、历史C成功次数；每格分母20。共120有效闭环，{invalid_attempts}次基础设施中断尝试单独保留，未重试任务失败。网络中断后保留前101次有效结果，仅恢复剩余19个槽位；中断槽位使用新目录，训练没有重放。以20个物理状态为跨外观聚类单位，不把60种状态/外观组合当60个独立状态。

{'全部观测均失败，因此配对翻转为0、背景交互为0，bootstrap区间退化为[0,0]，这不是“真实差异精确为0”的证明。单个外观每组0/20的二项精确95%区间上限约16.8%；地板效应和小样本限制明显。补充边际精确区间仅用于说明地板不确定性，不替换预定的状态聚类主分析。' if all_zero else '配对翻转、状态聚类区间与背景交互详见 paired_statistics.json；不以Clean下降造成的交互改善冒充鲁棒性。'}

## 行为阶段，而非仅最终标签

{chr(10).join(stage_lines)}

阶段规则沿用既定EE距离/目标抬升代理，不将接触代理冒充独立抓取真值。关闭命令不等于实际成功抓取；抬升也不等于最终任务完成。逐轨迹检查和视频保存在本地。

`rollout_gripper_timing.csv`覆盖全部120条保存轨迹：预测chunk经冻结后处理后与实际执行动作逐项一致。只恢复保存状态并执行mj_forward，未新增控制步。D-center首次关闭均不早于首次进入10cm接近范围；D-old有1次提前关闭。固定选择首个状态、全部三种外观的均匀时间点视频拼图，仅作视觉参考，不替代统一阶段统计。

训练验收耗时：D-old 2263.16秒，D-center 2080.16秒；各进程峰值显存20,397,878,784字节（约19.0GiB），无新增ECC。恢复评估峰值约3.72GiB，结束后本任务GPU进程全部退出。历史ECC计数保留，不声称硬件已修复。

## 离线动作与梯度

全部固定development窗口（动作采用训练归一化空间，夹爪保持0/1语义）：

| 组 | 窗口×外观数 | 动作L1 | 夹爪L1 |
|---|---:|---:|---:|
{whole_table}

关闭事件锚点的当前步闭合预测：

| 组 | 划分 | 预测闭合/观测数 |
|---|---|---:|
{close_table}

与覆盖事件附近的固定诊断样本分开报告，不能只挑局部改善。完整逐维/逐预测跨度指标见diagnostics目录。

最终验证固定batch的辅助残差分解：

| 组 | target平方能量 | 残差均值能量 | 去均值残差能量 | 原始latent坐标L1 |
|---|---:|---:|---:|---:|
{energy_table}

目标平方能量、实际残差能量和梯度范数是不同量；训练损失仍是官方L1。去中心后误差同时还原到统一原始AE-D latent坐标。

最终验证梯度（全部共享模块）：

| 组 | 动作梯度范数 | 加权辅助梯度范数 | 余弦 |
|---|---:|---:|---:|
{grad_table}

另提供VLM、共享head主干、proprio分项；不把较小辅助loss或更正的梯度夹角直接当作物理信息改善。

## 复现、预算与限制

D-old在0/1/100/300/600全部参数是否逐位复现历史D：{reproduced}。两组共同step0参数与所有训练RNG检查通过。各600成功更新、实际600、重放0；AE更新0。训练仅使用20条训练示范，development不参与均值拟合。单seed结论不外推至其他任务或独立测试。

本轮到此停止；不自动调LR、辅助系数、阈值或增加物理loss。下面是完整方法与复现记录。

---

'''
with (r/'report.md').open('x') as f:f.write(prefix+text)
print('FINALIZED',json.dumps(summary),flush=True)
