"""Small post-run summaries; never executes a model, optimizer or rollout."""
import csv,json,re,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_shared_reference_target.experiment import R,HERE,C_RUN,SEEDS,read,save,table
out=Path(sys.argv[1]);dest=HERE/'delivery'/out.name
assert read(out/'delivery_verification.json')['valid_rollouts']==120
lossrows=[]
for seed in SEEDS:
 for method,path in [('C',C_RUN/f'seed_{seed}/C/formal/training_metrics.csv'),('C-shared-target',out/f'seed_{seed}/C-shared-target/formal/training_metrics.csv')]:
  records=list(csv.DictReader(path.open()));assert len(records)==600
  for lo,hi in [(1,100),(101,300),(301,600),(501,600)]:
   q=[r for r in records if lo<=int(r['step'])<=hi]
   lossrows.append(dict(order_seed=seed,method=method,start_step=lo,end_step=hi,updates=len(q),**{k:float(np.mean([float(r[k]) for r in q])) for k in ['action_loss','auxiliary_loss','weighted_auxiliary','total_loss','gradient_vla','gradient_aux_head']}))
table(out/'loss_contribution_comparison.csv',lossrows)
rates=list(csv.DictReader((out/'success_summary.csv').open()));rep=list(csv.DictReader((out/'representation_summary.csv').open()));stats=read(out/'paired_statistics.json');flips=list(csv.DictReader((out/'paired_results.csv').open()))
lines=['# 共享Clean参考目标：结果解释','',
 'AE-C冻结，只有Background样本的辅助target改为同一物理转移的Clean target。旧μC/sC和λaux保持不变。不是AE-D方法，也没有教师策略或显式一致性loss。','',
 '| 顺序 | 方法 | Clean | 已见背景 | 未见背景 | 三外观均值 |','|---|---|---|---|---|---|']
for seed in SEEDS:
 for method in ['C','D-center','C-shared-target']:
  rr=[r for r in rates if int(r['order_seed'])==seed and r['method']==method];assert len(rr)==3
  lines.append(f'| {seed} | {method} | '+' | '.join(r['successes']+'/20' for r in rr)+f" | {np.mean([float(r['SR']) for r in rr]):.2%} |")
primary=[s for s in stats['results'] if 'minus C,' in s['comparison'] and 'equal appearance' in s['comparison']]
lines+=['','## 主比较与限制','']
for s in stats['results']:
 if 'equal appearance' in s['comparison'] or 'two orders' in s['comparison']:lines.append(f"- {s['comparison']}: {100*s['difference']:+.2f}个百分点；物理状态聚类95%区间[{100*s['CI95'][0]:+.2f}, {100*s['CI95'][1]:+.2f}]。")
effects=[s['difference'] for s in primary];assert len(effects)==2
if all(x<0 for x in effects):lines.append('- 两种顺序均低于对应C：本轮没有观察到共享参考target的控制增益，保留负结果，不自动调权重。')
elif min(effects)<0<max(effects):lines.append('- 两种顺序的增量方向相反：结果依赖采样顺序，不能只展示有利顺序或宣称稳定收益。')
elif all(x>=0 for x in effects) and max(effects)>0:lines.append('- 当前固定条件下出现不劣/改善的探索性结果；仍不能据两个顺序证明普遍稳定或独立泛化。')
else:lines.append('- 没有两种顺序一致的正增量证据；相对D的恢复不能替代相对C的比较。')
lines+=['- development已反复用于开发；仅两个顺序、固定初始化与训练RNG、一个任务。区间将同一物理状态的全部外观和顺序一起重采样，不包含完整训练seed或任务不确定性。',
 '- 新方法及D参考的闭环在GPU0；历史C闭环在GPU1。设备差异明确记录，未声称执行硬件完全相同。','',
 '| 顺序 | 外观 | 新方法成功/C失败 | C成功/新方法失败 |','|---|---|---|---|']
for seed in SEEDS:
 for c in ['clean','seen_background','unseen_background']:
  rr=[r for r in flips if int(r['order_seed'])==seed and r['appearance']==c and r['reference']=='C']
  lines.append(f"| {seed} | {c} | {sum(int(r['difference'])>0 for r in rr)} | {sum(int(r['difference'])<0 for r in rr)} |")
lines+=['','## 背景交互','']
for s in stats['background_interactions']:
 if 'minus C ' in s['comparison']:lines.append(f"- {s['comparison']}: {100*s['difference']:+.2f}个百分点，95%区间[{100*s['CI95'][0]:+.2f}, {100*s['CI95'][1]:+.2f}]。")
lines+=['- 交互必须结合上表绝对成功率：Clean下降不能解释为鲁棒性提升。','',
 '## 实际预测表示（不是目标相等性）','',
 '固定每条训练/development示范首个及中间有效窗口，两种方法读相同观测。以下为development均值，均在既有C规范化坐标计算；误差参考为同一Clean target。','',
 '| 顺序 | 方法 | 跨外观预测L1 | 相对L2 | Clean参考误差L1 | Background参考误差L1 |','|---|---|---|---|---|---|']
for r in rep:
 if r['split']=='validation':lines.append('| '+r['order_seed']+' | '+r['method']+' | '+' | '.join(f"{float(r[k]):.6f}" for k in ['cross_prediction_L1','cross_prediction_relative_L2','clean_reference_L1','background_reference_L1'])+' |')
lines+=['- 目标相同是人为定义，不能作为学到不变性的证据；预测差异减小也可能伴随预测误差或控制能力变差。不能用表示指标替代闭环结果。',
 '- loss_contribution_comparison.csv提供原C/新方法各训练区间的动作loss、原始/加权辅助loss和已记录梯度范数。两个方法监督目标有变化，不按辅助loss绝对值给方法排名。',
 '- failure_stage_summary.csv保留抓取/接触和抬升后/放置等冻结代理分类；这些不是精确物理因果标签。','',
 '## 执行与保护','',
 '- 两组各600次成功更新，共1200；实际更新计算1200，重放0。120次有效闭环、120次尝试、无效0。',
 '- AE/C/D更新0，封存集读取0。关闭共享模式输出/loss/梯度与旧C路径精确一致；Background图像保留；target无梯度且不进入推理输出。',
 '- 两组初始参数与RNG、600步实际采样ID及除方法名外的执行配置均与对应C匹配。历史受保护文件哈希未变。完成后停止，不自动扩量或搜索。',
 '- 一次冻结表示诊断因独立随机补齐的动作token ID不相同而断言停止。已保留失败日志；官方RLDSBatchTransform会random.choices补齐64个动作token位置，模型随后将这些位置替换为action queries，连续BC标签另取batch actions。修正仅匹配诊断的两路预处理Python RNG及方法间样本随机序列，并验证原差异只位于替换位置；没有修改训练、权重、部署或闭环结果。重试新增训练更新0、backward0、闭环0。']
(out/'results_interpretation.md').write_text('\n'.join(lines)+'\n')
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True);workers=[s for s in processes.splitlines() if 'bootstrap.py' in s and str(out) in s];assert not workers
gpu=subprocess.check_output(['nvidia-smi','-i','GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e','--query-gpu=memory.free,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True)
save(out/'resource_release_verification.json',dict(task_GPU_workers_remaining=workers,task_GPU_processes_released=True,GPU0_final_check=gpu,other_users_processes_untouched=True))
save(out/'token_padding_audits.json',{str(seed):read(out/f'seed_{seed}/representation'/('matched_preprocessing' if seed==20261101 else 'formal')/'token_padding_audit.json') for seed in SEEDS})
for name in ['results_interpretation.md','loss_contribution_comparison.csv','resource_release_verification.json','diagnostic_retry_audit.json','token_padding_audits.json']:
 text=(out/name).read_text().replace(str(R),'<PROJECT_ROOT>');text=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',text)
 assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',text)
 with (dest/name).open('x') as f:f.write(text)
print('RESULT_INTERPRETATION_COMPLETE')
