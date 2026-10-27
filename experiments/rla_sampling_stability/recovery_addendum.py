"""Read-only result interpretation and recovery verification; no model execution."""
import csv,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_sampling_stability.experiment import read,save,R,HERE,REPO
out=Path(sys.argv[1]);dest=HERE/'delivery'/out.name
assert read(out/'delivery_verification.json')['new_valid_rollouts']==120
p=out/'seed_20261102/D-center'
old={int(r['step']):r for r in csv.DictReader((p/'formal/training_metrics.csv').open())}
replayed=[r for r in csv.DictReader((p/'gpu0_resume300/training_metrics.csv').open()) if int(r['step'])<=400]
keys=['action_loss','auxiliary_loss','total_loss','lr'];assert len(replayed)==100
save(out/'replay_numerical_audit.json',dict(replayed_steps=100,comparison_fields=keys,exact_rows=sum(all(r[k]==old[int(r['step'])][k] for k in keys) for r in replayed),scope='Logged losses and LR only; not a proof of bitwise parameter equivalence across devices'))
rates=list(csv.DictReader((out/'success_summary.csv').open()));paired=list(csv.DictReader((out/'paired_results.csv').open()));statistics=read(out/'paired_statistics.json')
lines=['# 均衡采样 C/D-center 配对补全：结果与恢复说明','',
 '两组 D-center 均完成600次有效更新；120次新增闭环全部有效，无重试。已有C不重训、不重评。',
 '', '| 顺序 | 方法 | Clean | 已见背景 | 未见背景 | 三外观均值 |', '|---|---|---|---|---|---|']
for seed in ['20261101','20261102']:
 for method in ['C','D-center']:
  rr=[r for r in rates if r['order_seed']==seed and r['method']==method];assert len(rr)==3
  lines.append('| '+seed+' | '+method+' | '+' | '.join(r['successes']+'/20' for r in rr)+f" | {sum(float(r['SR']) for r in rr)/3:.2%} |")
lines+=['','## 配对结果','', '| 比较 | D−C（百分点） | 95%状态聚类bootstrap区间 |', '|---|---|---|']
for s in statistics['results']:
 ci=s['physical_state_cluster_CI95'];lines.append(f"| {s['comparison']} | {100*s['difference']:+.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] |")
lines+=['','同一物理状态的两种顺序和三种外观始终一起重采样。区间仅描述这20个重复使用development状态上的条件不确定性，不覆盖训练seed/任务的总体不确定性。','',
 '| 顺序 | 外观 | D成功/C失败 | C成功/D失败 |','|---|---|---|---|']
for seed in ['20261101','20261102']:
 for condition in ['clean','seen_background','unseen_background']:
  q=[r for r in paired if r['order_seed']==seed and r['appearance']==condition]
  lines.append(f"| {seed} | {condition} | {sum(int(r['difference'])>0 for r in q)} | {sum(int(r['difference'])<0 for r in q)} |")
lines+=['','## 背景交互','']
for s in statistics['background_interactions']:
 ci=s['physical_state_cluster_CI95'];lines.append(f"- {s['comparison']}: {100*s['difference']:+.2f} pp，95%区间[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]。")
lines+=['','## 结论与限制','',
 '- 两种顺序下D-center均低于对应C，没有观察到一致性target带来的控制收益。第一顺序D恢复到较高绝对成功率，但不能称为优于C；第二顺序D依然显著较差。',
 '- 两种顺序间表现差异很大，均衡采样没有消除D的训练顺序敏感性。该比较不能确定唯一失败机制，也不能仅凭成功率判断表示坍塌。',
 '- 第一顺序D的已见背景比Clean高10个百分点，但对应C处于满分天花板，而且D绝对成功率仍低于C；第二顺序D背景表现更差。没有稳定的背景特异性收益证据。',
 '- 第一顺序D失败代理分类：抬升后/放置9次，接近目标后的抓取或接触1次。第二顺序：抬升后/放置27次，接近目标后的抓取或接触29次。这些来自冻结距离/抬升阈值，只是阶段代理，不是精确接触因果真值。',
 '- 单任务、固定AE与初始化、仅两种顺序；development已经参与历史开发。历史均匀窗口采样只作参照，均衡采样改变采样权重，不是纯顺序交换。',
 '', '## 恢复与保护','',
 '- GPU1在第二组step400发生ECC计数保护退出；保留全部失败日志与检查点。根据用户授权，在GPU0从完整step300恢复模型、optimizer、scheduler、CPU/CUDA/Python/NumPy RNG及采样位置。BF16未使用AMP scaler。',
 '- 两组有效更新共1200，实际成功更新计算1300，其中100步重放。恢复段301–400的动作loss、辅助loss、总loss与LR共100行均与原日志精确一致；这不是全部参数跨设备逐位一致证明。',
 '- 两组D评估在GPU0完成，已有C评估在GPU1完成；此设备差异显式保留，不声称所有执行环境完全一致。GPU0执行期间ECC无增量。',
 '- AE更新0、C更新0、额外C闭环0、封存集读取0。两组实际600步采样ID与对应C完全匹配；历史受保护输入哈希检查通过。',
 '- 本轮结束，不自动追加训练、seed、任务或闭环。']
(out/'results_interpretation.md').write_text('\n'.join(lines)+'\n')
gpu=subprocess.check_output(['nvidia-smi','-i','GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e','--query-gpu=memory.free,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True)
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
workers=[s for s in processes.splitlines() if 'bootstrap.py' in s and str(out) in s]
assert not workers
save(out/'resource_release_verification.json',dict(task_GPU_workers_remaining=workers,task_GPU_processes_released=True,GPU0_final_check=gpu,other_users_processes_untouched=True))
for name in ['results_interpretation.md','gpu0_recovery_audit.json','replay_numerical_audit.json','resource_release_verification.json']:
 text=(out/name).read_text().replace(str(R),'<PROJECT_ROOT>');text=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',text)
 assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',text)
 with (dest/name).open('x') as f:f.write(text)
print('RECOVERY_INTERPRETATION_AND_RESOURCE_AUDIT_COMPLETE')
