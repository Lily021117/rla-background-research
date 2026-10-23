"""Summarize fixed compute coverage experiment, without modifying source outputs."""
import argparse,json,csv,hashlib,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
def write(name,rows):
    with (out/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
metrics=[];training=[];inventory=[];timing=[];plans={};sampleaudit={}
for arm in ('B4','B20'):
    root=out/arm/'formal_B';plan=json.loads((out/arm/'training_plan.json').read_text());plans[arm]=plan
    assert json.loads((root/'training_acceptance.json').read_text())['successful_updates']==600
    samples=json.loads((root/'sampling_order.json').read_text());sampleaudit[arm]=dict(samples=len(samples),by_episode=dict(collections.Counter(x[0] for x in samples)),parameter_hash_step0=json.loads((root/'checkpoint_step0000_audit.json').read_text())['parameter_hash'])
    training.extend(dict(arm=arm,**r) for r in csv.DictReader((root/'training_metrics.csv').open()))
    groups=collections.defaultdict(list)
    for r in csv.DictReader((root/'validation_metrics.csv').open()):groups[(r['step'],r['set'],r['episode'],r['appearance'],r['position'])].append(r)
    for (step,role,ep,c,h),rows in groups.items():metrics.append(dict(arm=arm,step=int(step),set=role,episode=ep,appearance=c,position=int(h),windows=len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in ('action_L1','translation_L1','rotation_L1','gripper_L1')}))
    man=json.loads(Path(plan['manifest']).read_text());epmeta={e['episode']:e for e in man['episodes']}
    for step in (0,100,300,600):
        path=root/f'step{step:04}.pt';inventory.append(dict(arm=arm,step=step,path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        for role in ('development','common_B4'):
            fixed=json.loads((root/f'{role}_samples.json').read_text());d=np.load(root/f'{role}_step{step:04}.npz');pr=d['prediction'];gt=d['target']
            for ep in sorted(set(x[0] for x in fixed)):
                raw=np.load(Path(epmeta[ep]['root'])/'trajectory_labels.npz')['actions'][:,6];events=np.flatnonzero((raw[:-1]<0)&(raw[1:]>0))+1
                if not len(events):continue
                event=int(events[0])
                for c in ('clean','background'):
                    indices={int(x[1]):i for i,x in enumerate(fixed) if x[0]==ep and x[2]==c}
                    for h in range(8):
                        ideal=event-h
                        if ideal not in indices:continue
                        scan=[t for t in range(max(0,event-8),event+9) if t in indices];closed=[t for t in scan if pr[indices[t],h,6]<.5];onset=closed[0] if closed else None
                        timing.append(dict(arm=arm,step=step,set=role,episode=ep,appearance=c,event=event,position=h,prediction=float(pr[indices[ideal],h,6]),target=float(gt[indices[ideal],h,6]),correct_close=bool(pr[indices[ideal],h,6]<.5),onset_error_steps=onset-ideal if onset is not None else None))
assert sampleaudit['B4']['parameter_hash_step0']==sampleaudit['B20']['parameter_hash_step0']
assert plans['B4']['statistics_sha256']==plans['B20']['statistics_sha256']
import torch
s0=[torch.load(out/arm/'formal_B/step0000.pt',map_location='cpu') for arm in ('B4','B20')]
assert torch.equal(s0[0]['torch_rng'],s0[1]['torch_rng'])
assert all(torch.equal(x,y) for x,y in zip(s0[0]['cuda_rng'],s0[1]['cuda_rng']))
assert s0[0]['python_rng']==s0[1]['python_rng']
assert np.array_equal(s0[0]['numpy_rng'][1],s0[1]['numpy_rng'][1])
del s0
write('training_metrics.csv',training);write('validation_metrics.csv',metrics);write('checkpoint_inventory.csv',inventory);write('closure_timing_metrics.csv',timing)
with (out/'initialization_and_sampling_audit.json').open('x') as f:json.dump(dict(common_step0_parameters=True,common_CPU_CUDA_Python_NumPy_RNG=True,common_seed=20261023,common_normalization=True,sampling='same seeded permutation algorithm, not same sample sequence',arms=sampleaudit),f,indent=2)
trials=json.loads((out/'per_trial_results.json').read_text());assert len(trials)==20 and all(r['status']=='COMPLETE' for r in trials);paired=[];summary=[]
for arm in ('B4','B20'):
    for c in ('clean','background'):
        r=[t for t in trials if t['arm']==arm and t['appearance']==c];summary.append(dict(arm=arm,appearance=c,successes=sum(x['success'] for x in r),trials=len(r)))
for ep in sorted(set(t['episode'] for t in trials)):
    for c in ('clean','background'):
        rs={t['arm']:t for t in trials if t['episode']==ep and t['appearance']==c};paired.append(dict(episode=ep,appearance=c,B4_success=rs['B4']['success'],B20_success=rs['B20']['success'],difference=int(rs['B20']['success'])-int(rs['B4']['success'])))
write('paired_results.csv',paired)
failures=['# 固定development失败阶段诊断','','仅分析本轮已有20次试验；动作接近与目标物体位移为诊断，不把接触/抬升代理当成功判定。成功只取env.check_success()。','']
for r in trials:
    path=Path(r['trajectory']);d=np.load(path);cmd=d['executed_actions'];closing=int((cmd[:,6]>0).sum());lift=r.get('target_max_lift_m',0)
    stage='完整任务成功' if r['success'] else ('已有目标抬升但任务未完成，搬运/释放阶段需视频核对' if lift>.02 else ('接近/抓取未形成明显抬升，不能仅归因为夹爪' if closing else '无关闭命令且无明显抬升；接近位姿与关闭时机尚不可分离'))
    failures.append(f"- {r['arm']} / {r['source_demo']} / {r['appearance']}: {stage}；close命令{closing}/{r['steps']}，目标最大抬升{lift:.4f}m，最近末端到目标中心{r.get('min_eef_to_target_center_m',float('nan')):.4f}m，平移饱和比例{r.get('translation_saturation_fraction',float('nan')):.3f}。")
(out/'failure_stage_analysis.md').write_text('\n'.join(failures)+'\n')
with (out/'success_summary.json').open('x') as f:json.dump(summary,f,indent=2)
lines=['# B4/B20固定计算预算对照','', '实验为基于LIBERO示范构建的派生数据与自定义背景材质干预；不是官方LIBERO-Plus。', '', '## 预算和数据', '', 'B4原4条是B20的子集；5条development按episode隔离。共同20train-only Q99统计，B4也获得其聚合统计信息；历史270步不是严格对照。两组共同官方Prismatic/seeded head初始化、Math SDPA、BF16、AdamW LR2e-4、microbatch8/accumulation2，各600次成功更新。每组4800物理窗口曝光/9600外观观测；B4约11.14窗口epoch，B20约2.54。固定预算比较，不声称收敛。', '', '## 闭环', '', '|模型|外观|成功/次数|','|---|---|---|']
lines.extend(f"|{r['arm']}|{r['appearance']}|{r['successes']}/{r['trials']}|" for r in summary)
lines+=['','全部使用最终step600、5个development保存状态、seed0、220步、chunk8、相同成功判定与材质协议；按状态/外观B4-B20配对，完整reset，无静默重跑。视频及状态轨迹留本地。5状态仅探索性证据，不是独立测试或稳定泛化结论。','','## 离线诊断','', 'validation_metrics.csv含step0/100/300/600、每示范、每chunk位置与xyz/旋转/夹爪分项。common_B4为两组相同样本；own_train为各训练池固定步长抽样，development为全部466窗口两外观。closure_timing_metrics.csv固定首个闭合事件，不把整体L1下降等同于闭合时序或任务成功。','','## 资源与边界','','原GPU1预检显存不足退出，0更新；用户授权GPU0后独立run启动，无中断重放。未训练AE-C/AE-D或策略C/D；未读取封存内容。正式更新总数1200；新增有效闭环20。历史文件保持只读。源码/配置/脱敏汇总同步研究分支，权重/图像/视频/轨迹不上传。']
(out/'report.md').write_text('\n'.join(lines)+'\n')
with (out/'delivery_verification.json').open('x') as f:json.dump(dict(status='PASS_EXECUTION_NOT_METHOD_PROOF',B4_updates=600,B20_updates=600,new_closed_loops=20,heldout_read=False,AE_C_D=False,common_initial_parameters=True,statistics_shared=True),f,indent=2)
