"""CPU-only evidence ledger and arithmetic checks; no ML/simulator imports.

Usage: python build_evidence.py --project /path/to/rla-wm
Only explicit small report/config/result files are read. No checkpoint/data scan.
"""
import argparse,csv,hashlib,json,statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args()
P=args.project;OUT=Path(__file__).resolve().parent
roots={
'A':'rla_ae_from_scratch/run_20261029_01',
'P':'rla_scratch_policy_aux/run_20261029_01',
'Q':'rla_scratch_policy_aux/run_20261030_01_c_low',
'S':'rla_scratch_low_policy_seeds/run_20261030_01',
'L':'rla_low_lr_continuation/run_20261030_01',
'I':'rla_frozen1800_independent_states/run_20261031_02',
'R':'rla_drawer_baseline_gap/run_20261101_01',
'N':'rla_frozen_nuisance_pilot/run_20261101_01'}
refs={}
def reg(id,root,file):refs[id]='experiments/'+roots.get(root,root)+'/'+file
for root in ['A','S','L','I','R','N']:reg(root+'1',root,'delivery/report.md' if root not in ['S','L','R','N'] else 'protocol.json')
reg('A2','A','frozen_plan.json');reg('A3','A','delivery/reconstruction_summary.csv');reg('A4','A','delivery/representation_summary.csv');reg('A5','A','code/experiments/rla_ae_from_scratch/run.py');reg('A6','A','initialization_audit.json')
reg('P1','P','protocol.json');reg('P2','P','delivery/success_summary.csv');reg('P3','S','protocol.json');reg('P4','Q','delivery/success_summary.csv');reg('P5','Q','delivery/delivery_verification.json')
for r in ['S','L','I']:
 reg(r+'2',r,'delivery/success_summary.csv');reg(r+'3',r,'delivery/delivery_verification.json')
reg('I4','I','delivery/paired_statistics.json')
reg('R2','R','delivery/success_by_seed_appearance.csv');reg('R3','R','delivery/method_summary.json');reg('R4','R','delivery/delivery_verification.json')
reg('N2','N','clean_reuse.json');reg('N3','N','delivery/main_statistics.json');reg('N4','N','delivery/offline_stage_summary.csv');reg('N5','N','delivery/offline_by_demo.csv');reg('N6','N','delivery/success_by_seed_condition.csv');reg('N7','N','delivery/delivery_verification.json')
history={
'H1':('official_libero_rla_consistency/run_20261023_10/recovery_20261024/completion','report.md'),
'H2':('official_libero_rla_consistency/run_20261023_09','report.md'),
'H3':('official_libero_rla_consistency/stove_centering/run_20261024_01','report.md'),
'H4':('official_libero_rla_consistency/stove_aux_weight/run_20261025_01','completion_report.md'),
'H5':('rla_ae_consistency_strength/run_20261026_01','report.md'),
'H6':('control_preserving_rla/run_20261025_01/results','report.md'),
'H7':('rla_appearance_gap_eval/run_20261028_01','report.md'),
'H8':('rla_aux_interference_diagnostic/run_20261028_01','report.md'),
'H9':('rla_trajectory_divergence_analysis/run_20261028_01/analysis','report.md'),
'H10':('rla_task_geometry_target/run_20261028_01','report.md')}
for id,(r,f) in history.items():reg(id,r,f)
# Source-level method checks, identity receipts and exact result files.
for r in ['P','Q','S','L','I','R','N']:
 reg(r+'report',r,'delivery/report.md')
 reg(r+'verification',r,'delivery/delivery_verification.json')
reg('Averification','A','delivery/delivery_verification.json')
reg('Adata','A','data_inventory.csv')
reg('PnormC','P','targets/C/statistics.json');reg('PnormD','P','targets/D/statistics.json')
reg('H10stats','rla_task_geometry_target/run_20261028_01','success_summary.csv')
reg('H7stats','rla_appearance_gap_eval/run_20261028_01','complete_comparison_table.csv')
reg('H8stats','rla_aux_interference_diagnostic/run_20261028_01','target_switch_summary.csv')
reg('H9stats','rla_trajectory_divergence_analysis/run_20261028_01/analysis','paired_summary.csv')
reg('H1stats','official_libero_rla_consistency/run_20261023_10/recovery_20261024/completion','per_trial_results_combined.csv')
reg('Nrender','N','offline_render_acceptance.json');reg('Nae','N','offline_ae_acceptance.json');reg('Npolicy','N','offline_policy_acceptance.json')
commits={'R':'752fbdd2bcf17361a00a5ff0cedaf9c18c05ae2e','N':'44615033c8d6227f1daf17dec6230136076c8a03'}
evidence=[];content={}
for id,rel in refs.items():
 f=P/rel
 if not f.is_file():
  evidence.append(dict(id=id,path=rel,status='未核验：该定位文件缺失',sha256=None,related_commit=commits.get(id[0],'未核验')));continue
 assert f.stat().st_size<25*1024**2,(id,'unexpectedly large evidence')
 b=f.read_bytes();content[id]=b.decode();evidence.append(dict(id=id,path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='已读取；引用材料hash核验，不代表重审原始轨迹',related_commit=commits.get(id[0],'未核验；以现存文件hash定位')))
def rows(id):return list(csv.DictReader(content[id].splitlines()))
def j(id):return json.loads(content[id])
def write(name,obj):
 with (OUT/name).open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
checks={}
for id in ['P2','P4','S2','L2','I2','R2','N6']:
 rs=rows(id);summ=[]
 for r in rs:
  n=int(r.get('trials',r.get('denominator',20)));s=int(r['successes']);assert 0<=s<=n
  for k in ['success_rate','SR','rate']:
   if k in r:assert abs(float(r[k])-s/n)<1e-10,(id,r)
  summ.append((s,n))
 checks[id]=dict(rows=len(rs),successes=sum(x[0] for x in summ),displayed_trials=sum(x[1] for x in summ),note='展示总数包含复用，不能当新增闭环')
rs=rows('N6');nv=j('N3');nuisance=[]
for kind in ['clean','background','lighting']:
 selected=[r for r in rs if r['condition'].startswith(kind)]
 rates={m:statistics.mean(float(r['rate']) for r in selected if r['method']==m) for m in ['C-low','D-low']}
 assert abs(rates['D-low']-rates['C-low']-nv[kind]['D_minus_C'])<1e-12
 nuisance.append(dict(category=kind,**rates,D_minus_C=rates['D-low']-rates['C-low']))
checks['nuisance_recomputed']=nuisance
historical={}
for r in rows('H1stats'):
 assert r['status']=='COMPLETE'
 key=r['task']+'/'+r['model'];v=historical.setdefault(key,dict(successes=0,trials=0))
 v['trials']+=1;v['successes']+=int(r['success'].lower()=='true')
assert historical['stove/C']==dict(successes=59,trials=60)
assert historical['stove/D']==dict(successes=0,trials=60)
assert historical['cookie_historical/C']==dict(successes=59,trials=60)
assert historical['cookie_historical/D']==dict(successes=60,trials=60)
checks['early_task_raw_trial_recount']=historical
for step in ['600','1800']:
 rs=[r for r in rows('L2') if r['step']==step];stats={}
 for method in ['C-low','D-low']:
  for category in ['clean','background']:
   selected=[r for r in rs if r['method']==method and ((r['appearance']=='clean')==(category=='clean'))]
   assert len(selected)==(3 if category=='clean' else 12)
   stats[method+'_'+category]=statistics.mean(int(r['successes'])/int(r['trials']) for r in selected)
 checks['continuation_'+step]=stats
ae=[r for r in rows('A4') if r['step']=='2000' and r['split']=='validation' and r['appearance']=='clean']
checks['AE_final_representation']=ae
assert j('A2')['updates_each']==2000 and j('A2')['AE_pretrained_load'] is False
assert abs(j('A2')['lambda_cons']-4.362407926306656e-05)<1e-15
checks['AE_configuration']=dict(seed=j('A2')['seed'],formal_updates=2*j('A2')['updates_each'],pilot_updates=j('A2')['pilot_updates'],lambda_cons=j('A2')['lambda_cons'])
counts={k:0 for k in ['policy_updates','AE_updates','DINO_updates','model_forwards','backward','new_rollouts','simulation_replays','new_sealed_reads']}
write('arithmetic_checks.json',dict(status='PASS',checks=checks,this_review=counts))
write('evidence_manifest.json',dict(project_root='<PROJECT_ROOT>',base_commit=commits['N'],scope='explicit cited small files only; no weights/raw states/sealed scan',files=evidence,this_review=counts))

ledger=[]
def entry(id,task,split,ae,policy,budget,result,identity,reuse,extra='未逐项复核；见证据',new='未核验'):
 ledger.append(dict(experiment=id,task=task,data_split=split,AE_source=ae,policy_initialization_seed=policy,training_budget=budget,main_result=result,evaluation_identity=identity,new_valid_rollouts=new,reused_results=reuse,replays_invalid_attempts=extra,evidence_paths=';'.join(refs[x] for x in id.split('+') if x in refs)))
entry('H2','cookie取碗放盘','20train+5dev','step90000后C/D各600','共同原始策略；单seed','策略C/D各600','C59/60 D60/60；天花板','20development×3外观','B20冻结结果纳入本次统一评估','C重放1；另OOM未完成更新',180)
entry('H1','抽屉取碗、炉灶取碗放盘','每任务20/5','每任务step90000后C/D各600','共同原始策略；单seed','AE2400；策略3600','drawer C1/60 D0/60；stove C59/60 D0/60','每任务20development×3外观','cookie180历史结果复用','炉灶两AE各44重放；在途各至多1次未确认',360)
entry('H3','炉灶','20/5','冻结历史AE-D','原始共同step0','D-old/D-center各600','均0/60；关闭/抬升代理改变','原20development×3外观','历史C60','技术无效1；有效120',120)
entry('H4','炉灶','20/5','冻结历史AE-C/D','共同step0 seed20261023','两策略各600；lambda_aux原值0.1倍','C5/60 D9/60，低于历史C59/60','原20development×3外观','历史C/D-center120','重放0，无效0',120)
entry('H5','炉灶','20/5','step90000起，lambda_inv×0.3，600步','共同原始step0','AE600+策略600','D-inv03 0/60','原development','历史C/D/lowLR参照','本次精确重放未核验',60)
entry('H6','炉灶','20/5','冻结历史AE','从C成品共同初始化；重置辅助读出','四学生各600','E-C59 E-D31 E-C-keep27 E-D-keep0 /60','原development','原C作为共同成本','本次精确重放未核验',240)
entry('H7','炉灶','20/5','冻结历史AE-C/D','两个采样顺序20261101/02，已有600模型','本评估0','shared背景对C +3.75/-53.75pp；D -2.5/-60pp','原20状态四新背景；Clean历史','C160新背景、各方法Clean复用','历史D重放100不计本评估',320)
entry('H8','炉灶','固定合法训练batch，已有validation','历史AE','已有C/shared0/100/300/600快照，实际覆盖缩减','0更新；历史有限诊断backward','PARTIAL，未确认持续梯度冲突','非闭环机制诊断','已有训练/验证曲线','原315确认目标提取+中断单元，上限318',0)
entry('H9','炉灶','仅已有评估轨迹','不加载','四个已训练策略已有结果','0模型计算','200对，局部放置模式只在一个顺序','重复development轨迹分析','全部400轨迹复用','无新增闭环',0)
entry('H10','炉灶','20/5','不新增AE；额外仿真几何标签','共同step0 两采样顺序','4×600','Geometry优于action-only-match，但低于C','20development×5外观','C200复用','详细重放/无效未重核',400)
entry('A1+A2','cookie/抽屉取碗/炉灶','60train/15dev；7231窗口','官方构造共同未训练encoder/decoder；DINO冻结','AE seed20261029；不训策略','C/D各2000+校准60','D重建略低，单位范数距离改善，尺度/有效秩变化','探索性固定示范验证','历史已验收配对缓存复用','见Averification；不是策略训练',0)
entry('P1+P2','炉灶','20/5','新从零AE-C/D最终2000','历史共同step0；仅顺序20261101/02','8×600=4800','D-low优于D-fixed；D-decay差；C-fixed顺序反转','原20development×5外观','无闭环复用','更新实际4800；重放0，无效0',800)
entry('P4+P5','炉灶','20/5','同一新AE-C','匹配两顺序共同step0','2×600=1200','C-low Clean0/20与16/20，结论反转','原development','C-fixed/D-low400复用','详见P5',200)
entry('S1+S2','炉灶','20/5','同一新AE对固定','完整策略seed31001/02/03；预训练不重置','6×600=3600','C背景13.75% D60%；D也有失败seed','原development；非独立测试','两个历史顺序不混入主结果','实际3649；D31002重放49',600)
entry('L1+L2','炉灶','20/5','同一新AE对固定','上述六step600完整恢复','新增6×1200=7200，总步1800','C/D背景85.42/95%；Clean96.67/95%','原development','step600的600评估复用','本次实际7200，重放0；历史49保留',600)
entry('I1+I2','炉灶','不改训练','同一新AE对','同六step1800；0新训练seed','0','背景D-C -0.67pp，优势未复现','50新生成物理状态；旧四背景','旧dev不并入','见I3；候选来源审计曾读取封存，不等于本轮新增读取',1500)
entry('R1+R2','打开中层抽屉','本任务20/5','迁移三取碗任务AE-C最终2000','BC/C各完整seed31001/02/03从原始起点','6×1800=10800','均过Clean；背景变化seed依赖，未达推进门槛','本任务20development×5外观','历史初态/纹理复用；此次评分为新增','无D训练；见R4',600)
entry('N1+N3','炉灶','不改训练；离线5dev×20anchor','同一新AE对','同六step1800','0','新背景-2.67pp；静态光照-1.33pp','50中哈希选25已评状态；新6条件','Clean150复用；旧四背景不混入','新增尝试900有效900无效0',900)
with (OUT/'experiment_ledger.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
report=OUT/'decision_report.md';text=report.read_text().split('\n## CPU核验附表')[0].split('\n## 引用文件定位')[0]
text+='\n## CPU核验附表\n\n从A4原统计筛选最终2000步、validation、Clean行；配对距离本身涉及两外观。各任务45个固定anchor/horizon样本，非45条独立示范。\n\n|任务|AE|latent范数|逐坐标方差|有效秩|单位范数配对L2|\n|---|---|---:|---:|---:|---:|\n'
for r in ae:text+=f"|{r['task']}|{r['arm']}|{float(r['latent_norm']):.3f}|{float(r['mean_coordinate_variance']):.4f}|{float(r['entropy_effective_rank']):.4f}|{float(r['paired_unit_L2']):.5f}|\n"
text+='\n## 引用文件定位\n\n编号精确定位本次读取文件；内容hash、大小、相关提交和未核验项见manifest。下列路径相对 `<PROJECT_ROOT>`。\n\n'
for e in evidence:text+=f"- **{e['id']}**：`{e['path']}`（{e['status']}）。\n"
report.write_text(text)
unchanged=all(hashlib.sha256((P/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in evidence if e['sha256'])
assert unchanged
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name not in ['delivery_verification.json','github_sync.json']}
write('delivery_verification.json',dict(status='PASS_READ_ONLY_REVIEW',historical_cited_files_unchanged=unchanged,actual_cited_files=len(content),ledger_experiments=len(ledger),new_model_or_simulator_processes_started=False,counts=counts,missing_files=[e['id'] for e in evidence if not e['sha256']],limitations=['Historical related commits not individually verified unless specified','Partial historical gradient coverage not repaired','No complete historical rollout/replay grand total claimed','Same AE/models reused across evaluation rounds, not independent training replications'],files=files))
print(json.dumps(dict(evidence_files=len(content),missing=[e['id'] for e in evidence if e['sha256'] is None],ledger_rows=len(ledger),checks='PASS'),ensure_ascii=False))
