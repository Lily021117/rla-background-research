"""Task-level, incremental evidence delivery; no learned computation."""
from common import *
from complete_sources import PREVIOUS
import collections,time,re

def main():
 out=RUN/'delivery';out.mkdir(exist_ok=True)
 public=Path(__file__).resolve().parents[1]/'rla_official_demo_source_completion/delivery';public.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(open(RUN/'source_manifest.csv')))
 audit={(x['suite'],x['trajectory_id']):x for x in csv.DictReader(open(RUN/'lineage_audit.csv'))} if (RUN/'lineage_audit.csv').exists() else {}
 reuse=json.loads((RUN/'reuse_audit.json').read_text()) if (RUN/'reuse_audit.json').exists() else []
 reused={(x['suite'],x['trajectory_id']) for x in reuse if x['status']=='REUSED_VERIFIED'}
 taskrows=[];frozen=[];render_failures=[];counts=dict(new_trajectories=0,reused_trajectories=0,new_source_frames=0,reused_source_frames=0,loader_pass=0,state_restores=0,kinematic_forwards=0,rgb_renders=0,segmentation_renders=0,depth_renders=0,physics_steps=0,action_replay=0,model_forwards=0,updates=0,backward=0,policy_rollouts=0)
 batch_records=json.loads((RUN/'batch_results.json').read_text()) if (RUN/'batch_results.json').exists() else []
 batch_by_id={(x['suite'],x['trajectory_id']):x for x in batch_records}
 for x in rows:
  key=x['suite'],x['trajectory_id'];p=RUN/'trajectories'/x['suite']/f"traj_{x['trajectory_id']}";r=p/'result.json';l=p/'loader_audit.json'
  x.update(result_path=str(r) if r.exists() else '',output_reused=key in reused,paired_complete=r.exists(),loader_status=json.loads(l.read_text())['status'] if l.exists() else 'NOT_TESTED',remaining_reason=audit.get(key,{}).get('decision_reason','PENDING_REVIEW'))
  if r.exists():
   d=json.loads(r.read_text());prefix='reused' if key in reused else 'new';counts[prefix+'_trajectories']+=1;counts[prefix+'_source_frames']+=d['frames']
   if prefix=='new':
    for k,v in d['counts'].items():counts[k]+=v
  elif (p/'failure.json').exists():
   failed=json.loads((p/'failure.json').read_text());render_failures.append(dict(suite=x['suite'],trajectory_id=x['trajectory_id'],**failed))
   x['remaining_reason']='RENDER_FAILURE: '+str(failed.get('error','see preserved failure.json'))
   for k,v in failed.get('counts',{}).items():counts[k]+=v
  elif key in batch_by_id:
   x['remaining_reason']=batch_by_id[key]['status']
  if l.exists() and x['loader_status']=='PASS':counts['loader_pass']+=1
  frozen.append(x)
 for suite,task in sorted({(x['suite'],x['task']) for x in rows}):
  group=[x for x in rows if x['suite']==suite and x['task']==task]
  for split in ['train','val']:
   g=[x for x in group if x['split']==split]
   taskrows.append(dict(suite=suite,task=task,split=split,target=len(g),source_file_status='VERIFIED' if any(x['source_file'] for x in group) else 'MISSING_OR_UNVERIFIED',source_confirmed=sum(x['source_status']=='MULTI_EVIDENCE_JOIN' for x in g),paired_complete=sum(x['paired_complete'] for x in g),reused=sum(x['paired_complete'] and x['output_reused'] for x in g),new=sum(x['paired_complete'] and not x['output_reused'] for x in g),loader_pass=sum(x['loader_status']=='PASS' for x in g),unresolved=sum(x['source_status']=='UNRESOLVED' for x in g),missing=sum(x['source_status']=='MISSING_OFFICIAL_HDF5' for x in g),split_conflicts=sum(x['split_conflict']=='True' for x in g)))
 assert len(taskrows)==80
 csvout(out/'coverage_by_task_split.csv',taskrows);csvout(out/'frozen_data_manifest.csv',frozen)
 csvout(out/'unfinished_entries.csv',[x for x in frozen if not x['paired_complete'] or x['loader_status']!='PASS'] or [dict(status='NONE')])
 dump(out/'counts.json',counts);dump(out/'render_failures.json',render_failures)
 failures=[];downloads=[]
 for p in (RUN/'download_attempts').glob('*/*/attempt_*.json'):
  d=json.loads(p.read_text());downloads.append(d)
  if d['status']=='FAILED_PRESERVED':failures.append(d)
 dump(out/'download_attempts.json',downloads);dump(out/'download_failures.json',failures)
 dump(out/'download_accounting.json',dict(new_attempts=len(downloads),network_bytes_recorded=sum(x.get('network_bytes',0) for x in downloads),note='Application-delivered bytes only; interrupted partial chunks/TLS retransmissions not measured; old resume prefixes not new network bytes.'))
 for name in ['protocol.json','download_plan.json','download_complete.json','source_files.json','rejected_source_files.json','lineage_audit.csv','previous_210_audit.csv','phase_audit.csv','source_rule_diff.csv','lineage_rule_v2.json','generation_code_evidence.json','split_conflicts.json','reuse_audit.json','batch_complete.json']:
  p=RUN/name
  if p.exists():(out/name).write_bytes(p.read_bytes())
 complete=(RUN/'loader_audit.json').exists() and json.loads((RUN/'loader_audit.json').read_text()).get('all_pass')
 state='PARTIAL_DATA_BUILD_COMPLETE' if complete else 'IN_PROGRESS'
 reasons=collections.Counter()
 for x in audit.values():
  if x['old_status']=='UNRESOLVED':reasons.update(filter(None,x['old_numeric_failed_conditions'].split(';')))
 completed=counts['new_trajectories']+counts['reused_trajectories'];missing=sum(x['source_status']=='MISSING_OFFICIAL_HDF5' for x in rows);unresolved=sum(x['source_status']=='UNRESOLVED' for x in rows)
 text=f'''# 官方示范统一重渲染派生版：来源补齐与统一复核

状态：**{state}**。目标不变：1653训练＋40验证。已完成{completed}条，其中本轮新增{counts['new_trajectories']}、复用{counts['reused_trajectories']}；loader通过{counts['loader_pass']}。源文件未通过核验涉及{missing}条，来源证据仍不足{unresolved}条。逐任务、逐划分见 `coverage_by_task_split.csv`（40任务×2划分）。没有用其他demo补数，没有自动开训。

## 210条究竟为何未通过

旧210条都有唯一完整过滤动作匹配，不是多个动作候选。旧诊断失败项（可重叠）：{dict(reasons)}。每条候选数、语言/任务路径、最近/次近demo、距离比、最近及候选RMSE详见 `previous_210_audit.csv`。历史955条也接受完全相同的重新核验，差异见 `source_rule_diff.csv`。

## 相位、判定规则及证据边界

固定官方源转换代码先执行动作再记录obs；其states仍为动作前状态。再生成代码从原始states[0]开始，执行10个稳定等待动作，跳过no-op，记录动作前obs再执行动作。再生成第一行states特意复制等待前原始状态，不能将它与等待后的首个obs当作同相位。源obs与RLDS joints逐行或归一化进度比较混合了相位和不同积分路径，不能用阈值不通过直接证明身份不同。

v2修正上述解释，不降低ratio=2或提高RMSE=.1，也没有只给失败条目更宽规则。原有路径一致性仅作为保守的正向佐证；缺少它时，本轮仍没有取得独立的再生成demo映射/初态元信息来补足来源证据。因此保留隔离，而不是把所有唯一动作匹配自动升级。这里承认原判定可能造成假阴性，不把隔离写成身份已被否定。不同轨迹归一化进度不代表相同时刻，也不能用于输出状态插值。

源完整初态哈希、RLDS首个机器人观测哈希和差异均记录；二者不是同相位，未将其差异用于自动放行。`phase_audit.csv`同时给出源动作后obs与同列/下一列pre-state、原始pre-state与RLDS的诊断距离，不根据最小距离猜移位。没有新的动作重放或环境版本搜索。缺少的最小资料是发布者的regenerated HDF5或逐RLDS episode→原demo映射、初态及生成配置；原始demo本身的初态不能证明RLDS的初态。

## 下载、复用与增量

按父download_plan固定revision、大小与SHA256，每个未完成文件最多两次新增网络尝试。旧28个合格文件只重新核验，不重新下载。截断文件只在独立目录复制已知SHA关联的断点；大小正确但哈希错误的旧文件不作为断点、不作为HDF5读取。完整失败现场保留。网络错误、响应范围、字节数、实际hash见下载记录；无法获得完整新文件时，不能断言已确定旧坏文件是缓存还是断点污染。

复用核验来源身份、原始过滤索引及划分不变，检查记录的模型/状态缓存哈希、所有PNG文件结构及数量，保留旧逐帧数值验收。旧PNG没有逐文件历史hash，不声称本次重新证明其字节完全一致。所有旧文件只读；新的loader检查写在增量目录，不改旧报告。无来源变化且验收有效的产物不重渲染。

新增仍使用原始HDF5状态、逐demo XML、已验收静态同步路径，固定H+S和背景白名单。两路不推进物理时间；真实时间戳/索引不变。模型前向、训练更新、backward、策略闭环、源动作重放均为0。运动学forward单独计数，不属于模型前向。

## 训练决策

这不是1693条全量准入。缺口分布及未完成原因见逐任务表，任务内来源证据筛选会改变demo组成、帧数/跨度分布和任务权重；源文件缺失还可能让整个任务缺席。即使训练/验证边界保持，也不能把子集结果称为原官方数据范围复现。必须先补齐或明确另定子集协议，不能自动按子集开训。历史原视频精确对应失败结论保持不变。
'''
 (out/'report.md').write_text(text)
 evidence=[dict(path=str(p),sha256=sha(p)) for p in out.iterdir() if p.is_file() and p.name!='evidence_manifest.json'];dump(out/'evidence_manifest.json',evidence)
 for p in out.iterdir():
  if p.suffix in ['.md','.csv','.json']:
   value=p.read_text().replace(str(ROOT),'<PROJECT>').replace('/data/lyc/projects/vla-adapter-ref','<ENV_PROJECT>')
   value=re.sub(r'(https?://[^\s"?]+)\?[^\s"]+',r'\1?<QUERY_REDACTED>',value)
   (public/p.name).write_text(value)
 dump(RUN/'delivery_status.json',dict(status=state,counts=counts,time_unix=time.time()))
 print(state,counts,flush=True)

if __name__=='__main__':main()
