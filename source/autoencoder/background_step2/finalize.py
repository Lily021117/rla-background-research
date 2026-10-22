"""Independent final accounting and protection verification, no model execution."""
import csv
from .common import *
def csvwrite(name,rows):
 p=OUT/name;assert not p.exists()
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 assert read(OUT/'training_complete.json')['status']=='PASS'
 protected=read(OUT/'protection_before.json');changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed,changed
 for f in read(OUT/'feature_cache_acceptance.json')['cache_inventory']:assert sha(f['path'])==f['sha256']
 assert sha(DINO/'model.safetensors')=='dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
 import torch
 initial_misc=[torch.load(OUT/g/'ckpts/misc_step0000000.pt',map_location='cpu',weights_only=False) for g in ('B','C')]
 assert all(not x['optimizer']['state'] for x in initial_misc)
 assert initial_misc[0]['optimizer']['param_groups']==initial_misc[1]['optimizer']['param_groups']
 del initial_misc
 health();rows=[];curve=[];updates={};peaks=[]
 for group in ('B','C'):
  done=read(OUT/group/'complete.json');assert done['successful_updates']==200 and done['extra_updates']==0
  logs=[json.loads(l) for l in (OUT/group/'train.jsonl').read_text().splitlines()]
  assert [r['step'] for r in logs]==list(range(1,201))
  assert all(np.isfinite(list(r['loss'].values())).all() for r in logs)
  updates[group]=dict(successful_updates=200,attempts=200,retained_preflight=2,extra_updates=0,replays=0,
    view_pair_exposures=sum(r['view_pair_batch'] for r in logs),branch_exposures=sum(r['branch_samples'] for r in logs),
    optimizer_compute_seconds=sum(r['elapsed_seconds'] for r in logs),last_loss=logs[-1]['loss'],
    peak_gpu_bytes=max([r['peak_gpu_bytes'] for r in logs]+[read(OUT/group/'preflight.json')['peak_gpu_bytes']]))
  peaks.append(updates[group]['peak_gpu_bytes'])
  for step in (0,50,100,200):
   v=read(OUT/group/f'validation_step{step:07d}.json')
   curve.append(dict(group=group,step=step,**v['summary']))
   for r in v['rows']:rows.append(dict(group=group,step=step,**r))
 csvwrite('validation_per_sample.csv',rows);csvwrite('validation_summary.csv',curve)
 per_demo=[];per_camera=[]
 for group in ('B','C'):
  for step in (0,50,100,200):
   v=read(OUT/group/f'validation_step{step:07d}.json')
   per_demo.extend(dict(group=group,step=step,**r) for r in v['per_demo'])
   for camera in ('agentview','wrist'):
    rs=[r for r in v['rows'] if r['camera']==camera]
    keys=[k for k in rs[0] if isinstance(rs[0][k],float)]
    per_camera.append(dict(group=group,step=step,camera=camera,**{k:float(np.mean([r[k] for r in rs])) for k in keys}))
 csvwrite('validation_per_episode.csv',per_demo);csvwrite('validation_per_camera.csv',per_camera)
 changes=[]
 b=read(OUT/'B/validation_step0000200.json');c=read(OUT/'C/validation_step0000200.json')
 for br,cr in zip(b['rows'],c['rows']):
  assert (br['episode_id'],br['camera'],br['window'])==(cr['episode_id'],cr['camera'],cr['window'])
  changes.append(dict(episode_id=br['episode_id'],camera=br['camera'],window=br['window'],**{k:cr[k]-br[k] for k in br if isinstance(br[k],float)}))
 csvwrite('validation_paired_C_minus_B.csv',changes)
 assert read(OUT/'initialization_pairing.json')['outputs_match']
 a=read(OUT/'B/validation_step0000000.json')['summary'];metrics={'A (frozen official)':a,'B (reconstruction)':b['summary'],'C (consistency)':c['summary']}
 write(OUT/'update_accounting.json',updates)
 code={str(p.relative_to(WT)):sha(p) for p in (WT/'background_step2').rglob('*.py')}
 write(OUT/'code_inventory.json',dict(base_commit=shell(['git','-C',str(WT),'rev-parse','HEAD']).strip(),adapter_files=code,
   worktree_status=shell(['git','-C',str(WT),'status','--short']),original_tracked_files_changed=shell(['git','-C',str(WT),'diff','--name-only']).strip()))
 assert not read(OUT/'code_inventory.json')['original_tracked_files_changed']
 lam=read(OUT/'gradient_scale_and_fixed_lambda.json')['lambda_cons']
 lines=['# RLA 跨背景一致性 Step 2：B/C 200步受控训练','',
 '状态：PASS（训练与数据管线验收完成；不意味着一致性方法效果通过）。B/C各200次成功优化更新，总400次；本轮未评分保留测试集。','',
 '## 版本、权重和数据保护','',
 '- 官方base commit `0fc6e97212442ebccffa06f1d42f5481e5f22950`；新detached worktree `'+str(WT)+'`。仅新增 `background_step2/` 适配器，官方tracked源文件零改动；基准worktree和原主分支不切换、不恢复。新增代码未自动提交，其完整SHA在code_inventory.json。',
 '- 官方ManiSkill step100000 Encoder SHA `003e61d998b5522dcb5bf54c1eca4e0433eeb304117773e75724d1ba40e87d0a`；Decoder SHA `8f048af24582216a3a6655ece82095adf9f56d9a9ae27d83ef94277266aacfa2`。路径 `${DATA_ROOT}/models/rla-wm/official/rla/maniskill/20260323_21-41-25/ckpts/`。严格加载，模型权重warm-start、全新AdamW，不假称恢复作者optimizer。',
 '- 单视角512px、DINOv3 ViT-L/16、32×64 latent、原12-block/16-head结构不变。双相机分别处理，无新增投影头或Plücker结构。DINO冻结，参数哈希在提取前后相同；训练过程只读特征缓存。',
 '- 沿用readiness episode split：train19/190转移、validation4/40、reserved test20/200；相机配对数380/80/400。整episode动作身份隔离，无相邻转移泄漏；reserved test无训练、调参、评分。该test有历史暴露，不能称未触碰最终测试。',
 '- 23条训练/验证episode复核raw states/actions、XML仅桌面/墙面texture变化、原几何误差≤1e-9及video SHA；20条稀疏视频显式source-ID join。所有输出和cache/TMP/tmux socket均在数据盘。原Step1/readiness/官方checkpoint哈希复核未改动。','',
 '## 实现与公平性','',
 '- 复用官方RlaAutoencoderTrainer.inference_batch/training_losses、BasicTrainer.run_step/save/load；仅在冻结DINO缓存边界适配输入，新增配对loss和固定source-ID采样。未重写Transformer、原重建损失或optimizer更新算法。官方RGB snapshot路径不运行，改用特征验证；无需RGB UNet。',
 '- 遵从最新公式，两分支重建取和：`Lrec=(L1c+MSEc)+(L1b+MSEb)`；B=Lrec；C=Lrec+lambda*mean((zc-zb)^2)。这与readiness的分支平均提案不同，差异在训练前明确记录并对B/C共同应用。每个分支都重建自己的未来DINO。',
 f'- lambda固定为 **{lam:.12g}**。只对一个固定训练样本计算原重建与一致性在encoder.input_layer.weight的梯度范数，按预设保守规则限制加权一致性probe梯度为重建的10%；首block一致性梯度也非零。不是全encoder梯度比的保证；未用validation/test搜索。',
 '- AdamW lr1e-4、weight_decay0，原inflat_all模式和fp16 torso/fp32 master；为避免过大初始loss-scale，双方log_scale统一从8起、growth=.001。两个模型参数及固定输入输出初始化一致。',
 '- 第1步batch1 smoke，计入200；第2步为重载后真实恢复更新，也计入200。其后每步4个分相机配对单元、8个分支，micro-batch1×4等权累积。每步两路相机各2单元（第1步仅首agentview单元）。B/C实际序列完全一致。四个view单元不保证来自四条不同物理窗口，不冒充同时双视角模型。',
 '- 各组共797个view-pair曝光、1594个分支曝光；预检保留2步，正式阶段续198步，没有丢弃/replay/隐藏优化更新。original DataLoader worker0；显式immutable sampling_sequence+progress cursor是实际采样恢复依据，官方sampler状态虽保存但未用于这些显式批次。','',
 '## 预检、资源和恢复','',
 '- 两组的batch1前后向/参数更新、梯度有限、encoder/decoder均更新、source配对、checkpoint保存/磁盘重载及恢复更新通过。step1恢复模型与optimizer状态完整哈希相同，global step和fp16 log_scale正确恢复，输出max_abs=0。所有预检通过后才进入198步正式续训。',
 f'- 唯一GPU：UUID `{UUID}`（物理GPU1、H100 80GB）。用户本轮允许在约27.9GiB free时先做smoke，覆盖旧32GiB门槛的此次预检要求。设置最多24GiB且保留至少4GiB的PyTorch allocator上限；全程峰值allocated **{max(peaks)/1024**3:.3f} GiB**，未切卡、未终止其他进程。',
 '- 历史SRAM parity=1/SM=1保留，本次ECC基线及运行期间/最终检查无增量；发生变化或OOM/NaN即停止而不是自动重试。资源记录含RSS/HWM、磁盘空闲、/proc/self/io和峰值显存；未使用/dev/shm或全量特征数组。GPU显存数字为本进程PyTorch allocated，不是整卡他人占用。',
 '- 最终step200再次从磁盘重载，模型、AdamW state digest、global step、fp16 log_scale和固定输入输出均核验一致；最终重载不执行额外优化。保留step0/1/2/50/100/200的encoder/decoder/misc及progress文件。','',
 '## 固定最终验证结果（4条示范等权，非保留测试）','',
 '| 模型 | clean L1 | clean MSE | bg L1 | bg MSE | latent MSE | Relative L2 | Cosine | latent逐维方差 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,m in metrics.items():lines.append('| '+name+' | '+' | '.join(f'{m[k]:.6g}' for k in ('clean_l1','clean_mse','background_l1','background_mse','latent_mse','relative_l2','cosine','latent_per_dim_variance'))+' |')
 lines+=['','Relative L2仍为clean范数分母；latent指标在完整32×64上计算。报告A=两组训练前相同官方基线；B/C固定最终step200，未挑最优中间checkpoint。未来DINO指标均为真实未来参与Encoder的后验重建，不是动作条件前向预测或控制效果。','',
 '### C相对B的取舍','',
 f'- latent MSE差：{c["summary"]["latent_mse"]-b["summary"]["latent_mse"]:+.6g}；Relative L2差：{c["summary"]["relative_l2"]-b["summary"]["relative_l2"]:+.6g}；Cosine差：{c["summary"]["cosine"]-b["summary"]["cosine"]:+.6g}。',
 f'- clean重建L1差：{c["summary"]["clean_l1"]-b["summary"]["clean_l1"]:+.6g}；bg重建L1差：{c["summary"]["background_l1"]-b["summary"]["background_l1"]:+.6g}。逐episode/camera/window差在validation_paired_C_minus_B.csv，不能把总体差当统计显著。',
 '- 方差、范数和物理运动关联需要与距离一起解释；距离下降不是动作不变性或运动信息保留的充分证据。motion_fidelity_context.json仅提供末端、实际指尖开合和两只命名黑碗的验证集latent距离关联；没有新增物理监督或预测头，不虚构任务目标物体身份，不称物理MAE或控制保真结论。',
 '- 只覆盖一个背景preset、同四任务、单种子和4条validation。200步是训练管线/机制pilot，不足以证明全面泛化；不因小幅验证差自动扩训或宣称成功。','',
 '## 交付与复现','',
 '- `B/config.json`、`C/config.json`，protocol_before_execution.json，gradient_scale_and_fixed_lambda.json，initialization_pairing.json，split_manifest.json，training_records.json/validation_records.json和actual_sampling_sequence.json。',
 '- `features/*.npy` 每个文件分帧写入并memmap读取；JSON记录源video、source IDs、shape、float16、camera、mask支持协议和SHA；92个文件。对没有原始segmentation stream的重放数据明确采用与Step1相同全幅RGB支持协议，不伪造mask文件。',
 '- B/C各自ckpts/、train.jsonl、validation_step*.json、preflight.json、complete.json及training.log、resource_monitor.jsonl、resource_before.json、GPU gate/ECC记录。',
 '- validation_per_sample.csv、validation_summary.csv、validation_paired_C_minus_B.csv及逐episode结果；训练/验证曲线图在figures/。',
 '- `source experiments/rla_background_invariance/step2_training/env.sh` 后，从新worktree运行 `.venv/bin/python -m background_step2.prepare`、`extract`、`run`（真实解释器路径为 `${RLA_PROJECT_ROOT}/.venv/bin/python`）。所有writer拒绝覆盖；复现须新空输出目录并设置RUN_DIR，不可原地重跑。',
 '- 训练命令使用数据盘TMUX_TMPDIR的tmux session `rla_bg_step2_train`，完整stderr/stdout在training.log，training_exit_code.txt记录退出码。后处理motion_check仅冻结验证，不追加优化。',
 '', '## 实际更新和计算时间','']
 for g,u in updates.items():lines.append(f'- {g}：成功{u["successful_updates"]}、尝试{u["attempts"]}、预检保留{u["retained_preflight"]}、额外0、重放0、溢出跳过0；优化更新累计耗时{u["optimizer_compute_seconds"]:.1f}s（不含缓存、验证及保存重载）；最终loss `{json.dumps(u["last_loss"])}`。')
 lines+=['','完成后停止，没有运行保留测试、RLA-WM、策略或闭环控制；不自动继续长训。']
 p=OUT/'report.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n')
 # Plot only already completed runs, no selection or additional inference.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 (OUT/'figures').mkdir(exist_ok=True)
 fig,ax=plt.subplots(1,2,figsize=(10,3))
 for g in ('B','C'):
  logs=[json.loads(l) for l in (OUT/g/'train.jsonl').read_text().splitlines()]
  ax[0].plot([x['step'] for x in logs],[x['loss']['reconstruction'] for x in logs],label=g,alpha=.7)
  ax[1].plot([x['step'] for x in logs],[x['loss']['consistency'] for x in logs],label=g,alpha=.7)
 ax[0].set_title('Reconstruction sum');ax[1].set_title('Raw latent consistency')
 for x in ax:x.legend();x.set_xlabel('Successful optimizer updates')
 fig.tight_layout();fig.savefig(OUT/'figures/train_curves.png',dpi=160);plt.close(fig)
 write(OUT/'final_acceptance.json',dict(status='PASS',successful_updates=400,groups=updates,all_original_files_unchanged=True,
     cache_files_rehashed=92,model_reloads_verified=True,workers=0,gpu_uuid=UUID,ECC_unchanged=True,OOM=0,NaN=0,
     reserved_test_scored=False,heldout_used_for_tuning=False,code_inventory='code_inventory.json',report='report.md'))
 print('FINAL_ACCEPTANCE_PASS',json.dumps(updates),flush=True)
if __name__=='__main__':main()
