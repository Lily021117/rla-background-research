"""Supplement AE sample identity and inspect the exact small Git delivery."""
import argparse,ast,re
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--package',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);es=read(root/'frozen_dataset_manifest.json')['episodes'];train={e['episode']:e for e in es if e['split']=='train'};ae=root/'ae_run';reg=read(sorted((ae/'model_registry_versions').glob('*.json'))[-1]);orders={arm:read(Path(reg[arm])/'sampling_order.json') for arm in ('C','D')};assert orders['C']==orders['D'] and len(orders['C'])==4800
for e,t,h in orders['C']:assert e in train and 0<=t<train[e]['valid_chunks'] and 1<=h<=8
record=dict(AE_C_D_order_exact=True,sampled_physical_transitions=4800,unique_source_train_episodes=len(set(e for e,t,h in orders['C'])),all_targets_h1_to8_valid=True,no_validation_in_AE_updates=True,sampling_order_files={arm:dict(path=str(Path(reg[arm])/'sampling_order.json'),sha256=sha(Path(reg[arm])/'sampling_order.json')) for arm in ('C','D')})
save(a.out/'AE_sampling_identity.json',record)
delivery=a.package/'delivery/run_20261024_02';clean=lambda s:s.replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc','${DATA_ROOT}')
for name in ('initialization_source_verification.json','AE_sampling_identity.json'):
 with (delivery/name).open('x') as f:f.write(clean((a.out/name).read_text()))
files=[]
for path in sorted(a.package.rglob('*')):
 if not path.is_file():continue
 assert path.suffix in ('.py','.json','.csv','.md'),path
 assert path.stat().st_size<2*1024**2,path
 content=path.read_text()
 assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY',content),path
 if 'delivery' in path.parts:assert '/data/lyc' not in content and '/root/' not in content,path
 if path.suffix=='.py':ast.parse(content,filename=str(path))
 files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
save(a.out/'publish_content_audit.json',dict(status='PASS_ALLOWLIST_SECRET_PATH_SIZE_AST_CHECKS',files=files,no_weights_images_sidecars_trajectories=True,new_history_base='9da6484d09c6faea07f6c8f89fea65ee6a8cb730'));print('PACKAGE CHECKED',len(files),flush=True)
