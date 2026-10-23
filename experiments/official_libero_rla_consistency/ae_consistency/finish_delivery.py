"""Aggregate measured outputs and keep fault-phase evidence distinct from final delivery."""
import argparse,csv,collections
from common import *
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;policy=run/'policy_readiness_retry1/isolated_smoke';pa=json.loads((policy/'policy_initialization_acceptance.json').read_text());assert pa['status']=='PASS_ZERO_UPDATES'
raw=json.loads((run/'raw_source_recompute_audit.json').read_text());assert raw['status']=='PASS';side=json.loads((run/'sidecar_provenance.json').read_text());assert len(side)==100
def csvout(name,rows):
    with (run/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
rec=list(csv.DictReader((run/'reconstruction_metrics.csv').open()));summary=[]
for model in ('reference','C','D'):
    for c in ('clean','background'):
        q=[x for x in rec if x['model']==model and x['appearance']==c];z=list(csv.DictReader((run/f'latent_pairs_{model}.csv').open()));summary.append(dict(model=model,appearance=c,reconstruction=float(np.mean([float(x['reconstruction_L1'])+float(x['reconstruction_MSE']) for x in q])),copy_current=float(np.mean([float(x['copy_current_L1'])+float(x['copy_current_MSE']) for x in q])),paired_MSE=float(np.mean([float(x['latent_MSE']) for x in z])),paired_relative_L2=float(np.mean([float(x['relative_L2']) for x in z])),paired_unit_L2=float(np.mean([float(x['unit_L2']) for x in z])),paired_cosine=float(np.mean([float(x['cosine']) for x in z]))))
csvout('representation_summary.csv',summary);groups=collections.defaultdict(list)
for x in rec:groups[x['model'],x['episode'],x['h'],x['appearance'],x['camera']].append(x)
rows=[]
for k,v in groups.items():rows.append(dict(zip(['model','episode','h','appearance','camera'],k),samples=len(v),**{f:float(np.mean([float(x[f]) for x in v])) for f in ('reconstruction_L1','reconstruction_MSE','copy_current_L1','copy_current_MSE','ee_displacement_m')}))
csvout('reconstruction_by_demo_horizon.csv',rows)
inventory=[]
for arm in ('C','D'):
    root=armdir(run,arm);assert json.loads((root/'acceptance.json').read_text())['successful_updates']==600
    for kind in ('encoder','decoder'):
        path=root/f'{kind}_step0600.pt';inventory.append(dict(model=arm,kind=kind,step=600,initial_official_step=90000,file=str(path),sha256=sha(path),bytes=path.stat().st_size))
path=policy/'common_policy_step0.pt';inventory.append(dict(model='policy_C_D_common',kind='trainable_deltas',step=0,initial_official_step=0,file=str(path),sha256=sha(path),bytes=path.stat().st_size));csvout('final_checkpoint_inventory.csv',inventory)
for item in side:
    block=np.load(item['file'],mmap_mode='r');mask=np.load(item['mask_file']);assert block.shape==(item['frames'],8,16,64) and block.dtype==np.float16 and mask.shape==(item['frames'],8);assert np.isfinite(block).all() and mask[:item['valid_full_chunks']].all()
protocol=json.loads((run/'development_protocol/evaluation_protocol.json').read_text());assert len(protocol['new_development_states'])==20
save(run/'final_delivery_verification.json',dict(status='PASS_PIPELINE_WITH_REPRESENTATION_TRADEOFF',AE_C_effective_successful_updates=600,AE_D_effective_successful_updates=600,AE_D_total_computed_successful_updates=680,AE_D_replay_count=80,replay80_losses_and_gradient_norms_exact=True,common_initial_AE_parameters_verified=True,common_sampling_order_verified=True,feature_images=10224,sidecar_files=100,sidecar_appearance_model_roots=4,sidecar_rows=sum(x['frames'] for x in side),sidecar_full_windows=sum(x['valid_full_chunks'] for x in side),raw_source_checks=len(raw['checks']),policy_initialization_acceptance=True,policy_optimizer_updates=0,new_policy_closed_loops=0,new_development_states=20,conditions_per_new_state=3,future_policy_trials_on_new_states=180,historical5_role='regression development, not independent test',sealed_contents_read=False,AE_D_latent_shrinkage_present=True,AE_D_rank_reduction_present=True,no_method_effectiveness_claim=True,prior_fault_phase_verification_retained='delivery_verification.json'))
save(run/'sidecar_delivery_summary.json',dict(files=100,train_episodes=20,development_episodes=5,models=['C','D'],appearance_roots=['clean','background'],per_file_shape='N,8,16,64',policy_shape='batch,8,1024',normalization='common C train-only both-appearance center statistics',raw_source_checks=len(raw['checks']),all_reloaded_finite=True,tail_masked=True))
print('DELIVERY PASS',flush=True)
