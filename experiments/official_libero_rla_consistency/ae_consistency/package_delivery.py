"""Sanitized small delivery only; weights/images/states/sidecars remain local."""
import argparse,json,re,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();a.destination.mkdir(exist_ok=False)
def clean(text):
    text=text.replace(str(a.run),'${RUN_DIR}')
    return re.sub(r'/data/[^\s"\n,;`]+','${LOCAL_PATH}',text)
names=['report.md','training_plan.json','consistency_scale_calibration.json','official_initialization.json','representation_summary.csv','latent_distribution_metrics.csv','decoder_parameter_scale.csv','final_checkpoint_inventory.csv','final_delivery_verification.json','resource_reauthorization.json','recovery_replay_verification.json','hardware_incident.json','sidecar_delivery_summary.json']
for name in names:(a.destination/name).write_text(clean((a.run/name).read_text()))
policy=a.run/'policy_readiness_retry1/isolated_smoke'
for name in ('policy_config_C.json','policy_config_D.json','shared_training_protocol.json','policy_initialization_acceptance.json'):(a.destination/name).write_text(clean((policy/name).read_text()))
protocol=json.loads((a.run/'development_protocol/evaluation_protocol.json').read_text());plan=json.loads((a.run/'development_protocol/protocol_plan.json').read_text())
summary=dict(status=protocol['status'],new_seeds=[x['seed'] for x in protocol['new_development_states']],conditions=protocol['conditions'],historical_regression_episodes=protocol['historical_development'],state_checks=[dict(seed=x['seed'],status=x['status'],robot_pose_fixed=x['robot_pose_fixed'],conditions=[dict(condition=c['condition'],RGB_MAE_vs_clean=c['RGB_MAE_vs_clean'],geometry_match=c['geometry_match'],segmentation_match=c['segmentation_match'],depth_match=c['depth_match']) for c in x['conditions']]) for x in protocol['new_development_states']],unseen_RGB=plan['unseen_RGB'],independence=protocol['independence'],policy_rollouts=0)
(a.destination/'development_protocol_summary.json').write_text(json.dumps(summary,indent=2))
(a.destination/'evaluation_freeze_supplement.json').write_text((a.run/'development_protocol/evaluation_freeze_supplement.json').read_text())
for arm,root in [('C',a.run/'AE_C'),('D_effective301_600',a.run/'AE_D_recovery_01')]:
    (a.destination/f'training_metrics_{arm}.csv').write_text((root/'training_metrics.csv').read_text())
# The original D run is retained, with replay rows explicitly identified rather than merged silently.
(a.destination/'training_metrics_D_initial1_380.csv').write_text((a.run/'AE_D/training_metrics.csv').read_text())
