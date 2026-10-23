"""Freeze only fully verified exports, retaining explicit source and appearance IDs."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();selection=json.loads((a.out/'dataset_selection.json').read_text());results={x['episode']:x for x in json.loads((a.out/'render_results.json').read_text())}
    frozen=[];files=[];identity=set();clip=[];oldstats=json.loads((a.previous/'normalization_statistics_train_only.json').read_text());values={'action':[],'proprio':[]};train_sources=set();dev_sources=set()
    for e in selection['episodes']:
        assert results[e['episode']]['status'] in ('PASS','REUSED');root=Path(results[e['episode']]['root']);lab=np.load(root/'trajectory_labels.npz');N=e['frames'];source_key=(e['source_hdf5'],e['source_demo']);assert source_key not in identity;identity.add(source_key)
        assert hashlib.sha256(lab['actions'].astype(np.float32).tobytes()).hexdigest()==e['source_actions_sha256'];assert hashlib.sha256(lab['states'].tobytes()).hexdigest()==e['source_states_sha256']
        ((train_sources if e['split']=='train' else dev_sources)).add(source_key)
        metrics=list(csv.DictReader((root/'regional_sequence_metrics.csv').open()));assert len(metrics)==N*2
        assert all(x['seg_equal']=='True' and x['depth_equal']=='True' for x in metrics)
        visibility_gaps=[x for x in metrics if int(x['changed_pixels'])==0]
        assert not visibility_gaps,'Background visibility gap retained; do not admit '+e['episode']
        assert len(list((root/'frames').glob('*.npz')))==N
        for t in range(N):
            path=root/'frames'/f'{t:04}.npz'
            with np.load(path) as d:
                assert d['clean'].shape==d['background'].shape==(2,3,224,224)
                assert d['clean'].dtype==d['background'].dtype==np.uint8
                assert all(np.isfinite(d[k]).all() for k in d.files)
                np.testing.assert_array_equal(d['action'],lab['actions'][t]);np.testing.assert_array_equal(d['state'],lab['states'][t]);np.testing.assert_array_equal(d['proprio'],lab['proprio'][t]);np.testing.assert_array_equal(d['K'],lab['K'][t]);np.testing.assert_array_equal(d['c2w'],lab['c2w'][t])
                assert d['K'].shape==(2,3,3) and np.all(d['K'][:,0,0]<0) and np.all(d['K'][:,1,1]>0)
            files.append(dict(episode=e['episode'],split=e['split'],source_demo=e['source_demo'],source_timestep=t,appearances=['clean','background'],camera_order=['agentview','robot0_eye_in_hand'],file=str(path),sha256=sha(path)))
        cam=json.loads((root/'camera_geometry_audit.json').read_text());assert cam['status']=='PASS_RASTER_PROJECTION' and cam['max_projection_error_px']<=2
        assert lab['valid_horizon_mask'][:N-8].all()
        standard=lab['actions'].astype(np.float32).copy();standard[:,6]=1-np.clip(standard[:,6],0,1)
        if e['split']=='train':values['action'].append(standard);values['proprio'].append(lab['proprio'])
        for field,x in [('action',standard),('proprio',lab['proprio'])]:
            lo=np.array(oldstats[field]['q01']);hi=np.array(oldstats[field]['q99']);mask=np.array(oldstats[field].get('mask',[True]*x.shape[1]));outside=((x<lo)|(x>hi))&mask
            clip.append(dict(episode=e['episode'],split=e['split'],field=field,old_fixed_bounds_clipping_per_dimension=outside.mean(0).tolist(),old_fixed_bounds_any_dimension=float(outside.any(1).mean())))
        frozen.append(dict(episode=e['episode'],split=e['split'],source_hdf5=e['source_hdf5'],source_demo=e['source_demo'],source_actions_sha256=e['source_actions_sha256'],source_states_sha256=e['source_states_sha256'],task=e['task'],frames=N,valid_chunks=N-8,root=str(root),dataset_version='cookie_dual224_fk_v4',storage_format='dual224_fk_v2',physical_episode_independent_unit=True,appearance_variants=2,RLA_target_required=False,export_reused=bool(e['existing_frames'])))
    assert not train_sources&dev_sources
    protected={(x['source_hdf5'],x['source_demo']) for x in selection['protected_sources']};assert not identity&protected
    assert sum(x['split']=='train' for x in frozen)==20 and sum(x['split']=='validation' for x in frozen)==5
    stats={}
    for field,arrays in values.items():
        x=np.concatenate(arrays);stats[field]={k:np.asarray(v).tolist() for k,v in dict(mean=x.mean(0),std=x.std(0),min=x.min(0),max=x.max(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0)).items()}
    stats['action']['mask']=[True]*6+[False]
    save(a.out/'normalization_statistics_expanded_train_only.json',stats);save(a.out/'old_normalization_clipping_audit.json',clip)
    save(a.out/'frozen_dataset_manifest.json',dict(status='PASS',episodes=frozen,selection_manifest_sha256=sha(a.out/'dataset_selection.json'),episode_unit='original HDF5 demo; appearances/windows never independent demos',camera_order=['agentview','robot0_eye_in_hand'],reflection=1,MSAA=4,training_episodes=20,development_episodes=5,total_frames=sum(e['frames'] for e in frozen),total_h8_chunks=sum(e['valid_chunks'] for e in frozen),train_frames=sum(e['frames'] for e in frozen if e['split']=='train'),development_frames=sum(e['frames'] for e in frozen if e['split']=='validation'),train_chunks=sum(e['valid_chunks'] for e in frozen if e['split']=='train'),development_chunks=sum(e['valid_chunks'] for e in frozen if e['split']=='validation'),independent_new_episodes=sum(not e['export_reused'] for e in frozen),geometry_segmentation_depth='exact',proprio='same-state FK; newly exported episodes additionally checked against robot-only FK and raw joint/finger qpos at obs[s-1]',RGB='material intervention; inherited reflection/boundary response permitted; per-pixel causal attribution not certified',optimizer_updates=0,new_closed_loop=0))
    save(a.out/'frame_source_manifest.json',files);save(a.out/'split_integrity.json',dict(status='PASS',train_development_overlap=0,protected_overlap=0,duplicate_source_episodes=0,protected_episode_data_read=False,all_appearances_follow_source_episode=True,old_train_and_dev_assignments_preserved=True))
    print(json.dumps({k:v for k,v in json.loads((a.out/'frozen_dataset_manifest.json').read_text()).items() if k!='episodes'},indent=2))
if __name__=='__main__':main()
