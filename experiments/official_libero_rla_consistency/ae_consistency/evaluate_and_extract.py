"""Matched AE representation diagnostics and official anchored sidecars."""
import argparse,csv,time,collections
from common import *
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['evaluate','sidecar'],required=True);p.add_argument('--models',nargs='+',choices=['reference','C','D']);p.add_argument('--common-stats',type=Path);a=p.parse_args();ext,init=load(a.project);before=ecc();ext.dino_extractor.cpu();del ext.dino_extractor;torch.cuda.empty_cache();es=entries(a.run)
def loadarm(arm):
    for k,m in ext.models.items():
        path=init/'ckpts'/f'{k}_step0090000.pt' if arm=='reference' else armdir(a.run,arm)/f'{k}_step0600.pt'
        state=torch.load(path,map_location='cuda',weights_only=True);m.load_state_dict(state,strict=True);m.eval().requires_grad_(False)
def csvout(name,rows):
    with (a.run/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if a.mode=='evaluate':
    fixed=[dict(episode=e['episode'],t=int(t),h=h) for e in es if e['split']=='validation' for t in np.linspace(0,e['valid_chunks']-1,10,dtype=int) for h in range(1,9)];save(a.run/'representation_sample_manifest.json',dict(selection='10 evenly spaced valid anchors per development episode x8 horizons, fixed before scoring',samples=fixed));byid={e['episode']:e for e in es};rows=[];distribution=[];ablations=[];parameter_scale=[]
    with torch.inference_mode():
        for arm in (a.models or ('reference','C','D')):
            loadarm(arm);zs={c:[] for c in ('clean','background')}
            for x in fixed:
                e=byid[x['episode']];vals={};lab=np.load(Path(e['root'])/'trajectory_labels.npz');ee=float(np.linalg.norm(lab['proprio'][x['t']+x['h'],:3]-lab['proprio'][x['t'],:3]))
                for c in zs:
                    x0,x1,r=pair(a.run,e,x['t'],x['h'],c);z,pred=forward(ext,ext.models,x0,x1,r);assert torch.isfinite(z).all() and torch.isfinite(pred).all();zs[c].append(z.cpu().numpy()[0]);vals[c]=(z,pred,x0,x1,r)
                    for cam in range(2):rows.append(dict(model=arm,episode=e['episode'],t=x['t'],h=x['h'],appearance=c,camera=cam,ee_displacement_m=ee,DINO_delta_RMS=float((x1[:,cam]-x0[:,cam]).square().mean().sqrt()),reconstruction_L1=float(F.l1_loss(pred[:,cam],x1[:,cam])),reconstruction_MSE=float(F.mse_loss(pred[:,cam],x1[:,cam])),copy_current_L1=float(F.l1_loss(x0[:,cam],x1[:,cam])),copy_current_MSE=float(F.mse_loss(x0[:,cam],x1[:,cam]))))
            zc=np.stack(zs['clean']).reshape(len(fixed),-1);zb=np.stack(zs['background']).reshape(len(fixed),-1);nc=np.linalg.norm(zc,axis=1);nb=np.linalg.norm(zb,axis=1);delta=np.linalg.norm(zc-zb,axis=1)
            paired=[dict(model=arm,**x,latent_MSE=float(np.mean((zc[i]-zb[i])**2)),relative_L2=float(delta[i]/(nc[i]+1e-8)),unit_L2=float(np.linalg.norm(zc[i]/max(nc[i],1e-8)-zb[i]/max(nb[i],1e-8))),cosine=float(np.dot(zc[i],zb[i])/(nc[i]*nb[i]+1e-8)),clean_norm=float(nc[i]),background_norm=float(nb[i])) for i,x in enumerate(fixed)]
            csvout(f'latent_pairs_{arm}.csv',paired)
            episodes=np.array([x['episode'] for x in fixed]);different=episodes[:,None]!=episodes[None,:]
            for c,z in [('clean',zc),('background',zb)]:
                centered=z.astype(np.float64)-z.mean(0);variances=centered.var(0);s=np.linalg.svd(centered,compute_uv=False);eig=s*s;prob=eig/max(eig.sum(),1e-20);norm=np.linalg.norm(z,axis=1);sq=np.maximum(norm[:,None]**2+norm[None,:]**2-2*z@z.T,0);between=np.sqrt(sq[different]);u=z/np.maximum(norm[:,None],1e-8);unit=np.sqrt(np.maximum(2-2*u@u.T,0))[different]
                distribution.append(dict(model=arm,appearance=c,samples=len(z),norm_mean=float(norm.mean()),norm_std=float(norm.std()),norm_median=float(np.median(norm)),dimension_variance_mean=float(variances.mean()),dimension_variance_median=float(np.median(variances)),dimension_variance_p05=float(np.quantile(variances,.05)),dimension_variance_p95=float(np.quantile(variances,.95)),participation_ratio=float(eig.sum()**2/max(np.sum(eig*eig),1e-20)),entropy_effective_rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))),different_episode_L2_mean=float(between.mean()),different_episode_unit_L2_mean=float(unit.mean())))
                np.save(a.run/f'latent_dimension_variance_{arm}_{c}.npy',variances)
                distribution[-1].update(latent_mean_scalar=float(z.mean()),latent_mean_vector_norm=float(np.linalg.norm(z.mean(0))))
            for i,x in enumerate(fixed):
                anchors=np.linspace(0,byid[x['episode']]['valid_chunks']-1,10,dtype=int)
                if x['h']!=8 or x['t'] not in anchors[[0,5,9]]:continue
                e=byid[x['episode']];other=next(j for j,y in enumerate(fixed) if y['episode']!=x['episode'] and y['h']==8)
                for c in ('clean','background'):
                    x0,x1,r=pair(a.run,e,x['t'],8,c);z=torch.from_numpy(np.stack(zs[c])[i:i+1]).cuda();wrong=torch.from_numpy(np.stack(zs[c])[other:other+1]).cuda();errors={}
                    for name,tokens in [('true',z),('zero',torch.zeros_like(z)),('other_episode',wrong)]:
                        _,pred=ext._forward_views(ext.models['decoder'],x0.reshape(1,-1,1024),tokens,2,plucker=r);errors[name]=float(F.l1_loss(pred.reshape_as(x1),x1)+F.mse_loss(pred.reshape_as(x1),x1))
                    ablations.append(dict(model=arm,episode=e['episode'],appearance=c,t=x['t'],h=8,**errors))
            for name,p in ext.models['decoder'].named_parameters():
                if 'token_proj' in name or 'out_layer.weight'==name:parameter_scale.append(dict(model=arm,parameter=name,norm=float(p.float().norm())))
            print('REPRESENTATION',arm,'DONE',flush=True);assert counters(before)==counters(ecc())
    csvout('reconstruction_metrics.csv',rows);csvout('latent_distribution_metrics.csv',distribution);csvout('decoder_ablation_metrics.csv',ablations);csvout('decoder_parameter_scale.csv',parameter_scale);save(a.run/'representation_acceptance.json',dict(status='PASS_DIAGNOSTIC_NOT_METHOD',samples=len(fixed),episodes=sum(e['split']=='validation' for e in es),models=a.models or ['reference','C','D'],optimizer_updates=0,policy_rollouts=0))
else:
    roots={};provenance=[];checks=[];moments={}
    with torch.inference_mode():
        for arm in (a.models or ('C','D')):
            loadarm(arm);enc=armdir(a.run,arm)/'encoder_step0600.pt';encsha=sha(enc);total=np.zeros((16,64),dtype=np.float64);total2=total.copy();count=0
            for c in ('clean','background'):
                root=a.run/'sidecars'/arm/c;(root/'libero_spatial').mkdir(parents=True);keys={};roots[arm,c]=root
                for idx,e in enumerate(es):
                    tokens=torch.from_numpy(np.array(cached(a.run,e,c),copy=True)).cuda();ray=torch.from_numpy(np.array(rays(a.run,e),copy=True)).cuda();z=ext.encode_episode(tokens,ray,chunk=8,batch_pairs=8,pairs_mode='anchored').float().cpu().numpy();N=e['frames'];assert z.shape==(N,8,16,64) and np.isfinite(z).all();lab=np.load(Path(e['root'])/'trajectory_labels.npz');key=hashlib.sha1(np.ascontiguousarray(lab['proprio'],dtype=np.float32).tobytes()).hexdigest();assert key not in keys
                    path=root/'libero_spatial'/f'ep_{idx:06}.npy';np.save(path,z.astype(np.float16));np.save(root/'libero_spatial'/f'mask_{idx:06}.npy',lab['valid_horizon_mask']);keys[key]=dict(file=path.name,episode_index=idx,traj_len=N)
                    if e['split']=='train':v=z[:e['valid_chunks']].reshape(-1,16,64).astype(np.float64);total+=v.sum(0);total2+=(v*v).sum(0);count+=len(v)
                    for t,h in [(0,1),(0,8),(e['valid_chunks']//2,4)]:
                        ref,_=ext._forward_views(ext.models['encoder'],ext._compose_inverse_input(x_t_flat=tokens[t:t+1].reshape(1,-1,1024),x_T_flat=tokens[t+h:t+h+1].reshape(1,-1,1024)),None,2,plucker=ray[t:t+1]);saved=z[t,h-1].astype(np.float16).astype(np.float32);error=float(np.max(abs(ref.float().cpu().numpy()[0]-saved)));tol=max(.001,.02*float(ref.float().abs().max()));assert error<tol;checks.append(dict(model=arm,episode=e['episode'],appearance=c,anchor=t,h=h,future=t+h,max_error=error,tolerance=tol))
                    provenance.append(dict(model=arm,episode=e['episode'],source_demo=e['source_demo'],split=e['split'],appearance=c,frames=N,valid_full_chunks=e['valid_chunks'],key=key,file=str(path),sha256=sha(path),encoder_sha256=encsha,mask_file=str(root/'libero_spatial'/f'mask_{idx:06}.npy')))
                save(root/'libero_spatial/keys.json',keys);save(root/'manifest.json',dict(chunk=8,num_tokens=16,token_dim=64,pairs='anchored',format='npy',suites=['libero_spatial'],appearance=c,model=arm,encoder_sha256=encsha,encoder=str(enc),source='derived LIBERO full sequences',img_size=224,normalization='common AE-C train-only clean+background center statistics'))
                print('SIDECAR',arm,c,'DONE',flush=True)
            mean=total/count;std=np.sqrt(np.maximum(total2/count-mean*mean,0));moments[arm]=dict(count=count,mean=mean.tolist(),std=std.tolist(),std_floor=1e-3,scope='training complete windows, both appearances only')
        common_stats=json.loads(a.common_stats.read_text()) if a.common_stats else moments['C']
        for (arm,c),root in roots.items():save(root/'raw_target_statistics.json',moments[arm]);save(root/'stats.json',common_stats)
    assert counters(before)==counters(ecc())
    save(a.run/'sidecar_provenance.json',provenance);save(a.run/'sidecar_alignment_audit.json',dict(status='PASS_CACHE_RECOMPUTE',formula='future=min(t+h,N-1), h1..8 anchored on t',shape='(N,8,16,64)',appearances_separate_roots=True,models_separate_roots=True,camera_order=['agentview','robot0_eye_in_hand'],invalid_tail_masked_not_real_targets=True,common_normalization='Base store stats are AE-C training statistics; raw_target_statistics contains each encoder own training mean. Policy reader centering override is audited separately in target_readiness.json.',checks=checks,source_image_recompute='pending independent raw RGB check in separate audit',policy_updates=0))
