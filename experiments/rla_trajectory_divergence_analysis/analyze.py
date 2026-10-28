"""CPU-only existing-record analysis. Never imports policy or simulator code.

Usage: python analyze.py --source GAP_RUN --out NEW_DIRECTORY
Artifacts are exclusive-created. Raw paths stay local; publish only sanitized summaries.
"""
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def save(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def table(p,x):
    with Path(p).open('x') as f: pd.DataFrame(x).to_csv(f,index=False)
def first(x):
    v=np.flatnonzero(x); return int(v[0]+1) if len(v) else None
def sustained(x,n=8):
    if len(x)<n:return None
    v=np.flatnonzero(np.convolve(np.asarray(x,int),np.ones(n,int),'valid')==n)
    return int(v[0]+1) if len(v) else None

def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();src=args.source;out=args.out
    out.mkdir(exist_ok=True);(out/'curves').mkdir(exist_ok=False)
    conditions=['clean']+[f'new_background_{i}' for i in range(1,5)]
    rules=dict(base_commit='3a3a3c091e707cd5372d763ed80ff1f86eee9849',scope='C vs C-shared-target, final600; two orders; 20 states; clean + four frozen new backgrounds',
      sequence='freeze -> inventory -> validation -> clean pairs -> background pairs',
      time='control step 1 = first post-action state; video frame k is pre-action at zero-based control 8*k; 20Hz control,2.5fps video',
      existing_proxies=dict(lift_m=.03,near_ee_bowl_m=.10,center_xy_m=.03,goal='recorded height AND contact AND xy<0.03; original success label retained'),
      new_descriptive_rules=dict(main=dict(position_separation_m=.03,persistence_steps=8),sensitivity=dict(position_separation_m=.05,persistence_steps=8),basis='3cm uses existing target-centering tolerance;5cm fixed sensitivity;8 steps is one deployed chunk, not optimized for outcomes'),
      candidate_rules=['post-lift first open while outside xy goal','post-lift finger-contact loss before goal (contact is not stable grasp truth)','post-lift outward XY change >=3cm (sensitivity5cm) during at least8 consecutive bowl-descending intervals','center entered then lost before goal','EE re-approach: near->far->near using existing0.10m threshold, not confirmed regrasp'],
      grouping=['order','condition','both_success/C_only/shared_only/both_failure'],
      video_selection='one per order x outcome group; conditions in clean,bg1,bg2,bg3,bg4 order then ascending state; empty groups not replaced; at most8 pairs',
      no_inference_or_simulation=True,no_threshold_search=True,no_C_expert_assumption=True,intervals='none; descriptive census; no independence claims',
      counters=dict(model_forward=0,backward=0,optimizer_updates=0,AE_updates=0,new_closed_loop=0,simulation_replay=0))
    save(out/'frozen_analysis_rules.json',rules)
    sources={};rows=[]
    for name in ['reused_results.json','per_trial_results.json','reuse_sources.json','comparability_audit.json','complete_comparison_table.csv']:
        sources[str(src/name)]=sha(src/name)
    for r in read(src/'reused_results.json'):
        if r['method'] in ('C','C-shared-target') and r['condition'] in conditions:
            r=dict(r);r['ledger']=r['source_results'];rows.append(r)
    for r in read(src/'per_trial_results.json'):
        if r['model'].startswith('C-shared-target_order'):
            r=dict(r,method='C-shared-target',order=int(re.search(r'order(\d+)',r['model'])[1]),ledger=str(src/'per_trial_results.json'));rows.append(r)
    index={}
    for r in rows:
        key=(r['order'],r['seed'],r['condition'],r['method']);assert key not in index;index[key]=r
        sources[r['ledger']]=sha(r['ledger'])
    manifest=[];video=[]
    for order in (20261101,20261102):
        for cond in conditions:
            for seed in range(2026102500,2026102520):
                pair=dict(order=order,state=seed,condition=cond,pair_id=f'{order}_{seed}_{cond}',status='PENDING')
                rr=[index.get((order,seed,cond,m)) for m in ('C','C-shared-target')]
                if any(r is None for r in rr):pair['status']='MISSING_RESULT'
                else:
                    c,s=rr;pair['group']='both_success' if c['success'] and s['success'] else 'C_only' if c['success'] else 'shared_only' if s['success'] else 'both_failure'
                    pair['matching_metadata']=all(c[k]==s[k] for k in ('state_sha256','xml_sha256','language','condition','seed'))
                    for label,r in zip(('C','shared'),rr):
                        pair[label+'_checkpoint_sha256']=r['checkpoint_sha256'];pair[label+'_result_ledger']=r['ledger'];pair[label+'_trajectory']=r['trajectory'];pair[label+'_video']=r['video']
                        for p in [Path(r['trajectory']),Path(r['trajectory']).with_name('phase_timeline.json'),Path(r['trajectory']).with_name('result.json'),Path(r['video'])]:
                            if p.is_file():sources[str(p)]=sha(p)
                            else:pair['status']='MISSING_FILE'
                    if pair['status']=='PENDING':pair['status']='READY' if pair['matching_metadata'] else 'METADATA_MISMATCH'
                manifest.append(pair)
    # Freeze actual count and deterministic video candidates before any behavior statistics.
    save(out/'paired_manifest.json',manifest);save(out/'input_hashes.json',sources)
    save(out/'inventory_acceptance.json',dict(planned_pairs=200,ready_pairs=sum(p['status']=='READY' for p in manifest),results=len(rows),missing=[p['pair_id'] for p in manifest if p['status']!='READY']))
    for order in (20261101,20261102):
        for group in ('both_success','C_only','shared_only','both_failure'):
            choices=[p for p in manifest if p['order']==order and p.get('group')==group and p['status']=='READY']
            if choices:video.append(dict(choices[0],review_status='PENDING'))
    save(out/'video_review_selection.json',video)
    audits=[];traj=[];pairs=[];events=[];continuous=[];boundary=[]
    for pair in sorted(manifest,key=lambda p:(conditions.index(p['condition']),p['order'],p['state'])):
        if pair['status']!='READY':continue
        data=[];valid=True
        for method,label in [('C','C'),('C-shared-target','shared')]:
            r=index[(pair['order'],pair['state'],pair['condition'],method)];p=Path(r['trajectory'])
            try:
                z=dict(np.load(p,allow_pickle=False));ph=read(p.with_name('phase_timeline.json'));n=r['steps'];act=z['executed_actions'];pred=z['predicted_chunks'];ee=z['eef_positions'];b=z['target_bowl_positions']
                assert act.shape==(n,7) and pred.shape==((n+7)//8,8,7) and ee.shape==b.shape==(n,3)
                assert len(z['states'])==n and len(ph)==n and all(np.isfinite(v).all() for v in z.values())
                assert [x['step'] for x in ph]==list(range(1,n+1))
                decoded=pred.reshape(-1,7)[:n].copy();decoded[:,6]=-np.sign(2*decoded[:,6]-1)
                assert np.array_equal(decoded,act),'predicted/deployed mismatch'
                assert np.allclose(ee,[[x[f'ee_{i}'] for i in range(3)] for x in ph],atol=1e-12,rtol=0)
                # Body center in NPZ and geometry center in timeline are not assumed identical.
                assert bool(ph[-1]['goal'])==r['success']
                assert all(x['goal']==(x['height_condition'] and x['contact_condition'] and x['xy_distance_m']<.03) for x in ph)
                audit=dict(pair_id=pair['pair_id'],method=method,status='PASS',steps=n,chunk_count=len(pred),state_width=z['states'].shape[1],body_vs_geom_bowl_max_m=float(np.max(abs(b-np.array([[x[f'bowl_{i}'] for i in range(3)] for x in ph])))))
                audits.append(audit)
            except Exception as exc:
                audits.append(dict(pair_id=pair['pair_id'],method=method,status='INVALID',reason=str(exc)));valid=False;continue
            xy=np.array([x['xy_distance_m'] for x in ph]);height=np.array([x['vertical_bowl_plate_m'] for x in ph]);near=np.linalg.norm(ee-b,axis=1)<=.10;close=act[:,6]>0;lift=r.get('first_lift_step');start=lift-1 if lift is not None else n;post=ph[start:];goal=np.array([x['goal'] for x in ph]);center=xy<.03
            def event(test):return next((x['step'] for x in post if test(x)),None)
            opened=event(lambda x:not x['close_command']);openoutside=event(lambda x:not x['close_command'] and not x['xy_condition'] and not x['goal'])
            losses=[y['step'] for x,y in zip(post[:-1],post[1:]) if x['finger_contact'] and not y['finger_contact'] and not y['goal']]
            centerloss=[i+1 for i in range(max(1,start),n) if center[i-1] and not center[i] and not goal[i]]
            outward={}
            for suffix,threshold in [('main',.03),('sensitivity',.05)]:
                hit=[]
                for i in range(start,max(start,n-8)):
                    seg=ph[i:i+9]
                    if len(seg)==9 and all(y['bowl_2']<x['bowl_2'] for x,y in zip(seg[:-1],seg[1:])) and xy[i+8]-xy[i]>=threshold and not goal[i:i+9].any():hit.append(i+9)
                outward[suffix]=hit[0] if hit else None
            reapproach=[];seen=False;left=False
            for i,v in enumerate(near):
                if v and left:reapproach.append(i+1);left=False
                if v:seen=True
                elif seen:left=True
            metric=dict(pair_id=pair['pair_id'],order=pair['order'],state=pair['state'],condition=pair['condition'],group=pair['group'],method=method,success=r['success'],steps=n,stage=r['stage'],first_close=first(close),first_near=first(near),first_lift=lift,first_postlift_open=opened,open_outside_goal_step=openoutside,contact_loss_step=losses[0] if losses else None,center_loss_step=centerloss[0] if centerloss else None,outward_descent_step=outward['main'],outward_descent_sensitivity_step=outward['sensitivity'],reapproach_count=len(reapproach),first_reapproach=reapproach[0] if reapproach else None,final_xy_m=float(xy[-1]),min_postlift_xy_m=float(xy[start:].min()) if start<n else None,final_height_m=float(height[-1]),eef_path_m=float(np.linalg.norm(np.diff(ee,axis=0),axis=1).sum()),bowl_path_m=float(np.linalg.norm(np.diff(b,axis=0),axis=1).sum()),close_fraction=float(close.mean()),postlift_goal=any(x['goal'] for x in post),postlift_center=any(x['xy_condition'] for x in post),postlift_contact=any(x['contact_condition'] for x in post))
            for key in ('open_outside_goal_step','contact_loss_step','center_loss_step','outward_descent_step','outward_descent_sensitivity_step'):
                metric[key+'_before_terminal']=metric[key] is not None and metric[key]<n
                events.append(dict(pair_id=pair['pair_id'],method=method,event=key,step=metric[key],relative_to_lift=None if metric[key] is None or lift is None else metric[key]-lift,before_terminal=metric[key+'_before_terminal']))
            traj.append(metric);data.append((z,ph,metric))
        if not valid or len(data)!=2:continue
        (c,cp,cm),(s,sp,sm)=data;n=min(cm['steps'],sm['steps']);de=np.linalg.norm(c['eef_positions'][:n]-s['eef_positions'][:n],axis=1);db=np.linalg.norm(c['target_bowl_positions'][:n]-s['target_bowl_positions'][:n],axis=1);da=abs(c['executed_actions'][:n]-s['executed_actions'][:n]);d0=abs(c['predicted_chunks'][0]-s['predicted_chunks'][0]);rel=np.linalg.norm((c['eef_positions'][:n]-c['target_bowl_positions'][:n])-(s['eef_positions'][:n]-s['target_bowl_positions'][:n]),axis=1)
        pm={k:pair[k] for k in ('pair_id','order','state','condition','group')};pm.update(common_steps=n,C_steps=cm['steps'],shared_steps=sm['steps'],eef_separation_mean_m=float(de.mean()),bowl_separation_mean_m=float(db.mean()),relative_position_separation_mean_m=float(rel.mean()),first_sustained_3cm=sustained(de>.03),first_sustained_5cm=sustained(de>.05),first_chunk_translation_mae=float(d0[:,:3].mean()),first_chunk_rotation_command_mae=float(d0[:,3:6].mean()),first_chunk_gripper_raw_mae=float(d0[:,6].mean()),executed_translation_mae=float(da[:,:3].mean()),executed_rotation_command_mae=float(da[:,3:6].mean()),executed_gripper_disagreement=float((da[:,6]>0).mean()),lift_delay_shared_minus_C=None if cm['first_lift'] is None or sm['first_lift'] is None else sm['first_lift']-cm['first_lift'])
        pairs.append(pm)
        for i in range(n):continuous.append(dict(pair_id=pair['pair_id'],step=i+1,chunk_position=i%8,eef_separation_m=de[i],bowl_separation_m=db[i],relative_separation_m=rel[i],translation_command_mae=da[i,:3].mean(),rotation_command_mae=da[i,3:6].mean(),gripper_disagrees=bool(da[i,6]>0)))
        for position in range(8):
            q=da[position::8];boundary.append(dict(pair_id=pair['pair_id'],order=pair['order'],condition=pair['condition'],group=pair['group'],chunk_position=position,count=len(q),translation_command_mae=float(q[:,:3].mean()),rotation_command_mae=float(q[:,3:6].mean()),gripper_disagreement=float((q[:,6]>0).mean())))
        fig,axs=plt.subplots(3,2,figsize=(12,9));
        for z,ph,m in data:
            t=np.arange(1,m['steps']+1);xy=[x['xy_distance_m'] for x in ph];h=[x['vertical_bowl_plate_m'] for x in ph];name=m['method'];axs[0,0].plot(t,xy,label=name);axs[0,1].plot(t,h,label=name);axs[1,0].plot(t,z['executed_actions'][:,6],label=name);axs[1,1].plot(t,np.linalg.norm(z['eef_positions']-z['target_bowl_positions'],axis=1),label=name)
            if m['first_lift'] is not None:axs[2,0].plot(t-m['first_lift'],xy,label=name)
        axs[2,1].plot(np.arange(1,n+1),de,label='EE');axs[2,1].plot(np.arange(1,n+1),db,label='bowl');
        for ax,title in zip(axs.flat,['Bowl-plate XY (m)','Bowl-plate vertical (m)','Executed gripper (+close)','EE-body bowl distance (m)','XY aligned to first lift','Pair separation (m)']):ax.set_title(title);ax.legend(fontsize=7);ax.set_xlabel('control step (post-action; lift-relative only bottom-left)')
        fig.suptitle(pair['pair_id']+' / '+pair['group']);fig.tight_layout();fig.savefig(out/'curves'/f"{pair['pair_id']}.png",dpi=90);plt.close(fig)
    for name,x in [('field_time_audit',audits),('trajectory_metrics',traj),('event_table',events),('paired_summary',pairs),('continuous_pair_curves',continuous),('chunk_position_metrics',boundary)]:table(out/(name+'.csv'),x)
    df=pd.DataFrame(traj);pf=pd.DataFrame(pairs)
    summary=[]
    for key,g in df.groupby(['order','condition','group','method']):
        row=dict(zip(['order','condition','group','method'],key));row['n']=len(g)
        for c in ['open_outside_goal_step_before_terminal','contact_loss_step_before_terminal','center_loss_step_before_terminal','outward_descent_step_before_terminal','outward_descent_sensitivity_step_before_terminal']:row[c]=int(g[c].sum())
        row['lift_available']=int(g.first_lift.notna().sum());row['reapproach_any']=int((g.reapproach_count>0).sum());row['median_final_xy_m']=float(g.final_xy_m.median());summary.append(row)
    table(out/'candidate_group_summary.csv',summary)
    table(out/'outcome_group_counts.csv',pf.groupby(['order','condition','group']).size().rename('pairs').reset_index().to_dict('records'))
    save(out/'analysis_acceptance.json',dict(planned_pairs=200,valid_pairs=len(pairs),valid_trajectories=len(traj),invalid_field_records=[r for r in audits if r['status']!='PASS'],clean_pairs=sum(p['condition']=='clean' for p in pairs),background_pairs=sum(p['condition']!='clean' for p in pairs),manual_video_pairs_selected=len(video),counters=rules['counters']))
    print(json.dumps(read(out/'analysis_acceptance.json'),indent=2))

if __name__=='__main__':main()
