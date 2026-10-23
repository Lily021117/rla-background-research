"""CPU visible-GEOM regional RGB accounting, no scene intervention or rollout."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
import argparse,json,csv
from pathlib import Path
import numpy as np,mujoco
from scipy.ndimage import binary_erosion
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output-name',default='regional_audit');a=p.parse_args()
rows=json.loads((a.run/'episode_manifest.json').read_text());out=a.run/a.output_name;out.mkdir();(out/'heatmaps').mkdir()
f=(out/'metrics.csv').open('x');writer=csv.DictWriter(f,fieldnames=['episode','timestep','camera','region','pixels','changed_pixels','MAE_255','max_channel_255']);writer.writeheader()
for e in rows:
    m=mujoco.MjModel.from_xml_path(str(Path(e['geometry_path']).parent/'clean.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
    tex={m.texture(n).id for n in ('tex-table','tex-wall')};root=a.run/'episodes'/e['episode']
    for t in range(e['frames']):
        with np.load(root/'frames'/f'{t:04}.npz') as x:s=x['state'];clean=x['clean'].transpose(0,2,3,1);bg=x['background'].transpose(0,2,3,1)
        d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
        for ci,camera in enumerate(('agentview','robot0_eye_in_hand')):
            r.enable_segmentation_rendering();r.update_scene(d,camera=camera,scene_option=opt);r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;seg=r.render()[:,::-1].copy();r.disable_segmentation_rendering()
            masks={k:np.zeros((224,224),bool) for k in ('background','robot_and_gripper_interior','robot_base_interior','stove_interior','other_opaque_entities_interior','entity_boundaries','transparent_or_unclassified')}
            scene={int(g.objid):g for g in r.scene.geoms[:r.scene.ngeom] if g.objtype==mujoco.mjtObj.mjOBJ_GEOM}
            for gid in np.unique(seg[:,:,0][seg[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM]):
                pix=(seg[:,:,0]==gid)&(seg[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM);mat=int(m.geom_matid[gid]);name=(m.geom(int(gid)).name or '')+' '+(m.body(int(m.geom_bodyid[gid])).name or '')
                if mat>=0 and tex.intersection(np.asarray(m.mat_texid[mat]).ravel().tolist()):masks['background']|=pix;continue
                if gid not in scene or scene[gid].rgba[3]<.999999:masks['transparent_or_unclassified']|=pix;continue
                yy,xx=np.where(pix);y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
                interior=np.zeros_like(pix);interior[y0:y1,x0:x1]=binary_erosion(pix[y0:y1,x0:x1],iterations=1);masks['entity_boundaries']|=pix&~interior
                # XML root robot0_link0 is the static base; the geometry names
                # do not contain 'base'. Preserve the same pixels, only split
                # this anatomically identified category from the robot union.
                category='robot_base_interior' if ('base' in name and 'robot' in name) or m.body(int(m.geom_bodyid[gid])).name=='robot0_link0' else 'robot_and_gripper_interior' if any(k in name for k in ('robot0','gripper','finger')) else 'stove_interior' if 'stove' in name else 'other_opaque_entities_interior'
                masks[category]|=interior
            delta=abs(clean[ci].astype(np.int16)-bg[ci].astype(np.int16))
            for region,mask in masks.items():
                n=int(mask.sum());v=delta[mask];writer.writerow(dict(episode=e['episode'],timestep=t,camera=camera,region=region,pixels=n,changed_pixels=int(np.any(v>0,axis=-1).sum()),MAE_255=float(v.mean()) if n else '',max_channel_255=int(v.max()) if n else ''))
            if t==0:Image.fromarray(np.uint8(np.clip(delta.max(-1)*4,0,255))).save(out/'heatmaps'/f"{e['episode']}_{camera}.png")
    r.close();f.flush();print('REGIONS',e['episode'],flush=True)
f.close()
with (out/'limitations.json').open('x') as f:json.dump(dict(shadow_contact_masks='Unavailable: no separate shadow/contact attribution claimed',reflection='Original reflection retained; RGB response measured, not counterfactually isolated per pixel',boundaries='Per-visible-opaque-GEOM one-pixel erosion, unchanged rule',policy_rollouts=0,GPU=False),f,indent=2)
