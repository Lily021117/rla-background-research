"""Restore frozen development states without policy inference or dynamics."""
import argparse,json,os
from pathlib import Path
import numpy as np
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero import get_libero_path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/'evaluation_protocol.json').read_text());rows=[];seen=set()
for t in protocol['trials']:
    if t['episode'] in seen:continue
    seen.add(t['episode']);saved=np.load(Path(t['root'])/'trajectory_labels.npz')['states'][0];baseline=None
    for c in ('clean','background'):
        env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/'libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl'),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,ignore_done=False)
        try:
            env.seed(0);env.reset();env.reset_from_xml_string((Path(t['xml_directory'])/(c+'.xml')).read_text());env.set_init_state(saved);env.sim.forward();np.testing.assert_allclose(env.get_sim_state(),saved,atol=1e-8,rtol=0)
            d=env.sim.data._data;geometry={k:np.array(getattr(d,k)).copy() for k in ('qpos','qvel','xpos','xquat','cam_xpos','cam_xmat')}
            if baseline is None:baseline=geometry
            else:
                for k in baseline:np.testing.assert_allclose(baseline[k],geometry[k],rtol=0,atol=1e-8)
            assert not env.check_success();rows.append(dict(episode=t['episode'],appearance=c,restored=True,geometry_paired=True,initial_success=False,dynamics_steps=0))
        finally:env.close()
with (a.run/'evaluation_state_preflight.json').open('x') as f:json.dump(dict(status='PASS',states=5,conditions=10,checks=rows,policy_rollouts=0,dynamics_steps=0),f,indent=2)
