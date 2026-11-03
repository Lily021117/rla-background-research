"""Compile exact fixed environment models only. No action step/replay."""
from common import *
import sys,argparse
sys.path.insert(0,str(OLD/'libero_official'));os.environ['LIBERO_CONFIG_PATH']=str(OLD/'libero_config')
def main(suite):
 out=RUN/'model_snapshot'/suite;out.mkdir(parents=True,exist_ok=False)
 from libero.libero.envs import OffScreenRenderEnv
 from robosuite.utils import binding_utils as binding
 counts=dict(render_calls=0,physics_substeps=0)
 original_render=binding.MjSim.render;original_step=binding.MjSim.step
 def render(self,*a,**k):counts['render_calls']+=1;return original_render(self,*a,**k)
 def step(self,*a,**k):counts['physics_substeps']+=1;raise RuntimeError('No stepping allowed in model snapshot')
 binding.MjSim.render=render;binding.MjSim.step=step
 bddl=json.loads((OLD/'pilot_replay'/suite/'environment.json').read_text())['bddl']
 env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256)
 try:
  env.seed(0);env.reset();model=env.sim.model._model
  xml=env.sim.model.get_xml();(out/'runtime.xml').write_text(xml)
  assert sha(out/'runtime.xml')==sha(OLD/'pilot_replay'/suite/'runtime_model.xml'),'same model construction not confirmed'
  mujoco.mj_saveModel(model,str(out/'model.mjb'),None)
  dump(out/'result.json',dict(suite=suite,counts=counts,source_actions=0,wait_steps=0,xml_hash_identical_to_previous=True,
   model_sha256=sha(out/'model.mjb'),purpose='Recover exact compiled model, no trajectory recreation or alternate environment'))
 finally:env.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=SUITES);main(p.parse_args().suite)
