"""Configuration and inference source equivalence; no model forward."""
import ast,subprocess,importlib.metadata as metadata
from common import *
class Rename(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id=='inputs':n.id='inp'
  return n
def nested(path,name):
 tree=ast.parse(Path(path).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute');part=next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name==name);return ast.dump(Rename().visit(part),include_attributes=False)
def main():
 a=HERE.parent/'rla_scratch_policy_aux/evaluation.py';b=HERE.parent/'rla_new_task_baseline/evaluation.py'
 assert nested(a,'infer')==nested(b,'infer') and nested(a,'proprio')==nested(b,'proprio')
 rows=[]
 for e in read(ROOT/'checkpoint_inventory.json'):
  p=STOVE/f"seed_{e['policy_seed']}/C-low/run_plan.json" if e['task']=='stove' else DRAWER/f"seed_{e['policy_seed']}/run_plan.json"
  cfg=read(p);rows.append(dict(model=e['model_id'],configuration=str(p),configuration_sha256=sha(p),statistics=e['statistics'],statistics_sha256=e['statistics_sha256'],checkpoint_sha256=e['sha256']))
 save(ROOT/'configuration_identity.json',dict(models=rows,infer_AST_equivalent=True,proprio_AST_equivalent=True,only_AST_normalization='rename local inputs variable to inp',sources=[dict(path=str(p),sha256=sha(p)) for p in [a,b,HERE.parent/'rla_lr_budget_sweep/bootstrap.py']],packages={n:metadata.version(n) for n in ['torch','mujoco','transformers','robosuite','numpy']},LIBERO_commit=subprocess.check_output(['git','-C',str(P.parent/'LIBERO-plus'),'rev-parse','HEAD'],text=True).strip()))
 print('Configuration identity PASS')
if __name__=='__main__':main()
