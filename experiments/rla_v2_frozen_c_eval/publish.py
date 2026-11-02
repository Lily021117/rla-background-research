"""Sanitize small protocol/results only. No source states/XML/videos/weights."""
import sys
from common import *
def scrub(s):return s.replace(str(ROOT),'${RUN_ROOT}').replace(str(P),'${PROJECT_ROOT}').replace(str(P.parent/'LIBERO-plus'),'${LIBERO_ROOT}')
def main():
 stage=sys.argv[1];out=HERE/('frozen_protocol' if stage=='protocol' else 'delivery');out.mkdir(exist_ok=False)
 if stage=='protocol':files=[ROOT/n for n in ['evaluation_protocol.json','checkpoint_inventory.json','state_manifest.json','integration_static_checks.json','clean_reuse_audit.json','configuration_identity.json']]
 else:files=list((ROOT/'delivery').glob('*'))+[ROOT/t/n for t in ['stove','drawer'] for n in ['runtime_integration.json','inference_identity.json','completion.json']]
 for p in files:
  name=p.name if p.parent.name in ['delivery',ROOT.name] else p.parent.name+'_'+p.name
  (out/name).write_text(scrub(p.read_text()))
 save(out/'evidence_hashes.json',{scrub(str(p)):sha(p) for p in files})
 print(stage,len(files))
if __name__=='__main__':main()
