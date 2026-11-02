"""Small sanitized artifacts only; no weights/raw states/trajectory/video."""
import sys
from common import *
def scrub(s):return s.replace(str(BASE),'${RUN_ROOT}').replace(str(P),'${PROJECT_ROOT}').replace(str(P.parent/'LIBERO-plus'),'${LIBERO_ROOT}')
def main():
 stage=sys.argv[1];out=HERE/('frozen_protocol' if stage=='protocol' else 'delivery');out.mkdir(exist_ok=False)
 if stage=='protocol':files=[BASE/n for n in ['calibration_protocol.json','checkpoint_inventory.json','state_manifest.json']]+list((BASE/'image_acceptance').glob('*.json'))
 else:files=list((BASE/'delivery').glob('*'))+[BASE/n for n in ['single_factor_selection.json','combination_selection.json','selected_candidates.json','completion.json','resource_release.json'] if (BASE/n).exists()]+list(BASE.glob('*/stove/clean_identity.json'))+list(BASE.glob('*/drawer/clean_identity.json'))
 manifest={}
 for p in files:
  name=p.name if p.parent in [BASE,BASE/'delivery'] else '_'.join(p.relative_to(BASE).parts)
  if p.suffix=='.png':(out/name).write_bytes(p.read_bytes())
  else:(out/name).write_text(scrub(p.read_text()))
  manifest[scrub(str(p))]=sha(p)
 save(out/'evidence_hashes.json',manifest)
if __name__=='__main__':main()
