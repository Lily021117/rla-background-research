"""Recheck frozen full Prismatic source and token/processor configuration."""
import argparse,subprocess
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);project=root.parents[3];base=project/'experiments/official_libero_rla_consistency';ref=base/'run_20261023_02';evidence=read(base/'run_20261023_03/initialization_provenance.json');path=ref/'resolved_prismatic_initialization.pt';actual=sha(path);assert actual==evidence['resolved_state_sha256']
official=project/'worktrees/rla-official-0fc6e97';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip();assert commit=='0fc6e97212442ebccffa06f1d42f5481e5f22950'
files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((ref/'vla-official/pretrained_models/configs').iterdir()) if p.is_file()]
save(a.out/'initialization_source_verification.json',dict(resolved_prismatic_path=str(path),resolved_sha256=actual,matched_prior_run03=True,source_repo=evidence['repo'],source_revision=evidence['revision'],source_checkpoint_sha256=evidence['checkpoint_SHA256'],official_RLA_commit=commit,tokenizer_processor_config_files=files,full_prismatic_not_spatial_pro=True))
print('FROZEN FULL PRISMATIC VERIFIED',actual,flush=True)
