from pathlib import Path
from rla_scratch_policy_aux.common import read,save,sha,table,P,GPU
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
ROOT=CODE.parent
BASE=P/'experiments/rla_scratch_policy_aux/run_20261029_01'
PRIOR=P/'experiments/rla_scratch_policy_aux/run_20261030_01_c_low'
BRANCH='experiments/rla-scratch-low-policy-seeds'
LAMBDA=0.11204268292682927
ARMS=['C-low','D-low']
def weight(arm,k):
 assert arm in ARMS+['preflight'] and 0<=k<600
 return LAMBDA
