from pathlib import Path
from rla_scratch_policy_aux.common import read,save,sha,table,P,GPU
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
ROOT=CODE.parent
SOURCE=P/'experiments/rla_scratch_low_policy_seeds/run_20261030_01'
BRANCH='experiments/rla-low-lr-continuation'
SEEDS=[31001,31002,31003]
ARMS=['C-low','D-low']
LAMBDA=0.11204268292682927
def weight(arm,k):
 assert arm in ARMS and 0<=k<1800
 return LAMBDA
