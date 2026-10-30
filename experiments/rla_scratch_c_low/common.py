"""Append-only C-low completion of the frozen scratch-AE policy comparison."""
from pathlib import Path
from rla_scratch_policy_aux.common import read,save,sha,table,P,GPU,ORDERS,LAMBDA
BASE=P/'experiments/rla_scratch_policy_aux/run_20261029_01'
ROOT=P/'experiments/rla_scratch_policy_aux/run_20261030_01_c_low'
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
def weight(arm,k):
 assert arm=='C-low' and 0<=k<600
 return LAMBDA/2
