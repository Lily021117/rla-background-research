"""Original shared policy step0; no teacher, no auxiliary reinitialization."""
from pathlib import Path
from control_preserving_rla.common import read
from stove_centering.training import initialize as original_initialize

def initialize(g):
 plan,prev,ck=original_initialize(g)
 assert ck['global_step']==0
 return plan,prev,ck
