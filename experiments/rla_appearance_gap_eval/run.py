"""Only the frozen200 evaluations, analysis and sanitized delivery."""
import subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_appearance_gap_eval.common import HERE,REPO
from rla_appearance_gap_eval.analyze import analyze,publish
out=Path(sys.argv[1])
subprocess.run([sys.executable,str(HERE/'launch.py'),'--out',str(out)],cwd=REPO,check=True)
analyze(out);publish(out)
