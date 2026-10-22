from pathlib import Path
p=Path(__file__).parent/'audit.py'
s=p.read_text()
prefix=s[:s.index('start=time.time();')]
remaining=s[s.index('# Compare already saved parameters'):]
setup="start=time.time();trans=read(F/'selection_manifest.json')['transitions']\n"
exec(compile(prefix+setup+remaining,str(p),'exec'))
