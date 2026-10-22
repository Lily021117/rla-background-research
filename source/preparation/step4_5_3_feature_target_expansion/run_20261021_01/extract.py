from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent
src=Path('${RLA_PROJECT_ROOT}/experiments/rla_background_invariance/step4_5_feature_target_pilot/run_20261021_02/extract.py');s=src.read_text()
changes={
"assert len(rows)==32 and len(trans)==8":"assert len(rows)==876 and len(trans)==220",
"assert len(native)==64":"assert len(native)==1752",
"free>=10*1024**3":"free>=13*1024**3",
"'latest user explicitly authorizes GPU0 measured risk preflight below former26GiB gate; not a permanent global threshold change'":"'user-authorized GPU0; current minimum 13GiB'",
"episode=r['episode'],state_id=":"episode=r['episode'],split=r['split'],state_id=",
"episode=t['episode'],camera=":"episode=t['episode'],split=t['split'],camera=",
"'/64'":"'/1752'","'/8'":"'/220'",
"features=64,physical_transitions=8":"features=1752,physical_transitions=220",
"proprio_alignment=False":"proprio_alignment=True",
"tab('proprio_alignment_audit.csv',props)":"assert len(props)==220 # existing exact-state audit preserved",
}
for a,b in changes.items():assert a in s,a;s=s.replace(a,b)
# Strict CPU loading of matched decoders, without decoder inference or checkpoint writes.
needle="targetrows=[];pairs=[];models=[]"
replacement=needle+"\n  decoder_checks=[]\n  for dg in ['B','C']:\n   dp=X/f'step2_training/{dg}/ckpts/decoder_step0000200.pt';dm=SimpleTokenTransformer(**cfg['models']['decoder']['args']);dm.load_state_dict(torch.load(dp,map_location='cpu',weights_only=True),strict=True);decoder_checks.append(dict(group=dg,path=str(dp),SHA256=sha(dp),strict=True));del dm\n  js('decoder_compatibility.json',decoder_checks)"
assert needle in s;s=s.replace(needle,replacement)
(O/'extraction_reuse.json').write_text(json.dumps({'source':str(src),'SHA256':hashlib.sha256(src.read_bytes()).hexdigest(),'replacements':changes,'decoder_check_added':True},indent=2))
ns={'__name__':'expanded','__file__':str(O/'runtime_extract.py')};exec(compile(s,str(src),'exec'),ns);ns['P']=O;ns['PRE']=O/'preflight';ns['main']()
