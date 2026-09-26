from pathlib import Path
import json,sys,datetime,hashlib
import numpy as np,cv2
from PIL import Image
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R))
from genai_lab.garment_edit_plan import build_garment_edit_plan
D=json.loads((O/'components.json').read_text(encoding='utf-8'))
def a(p):return np.asarray(Image.open(p).convert('L'))>=128
for c in D['cases']:
 if c['protection_equality']['same_run_saved']:
  m=Path(c['protection_equality']['comparison_path']).parent;h=a(m/'hard_protection.png');ex=cv2.dilate(h.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
  plan=build_garment_edit_plan(Image.open(c['base']),Image.open(m/'source_garment_removal.png'),Image.open(m/'target_garment_coverage.png'),Image.open(m/'hard_protection.png'),Image.open(c['foreground']),garment_growth_envelope=Image.open(m/'garment_growth_envelope.png'),conditional_protection=Image.open(m/'conditional_protection.png'),soft_boundary_protection=Image.fromarray((ex&~h).astype('uint8')*255),feather_radius=10,soft_boundary_strength=.5)
  c['saved_plan_reconstruction']={n:int(np.count_nonzero(np.asarray(im)!=np.asarray(Image.open(m/f'{n}.png')))) for n,im in plan.images.items() if n!='overlay' and (m/f'{n}.png').exists()}
  print(c['id'],c['saved_plan_reconstruction']);assert all(x==0 for x in c['saved_plan_reconstruction'].values())
 if c['garment']!='v2':
  case=R/f"outputs/generalization-text-20260926/cases/{c['character']}/{c['garment']}/off"
  start=json.loads((case/'attempt-started.json').read_text(encoding='utf-8'))['time'];dur=json.loads((case/'arm-summary.json').read_text(encoding='utf-8'))['seconds'];t=Path(c['detector_record']).stat().st_mtime
  c['temp_record_attribution']={'basis':'parts.json mtime inside attempt window + Base dimensions + component pixels and hashes; no run_id in parts.json','start_local':datetime.datetime.fromtimestamp(start).isoformat(),'end_local':datetime.datetime.fromtimestamp(start+dur).isoformat(),'parts_time_local':datetime.datetime.fromtimestamp(t).isoformat(),'inside_window':start<=t<=start+dur};print(c['id'],c['temp_record_attribution']);assert c['temp_record_attribution']['inside_window']
D['measurements_note']='No detection rerun: original temporary query records and masks survived. Failed runs did not save final effective masks. Their original query masks compose exactly the completed Ferrari saved protection; own-run pixel comparison is unavailable.'
(O/'components.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
