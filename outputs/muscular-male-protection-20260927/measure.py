from pathlib import Path
import json,hashlib,sys,itertools,datetime
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
R=Path(__file__).resolve().parents[2]; O=Path(__file__).parent
sys.path.insert(0,str(R))
from genai_lab.garment_edit_plan import project_target_garment_coverage,build_garment_edit_plan
G=R/'outputs/generalization-text-20260926'; P=json.loads((G/'preflight.json').read_text(encoding='utf-8'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def mask(p):return np.asarray(Image.open(p).convert('L'))>=128
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pix(a):return hashlib.sha256(np.asarray(a,dtype=np.uint8).tobytes()).hexdigest()
def bbox(a):
 y,x=np.where(a)
 return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def stats(a,fg):return {'pixels':int(a.sum()),'bbox_xyxy_exclusive':bbox(a),'mask_pixels_div_foreground_pixels':float(a.sum()/fg.sum()),'foreground_intersection_pixels':int((a&fg).sum()),'pixel_sha256':pix(a)}
def mapped(s):
 s=str(s).replace('\\','/')
 return R/s.split('/genai-lab/',1)[1] if '/genai-lab/' in s else Path(s)
old=R/'outputs/sdxl-local-garment-strength-v2-20260923/cases/muscular-male'
oldrun=read(old/'run.json');oldm=mapped(oldrun['local_refinement']['mask_diagnostic_directory'])
specs=[('male-ferrari','muscular-male','ferrari','nz9mdn_1'),('male-partial','muscular-male','partial','oeaf38qk'),('male-minimal','muscular-male','minimal','ohzm1j_5'),('male-v2','muscular-male','v2','3fw0ur67'),('ordinary-ferrari','ordinary-female','ferrari','724sww9u')]
result={'generation_calls':0,'model_loads':0,'detection_rerun':False,'bbox_convention':'xyxy right/bottom exclusive for masks; detector boxes floating xyxy','cases':[]}
for ident,char,gar,suffix in specs:
 case=G/f'cases/{char}/{gar}/off' if gar!='v2' else old
 base=case/'Base.png' if gar!='v2' else old/'base_candidates/candidate_1.png'
 m=case/'masks' if gar!='v2' else oldm
 reference_m=G/'cases/muscular-male/ferrari/off/masks' if gar in ('partial','minimal') else m
 fgpath=(case/'execution/cases'/char/'base_candidates/output_regions/candidate_1_base/foreground.png') if gar!='v2' else old/'base_candidates/output_regions/candidate_1_base/foreground.png'
 fg=mask(fgpath)
 dp=Path('C:/Users/user/AppData/Local/Temp')/('genai-parts-'+suffix)/'parts.json';d=read(dp)
 dest=O/'cases'/ident;dest.mkdir(parents=True,exist_ok=True)
 raw={};expanded={};components={};queries={};sources={str(base):sha(base),str(fgpath):sha(fgpath),str(dp):sha(dp)}
 hardnames={'output_face','output_hair','output_animal_ears','output_ears','output_hair_accessory'}
 for name,entry in d['parts'].items():
  item=dict(entry);item['candidates']=[]
  for i,path in enumerate(entry.get('candidate_mask_paths',[])):
   a=mask(path);sources[path]=sha(path)
   item['candidates'].append({'index':i,'path':path,'sha256':sources[path],**stats(a,fg)})
  queries[name]=item
  path=entry.get('mask_path')
  a=mask(path) if path and entry.get('status')=='detected' else np.zeros_like(fg)
  if path:sources[path]=sha(path)
  raw[name]=a
  ex=np.asarray(Image.fromarray(a.astype('uint8')*255).filter(ImageFilter.MaxFilter(5)))>=128
  expanded[name]=ex
  components[name]={'accepted':bool(path and entry.get('status')=='detected'),'hard_protection_member':name in hardnames,'conditional_member':name=='output_tail','raw':stats(a,fg),'maxfilter5':stats(ex,fg)}
 for name in ['output_ears','output_tail']:
  if name not in raw:
   raw[name]=np.zeros_like(fg);expanded[name]=np.zeros_like(fg);components[name]={'accepted':False,'query_run':False,'hard_protection_member':name in hardnames,'conditional_member':name=='output_tail','raw':stats(raw[name],fg),'maxfilter5':stats(expanded[name],fg)}
 hard=np.logical_or.reduce([a for n,a in expanded.items() if n in hardnames]);conditional=expanded['output_tail'];effective=hard|conditional
 stored=mask(reference_m/'effective_protection.png');sources[str(reference_m/'effective_protection.png')]=sha(reference_m/'effective_protection.png')
 check={'comparison_path':str(reference_m/'effective_protection.png'),'same_run_saved':gar not in ('partial','minimal'),'pixel_equal':bool(np.array_equal(effective,stored)),'different_pixels':int((effective!=stored).sum())}
 component_checks={n:bool(np.array_equal(raw[n],mask(reference_m/f'{n}.png'))) for n in raw if (reference_m/f'{n}.png').exists()}
 if not check['pixel_equal']:raise RuntimeError('saved effective mismatch '+ident)
 request_saved=gar not in ('partial','minimal')
 source=Image.open(reference_m/'source_garment_removal.png').convert('L');baseim=Image.open(base).convert('RGB')
 if request_saved:
  req=mask(m/'requested_edit.png');he=mask(m/'hard_edit_domain.png');con=mask(m/'mask_conflict.png');soft=np.asarray(Image.open(m/'soft_guidance.png').convert('L'));metrics=read(m/'garment_edit_plan.json')['metrics'];req_origin='saved'
 else:
  cov=project_target_garment_coverage(source,Image.open(fgpath),tuple(P['garments'][gar]['approved_tags']),growth_pixels=12)
  # Same plan computation only; no model and no pipeline execution.
  import cv2
  exp=cv2.dilate(hard.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
  plan=build_garment_edit_plan(baseim,source,cov.mask,Image.fromarray(hard.astype('uint8')*255),Image.open(fgpath),garment_growth_envelope=cov.growth_envelope,soft_boundary_protection=Image.fromarray((exp&~hard).astype('uint8')*255),conditional_protection=Image.fromarray(conditional.astype('uint8')*255),feather_radius=10,soft_boundary_strength=.5)
  metrics=plan.record['metrics'];req=np.asarray(plan.images['requested_edit'])>=128;he=np.asarray(plan.hard_edit_domain)>=128;con=np.asarray(plan.images['mask_conflict'])>=128;soft=np.asarray(plan.soft_guidance);req_origin='offline reconstruction from same Base/source, per-garment saved approved tags; compared with exception request count'
  expected={'partial':511625,'minimal':489874}[gar]
  if metrics['requested_pixels']!=expected:raise RuntimeError('request reconstruction mismatch '+ident+' '+str(metrics))
 overlaps={f'{a}:{b}':{'raw_pixels':int((raw[a]&raw[b]).sum()),'expanded_pixels':int((expanded[a]&expanded[b]).sum())} for a,b in itertools.combinations(raw,2)}
 request={'origin':req_origin,'metrics':metrics,'all_requested_protected_pixels':int((req&effective).sum()),'in_domain_conflict_pixels':int(con.sum()),'requested_outside_edit_domain_pixels':int(req.sum()-con.sum()-he.sum()),'effective_protection_pixels':int(effective.sum()),'hair_unique_protection_pixels':int((expanded['output_hair']&~np.logical_or.reduce([a for n,a in expanded.items() if n in hardnames and n!='output_hair'])).sum())}
 colors={'output_hair':(230,65,55),'output_face':(45,125,255),'output_hair_accessory':(230,180,25),'output_animal_ears':(90,200,80),'output_tail':(160,80,210)}
 image=np.asarray(baseim).copy().astype(float)
 for n in colors:
  if n in expanded:
   a=expanded[n];image[a]=image[a]*.55+np.array(colors[n])*.45
 overlay=Image.fromarray(np.uint8(image));draw=ImageDraw.Draw(overlay);draw.rectangle((0,0,baseim.width,50),fill='white');draw.text((5,5),ident+' | red=hair blue=face (MaxFilter 5)',fill='black');draw.text((5,25),'saved parts; NOT generated output',fill='black');overlay.save(dest/'components-overlay.png')
 boxes=baseim.copy();draw=ImageDraw.Draw(boxes)
 for name,entry in queries.items():
  for i,box in enumerate(entry.get('boxes',[])):
   color=colors.get(name,(90,180,110));draw.rectangle(tuple(box),outline=color,width=3);draw.text((box[0]+2,box[1]+2),name.replace('output_','')+':'+str(i)+' '+str(round(entry['detection_scores'][i],3)),fill=color,stroke_width=1,stroke_fill='white')
 boxes.save(dest/'detector-boxes.png')
 for n in ('output_hair','output_face'):
  for candidate in queries[n]['candidates']:
   a=mask(candidate['path']);im=np.asarray(baseim).astype(float);im[a]=im[a]*.55+np.array(colors[n])*.45
   im=Image.fromarray(im.astype('uint8'));dr=ImageDraw.Draw(im);box=queries[n]['boxes'][candidate['index']];dr.rectangle(tuple(box),outline=colors[n],width=3);dr.text((10,10),n+' candidate '+str(candidate['index']),fill='black',stroke_width=2,stroke_fill='white');im.save(dest/f'{n}-candidate-{candidate["index"]}.png')
 row={'id':ident,'character':char,'garment':gar,'base':str(base),'base_sha256':sha(base),'foreground':str(fgpath),'foreground_pixels':int(fg.sum()),'foreground_note':'stored same-Base output-region foreground, not source-reference foreground; refinement fallback temporary foreground was not persisted','detector_record':str(dp),'detector_record_modified_local':datetime.datetime.fromtimestamp(dp.stat().st_mtime).isoformat(),'components':components,'queries':queries,'overlap':overlaps,'protection_equality':check,'raw_component_equality':component_checks,'request':request,'source_sha256':sources}
 result['cases'].append(row)
 print(ident,'fg',int(fg.sum()),'hair',components['output_hair'],'face',components['output_face'],'request',request,'equal',check)
(O/'components.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
