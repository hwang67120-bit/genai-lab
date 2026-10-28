"""Process-lifetime read-only profiling; no production edits or tensor copies."""
import os,sys,time,json,gc,weakref,threading,atexit,traceback
from pathlib import Path
import psutil
O=Path(__file__).resolve().parent;ROOTPID=int(os.environ.get('RAM_ROOT_PID',os.getpid()));PID=os.getpid()
EVENTS=O/'events';EVENTS.mkdir(exist_ok=True)
REG={};SEEN=set();LOADED={};ONNX={};BEFORE={};stage='startup';active=False;previous=None;sampler=None;stop=threading.Event();samples=[]
def emit(x):
 x.update(pid=PID,root_pid=ROOTPID,time=time.time(),stage=stage)
 with (EVENTS/f'{PID}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+'\n')
def rss():
 rows=[];errors=[]
 try:p=psutil.Process(ROOTPID);procs=[p]+p.children(recursive=True)
 except psutil.Error as e:return {'rss_bytes':None,'processes':[],'errors':[str(e)]}
 for p in {p.pid:p for p in procs}.values():
  try:rows.append({'pid':p.pid,'rss_bytes':p.memory_info().rss})
  except psutil.Error as e:errors.append({'pid':p.pid,'error':str(e)})
 return {'rss_bytes':sum(r['rss_bytes'] for r in rows),'processes':rows,'errors':errors}
def module_info(mod):
 params=list(mod.parameters());bufs=list(mod.buffers());hist={};total=0
 for t in params+bufs:
  size=t.numel()*t.element_size();total+=size;k=str(t.dtype)+'@'+str(t.device);hist[k]=hist.get(k,0)+size
 p=params[0] if params else bufs[0] if bufs else None
 return {'bytes':total,'GB':total/1024**3,'dtype':str(p.dtype) if p is not None else None,'device':str(p.device) if p is not None else None,'dtype_device_bytes':hist,'parameter_count':sum(p.numel() for p in params),'buffer_count':sum(b.numel() for b in bufs),'class':type(mod).__name__}
def register(mod,name,group):
 t=sys.modules.get('torch');nn=getattr(t,'nn',None)
 if nn is None or not isinstance(mod,nn.Module):return
 old=REG.get(id(mod))
 if old and old['ref']() is mod:
  if group in ['sdxl','ip_adapter','automasker','similarity_encoder']:old.update(name=name,group=group)
  return
 REG[id(mod)]={'ref':weakref.ref(mod),'name':name,'group':group,'object_id':id(mod)}
def live_modules():
 rows=[]
 for entry in list(REG.values()):
  obj=entry['ref']()
  if obj is not None:
   try:rows.append(dict(name=entry['name'],group=entry['group'],object_id=entry['object_id'],**module_info(obj)))
   except Exception as e:rows.append({'name':entry['name'],'error':str(e)})
 return rows
def mark(label,code=None,extra=None):
 gc.collect();r=rss();emit({'event':'checkpoint','label':label,'code':code,'gc_pid':PID,**r,'local_modules':live_modules(),'live_onnx':[dict(object_id=k,**v['data']) for k,v in list(ONNX.items()) if v['ref']() is not None],'extra':extra or {}})
 if PID==ROOTPID:print('RAM_MARK',code,label,round((r['rss_bytes'] or 0)/1024**3,3),flush=True)
def capture_pipe(pipe):
 for name in ['unet','vae','text_encoder','text_encoder_2','image_encoder']:
  obj=getattr(pipe,name,None)
  if obj is not None:register(obj,name,'ip_adapter' if name=='image_encoder' else 'sdxl')
def walk_models(obj,prefix,depth=0,seen=None):
 if seen is None:seen=set()
 if obj is None or id(obj) in seen or depth>4:return
 seen.add(id(obj));t=sys.modules.get('torch');nn=getattr(t,'nn',None)
 if nn is not None and isinstance(obj,nn.Module):register(obj,prefix,'automasker');return
 if hasattr(obj,'__dict__'):
  for n,v in vars(obj).items():
   if n in ['model','predictor','densepose_processor','schp_processor_atr','schp_processor_lip']:walk_models(v,prefix+'.'+n,depth+1,seen)
def profiler(frame,event,arg):
 global stage
 if event not in ['call','return']:return
 name=frame.f_code.co_name
 if name not in ['<module>','from_pretrained','enable_model_cpu_offload','load_ip_adapter','__init__']:return
 file=frame.f_code.co_filename.replace('\\','/').lower()
 try:
  if name=='<module>' and event=='return':
   n=frame.f_globals.get('__name__')
   if n in ['torch','diffusers'] and n not in SEEN:SEEN.add(n);mark(n+' import complete','01' if n=='torch' else '02',{'file':file})
  elif name=='from_pretrained' and event=='return' and arg is not None:
   cls=type(arg).__name__;key=('pretrained',id(arg))
   if key in LOADED and LOADED[key]() is arg:return
   if cls.startswith('StableDiffusionXL'):
    LOADED[key]=weakref.ref(arg);capture_pipe(arg);mark('SDXL from_pretrained complete','03',{'class':cls,'file':file})
   elif cls.startswith(('GroundingDino','Sam2','CLIPVision','CLIPText','UNet2DCondition','AutoencoderKL')):
    LOADED[key]=weakref.ref(arg);kind='grounding_dino' if cls.startswith('Grounding') else 'sam2' if cls.startswith('Sam2') else 'clip_vision' if cls.startswith('CLIPVision') else 'sdxl_component'
    register(arg,cls,kind)
    if kind in ['grounding_dino','sam2']:mark(kind+' from_pretrained complete','06' if kind=='grounding_dino' else '07',{'file':file})
  elif name in ['enable_model_cpu_offload','load_ip_adapter'] and event=='return' and 'diffusers' in file:
   obj=frame.f_locals.get('self')
   if obj is not None and type(obj).__name__.startswith('StableDiffusionXL'):capture_pipe(obj);mark(name+' complete','04' if name=='enable_model_cpu_offload' else '05',{'file':file})
  elif name=='__init__':
   obj=frame.f_locals.get('self');cls=type(obj).__name__
   if cls=='InferenceSession' and 'onnxruntime' in file:
    if event=='call':
     model=str(frame.f_locals.get('path_or_bytes',''));kind='isnet' if 'isnet' in model.lower() else 'wd14' if 'wd-vit' in model.lower() or 'smilingwolf' in model.lower() else 'onnx_other'
     gc.collect();before=rss();BEFORE[id(frame)]=(time.time(),model,kind,before);mark(kind+' ONNX before load',None,{'model_path':model,'kind':kind})
    elif event=='return':
     st=BEFORE.pop(id(frame),None)
     if st:
      gc.collect();after=rss();delta=after['rss_bytes']-st[3]['rss_bytes'] if after['rss_bytes'] is not None and st[3]['rss_bytes'] is not None else None
      extra={'model_path':st[1],'kind':st[2],'load_rss_delta_bytes':delta,'before_rss':st[3],'providers':obj.get_providers(),'parameter_access':'unavailable; RSS load delta substitute'}
      ONNX[id(obj)]={'ref':weakref.ref(obj),'data':extra};emit({'event':'onnx_load','object_id':id(obj),**extra});mark(st[2]+' ONNX session created','09' if st[2]=='isnet' else '08' if st[2]=='wd14' else None,extra)
   elif event=='return' and cls=='AutoMasker':walk_models(obj,'AutoMasker');mark('CatVTON AutoMasker initialized','10',{'file':file})
   elif event=='return' and cls=='VisionEncoder' and file.endswith('run_native_refinement.py'):register(obj.model,'VisionEncoder.model','similarity_encoder');mark('similarity VisionEncoder initialized',None,{'file':file})
 except Exception as e:emit({'event':'instrumentation_error','error':str(e),'traceback':traceback.format_exc(),'file':file,'function':name})
def loop():
 tick=time.perf_counter()
 with (O/'samples.jsonl').open('a',encoding='utf8') as f:
  while not stop.is_set():
   row={'time':time.time(),'monotonic':time.perf_counter(),'stage':stage,**rss()};samples.append(row);f.write(json.dumps(row)+'\n');f.flush();tick+=1;stop.wait(max(0,tick-time.perf_counter()))
def install():
 global active,previous,sampler
 if active:return
 active=True;previous=sys.getprofile();mark('process start (instrumentation initialized)','00',{'argv':sys.argv,'executable':sys.executable});sys.setprofile(profiler)
 if PID==ROOTPID:sampler=threading.Thread(target=loop,daemon=True);sampler.start()
 atexit.register(close)
def close():
 global active
 if not active:return
 sys.setprofile(previous);mark('process exit / instrumentation removed','16');active=False
 if PID==ROOTPID:
  stop.set()
  if sampler:sampler.join(timeout=10)
 emit({'event':'instrumentation_removed','profile_restored':sys.getprofile() is previous})
