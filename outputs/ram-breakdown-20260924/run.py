import os,sys,json,time,hashlib,subprocess,importlib.util,traceback
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state():
 names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_hashes':{n:sha(ROOT/n) for n in names if n and (ROOT/n).is_file()},'status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)}
if '--check' in sys.argv:
 for n in ['run.py','ram_probe.py','sitecustomize.py']:compile((O/n).read_text(encoding='utf8'),str(O/n),'exec')
 import yaml
 cfg=yaml.safe_load((ROOT/'configs/animagine.yaml').read_text(encoding='utf8'))
 assert cfg['refinement_execution']['mode']=='sdxl_local'
 assert {k:cfg['staged_reference_generation']['garment'][k] for k in ['reference_scale','inpaint_strength','inference_steps']}=={'reference_scale':.8,'inpaint_strength':.9,'inference_steps':28}
 assert sha(Path(r'C:\Users\user\Downloads\참조 의상\096104f0613d833f0c975d118610910e.jpg'))=='6fdcdacee7f7e3e08cc8bc9dbb9921ff94a19fc6494b71462f54e5b929e4e8fa'
 write(O/'code-state.json',{'before':state(),'instrumentation_files':{n:sha(O/n) for n in ['run.py','ram_probe.py','sitecustomize.py']}})
 print('PRECHECK PASS: syntax, offline inputs, current mode/profile, code hashes; no model load/generation');sys.exit(0)
st=json.loads((O/'code-state.json').read_text(encoding='utf8'))
for n,h in st['before']['tracked_hashes'].items():assert sha(ROOT/n)==h,('source changed',n)
with (O/'attempt-started.json').open('x',encoding='utf8') as f:json.dump({'time':time.time(),'attempt':1,'no_retry':True},f)
keys=['RAM_BREAKDOWN_ACTIVE','RAM_ROOT_PID','PYTHONPATH'];oldenv={k:os.environ.get(k) for k in keys}
os.environ['RAM_BREAKDOWN_ACTIVE']='1';os.environ['RAM_ROOT_PID']=str(os.getpid());os.environ['PYTHONPATH']=str(O)+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path[:0]=[str(O),str(ROOT),str(ROOT/'scripts')]
import ram_probe as p
p.install();patches=[];counts={'base':0,'inpaint':0};result=None
class NoRetry(BaseException):pass
def patch(obj,name,func):patches.append((obj,name,getattr(obj,name)));setattr(obj,name,func)
try:
 import torch
 import diffusers
 p.stage='pipeline_setup'
 spec=importlib.util.spec_from_file_location('ram_existing_full_pipeline',ROOT/'outputs/strength-revert-20260924/stage2_run_090.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);s.OUT=O
 s.prior.configure_console_encoding();s.prior.configure_system_certificates()
 control=s.read(ROOT/'outputs/garment-strength-095-20260923/controls.json')['p1-ordinary-female']
 pre={'approved_tags':control['approved_tags'],'approved_detail_tags':control['approved_detail_tags'],'P0_f':{'boards':{'ordinary-female':{'sha256':control['hashes']['base_candidates/input_garment.png']}}}}
 C=s.prior.GenerationOrchestrator;ob=C.generate_base_candidates;of=C.finalize_selected_candidate;ov=C.prepare_visual_inputs
 def visual(self,*a,**k):
  p.stage='reference_analysis';p.mark('reference analysis before')
  try:return ov(self,*a,**k)
  finally:p.mark('reference analysis after');p.stage='tagging_and_approval'
 def base(self,*a,**k):
  p.stage='base';p.mark('Base stage before','11')
  try:return ob(self,*a,**k)
  finally:p.mark('Base stage after','12');p.stage='selection'
 def final(self,*a,**k):
  p.stage='refinement';p.mark('refinement stage before','13')
  try:return of(self,*a,**k)
  finally:
   p.mark('refinement and its internal gates after','14');p.stage='measurement_and_gates_complete';p.mark('measurement and gates complete (inside finalize boundary)','15');p.stage='cleanup'
 patch(C,'prepare_visual_inputs',visual);patch(C,'generate_base_candidates',base);patch(C,'finalize_selected_candidate',final)
 # No extra generation call is permitted, even if production would attempt a fallback.
 from diffusers import StableDiffusionXLImg2ImgPipeline,StableDiffusionXLInpaintPipeline
 for cls,kind in [(StableDiffusionXLImg2ImgPipeline,'base'),(StableDiffusionXLInpaintPipeline,'inpaint')]:
  original=cls.__call__
  def make_call(orig,kind):
   def call(self,*a,**k):
    counts[kind]+=1
    if counts[kind]>1:raise NoRetry('second generation call blocked: '+kind)
    p.mark(kind+' actual sampling before',extra={'call':counts[kind]})
    try:return orig(self,*a,**k)
    finally:p.mark(kind+' actual sampling after',extra={'call':counts[kind]})
   return call
  patch(cls,'__call__',make_call(original,kind))
 result=s.run_case('ordinary-female',pre)
 write(O/'run-result.json',{'execution':result,'generation_calls':counts})
except BaseException as e:
 write(O/'fatal.json',{'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'generation_calls':counts,'retry':False});print(traceback.format_exc(),flush=True)
finally:
 for obj,n,orig in reversed(patches):setattr(obj,n,orig)
 p.stage='exit';p.close()
 for k,v in oldenv.items():
  if v is None:os.environ.pop(k,None)
  else:os.environ[k]=v
 st['after']=state();st['tracked_changes']=[n for n,h in st['before']['tracked_hashes'].items() if st['after']['tracked_hashes'].get(n)!=h]
 st['process_hooks_removed']=True;st['environment_restored']=True;st['production_files_edited']=False;write(O/'code-state.json',st)
 print('MEASUREMENT_FINISHED',result['status'] if result else 'failed_no_retry',counts,'tracked_changes',st['tracked_changes'],flush=True)
