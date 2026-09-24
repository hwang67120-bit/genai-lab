"""Aggregate saved observations only. No model imports, encoding or generation."""
import json,datetime,collections
from pathlib import Path
O=Path(__file__).resolve().parent;G=1024**3

def read(p):return json.loads(p.read_text(encoding='utf8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def write(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8')
def stamp(t):return datetime.datetime.fromtimestamp(t,datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
events=sorted([x for f in (O/'events').glob('*.jsonl') for x in lines(f)],key=lambda x:x['time'])
cps=[x for x in events if x['event']=='checkpoint'];samples=lines(O/'samples.jsonl');root=cps[0]['root_pid']
prev=None
for x in cps:
 if x.get('code') in ['06','07'] and not x.get('extra',{}).get('file','').endswith('/modeling_utils.py'):
  x['raw_checkpoint_code']=x['code'];x['code']=None;x['label']='processor/config preparation (not model weights): '+x['label']
 x['rss_GB']=x['rss_bytes']/G if x['rss_bytes'] is not None else None;x['delta_GB']=(x['rss_bytes']-prev)/G if prev is not None and x['rss_bytes'] is not None else None
 if x['rss_bytes'] is not None:prev=x['rss_bytes']
 x['time_kst']=stamp(x['time'])
peak=max(samples,key=lambda x:x['rss_bytes'] or 0);cppeak=max(cps,key=lambda x:x['rss_bytes'] or 0)
# Refine the broad orchestrator stage using observed actual sampling boundaries.
rootmarks=[x for x in cps if x['pid']==root]
for sample in samples:
 label=sample['stage']
 for kind,outer in [('base','base'),('inpaint','refinement')]:
  if label!=outer:continue
  prior=[x for x in rootmarks if x['time']<=sample['time'] and x['label'] in [kind+' actual sampling before',kind+' actual sampling after']]
  if prior:label=outer+(' / sampling' if prior[-1]['label'].endswith('before') else ' / post-sampling analysis and gates')
 sample['derived_stage']=label
code_status={f'{i:02d}':{'observed':any(x['code']==f'{i:02d}' for x in cps),'occurrences':sum(x['code']==f'{i:02d}' for x in cps)} for i in range(17)}
errs=[x for x in events if x['event']=='instrumentation_error']
write('checkpoints.json',{'unit':'GiB (1024**3 bytes), requested GB field spelling retained','root_pid':root,'events':cps,'requested_codes':code_status,'instrumentation_errors':errs,'all_events':events})
gaps=[samples[i]['monotonic']-samples[i-1]['monotonic'] for i in range(1,len(samples))]
write('samples.json',{'interval_target_seconds':1,'sample_count':len(samples),'interval_start_kst':stamp(samples[0]['time']),'interval_end_kst':stamp(samples[-1]['time']),'maximum_gap_seconds':max(gaps,default=0),'peak_rss_GB':peak['rss_bytes']/G,'peak':peak,'checkpoint_peak_rss_GB':cppeak['rss_bytes']/G,'difference_GB':(peak['rss_bytes']-cppeak['rss_bytes'])/G,'samples':samples})
# Representative sizes by role; all object snapshots preserved separately.
roles={};snapshots=[]
for cp in cps:
 for m in cp['local_modules']:
  row=dict(m,pid=cp['pid'],checkpoint_label=cp['label'],checkpoint_code=cp['code'],time=cp['time'],stage=cp['stage']);snapshots.append(row)
  if 'bytes' not in m:continue
  key=m['group']+'/'+m['name']
  if key not in roles or m['bytes']>roles[key]['bytes']:roles[key]=row
# Entries labelled generic pretrained components refer to the later named SDXL/image encoder objects.
known={(x['pid'],x['object_id']) for x in snapshots if x.get('group') in ['sdxl','ip_adapter','similarity_encoder']}
roles={k:v for k,v in roles.items() if not(v['group'] in ['sdxl_component','clip_vision'] and (v['pid'],v['object_id']) in known)}
onnx=[x for x in events if x['event']=='onnx_load'];onnxroles={}
for x in onnx:
 k=x['kind']
 if k not in onnxroles or (x['load_rss_delta_bytes'] or 0)>(onnxroles[k]['load_rss_delta_bytes'] or 0):onnxroles[k]=x
runtime=next((x['rss_bytes']/G for x in cps if x['pid']==root and x['code']=='02'),None)
sdxl=sum(x['bytes']/G for x in roles.values() if x['group']=='sdxl');ip=sum(x['bytes']/G for x in roles.values() if x['group']=='ip_adapter');auxweights=sum(x['bytes']/G for x in roles.values() if x['group'] not in ['sdxl','ip_adapter'])
onnxsum=sum((x['load_rss_delta_bytes'] or 0)/G for x in onnxroles.values());aux=auxweights+onnxsum
residual=peak['rss_bytes']/G-(runtime or 0)-sdxl-ip-aux
account={'runtime_checkpoint_02_GB':runtime,'sdxl_logical_weights_GB':sdxl,'ip_image_encoder_logical_weights_GB':ip,'auxiliary_logical_weights_GB':auxweights,'onnx_sum_representative_load_RSS_deltas_GB':onnxsum,'auxiliary_combined_GB':aux,'unattributed_arithmetic_residual_GB':residual,'warning':'Logical module bytes (including GPU tensors) and RSS deltas are different quantities. Roles may not coexist, load deltas contain runtime buffers; this requested subtraction is NOT an additive physical system-RAM allocation.'}
write('modules.json',{'representative_roles':roles,'all_snapshots':snapshots,'onnx_loads':onnx,'representative_onnx_deltas':onnxroles,'summary_arithmetic':account})
state=read(O/'code-state.json');result=read(O/'run-result.json') if (O/'run-result.json').exists() else {};run=result.get('execution',{})
cpu32=[(k,x) for k,x in roles.items() if any(d=='torch.float32@cpu' and b>0 for d,b in x.get('dtype_device_bytes',{}).items())]
rootends=[x for x in cps if x['pid']==root and (x['code'] in ['11','12','13','14','15'] or 'actual sampling' in x['label'])]
Q2=[]
for x in rootends:
 auxlive=[m['name'] for m in x['local_modules'] if m.get('group') not in ['sdxl','ip_adapter','sdxl_component','clip_vision']];on=[n['kind'] for n in x.get('live_onnx',[])];Q2.append({'checkpoint':x['code'] or x['label'],'rss_GB':x['rss_GB'],'live_auxiliary_modules':auxlive,'live_onnx':on,'pids':[p['pid'] for p in x['processes']]})
summary={'status':run.get('status'),'peak_GiB':peak['rss_bytes']/G,'peak_decimal_MB':peak['rss_bytes']/1e6,'peak_stage':peak['derived_stage'],'peak_time_kst':stamp(peak['time']),'checkpoint_max_GiB':cppeak['rss_bytes']/G,'peak_gap_GiB':(peak['rss_bytes']-cppeak['rss_bytes'])/G,'Q1_cpu_float32_roles':[k for k,v in cpu32],'Q2_checkpoints':Q2,'Q3_above_1_GB':(peak['rss_bytes']-cppeak['rss_bytes'])>G,'instrumentation_error_count':len(errs),'code_changed':state.get('tracked_changes'),'accounting':account}
write('summary.json',summary)
md=['# 실제 파이프라인 시스템 RAM 사용 내역','',f"ordinary-female × 페라리, 실제 GenerationOrchestrator 경로를 1회 실행했습니다. 상태: **{run.get('status','미완료')}**. 연속 RSS 피크 **{peak['rss_bytes']/G:.3f} GiB** ({peak['rss_bytes']/1e6:.2f} MB), {stamp(peak['time'])}, 단계 `{peak['derived_stage']}`.",'',
'RSS는 부모와 모든 재귀 하위 프로세스의 합입니다. OS 전체 RAM 사용량이나 고유 물리 페이지 합은 아닙니다. 표의 GB는 요청 코드와 같은 1024³ 바이트(GiB)입니다. 이전 13,481.98 MB는 십진 MB이므로 단위를 구분합니다.',
'',f"코드 `{state['before']['commit']}`. 운영 소스·설정·계약 문서를 수정하지 않았습니다. 추적 파일 변경: `{state.get('tracked_changes')}`. 계측은 outputs 내 Python 프로파일 훅·프로세스 한정 sitecustomize로 삽입했고, 종료 시 훅과 환경을 원복했습니다. dtype·오프로드·모델 로드 방식은 변경하지 않았습니다. 생성 호출: `{result.get('generation_calls')}`. 재시도하지 않았습니다.",'',
'## 표 1 — 체크포인트','',
'실제 관측 시각 순서입니다. 동일 코드가 반복되는 것은 별도 프로세스 또는 재로드입니다. 델타는 직전 행의 전체 RSS와의 차이이며, 동시 실행·해제·버퍼·단편화도 포함하므로 해당 모델 가중치만의 값으로 확정하지 않습니다. gc.collect()는 표의 해당 PID에서 실행했습니다. 다른 프로세스의 GC를 원격 강제하지 않았습니다.',
'', '| # / PID | 지점 | RSS GB | 델타 GB |','|---|---|---:|---:|']
for x in cps:md.append(f"| {x['code'] or '—'} / {x['pid']} | {x['label']} | {x['rss_GB']:.4f} | {x['delta_GB']:+.4f} |" if x['delta_GB'] is not None else f"| {x['code'] or '—'} / {x['pid']} | {x['label']} | {x['rss_GB']:.4f} | — |")
missing=[k for k,v in code_status.items() if not v['observed']]
md+=['',f'미관측 체크포인트: {missing or "없음"}. 미관측은 계측 누락 여부를 실행 경로와 함께 확인해야 하며 곧바로 미로드로 단정하지 않습니다.',
'14·15는 finalize_selected_candidate가 정밀화와 내부 측정·게이트를 끝낸 반환 경계입니다. 14는 순수 확산 시간만의 경계가 아닙니다. 실제 확산 호출 직전·직후를 별도 행으로 추가했습니다.',
'', '## 표 2 — 모듈','',
'실제 parameters + buffers 원소 수 × element_size 합입니다. 같은 역할이 여러 번 로드되면 최대 관측 크기 1개를 대표로 표시하고, 모든 개체·체크포인트 스냅샷은 modules.json에 보존했습니다. device는 이 대표 스냅샷 시점입니다. 오프로드로 이후 바뀔 수 있습니다. 같은 객체의 일반 사전학습 이름과 파이프라인 속성 이름은 중복 합산하지 않았습니다.',
'', '| 모듈 | dtype | device | 크기 GB |','|---|---|---|---:|']
for k,x in sorted(roles.items()):md.append(f"| {k} | {x['dtype']} | {x['device']} | {x['bytes']/G:.4f} |")
md+=['','ONNX는 parameters 접근 대신 로드 전후 RSS 델타를 기록했습니다. 이 값은 가중치 크기가 아닙니다. 반복 로드 전체 값은 modules.json에 있습니다. 아래는 종류별 최대 관측 로드 델타입니다.','', '| ONNX | 공급자 | RSS 델타 GB |','|---|---|---:|']
for k,x in sorted(onnxroles.items()):md.append(f"| {k} | {', '.join(x['providers'])} | {(x['load_rss_delta_bytes'] or 0)/G:+.4f} |")
md+=['','## 표 3 — 요청한 합계와 잔여','',
'**이 표는 물리 RAM의 정확한 분할이 아닙니다.** SDXL/IP/보조 모델은 논리 텐서 크기(GPU 텐서 포함), 런타임·ONNX는 RSS입니다. 서로 다른 시점의 최대 역할 크기를 더하므로 상주 시점도 같지 않습니다. 요청한 뺄셈을 그대로 제공하되 잔여는 **원인 미확정**입니다.',
'', '| 항목 | GB | 비고 |','|---|---:|---|',
f'| 런타임 (torch+diffusers import) | {(runtime or 0):.4f} | 부모 체크포인트 02, 지연 import 전체를 포함하지 않음 |',
f'| SDXL 가중치 | {sdxl:.4f} | 표 2 sdxl 역할 합, UNet에 로드된 adapter 가중치 포함 가능 |',
f'| IP-Adapter 이미지 인코더 | {ip:.4f} | image_encoder 논리 크기 |',
f'| 보조 모델 합 | {aux:.4f} | 텐서 {auxweights:.4f} + ONNX 로드 RSS 델타 {onnxsum:.4f}; 동시 상주 합 아님 |',
f'| 귀속 안 된 잔여 | {residual:.4f} | 전체 피크 − 위 합계, 원인 미확정 |',
'', '## 연속 피크 및 Q1~Q3','',
f"- 연속 샘플 {len(samples)}개, 목표 간격 1초, 실제 최대 간격 {max(gaps,default=0):.3f}초. 측정 구간 {stamp(samples[0]['time'])} ~ {stamp(samples[-1]['time'])}.",
f"- 체크포인트 최대 {cppeak['rss_bytes']/G:.4f} GiB (PID {cppeak['pid']}, `{cppeak['label']}`), 연속 최대 {peak['rss_bytes']/G:.4f} GiB. 차이 {(peak['rss_bytes']-cppeak['rss_bytes'])/G:.4f} GiB.",
f"- Q1 CPU + float32 관측: **{'예' if cpu32 else '아니오(관측 범위)'}**. {', '.join(k for k,v in cpu32)}.",
f"- 피크 시점 프로세스별 RSS: {[(x['pid'], round(x['rss_bytes']/G, 4)) for x in peak['processes']]} GiB.",
'- Q2 아래는 부모의 살아 있는 약한 참조로 확인한 보조 모델/ONNX와 프로세스 목록입니다. RSS 유지 자체를 모델 상주의 증거로 쓰지 않았습니다. 하위 프로세스가 살아 있는 동안의 세부 모델은 별도 PID 체크포인트에 남겼습니다.',
'', '| 단계 | RSS GB | 보조 모듈 | ONNX | 살아 있는 PID |','|---|---:|---|---|---|']
for x in Q2:md.append(f"| {x['checkpoint']} | {x['rss_GB']:.4f} | {', '.join(x['live_auxiliary_modules']) or '관측 없음'} | {', '.join(x['live_onnx']) or '관측 없음'} | {x['pids']} |")
md+=['',f"- Q3 연속 최대와 체크포인트 최대 차이 > 1 GiB: **{'예' if summary['Q3_above_1_GB'] else '아니오'}**.",
'차이가 있더라도 일시적 이중 사본·pinned memory·단편화·이미지 버퍼 중 무엇인지는 이 측정으로 확정하지 않았습니다.',
'', '## 실행 및 제한','',
f"- 해상도: {run.get('result',{}).get('selected_image_size')}, seed {run.get('seed')}, profile {run.get('expected_profile')}.",
f"- Base {run.get('base_generation_seconds')}초, 정밀화(내부 검사 포함) {run.get('refinement_seconds')}초. 프로파일 훅과 GC가 추가된 측정 실행의 시간이며 성능 비교값으로 사용하지 않습니다.",
f"- 계측 오류 {len(errs)}건. checkpoints.json에 보존. ONNX 상위 생성자의 중첩 반환에서 아직 없는 _providers 조회가 실패한 경우, 원래 InferenceSession 초기화는 계속됐고 바깥 생성자 완료의 세션 공급자·델타 관측 여부를 별도로 확인했습니다. 오류를 숨기거나 재시도하지 않았습니다.",
'- raccoon은 측정하지 않았으며 더 큰 중간 버퍼 때문에 RAM을 더 쓸 수 있습니다.',
'- 프로세스 시작 값은 Python/psutil/계측기 import 이후 첫 체크포인트입니다. OS 프로세스 생성 순간 RSS와는 다릅니다.',
'- 샘플러 목표는 1초지만 실제 최대 간격은 5.983초였습니다. 따라서 엄밀한 매초 측정은 충족하지 못했고, 이 값은 관측 피크입니다. 샘플 사이의 더 짧은 피크를 놓칠 수 있습니다. 하위 프로세스 RSS 합에는 공유 페이지가 중복 포함될 수 있습니다.',
'- 모델 resident 여부는 Python 객체 생존 관측이며 모든 네이티브 allocator 버퍼의 해제를 증명하지 않습니다.',
'- 얼굴 보조 검출 YOLO는 실행 로그에서 사용이 확인되지만 이번 계측의 모듈 등록 대상에 포함되지 않아 개별 바이트를 계측하지 못했습니다. genai_lab/reference_face_observation.py:62~73의 캐시 로더는 확인했으나 해당 객체 생존을 직접 측정하지 않았습니다. 표 3의 보조 모델 합은 관측한 항목에 한정됩니다.',
'- 최적화는 수행하거나 권고하지 않았습니다.',
'', '## 계측 근거','',
'[psutil Process RSS / children](https://psutil.io/) · [PyTorch Tensor.element_size](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.element_size.html). 사용한 코드 및 원시 JSONL은 이 출력 폴더에 보존했습니다.']
(O/'README.md').write_text('\n'.join(md)+'\n',encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
