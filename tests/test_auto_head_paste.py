"""자동 붙이기 계약·일괄 수명·사용자 선택을 GPU 없이 확인한다."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QSettings
from genai_lab import studio_head_paste as service
from genai_lab import studio_head_paste_gui as gui
from genai_lab import studio_controller as ui
from genai_lab import head_paste_worker as worker
from genai_lab.studio_generation import StudioRuntime, StudioResults
from genai_lab.head_paste_rules import TAGS
from genai_lab.head_paste_semantics import save_parts
from genai_lab.proportion_inputs import sha
from genai_lab.qwen_record_io import write_json
from gui_main import GenAILabWindow
from test_studio_proportion import two_pass_batch, prepared
from test_studio_head_paste import product_at, preview_at
from test_studio_generation import Tokens, wait_for


@pytest.mark.parametrize('appearance,garments,skip', [
    (['short_hair'], ['hoodie'], False),
    (['short hair'], ['coat'], False),
    (['long_hair'], ['coat'], True),
    ([], ['coat'], True),
    *[(['short_hair'], [tag], True) for tag in
      ['hat', 'cap', 'hood', 'hood_up', 'helmet', 'beret', 'beanie', 'headwear']]])
def test_confirmed_tags_only_skip_exact_coverings(appearance, garments, skip):
    assert bool(service.automatic_skip_reason(appearance, garments)) is skip


def measured_preview(folder, color=255, eyes=True, padding=False):
    folder.mkdir()
    original = np.full((1024,1024,3), color, np.uint8)
    if padding: original[:20] = 128
    mask = np.zeros((1024,1024), np.uint8); mask[300:700,300:700]=255
    parts = {name:np.zeros((1024,1024),bool) for name in TAGS}
    parts['face'][350:650,350:650]=True
    if eyes: parts['eyes'][400:470,400:470]=True
    for name,array in [('original',original),('head-mask',mask)]: Image.fromarray(array).save(folder/(name+'.png'))
    save_parts(folder/'original-parts.npz',parts)
    files=[folder/'original.png',folder/'head-mask.png',folder/'original-parts.npz']
    return dict(directory=str(folder),identity={},original_file=str(files[0]),head_mask_file=str(files[1]),
                manifest={str(p):sha(p) for p in files})


@pytest.mark.parametrize('color,eyes,skip', [(255,True,False),(248,True,True),(255,False,True)])
def test_background_distance_and_missing_eyeballs_skip(tmp_path,color,eyes,skip):
    info=measured_preview(tmp_path/'preview',color,eyes,True)
    reason, measurement=service.preview_skip_reason(info)
    assert bool(reason) is skip
    assert measurement['distance_space']=='RGB'
    assert measurement['white_distance']==pytest.approx(np.sqrt(3)*(255-color))
    if not eyes: assert '눈알' in reason


def auto_results(tmp_path,failed=2):
    batch=two_pass_batch(tmp_path/'generation')
    result=StudioResults(batch)
    items=[]
    for i in range(4):
        result.select(i)
        if i==failed:
            items.append(dict(seed=i+1,status='failed',reason='눈알 없음'))
        else:
            info=product_at(tmp_path/('head-'+str(i)),result)
            items.append(dict(seed=i+1,status='completed',directory=info['directory']))
    return StudioResults(replace(batch,auto_head=dict(status='awaiting_user_review',items=items)))


@pytest.mark.parametrize('before',[False,True])
def test_four_candidates_cycle_view_rejection_and_export(tmp_path,before):
    results=auto_results(tmp_path)
    originals=[c.path.read_bytes() for c in results.batch.candidates]
    assert len(results.batch.candidates)==4
    assert results.current_path==Path(results.active_head['product_file'])
    if before: results.toggle_head_view()
    results.select(3)
    assert results.reject_and_next()==0
    assert results.show_before_head is before
    assert results.rejections[-1]['selected']==3
    assert results.rejections[-1]['head_view']==('before' if before else 'pasted')
    results.approve(dict(character=True,garment=True,exposure=True))
    results.export(tmp_path/'saved.png')
    record=json.loads((tmp_path/'saved.review.json').read_text(encoding="utf-8"))
    assert record['head_view']==('before' if before else 'pasted')
    assert ('head_paste' in record) is not before
    assert (tmp_path/'saved.png').read_bytes()==(originals[0] if before else Path(results.active_head['product_file']).read_bytes())
    assert [c.path.read_bytes() for c in results.batch.candidates]==originals
    results.select(2)
    assert results.current_path==results.candidate.path and '눈알 없음' in results.head_notice


def test_toggle_invalidates_approval_and_changed_paste_blocks_save(tmp_path):
    results=auto_results(tmp_path)
    results.approve(dict(character=True,garment=True,exposure=True))
    results.toggle_head_view()
    assert results.approved_sha is None and results.status=='awaiting_user_review'
    results.toggle_head_view()
    Path(results.active_head['product_file']).write_bytes(b'changed')
    with pytest.raises(ValueError):results.verify_current()


def test_actual_gui_next_button_and_before_toggle(tmp_path):
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow()
    try:
        result=auto_results(tmp_path)
        window.studio.generated(result.batch)
        assert window.studio_head_paste_button.isHidden()
        window.studio_before_head_button.click()
        assert window.studio.results.show_before_head
        window.studio_candidate_combo.setCurrentIndex(3)
        window.reject_candidate_button.click()
        assert window.studio_candidate_combo.currentIndex()==0
        assert window.studio.results.show_before_head
        assert window.studio.results.rejections[-1]['selected']==3
        window.studio_candidate_combo.setCurrentIndex(2)
        assert not window.studio_before_head_button.isEnabled()
        assert '눈알 없음' in window.status_label.text()
    finally:window.close()


@pytest.mark.parametrize('choice',['accept','reject','remembered','background'])
def test_hair_confirmation_precedes_generation_once(tmp_path,monkeypatch,choice):
    app=QApplication.instance() or QApplication([])
    analysis,pose,draft=prepared(tmp_path)
    analysis['groups']['appearance'].append('short_hair')
    window=GenAILabWindow()
    studio=window.studio
    studio.analysis=analysis;studio.runtime=StudioRuntime()
    studio.run_directory=tmp_path/'run';studio.run_directory.mkdir()
    studio.prepared_proportion_pose=pose
    studio.proportion_pending=('male',('coat',),ui.AppearanceOverrides())
    events=[]
    info=preview_at(tmp_path/'preview')
    if choice=='remembered':
        service.confirm_hair(info,True)
        info={**info,'remembered':True}
    monkeypatch.setattr(ui,'review_head_outline',lambda *a:'confirm')
    monkeypatch.setattr(gui,'source_request',lambda *a:events.append('source') or {})
    monkeypatch.setattr(gui,'prepare_hair',lambda *a,**k:events.append('hair') or info)
    monkeypatch.setattr(gui,'preview_skip_reason',lambda *a:('배경 확인 불가' if choice=='background' else '',{}))
    monkeypatch.setattr(gui,'review_hair',lambda *a:events.append('review') or choice=='accept')
    monkeypatch.setattr(studio,'begin_generation',lambda *a,**k:events.append('generate'))
    try:
        studio.proportion_head_ready(draft)
        wait_for(app,lambda:events and events[-1]=='generate')
        assert events==['source','hair']+(['review'] if choice in ('accept','reject') else [])+['generate']
        saved=json.loads((studio.run_directory/'auto-head-confirmation.json').read_text(encoding="utf-8"))
        assert saved['status']==('confirmed' if choice in ('accept','remembered') else 'skipped')
    finally:window.close()


def test_batch_worker_segments_all_before_one_redraw_and_keeps_partial_success(tmp_path,monkeypatch):
    calls=[]
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,mem_get_info=lambda:(100,200)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    class Segmenter:
        def __init__(self,*a):calls.append('load-segmenter')
        def close(self):calls.append('close-segmenter')
    monkeypatch.setattr(worker,'HeadSegmenter',Segmenter)
    monkeypatch.setattr(worker,'redraw_models',lambda request:('settings','options',{}))
    def prepare(request,directory,segmenter,context):
        calls.append('segment-'+str(request['seed']))
        if request['seed']==2:raise ValueError('눈알 없음')
        return dict(run=directory)
    monkeypatch.setattr(worker,'prepare_redraw',prepare)
    def redraw(prepared,output,state):
        calls.append('redraw-once')
        assert calls[-2]=='close-segmenter'
        state['completed']=[[p['run'].name[:8],seed] for p,seed in zip(prepared,[1,3,4])]
    monkeypatch.setattr(worker,'redraw_prepared',redraw)
    monkeypatch.setattr(worker,'complete_product',lambda job,p,out:dict(product_file='product',product_sha256='sha'))
    jobs=[dict(seed=i,settings={},options={},semantic_repo='repo',semantic_checkpoint='checkpoint') for i in [1,2,3,4]]
    report=worker.complete_batch(dict(jobs=jobs,manifest={},gui_memory_before_worker={}),tmp_path)
    assert calls==['load-segmenter','segment-1','segment-2','segment-3','segment-4','close-segmenter','redraw-once']
    assert [i['status'] for i in report['items']]==['completed','failed','completed','completed']
    assert report['items'][1]['reason']=='눈알 없음'
    assert report['memory_before_models']['device_free_bytes']==100
    assert json.loads((tmp_path/'result.json').read_text(encoding="utf-8"))==report


def test_auto_generation_cancellation_does_not_become_four_successes(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow()
    batch=two_pass_batch(tmp_path/'generation')
    window.studio.head_paste_controller.generation_context=dict(preview={},background=None)
    def cancelled(*a,**k):raise ui.OnePassCancelled('취소')
    monkeypatch.setattr(gui,'apply_head_batch',cancelled)
    try:
        with pytest.raises(ui.OnePassCancelled):
            window.studio.head_paste_controller.apply_automatically(batch,lambda:True,lambda _:None)
        assert not (batch.directory/'auto-head-result.json').exists()
    finally:window.close()


def test_source_confirmation_identity_matches_post_generation_request(tmp_path):
    from dataclasses import asdict
    from genai_lab.studio_proportion import confirmed_options
    from genai_lab.proportion_inputs import json_value
    from genai_lab.onepass_generation_settings import OnePassGenerationSettings
    analysis,pose,draft=prepared(tmp_path)
    runtime=StudioRuntime()
    reference=dict(source=analysis['source'],sha256=sha(analysis['source']))
    analysis['references']={'character':reference}
    options=confirmed_options(pose,draft,runtime)
    request=service.source_request(analysis,options.head,runtime)
    batch=two_pass_batch(Path(analysis['directory']).parent/'generation')
    inputs=dict(face_file=str(Path(analysis['directory'])/'face.png'),face_sha256=analysis['face_sha256'])
    settings=json_value(asdict(runtime.generation))
    lock=json_value(dict(options=asdict(options),inputs=inputs,settings=settings,seeds=[1,2,3,4]))
    write_json(batch.directory/'preflight.json',lock)
    (batch.directory/'preflight.sha256').write_text(sha(batch.directory/'preflight.json'))
    write_json(Path(analysis['directory'])/'references.json',analysis['references'])
    write_json(Path(analysis['directory'])/'analysis.json',analysis)
    batch=replace(batch,candidates=tuple(replace(c,raw=replace(c.raw,record={**c.raw.record,'settings':settings})) for c in batch.candidates))
    post=service.selected_request(StudioResults(batch),runtime)
    assert post['identity']==request['identity'] and post['cache_key']==request['cache_key']
    Path(options.head.mask_file).write_bytes(b'changed')
    with pytest.raises(ValueError):service.source_request(analysis,options.head,runtime)


def test_successful_task_clears_cache_before_next_callback(monkeypatch):
    from genai_lab import generation_cleanup
    app=QApplication.instance() or QApplication([])
    calls=[]
    monkeypatch.setattr(ui,'release_cuda_cache',lambda torch:calls.append('clear') or {'reserved_bytes':0})
    value=object()
    task=ui.StudioTask(lambda *a:value,None)
    task.run()
    assert calls==['clear'] and task.result is value and task.end_memory=={'reserved_bytes':0}


def test_partial_redraw_failure_keeps_other_products(tmp_path,monkeypatch):
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,mem_get_info=lambda:(100,200)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    monkeypatch.setattr(worker,'HeadSegmenter',lambda *a:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(worker,'redraw_models',lambda request:('settings','options',{}))
    monkeypatch.setattr(worker,'prepare_redraw',lambda job,folder,*a:dict(run=folder))
    def redraw(prepared,output,state):
        state['completed']=[[p['run'].name[:8],i+1] for i,p in enumerate(prepared) if i!=1]
        state['candidate_errors']={'2':'한 장 다시 그리기 실패'}
    monkeypatch.setattr(worker,'redraw_prepared',redraw)
    monkeypatch.setattr(worker,'complete_product',lambda *a:dict(product_file='file',product_sha256='sha'))
    jobs=[dict(seed=i,settings={},options={},semantic_repo='repo',semantic_checkpoint='checkpoint') for i in [1,2,3,4]]
    report=worker.complete_batch(dict(jobs=jobs,manifest={},gui_memory_before_worker={}),tmp_path)
    assert [i['status'] for i in report['items']]==['completed','failed','completed','completed']
    assert report['items'][1]['reason']=='한 장 다시 그리기 실패'


def test_actual_generation_applies_four_after_preconfirmation(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    analysis,pose,draft=prepared(tmp_path)
    analysis['groups']['appearance'].append('short_hair')
    window=GenAILabWindow(); studio=window.studio
    studio.analysis=analysis; studio.runtime=StudioRuntime()
    studio.run_directory=tmp_path/'run';studio.run_directory.mkdir()
    studio.prepared_proportion_pose=pose
    studio.proportion_pending=('male',('coat',),ui.AppearanceOverrides())
    events=[]; preview=preview_at(tmp_path/'preview')
    monkeypatch.setattr(QMessageBox,'warning',lambda *a:pytest.fail(str(a)))
    monkeypatch.setattr(ui,'review_head_outline',lambda *a:'confirm')
    monkeypatch.setattr(gui,'source_request',lambda *a:{})
    monkeypatch.setattr(gui,'prepare_hair',lambda *a,**k:preview)
    monkeypatch.setattr(gui,'preview_skip_reason',lambda *a:('',{}))
    monkeypatch.setattr(gui,'review_hair',lambda *a:events.append('confirm') or True)
    def build(a,tags,gender,**kw):
        from genai_lab.studio_generation import build_request
        return build_request(a,tags,gender,**kw,tokenizers=(Tokens(),Tokens()),seeds=(1,2,3,4),
            preferences=QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat))
    monkeypatch.setattr(ui,'build_request',build)
    def generate(request,folder,**kw):
        assert events==['confirm']
        events.append('generate')
        batch=two_pass_batch(folder);kw['on_directory_created'](folder)
        return batch
    monkeypatch.setattr(ui,'generate_onepass_request',generate)
    def apply(batch,info,runtime,**kw):
        assert events==['confirm','generate'] and studio.results is None
        events.append('paste-four')
        result=StudioResults(batch); items=[]
        for index in range(4):
            result.select(index)
            if index==1:items.append(dict(seed=index+1,status='failed',reason='눈알 없음'))
            else:
                item=product_at(batch.directory/('head-'+str(index)),result)
                items.append(dict(seed=index+1,status='completed',directory=item['directory']))
        return dict(status='awaiting_user_review',items=items)
    monkeypatch.setattr(gui,'apply_head_batch',apply)
    try:
        studio.proportion_head_ready(draft)
        wait_for(app,lambda:studio.results is not None and studio.task is None)
        assert events==['confirm','generate','paste-four']
        assert len(studio.results.batch.candidates)==4 and studio.results.head_view=='pasted'
        window.studio_candidate_combo.setCurrentIndex(1)
        assert studio.results.current_path==studio.results.candidate.path
        assert '눈알 없음' in window.status_label.text()
    finally:window.close()


@pytest.mark.parametrize('fatal',[False,True])
def test_redraw_reuses_model_and_isolates_only_recoverable_failure(tmp_path,monkeypatch,fatal):
    import accelerate
    from genai_lab import head_paste_redraw as redraw
    from genai_lab import finishing_offload, finishing_memory, onepass_generation
    calls=[]
    class OOM(RuntimeError):pass
    fake=SimpleNamespace(float16='fp16',device=lambda name:name,
        cuda=SimpleNamespace(is_initialized=lambda:False,empty_cache=lambda:None,
            reset_peak_memory_stats=lambda:None,max_memory_reserved=lambda:0,OutOfMemoryError=OOM))
    monkeypatch.setitem(sys.modules,'torch',fake)
    hook=SimpleNamespace(offload=lambda:calls.append('offload'),remove=lambda:None)
    encoder=SimpleNamespace(to=lambda target:encoder)
    pipe=SimpleNamespace(text_encoder=encoder,text_encoder_2=encoder,adapter=object(),
        unet=SimpleNamespace(register_forward_pre_hook=lambda callback:hook),enable_vae_tiling=lambda:None)
    offload=SimpleNamespace(observer_hooks=[],close=lambda:calls.append('closed'))
    monkeypatch.setattr(redraw,'load_redraw_pipeline',lambda *a:calls.append('load-once') or pipe)
    monkeypatch.setattr(redraw,'encode_initial_heads',lambda *a:{})
    monkeypatch.setattr(redraw,'memory_snapshot',lambda *a:{})
    monkeypatch.setattr(finishing_offload,'configure_offload',lambda *a:offload)
    monkeypatch.setattr(finishing_memory,'attention_policy',lambda pipe:{})
    monkeypatch.setattr(accelerate,'cpu_offload_with_hook',lambda *a:(None,hook))
    monkeypatch.setattr(onepass_generation,'encode_prompt_plan',lambda *a:{})
    def candidate(pipe,p,seed,initial,embeds,output,state,*a):
        calls.append(seed)
        if seed==2:raise OOM('한도') if fatal else ValueError('한 장 실패')
        state['completed'].append([p['run'].name,seed])
    monkeypatch.setattr(redraw,'redraw_candidate',candidate)
    settings=SimpleNamespace(model_root='model',ip_root='ip')
    prepared=[dict(settings=settings,options=SimpleNamespace(sketch_root='sketch'),
        rec=dict(seeds=[1,2,3,4]),prompt=object(),run=Path('source'))]
    state=dict(isolate_failures=True,completed=[])
    if fatal:
        with pytest.raises(OOM):redraw.redraw_prepared(prepared,tmp_path,state)
        assert [v for v in calls if isinstance(v,int)]==[1,2]
    else:
        redraw.redraw_prepared(prepared,tmp_path,state)
        assert [v for v in calls if isinstance(v,int)]==[1,2,3,4]
        assert state['candidate_errors']['2']=='한 장 실패'
    assert calls.count('load-once')==1 and calls.count('closed')==1
    assert state['pipeline_loads']==1
