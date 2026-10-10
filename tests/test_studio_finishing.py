"""마무리 연결·고정 문구·좌표·파일 소유권을 CPU로 검사한다."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from genai_lab import studio_finishing as finish
from genai_lab import studio_generation as service
from genai_lab import studio_background as bg
from genai_lab import studio_controller as ui
from genai_lab.finishing_prompt import quality_strings, apply_quality_format
from genai_lab.finishing_backend import FinishingBackend
from genai_lab.onepass_generation import OnePassBatch,OnePassCandidate,OnePassCancelled,OnePassGenerationError
from genai_lab.qwen_record_io import write_json
from genai_lab.proportion_inputs import sha
from test_onepass_generation import prompt
from test_studio_generation import analysis_at,Tokens
from test_studio_proportion import prepared

FIXTURES=json.loads((Path(__file__).parent/"fixtures/finishing_quality_prompts.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("item",FIXTURES)
def test_quality_matches_all_locked_diagnostic_strings(item):
    assert quality_strings(**item["a"]) == (item["b"]["positive"],item["b"]["negative"])
    before=set(item["a"]["negative"].split(", "))
    after=set(item["b"]["negative"].split(", "))
    for term in ("nsfw","panties","underwear","buruma","1boy","1girl","bad hands"):
        assert (term in before) == (term in after)


def batch_at(folder,count=1):
    folder.mkdir(parents=True)
    candidates=[]
    for seed in range(1,count+1):
        directory=folder/f"seed-{seed}";directory.mkdir()
        path=directory/"raw.png"
        Image.new("RGB",(736,1232),(30,80,110)).save(path)
        record={"valid":True,"completed":True,"raw_sha256":sha(path),"seed":seed,"prompt":asdict(prompt())}
        write_json(directory/"run.json",record)
        candidates.append(OnePassCandidate(seed,path,directory/"run.json",record))
    return OnePassBatch(folder,tuple(candidates))


class Backend:
    def __init__(self,settings): self.sizes=[];self.closed=False
    def prepare(self,prompt,cancelled): self.prompt=prompt
    def refine(self,image,seed,cancelled):
        self.sizes.append(image.size)
        return image.copy(),{"seconds":.01,"max_reserved_bytes":123,"max_allocated_bytes":100}
    def close(self): self.closed=True


class Foreground:
    def alpha(self,image): return np.full((image.height,image.width),255,np.uint8)
    def close(self): pass


def run_finish(batch,*,heads=(),cancelled=lambda:False):
    backend=Backend(None)
    finish.finish_batch(batch,prompt(),service.StudioRuntime(),backend_factory=lambda _:backend,
        detector=lambda *_:heads,cancelled=cancelled)
    return backend


def test_runtime_defaults_and_explicit_configuration(tmp_path,monkeypatch):
    default=service.StudioRuntime()
    assert not default.quality_tags and not default.finishing
    config=tmp_path/"runtime.json"
    write_json(config,{"quality_tags":True,"finishing":True})
    monkeypatch.setenv("GENAI_STUDIO_RUNTIME",str(config))
    runtime=service.StudioRuntime.from_environment()
    assert runtime.quality_tags is True and runtime.finishing is True
    with pytest.raises(ValueError):service.StudioRuntime(quality_tags="yes")


def test_quality_off_preserves_prompt_and_on_never_changes_proportion(tmp_path):
    analysis,pose,draft=prepared(tmp_path)
    prefs=QSettings(str(tmp_path/"prefs.ini"),QSettings.Format.IniFormat)
    kwargs=dict(confirmed=True,preferences=prefs,tokenizers=(Tokens(),Tokens()),seeds=(1,2,3,4))
    plain=service.build_request(analysis,analysis["garment_tags"],"male",runtime=service.StudioRuntime(),**kwargs)
    on=service.build_request(analysis,analysis["garment_tags"],"male",runtime=service.StudioRuntime(quality_tags=True),**kwargs)
    assert (on.inputs.prompt.positive,on.inputs.prompt.negative)==quality_strings(plain.inputs.prompt.positive,plain.inputs.prompt.negative)
    first=service.build_request(analysis,analysis["garment_tags"],"male",runtime=service.StudioRuntime(),proportion_pose=pose,**kwargs)
    second=service.build_request(analysis,analysis["garment_tags"],"male",runtime=service.StudioRuntime(quality_tags=True,finishing=True),proportion_pose=pose,**kwargs)
    assert first.inputs==second.inputs


def test_face_geometry_and_blending_matches_trial():
    region=finish.face_region([([338,19,754,428],"head",.918)],(1104,1848))
    assert region["face_box"] == [213,0,878,556]
    with Image.new("RGB",(1104,1848),"blue") as source, Image.new("RGB",(1024,1024),"red") as patch:
        with finish.blend_face(source,region["face_box"],patch) as result:
            assert result.size==(1104,1848) and result.getpixel((0,0))==(0,0,255)
            assert result.getpixel((550,270))==(255,0,0)


@pytest.mark.parametrize("heads,expected",[((),"skipped"),([([100,40,220,200],"head",.9)],"completed")])
def test_raw_hires_face_background_and_export_are_distinct(tmp_path,heads,expected):
    batch=batch_at(tmp_path/"run")
    candidate=batch.candidates[0]
    original=candidate.path.read_bytes();raw_record=candidate.record_path.read_bytes()
    backend=run_finish(batch,heads=heads)
    assert backend.closed and backend.sizes==([(1104,1848)] if expected=="skipped" else [(1104,1848),(1024,1024)])
    source,record=finish.finishing_source(candidate)
    assert record["face_detail"]==expected
    bg.prepare_candidate(candidate,Foreground(),bg.BackgroundOptions(Path("fake")),source_file=source)
    results=service.StudioResults(batch)
    assert results.raw_path==candidate.path and results.current_path.name=="product.png"
    with Image.open(results.current_path) as image: assert image.size==(1104,1848)
    results.approve(dict(character=True,garment=True,exposure=True))
    saved=results.export(tmp_path/"saved.png")
    assert saved.read_bytes()==results.current_path.read_bytes()
    assert candidate.path.read_bytes()==original and candidate.record_path.read_bytes()==raw_record
    assert json.loads(saved.with_suffix(".review.json").read_text(encoding="utf-8"))["finishing"]["status"]=="completed"


def test_background_failure_keeps_finished_size_and_says_so(tmp_path):
    batch=batch_at(tmp_path/"run");candidate=batch.candidates[0]
    run_finish(batch)
    source,_=finish.finishing_source(candidate)
    bg.prepare_candidate(candidate,None,bg.BackgroundOptions(Path("fake")),RuntimeError("missing"),source_file=source)
    results=service.StudioResults(batch)
    assert results.current_path==source and "마무리 결과" in results.background_notice
    assert results.verify_basis()==sha(source)


@pytest.mark.parametrize("name",["raw.png","finished_hires.png","finished.png","finishing.json"])
def test_changed_stage_prevents_approval(tmp_path,name):
    batch=batch_at(tmp_path/"run");candidate=batch.candidates[0]
    run_finish(batch)
    source,_=finish.finishing_source(candidate)
    bg.prepare_candidate(candidate,Foreground(),bg.BackgroundOptions(Path("fake")),source_file=source)
    results=service.StudioResults(batch)
    path=candidate.path.parent/name
    if name.endswith("json"):
        record=json.loads(path.read_text(encoding="utf-8"));record["seed"]=999;write_json(path,record)
    else:path.write_bytes(b"changed")
    with pytest.raises(ValueError):results.approve(dict(character=True,garment=True,exposure=True))


def test_detector_runtime_error_is_failure_not_skipped_and_keeps_hires(tmp_path):
    batch=batch_at(tmp_path/"run")
    backend=Backend(None)
    def failure(*_):raise RuntimeError("detector failed")
    with pytest.raises(RuntimeError):
        finish.finish_batch(batch,prompt(),service.StudioRuntime(),backend_factory=lambda _:backend,detector=failure)
    assert backend.closed
    directory=batch.candidates[0].path.parent
    record=finish.read_finishing(batch.candidates[0])
    assert record["status"]=="failed" and record["face_detail"]=="pending"
    assert (directory/"finished_hires.png").exists() and not (directory/"finished.png").exists()


def test_invalid_source_is_rejected_before_pipeline_loading(tmp_path):
    batch=batch_at(tmp_path/"run")
    batch.candidates[0].path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        finish.finish_batch(batch,prompt(),service.StudioRuntime(),backend_factory=lambda _:pytest.fail("GPU"),detector=lambda *_:[])


def test_cancel_before_finishing_loads_no_pipeline(tmp_path):
    batch=batch_at(tmp_path/"run")
    with pytest.raises(OnePassCancelled):
        finish.finish_batch(batch,prompt(),service.StudioRuntime(),cancelled=lambda:True,backend_factory=lambda _:pytest.fail("GPU"))
    assert json.loads((batch.directory/"finishing-status.json").read_text(encoding="utf-8"))["status"]=="cancelled"


def test_new_size_keeps_tail_coordinates_and_preview(tmp_path):
    from genai_lab.qwen_tail_gui import screen_box_to_source
    from genai_lab.qwen_pose_edit import product_preview
    from genai_lab.qwen_pose_settings import output_dimensions
    # 전체가 아닌 사각형도 화면 배율 50%에서 원본 픽셀로 정확히 대응해야 한다.
    assert screen_box_to_source((60,70),(210,320),(10,20,552,924),(1104,1848))==(100,100,400,600)
    assert output_dimensions(1104,1848)==output_dimensions(736,1232)
    raw=tmp_path/"qwen-raw.png";Image.new("RGB",output_dimensions(1104,1848),"blue").save(raw)
    record=product_preview(raw,tmp_path/"product.png",basis_size=(1104,1848))
    assert record["size"]==[1104,1848] and record["offset"]==[0,0]
    with Image.open(tmp_path/"product.png") as image:assert image.size==(1104,1848)


def test_backend_memory_guard_and_cancel_without_gpu():
    backend=object.__new__(FinishingBackend)
    backend.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:7*2**30))
    with pytest.raises(OnePassGenerationError,match="6.5"):backend.guard(lambda:False)
    with pytest.raises(OnePassCancelled):backend.guard(lambda:True)


@pytest.mark.parametrize("steps", [9, 8, 10])
def test_backend_calls_locked_img2img_without_ip_and_releases_hook(steps):
    seen={}
    class Hook:
        def remove(self): seen["removed"]=True
    class UNet:
        def register_forward_hook(self, callback):self.callback=callback;return Hook()
    class Pipe:
        unet=UNet()
        def __call__(self, *, callback_on_step_end, **kwargs):
            seen.update(kwargs)
            for index in range(steps):
                self.unet.callback()
                assert callback_on_step_end(self,index,None,{"latents":"test"})=={"latents":"test"}
            return SimpleNamespace(images=[kwargs["image"]])
    class Generator:
        def __init__(self,device):assert device=="cuda"
        def manual_seed(self,seed):seen["seed"]=seed;return self
    backend=object.__new__(FinishingBackend)
    backend.pipe=Pipe();backend.embeds={"prompt_embeds":"test"}
    backend.scheduler_type=SimpleNamespace(from_config=lambda config:SimpleNamespace(config=config))
    backend.torch=SimpleNamespace(Generator=Generator,cuda=SimpleNamespace(
        reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        max_memory_reserved=lambda:123,max_memory_allocated=lambda:100))
    with Image.new("RGB",(1104,1848)) as source:
        if steps==9:
            image,metrics=backend.refine(source,1234,lambda:False)
            assert image is source and metrics["ip_adapter"]=="not_loaded"
            assert metrics["unet_calls"]==9
        else:
            with pytest.raises(OnePassGenerationError,match="호출 수"):
                backend.refine(source,1234,lambda:False)
    assert seen["removed"] and seen["seed"]==1234
    assert seen["strength"]==.35 and seen["num_inference_steps"]==28 and seen["guidance_scale"]==5.5
    assert not any("ip_adapter" in key for key in seen)


@pytest.mark.parametrize("finishing,proportion,expected",[
    (False,False,["generate_closed","background"]),
    (True,False,["generate_closed","finish_closed","background"]),
    (True,True,["generate_closed"])])
def test_controller_runs_finishing_only_between_single_pass_release_and_background(tmp_path,monkeypatch,finishing,proportion,expected):
    from PySide6.QtWidgets import QApplication
    from gui_main import GenAILabWindow
    app=QApplication.instance() or QApplication([])
    window=GenAILabWindow();controller=window.studio
    controller.run_directory=tmp_path
    controller.analysis={}
    controller.runtime=service.StudioRuntime(finishing=finishing)
    events=[];batch=object()
    monkeypatch.setattr(ui,"build_request",lambda *a,**k:SimpleNamespace(inputs=SimpleNamespace(prompt=prompt())))
    def generate(*args,**kwargs):events.append("generate_closed");return batch
    def finishing_call(actual,*args,**kwargs):assert actual is batch;events.append("finish_closed");return batch
    def background(actual,*args,**kwargs):
        assert actual is batch and kwargs.get("finishing",False)==finishing
        events.append("background");return batch
    monkeypatch.setattr(ui,"generate_onepass_request",generate)
    monkeypatch.setattr(finish,"finish_batch",finishing_call)
    monkeypatch.setattr(ui,"prepare_backgrounds",background)
    def launch(work,done,**kwargs):assert work(lambda:False,lambda _:None) is batch
    monkeypatch.setattr(controller,"launch",launch)
    try:
        controller.begin_generation(("male",[],None),options=object() if proportion else None)
        assert events==expected
    finally:window.close()



def test_replay_defaults_to_cpu_preflight(monkeypatch,capsys):
    from scripts import verify_studio_finishing as replay
    monkeypatch.setattr(replay,"preflight",lambda root,cases,runtime:{"status":"passed"})
    monkeypatch.setattr(replay,"run_replay",lambda *args:pytest.fail("GPU needs --run"))
    assert replay.main([])==0
    assert json.loads(capsys.readouterr().out)["gpu_executed"] is False
    with pytest.raises(SystemExit):replay.parser().parse_args(["--cases","17"])


def test_replay_copies_locked_f0_without_claiming_new_generation(tmp_path):
    from scripts import verify_studio_finishing as replay
    source=tmp_path/"f0.png"
    Image.new("RGB",(736,1232),"blue").save(source)
    item={"seed":123,"b":FIXTURES[0]["b"]}
    entry={"files":{"f0":source},"item":item,"expected":{"f0_sha256":sha(source)}}
    batch,result=replay.copy_source_case(entry,tmp_path/"copy",service.StudioRuntime(),(Tokens(),Tokens()))
    assert batch.candidates[0].record["generated_here"] is False
    assert sha(source)==sha(batch.candidates[0].path)
    assert result.positive==item["b"]["positive"] and result.negative==item["b"]["negative"]
    with pytest.raises(FileExistsError):replay.copy_source_case(entry,tmp_path/"copy",service.StudioRuntime(),(Tokens(),Tokens()))


def test_replay_mismatch_stops_and_preserves_record(tmp_path,monkeypatch):
    from scripts import verify_studio_finishing as replay
    entries=[{"case":1,"expected":{"f1_sha256":"expected","f2_sha256":"expected"}}, {"case":2}]
    batch=batch_at(tmp_path/"source")
    seen=[]
    def copied(entry,*args):seen.append(entry["case"]);return batch,prompt()
    monkeypatch.setattr(replay,"load_onepass_tokenizers",lambda _:None)
    monkeypatch.setattr(replay,"copy_source_case",copied)
    monkeypatch.setattr(replay,"finish_batch",lambda *args,**kwargs:None)
    monkeypatch.setattr(replay,"read_finishing",lambda _:{"stages":{"hires":{"sha256":"different"}},"finished_sha256":"different"})
    with pytest.raises(ValueError,match="SHA 불일치"):
        replay.run_replay({"cases":entries},tmp_path/"replay",service.StudioRuntime())
    state=json.loads((tmp_path/"replay/replay.json").read_text(encoding="utf-8"))
    assert state["status"]=="failed" and seen==[1] and state["comparisons"][0]["f2_match"] is False
