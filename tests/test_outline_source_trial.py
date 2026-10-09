"""CPU-only original-outline branch and replay contracts."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import pytest
from PIL import Image
from genai_lab import proportion_generation as flow
from genai_lab import proportion_inputs as contract
from genai_lab.onepass_pose import InputDecision
from genai_lab.onepass_generation import OnePassGenerationError
from genai_lab.qwen_record_io import write_json
from scripts import body_outline_source_trial as trial
from test_proportion_generation import case, factories, Foreground


def test_unknown_source_rejected():
    assert contract.ProportionOptions().outline_source == "base"
    with pytest.raises(ValueError, match="base"):
        contract.ProportionOptions(outline_source="guess")


def test_default_and_explicit_base_keep_pixel_bytes(case, tmp_path):
    inputs, settings, options = case
    results=[]
    for name,opt in (("default",options),("explicit",replace(options,outline_source="base"))):
        folder=tmp_path/name
        batch=flow.generate_proportion_batch(inputs,(1,2),folder,settings=settings,options=opt,**factories(case,[]))
        results.append([(c.raw.path.read_bytes(),c.path.read_bytes(),(folder/f"sketch_{c.seed}.png").read_bytes()) for c in batch.candidates])
    assert results[0] == results[1]


def test_original_skips_base_infers_once_and_keeps_schedule(case,tmp_path):
    inputs,settings,options=case
    options=replace(options,outline_source="original")
    events=[]
    providers=factories(case,events)
    providers["base_factory"]=lambda _:pytest.fail("Unused BASE GPU load")
    class OriginalForeground(Foreground):
        def alpha(self,image):
            if "contour-open" not in events:
                assert image.getpixel((0,0)) == (30,80,110)
            return super().alpha(image)
    providers["foreground_factory"]=lambda o:OriginalForeground(o,events)
    batch=flow.generate_proportion_batch(inputs,(1,2),tmp_path/"run",settings=settings,options=options,**providers)
    assert events == ["cpu-open","alpha","cpu-close","contour-open","contour-close","cpu-open","alpha","alpha","cpu-close"]
    maps=trial.read(batch.directory/"maps.json")
    assert maps["1"] == maps["2"]
    assert maps["1"]["outline_source"] == "original"
    assert maps["1"]["original_sha256"] == options.head.normalized_sha256
    run=trial.read(batch.directory/"run.json")
    assert run["BASE"] == [] and run["base_stage"] == "skipped_unused_outline"
    assert run["outline_source"] == "original" and run["original_sha256"] == options.head.normalized_sha256
    assert run["sketch_sha256"] == {str(seed):maps[str(seed)]["sha256"] for seed in (1,2)}
    assert not list(batch.directory.glob("BASE_*"))
    for candidate in batch.candidates:
        assert candidate.record["outline_source"] == "original"
        assert candidate.record["original_sha256"] == options.head.normalized_sha256
        assert candidate.record["adapter_strengths"] == [1.2,.5]
        assert candidate.record["adapter_applied_calls"] == list(range(11))
        assert [c["ip_scales"] for c in candidate.record["unet_calls"]] == [[.5]]*11+[[.9]]*17


@pytest.mark.parametrize("failure",["source_tamper","expected_base"])
def test_original_rejects_before_output_or_gpu(case,tmp_path,failure):
    inputs,settings,options=case
    options=replace(options,outline_source="original")
    extra={}
    if failure == "source_tamper": options.head.normalized_file.write_bytes(b"changed")
    else: extra["expected_base"]={1:"0"*64}
    with pytest.raises(ValueError):
        flow.generate_proportion_batch(inputs,(1,),tmp_path/"rejected",settings=settings,options=options,
            base_factory=lambda _:pytest.fail("GPU"),contour_factory=lambda *a:pytest.fail("GPU"),**extra)
    assert not (tmp_path/"rejected").exists()


def test_original_wrong_sketch_preserves_failure_and_never_loads_gpu(case,tmp_path):
    inputs,settings,options=case
    events=[]
    providers=factories(case,events)
    providers["contour_factory"]=lambda *a:pytest.fail("GPU")
    with pytest.raises(ValueError,match="스케치 SHA"):
        flow.generate_proportion_batch(inputs,(1,),tmp_path/"bad",settings=settings,
            options=replace(options,outline_source="original"),expected_sketch={1:"0"*64},**providers)
    record=trial.read(tmp_path/"bad/run.json")
    assert record["status"] == "failed" and record["phase"] == "sketch"
    assert events == ["cpu-open","alpha","cpu-close"]


def studio_at(case,directory):
    inputs,settings,options=case
    inputs=replace(inputs,input_decision=InputDecision("pass",(),(),("proceed_with_pose",)),user_choice="proceed_with_pose")
    folder=directory/"inputs";folder.mkdir(parents=True)
    source=directory/"original.png"
    for path in (source,folder/"character.png"):
        path.write_bytes(options.head.normalized_file.read_bytes())
    refs={"character":{"source":str(source),"sha256":contract.sha(source),
                       "preprocessing":{"analysis_sha256":contract.sha(folder/"character.png")}}}
    pose={"source_file":str(folder/"character.png"),"source_sha256":contract.sha(folder/"character.png"),
          "normalized_file":str(options.head.normalized_file),"normalized_sha256":options.head.normalized_sha256,
          "control_file":str(inputs.control_file),"control_sha256":inputs.control_sha256,
          "ip_early":.5,"decision":asdict(inputs.input_decision)}
    write_json(options.head.contour_file.parent/"user-confirmation.json",contract.json_value(
        {"reviewer":"user","automatic_approval":False,"head":asdict(options.head),"pose":pose}))
    write_json(folder/"approval.json",{"approved":True,"proportion_mode":"two_pass","prompt":inputs.prompt.positive,
        "negative":inputs.prompt.negative,"face_sha256":inputs.face_sha256,"references":refs})
    flow.generate_proportion_batch(inputs,(1,2,3,4),directory/"generation",settings=settings,options=options,
                                   **factories((inputs,settings,options),[]))
    return inputs,settings,options


def test_studio_restores_exact_prompt_tokens_face_settings_seeds(case,tmp_path):
    folder=tmp_path/"studio"
    inputs,settings,options=studio_at(case,folder)
    restored,config,new,seeds,manifest=trial.load_studio(folder)
    assert contract.json_value(asdict(restored)) == contract.json_value(asdict(inputs))
    assert config == settings and new == replace(options,outline_source="original")
    assert seeds == (1,2,3,4)
    trial.verify_manifest(manifest)


@pytest.mark.parametrize("change",["face","normalized","head","source","approval","lock","raw","product","incomplete"])
def test_studio_changed_inputs_or_incomplete_comparison_rejected(case,tmp_path,change):
    folder=tmp_path/"studio"
    inputs,settings,options=studio_at(case,folder)
    paths={"face":inputs.face_file,"normalized":options.head.normalized_file,"head":options.head.mask_file,
           "source":folder/"original.png","raw":folder/"generation/BASE_1/raw.png",
           "product":folder/"generation/seed-1/product.png"}
    if change in paths: paths[change].write_bytes(b"changed")
    elif change == "approval":
        path=folder/"inputs/approval.json";data=trial.read(path);data["prompt"]="different";write_json(path,data)
    elif change == "lock":
        path=folder/"generation/preflight.json";data=trial.read(path);data["seeds"][0]=9;write_json(path,data)
    else:
        path=folder/"generation/run.json";data=trial.read(path);data["status"]="failed";write_json(path,data)
    with pytest.raises((ValueError,KeyError,OnePassGenerationError)):
        trial.load_studio(folder)


def test_cpu_cli_default_never_generates_and_checks_output_reuse(case,tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"check_trial_locks",lambda:{})
    monkeypatch.setattr(trial,"replay_inputs",lambda _:case)
    monkeypatch.setattr(trial,"generate_proportion_batch",lambda *a,**k:pytest.fail("GPU action"))
    original=trial.prepare_trial
    def prepare(*a,**kwargs):
        # The unit fixture has synthetic pixels; real R6 SHA comparison is a separate CPU run.
        args=list(a);args[-1]={}
        return original(*args,foreground_factory=lambda o:Foreground(o,[]),**kwargs)
    monkeypatch.setattr(trial,"prepare_trial",prepare)
    args=["--r6","--foreground-model",str(case[2].foreground_model),"--output",str(tmp_path/"preflight")]
    trial.main(args)
    data=trial.read(tmp_path/"preflight/trial-preflight.json")
    assert data["seeds"] == list(trial.R6_SEEDS) and data["outline_source"] == "original"
    assert data["base_stage"] == "skipped_unused_outline"
    assert trial.read(tmp_path/"preflight/trial-status.json")["gpu_generation"] is False
    with pytest.raises(ValueError,match="새 출력"):
        trial.main(args)


def test_run_requires_explicit_flag_and_uses_preflight_hashes(case,tmp_path,monkeypatch):
    inputs,settings,options=case
    monkeypatch.setattr(trial,"check_trial_locks",lambda:{})
    monkeypatch.setattr(trial,"load_studio",lambda _: (inputs,settings,replace(options,outline_source="original"),(1,2,3,4),{}))
    original=trial.prepare_trial
    monkeypatch.setattr(trial,"prepare_trial",lambda *a:original(*a,foreground_factory=lambda o:Foreground(o,[])))
    calls=[]
    monkeypatch.setattr(trial,"generate_proportion_batch",lambda *a,**k:calls.append((a,k)))
    trial.main(["--studio-run",str(tmp_path),"--output",str(tmp_path/"run"),"--run"])
    args,kwargs=calls[0]
    assert args[0] is inputs and args[1] == (1,2,3,4)
    assert kwargs["options"].outline_source == "original" and len(kwargs["expected_sketch"]) == 4


def test_source_changed_after_preflight_stops_before_gpu(case,tmp_path,monkeypatch):
    path=tmp_path/"source.json";path.write_text("locked")
    manifest={str(path):contract.sha(path)}
    path.write_text("changed")
    with pytest.raises(ValueError,match="사전 검사 뒤"):
        trial.verify_manifest(manifest)
