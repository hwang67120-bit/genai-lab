"""머리 출처·보호 규칙과 GPU 없이 6개 고정 합성 결과를 확인한다."""
from pathlib import Path
import io
import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from genai_lab.head_paste_rules import padding_mask, split_features, blend_result, head_crop, hair_mask, TAGS, prepare_original, paste_canvas
from genai_lab.head_paste_redraw import RestoreFrom, cleanup
from genai_lab.proportion_inputs import sha, checked_image


def png_sha(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_padding_connected_border_only():
    image = np.full((1024,1024,3), 255, np.uint8)
    image[:10] = 128
    image[100:110,100:110] = 128
    image[20:25] = 127
    pad = padding_mask(image)
    assert pad[:10].all() and not pad[100:110,100:110].any() and not pad[20:25].any()


def features():
    mask = np.zeros((1024,1024), bool)
    mask[400:460,250:310] = True
    mask[400:460,650:710] = True
    mask[370:380,250:310] = True
    mask[480:490,500:510] = True
    mask[520:530,480:540] = True
    return mask


def test_eyes_only_large_components_brows_excluded_nose_mouth_kept():
    keep, brows, top, bottom = split_features(features())
    assert top == 400 and bottom == 460
    assert keep[400:460,250:310].all() and keep[480:490,500:510].all()
    assert keep[520:530,480:540].all()
    assert brows[370:380,250:310].all() and not keep[370:380,250:310].any()


def test_no_eyeball_is_explicit_failure():
    with pytest.raises(ValueError, match="눈알"):
        split_features(np.zeros((1024,1024), bool))


def test_blend_removes_expanded_face_overlap_and_preserves_outside():
    raw = np.full((40,40,3), 60, np.uint8)
    mask = np.zeros((20,20), np.uint8); mask[4:9,4:9]=255
    face = np.zeros_like(mask); face[9:11,9:11]=255
    result, checks = blend_result(raw, Image.new("RGB", (20,20), "white"), mask, face, (10,10,30,30))
    assert checks["outside_equal"] and checks["face_equal"] and checks["expanded_face_overlap_removed"]>0
    assert np.array_equal(raw[:10],result[:10]) and not np.array_equal(raw,result)
    assert np.array_equal(raw[19:21,19:21],result[19:21,19:21])


def test_blend_rejects_mask_on_eyes():
    raw=np.zeros((20,20,3),np.uint8); mask=np.full((10,10),255,np.uint8)
    with pytest.raises(ValueError,match="얼굴"):
        blend_result(raw,Image.new("RGB",(10,10)),mask,mask,(0,0,10,10))


def test_restore_exact_last_four_and_final_noise_zero():
    import torch
    initial=torch.ones((1,4,2,2)); noise=torch.full_like(initial,2)
    mask=torch.zeros_like(initial,dtype=torch.bool); mask[...,0,0]=True
    sigmas=torch.linspace(1,0,29)
    restore=RestoreFrom(initial,noise,mask,sigmas,24)
    for index in range(28):
        value=torch.full_like(initial,9)
        restore(index,None,value)
        assert torch.equal(value[~mask],(initial+noise*sigmas[index+1])[~mask])
        if index<24: assert torch.equal(value,initial+noise*sigmas[index+1])
        else: assert (value[mask]==9).all()
    assert len(restore.calls)==28 and torch.equal(value[~mask],initial[~mask])


def test_cleanup_error_does_not_stop_remaining_release():
    errors=[]; seen=[]
    def fail(): raise RuntimeError("실패")
    cleanup([("first",fail),("next",lambda:seen.append(True))],errors)
    assert seen == [True] and errors[0]["step"]=="first"



def test_memory_snapshot_writes_actual_jsonl_and_counts_gpu_parameters(tmp_path, monkeypatch):
    """GPU 없이 실제 기록을 저장·재독해한다. CPU 파라미터는 상주량에서 제외한다."""
    from types import SimpleNamespace
    from genai_lab import head_paste_redraw

    stats = dict(allocated_bytes=1024, reserved_bytes=2048,
                 peak_allocated_bytes=1536, peak_reserved_bytes=4096)
    cuda = SimpleNamespace(
        memory_allocated=lambda: stats["allocated_bytes"],
        memory_reserved=lambda: stats["reserved_bytes"],
        max_memory_allocated=lambda: stats["peak_allocated_bytes"],
        max_memory_reserved=lambda: stats["peak_reserved_bytes"],
    )
    names = ("text_encoder", "text_encoder_2", "image_encoder", "adapter", "unet", "vae")
    def parameter(device, count, size):
        return SimpleNamespace(device=SimpleNamespace(type=device),
                               numel=lambda: count, element_size=lambda: size)
    modules = {}
    for index, name in enumerate(names, 1):
        values = (parameter("cuda", index * 10, 2), parameter("cpu", 999, 4))
        modules[name] = SimpleNamespace(parameters=lambda values=values: iter(values))
    pipe = SimpleNamespace(**modules)
    times = iter((1.5, 2.5))
    monkeypatch.setattr(head_paste_redraw.time, "perf_counter", lambda: next(times))

    before = head_paste_redraw.memory_snapshot(SimpleNamespace(cuda=cuda), pipe, tmp_path, "models_loaded_cpu")
    stats.update(allocated_bytes=512, reserved_bytes=1024)
    after = head_paste_redraw.memory_snapshot(SimpleNamespace(cuda=cuda), pipe, tmp_path, "models_released")

    lines = (tmp_path / "memory.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [before, after]
    assert before == dict(stage="models_loaded_cpu", seconds=1.5,
                          allocated_bytes=1024, reserved_bytes=2048,
                          peak_allocated_bytes=1536, peak_reserved_bytes=4096,
                          gpu_parameter_bytes={name: index * 20 for index, name in enumerate(names, 1)})
    assert after == {**before, "stage": "models_released", "seconds": 2.5,
                     "allocated_bytes": 512, "reserved_bytes": 1024}


def sample_parts():
    parts={t:np.zeros((1024,1024),bool) for t in TAGS}
    parts["eyes"][:]=features()
    parts["hair"][100:400,200:800]=True
    parts["hair"][460:600,300:700]=True
    parts["face"][300:600,300:700]=True
    return parts


def test_beard_original_removed_from_paste_head_below_eye_bottom():
    image=np.full((1024,1024,3),(200,160,130),np.uint8)
    parts=sample_parts(); hm=np.zeros((1024,1024),bool);hm[100:600,200:800]=True
    H=parts["hair"].copy()
    original=prepare_original(image,hm,parts,H)
    # 확인 H가 피부 분류에서 제거되어도 얼굴 볼록 껍질 안 아래쪽 털은 보존 대상으로 제외한다.
    assert original["beard"].any()
    assert not original["o_head"][original["beard"]].any()


def test_hair_missing_skin_is_failure_not_guess():
    parts={t:np.zeros((1024,1024),bool) for t in TAGS}
    with pytest.raises(ValueError,match="씨앗"):
        hair_mask(np.full((1024,1024,3),255,np.uint8),np.ones((1024,1024),bool),parts,np.zeros((1024,1024),bool))


ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/"outputs/head-hair-mask-confirm-20261010"
CASES=[("be2bacf5",2987514498599016057,"ca986e45d224","a450ae1a6e61","c086d7cf74af","5c9864b2744c"),
       ("be2bacf5",2987514498599016058,"ca986e45d224","047bbfb998d5","32fc150349af","22e09e907229"),
       ("b70aacb6",2343252627519382913,"651d5018da86","1a2f6786f7b8","01e281a1137b","a602310e3662"),
       ("b70aacb6",2343252627519382914,"651d5018da86","ddc624362f26","0a8d590ba0be","23e473e513a9"),
       ("21bc423c",2945843103555359005,"8cf88b5e8538","50965db76732","bf9456a362b8","836a5f015305"),
       ("21bc423c",2945843103555359006,"8cf88b5e8538","cee58b50bb47","d8ca0959cac9","2682ca69c8aa")]


@pytest.mark.parametrize("run,seed,h,init,m,paste",CASES)
def test_six_frozen_paste_blends(run,seed,h,init,m,paste):
    folder=TRIAL/"paste"/run
    if not (folder/"inputs.json").exists(): pytest.skip("로컬 사용자 고정 그림이 없는 환경")
    record=json.loads((folder/"inputs.json").read_text(encoding="utf-8"))
    row=record["per_seed"][str(seed)]
    preflight=json.loads((Path(row["raw"]).parent.parent/"preflight.json").read_text(encoding="utf-8"))
    head=preflight["options"]["head"]
    original=checked_image(head["normalized_file"],head["normalized_sha256"],(736,1232),"RGB")
    outline=checked_image(head["mask_file"],head["mask_sha256"],(736,1232),"L")
    cropped,_,box=head_crop(original,outline)
    assert tuple(box)==tuple(record["box"])
    assert png_sha(cropped)==sha(TRIAL/"out"/run/"orig_crop.png")
    assert sha(TRIAL/"out"/run/"H.png").startswith(h)
    assert sha(folder/f"G1_canvas_{seed}.png").startswith(init)
    assert sha(folder/f"M_{seed}.png").startswith(m)
    raw=np.asarray(Image.open(row["raw"]).convert("RGB"))
    canvas=Image.open(folder/f"G1_canvas_{seed}.png").convert("RGB")
    mask=np.asarray(Image.open(folder/f"M_{seed}.png").convert("L"))
    features=np.asarray(Image.open(folder/f"features_eyeball_{seed}.png").convert("L"))
    result,checks=blend_result(raw,canvas,mask,features,record["box"])
    assert png_sha(result).startswith(paste)
    assert checks["outside_equal"] and checks["face_equal"]


@pytest.mark.parametrize("color,expected", [((205,165,135),True),((65,80,100),False)])
def test_skin_shift_branch_changes_fill_only_not_protected_features(color,expected):
    parts=sample_parts()
    hm=np.zeros((1024,1024),bool); hm[100:600,200:800]=True
    H=parts["hair"].copy()
    original=np.full((1024,1024,3),255,np.uint8)
    original[parts["face"]]=(200,160,130)
    original[hm & ~H]=(200,160,130)
    original[H]=(50,90,130)
    generated_parts=sample_parts()
    generated=np.full((1024,1024,3),255,np.uint8)
    generated[generated_parts["face"]]=color
    generated[generated_parts["hair"]]=(70,80,90)
    canvas,mask,protected,measurement=paste_canvas(original,hm,parts,H,generated,generated_parts)
    assert measurement["forehead_blend"]==expected
    assert measurement["erase_fill"]==("orig_skin" if expected else "gen_inpaint")
    assert np.array_equal(canvas[protected],generated[protected])
    assert np.array_equal(canvas[~mask],generated[~mask])



def test_redraw_load_uses_locked_local_face_and_sketch(monkeypatch):
    import torch,diffusers,transformers
    from types import SimpleNamespace
    from genai_lab.head_paste_redraw import load_redraw_pipeline
    seen=[]
    adapter=object(); encoder=object()
    class Pipe:
        def load_ip_adapter(self,*a,**k): seen.append(("face",a,k))
        def set_ip_adapter_scale(self,scale): seen.append(("scale",scale))
    pipe=Pipe()
    monkeypatch.setattr(diffusers.T2IAdapter,"from_pretrained",lambda *a,**k:seen.append(("sketch",a,k)) or adapter)
    monkeypatch.setattr(transformers.CLIPVisionModelWithProjection,"from_pretrained",lambda *a,**k:seen.append(("encoder",a,k)) or encoder)
    monkeypatch.setattr(diffusers.StableDiffusionXLAdapterPipeline,"from_pretrained",lambda *a,**k:seen.append(("pipeline",a,k)) or pipe)
    settings=SimpleNamespace(model_root=Path("base"),ip_root=Path("ip"),image_encoder_subfolder="encoder",ip_subfolder="weights",ip_weight_name="face.safetensors")
    assert load_redraw_pipeline(settings,SimpleNamespace(sketch_root=Path("sketch"))) is pipe
    assert all(row[2]["local_files_only"] for row in seen if row[0]!="scale")
    assert seen[0][2]["variant"]=="fp16" and seen[0][2]["torch_dtype"]==torch.float16
    assert seen[-1]==("scale",.9)


def test_redraw_call_keeps_trial_numbers_and_face(monkeypatch,tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from genai_lab import finishing_memory
    from genai_lab.head_paste_redraw import execute_redraw
    Image.new("L",(1024,1024),0).save(tmp_path/"sketch_42.png")
    Image.new("RGB",(64,64),"red").save(tmp_path/"face.png")
    seen={}
    class Pipe:
        unet=object()
        def __call__(self,**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB",(1024,1024))])
    monkeypatch.setattr(finishing_memory,"unet_attention",lambda _:nullcontext())
    callback=object(); generator=object()
    prepared=dict(paste_dir=tmp_path,inputs=SimpleNamespace(face_file=tmp_path/"face.png"),settings=SimpleNamespace(guidance_scale=5.5))
    execute_redraw(Pipe(),prepared,42,{"prompt_embeds":"locked"},generator,callback)
    assert (seen["width"],seen["height"],seen["num_inference_steps"],seen["guidance_scale"],seen["adapter_conditioning_scale"],seen["adapter_conditioning_factor"])==(1024,1024,28,5.5,.8,1.)
    assert seen["callback"] is callback and seen["generator"] is generator and seen["prompt_embeds"]=="locked"
    assert seen["ip_adapter_image"].getpixel((0,0))==(255,0,0)
