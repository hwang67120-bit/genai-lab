from types import SimpleNamespace
from dataclasses import replace
import hashlib
import io
import pytest
from PIL import Image, ImageDraw
from genai_lab.reference_pose import ReferencePoseOptions, observe_reference_pose
from genai_lab.reference_contract import validate_reference_mode
from genai_lab.pose_estimation import PoseEstimationApprovedInput, PoseJointCoordinateCandidate, prepare_pose_control_input
from test_clothing_reference_generation import Tokenizer, request_data


@pytest.fixture
def pose():
    im = Image.new("RGB", (40, 80))
    ImageDraw.Draw(im).line((20, 5, 20, 75), fill="white", width=3)
    joints = tuple(PoseJointCoordinateCandidate(str(i), 20., 20., .9, True) for i in range(18))
    p = PoseEstimationApprovedInput(im, joints, 18, 0, .30, ("test-dwpose",))
    yield p
    p.close()


def test_default_contract_still_rejects_and_option_keeps_rgb_contract():
    c = {"clothing_reference_generation": {"enabled": True}}
    with pytest.raises(ValueError, match="외부 자세"):
        validate_reference_mode(c, True)
    validate_reference_mode(c, True, reference_pose_enabled=True)
    c["clothing_reference_generation"]["without_initial_image"] = True
    with pytest.raises(ValueError, match="RGB"):
        validate_reference_mode(c, True, reference_pose_enabled=True)


@pytest.mark.parametrize("strength", [0, -1, 1.01, float("nan"), float("inf")])
def test_invalid_strength_fails_before_model_load(pose, strength):
    with pytest.raises(ValueError):
        ReferencePoseOptions(pose, strength)


@pytest.mark.parametrize("strength", [.35, .65, 1.0])
def test_visual_path_preserves_condition_and_passes_projected_pose(pose, request_data, monkeypatch, tmp_path, strength):
    import torch
    from genai_lab.generator import generate_character_candidate
    from genai_lab.visual_reference import VisualInputs
    from genai_lab.approved_reference_run import approve_reference_run, seal_encoded_condition
    from genai_lab.clothing_reference_generation import prepare_design_reference_request
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    head = Image.new("L", (256,448)); head.paste(255,(0,0,256,128))
    body = Image.new("L", (256,448)); body.paste(255,(0,128,256,448))
    inputs = VisualInputs(Image.new("RGB",(256,448),"gray"), Image.new("RGB",(32,32),"red"),
                          Image.new("RGB",(32,32),"blue"), head, body)
    request_data = replace(request_data, width=256, height=448)
    embeds = [torch.ones(2,1,4)]
    req, _ = prepare_design_reference_request(request_data, ("blue jacket",), (Tokenizer(),Tokenizer()), character_tags=("blue hair",))
    sec = dict(enabled=True, approved_tags=("blue jacket",), approved_character_tags=("blue hair",),
               visual_inputs=inputs, candidate_count=2, identity_reference_scale=.85, garment_reference_scale=.7,
               visual_condition={"ip_adapter_image_embeds":embeds}, approved_prompt_pair=(req.prompt,req.negative_prompt),
               require_prompt_approval=True, source_name="outfit.png")
    config = {"generation":{"mode":"image_to_image"}, "clothing_reference_generation":sec,
              "detail_correction":{"enabled":False}, "pose_control":dict(enabled=False,model_id="test-openpose", conditioning_scale=.65, guidance_start=0., guidance_end=.8,original_image_change_strength=.35)}
    approve_reference_run(inputs, config, request_data)
    sec["encoded_reference_receipt"] = seal_encoded_condition(inputs, sec["visual_condition"])
    class Pipe:
        tokenizer=Tokenizer(); tokenizer_2=Tokenizer(); vae_scale_factor=8
        _genai_lab_pose_control_enabled=True
        def set_ip_adapter_scale(self, value): pass
        def __call__(self, **kw):
            self.kw=kw
            assert kw["ip_adapter_image_embeds"] is embeds
            assert kw["image"].tobytes()==inputs.source.tobytes()
            assert kw["strength"]==strength
            assert kw["prompt"]==req.prompt and kw["negative_prompt"]==req.negative_prompt
            assert kw["controlnet_conditioning_scale"]==.65
            assert kw["control_guidance_start"]==0 and kw["control_guidance_end"]==.8
            expected=prepare_pose_control_input(pose,256,448)
            try: assert kw["control_image"].tobytes()==expected.control_map_image.tobytes()
            finally: expected.close()
            return SimpleNamespace(images=[Image.new("RGB",(256,448),"blue")])
    pipe=Pipe()
    try:
        candidate=generate_character_candidate(pipe,config,request_data,tmp_path,reference_pose=ReferencePoseOptions(pose,strength))
        assert candidate.pose_control_status=="applied"
        candidate.image.close()
    finally: inputs.close()


def test_provenance_hashes_actual_projected_map(pose, monkeypatch):
    r=SimpleNamespace(loaded={},flush=lambda:None)
    monkeypatch.setattr("genai_lab.provenance.recorder",lambda config:r)
    prepared=prepare_pose_control_input(pose,256,448)
    try:
        observe_reference_pose({},ReferencePoseOptions(pose,.65),prepared,.65)
        b=io.BytesIO();prepared.control_map_image.save(b,format="PNG")
        assert r.loaded["reference_pose"]["control_image_sha256"]==hashlib.sha256(b.getvalue()).hexdigest()
        observe_reference_pose({},None,None,.25)
        assert r.loaded["reference_pose"]=={"enabled":False,"base_strength":.25,"control_image_sha256":None}
    finally:prepared.close()
