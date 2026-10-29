"""CPU 입력 분리와 잘못된 전달의 차단을 검사한다. 모델 가중치·GPU·네트워크 불필요."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
from genai_lab.character_style_inputs import (
    CharacterInputError, prepare_character_style_bundle, read_verified_bundle, sha256_bytes,
)
from genai_lab.character_style_conditioning import (
    ConditioningOptions, build_conditioning_call, dispatch_conditioning_once,
)


class TokenizerStub:
    model_max_length = 77
    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return {"input_ids": list(range(len(text.split())+2))}


class RecordingPipeline:
    """호출 내용을 기록하는 CPU 대역이다. 실제 모델 생성의 증거로 사용하지 않는다."""
    def __init__(self):
        self.unet = SimpleNamespace(encoder_hid_proj=SimpleNamespace(image_projection_layers=[object(), object()]))
        self.controlnet = object()
        self.tokenizer = TokenizerStub()
        self.tokenizer_2 = TokenizerStub()
        self.calls = []
        self.scales = None
        self.fail = False

    def set_ip_adapter_scale(self, scales):
        self.scales = scales

    def __call__(self, prompt, prompt_2, image, ip_adapter_image, cross_attention_kwargs,
                 controlnet_conditioning_scale, **kwargs):
        self.calls.append(dict(prompt=prompt, image=image, ip_adapter_image=ip_adapter_image,
                               masks=cross_attention_kwargs["ip_adapter_masks"], **kwargs))
        if self.fail:
            raise RuntimeError("sentinel model failure")
        return SimpleNamespace(images=[Image.new("RGB", image.size, "orange")])


class TestCharacterStyleInputs(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.style = self.root/"source.png", self.root/"style.png"
        Image.new("RGB", (64,96), (73,83,93)).save(self.source)
        Image.new("RGB", (80,100), (200,100,50)).save(self.style)
        joints = {name:[20+i,30+i] for i,name in enumerate((
            "nose","neck","right_shoulder","right_elbow","right_wrist","left_shoulder",
            "left_elbow","left_wrist","right_hip","right_knee","right_ankle","left_hip"))}
        self.profile = {
            "schema_version":1,"source_sha256":sha256_bytes(self.source.read_bytes()),
            "style_sha256":sha256_bytes(self.style.read_bytes()),"canvas":[64,96],
            "coordinate_system":"image_xy","height_cm":None,"face_box":[16,8,48,40],
            "traits":[dict(name=name,text=name,description_ko=name,evidence="fixture observation",
                           status="operator_observation") for name in ("eyes","skin","hair","clothes","build")],
            "scene_prompt":"standing","style_prompt":"painted",
            "layout":dict(origin="manual_image_annotation",evidence="test fixture",joints=joints),
        }
        self.bundle=self.root/"bundle"
        self.options=ConditioningOptions(.4,.7,.6)

    def prepare(self):
        return prepare_character_style_bundle(self.source,self.style,self.profile,self.bundle)

    def call(self, **overrides):
        self.prepare()
        return build_conditioning_call(self.bundle,ConditioningOptions(.4,.7,.6,**overrides))

    def test_source_unchanged_and_no_false_quality_approval(self):
        before=self.source.read_bytes()
        manifest=self.prepare()
        self.assertEqual(self.source.read_bytes(),before)
        self.assertIsNone(manifest["review"]["user_quality_approval"])
        self.assertFalse(manifest["routing"]["original_rgb_as_initial_latent"])
        with self.assertRaises(FileExistsError):self.prepare()

    def test_wrong_source_sha_rejected_before_output(self):
        self.profile["source_sha256"]="0"*64
        with self.assertRaisesRegex(CharacterInputError,"sha_mismatch"):self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_invalid_annotations_rejected(self):
        for key,value in (("face_box",[-1,0,20,20]),("height_cm",150),("canvas",[65,96])):
            original=deepcopy(self.profile)
            self.profile[key]=value
            with self.subTest(key=key),self.assertRaises(CharacterInputError):self.prepare()
            self.profile=original
        self.assertFalse(self.bundle.exists())

    def test_missing_feature_evidence_rejected(self):
        del self.profile["traits"][0]["evidence"]
        with self.assertRaisesRegex(CharacterInputError,"evidence_missing"):self.prepare()

    def test_missing_joint_origin_rejected(self):
        del self.profile["layout"]["origin"]
        with self.assertRaisesRegex(CharacterInputError,"origin_missing"):self.prepare()

    def test_tampered_face_fails_instead_of_falling_back(self):
        self.prepare()
        (self.bundle/"identity-face.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(CharacterInputError,"sha_mismatch"):build_conditioning_call(self.bundle,self.options)

    def test_manifest_path_escape_is_rejected(self):
        manifest=self.prepare()
        manifest["assets"]["identity"]["file"]="../source.png"
        (self.bundle/"manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(CharacterInputError,"escapes_bundle"):read_verified_bundle(self.bundle)

    def test_actual_masks_and_assets_are_separate(self):
        call=self.call()
        import torch
        masks=call.kwargs["cross_attention_kwargs"]["ip_adapter_masks"]
        self.assertEqual([tuple(m.shape) for m in masks],[(1,1,96,64)]*2)
        self.assertTrue(all(str(m.device)=="cpu" for m in masks))
        self.assertEqual(int(torch.count_nonzero(masks[1])),32*32)
        self.assertEqual(call.kwargs["ip_adapter_image"][1].size,(32,32))
        self.assertNotEqual(call.kwargs["image"].tobytes(),Image.open(self.source).tobytes())
        self.assertNotIn("strength",call.kwargs)
        self.assertEqual(call.evidence["postprocessing"],[])

    def test_disable_identity_zeros_source_reference_without_changing_style(self):
        self.prepare()
        baseline=build_conditioning_call(self.bundle,self.options)
        changed=build_conditioning_call(self.bundle,ConditioningOptions(.4,.7,.6,identity_visual=False))
        self.assertIsNone(changed.kwargs["ip_adapter_image"][1].getbbox())
        self.assertEqual(changed.adapter_scales[1],0)
        self.assertEqual(changed.kwargs["prompt"],baseline.kwargs["prompt"])
        self.assertEqual(changed.kwargs["ip_adapter_image"][0].tobytes(),baseline.kwargs["ip_adapter_image"][0].tobytes())
        self.assertEqual(changed.adapter_scales[0],baseline.adapter_scales[0])

    def test_structure_off_preserves_appearance_references(self):
        call=self.call(structure=False)
        self.assertIsNone(call.kwargs["image"].getbbox())
        self.assertEqual(call.kwargs["controlnet_conditioning_scale"],0)
        self.assertEqual(call.kwargs["ip_adapter_image"][1].getpixel((0,0)),(73,83,93))

    def test_negative_nan_and_boolean_scales_rejected(self):
        for value in (-1,float("nan"),True,1.5):
            with self.subTest(value=value),self.assertRaises(CharacterInputError):ConditioningOptions(value,.5,.5)

    def test_generation_requires_separate_authorization(self):
        pipeline=RecordingPipeline()
        with self.assertRaises(PermissionError):dispatch_conditioning_once(pipeline,self.call())
        self.assertEqual(pipeline.calls,[])

    def test_wrong_adapter_count_stops_before_call(self):
        pipeline=RecordingPipeline();pipeline.unet.encoder_hid_proj.image_projection_layers=[object()]
        with self.assertRaisesRegex(CharacterInputError,"two_ordered"):
            dispatch_conditioning_once(pipeline,self.call(),execution_authorized=True)
        self.assertEqual(pipeline.calls,[])

    def test_img2img_pipeline_rejected_before_call(self):
        class WrongPipeline(RecordingPipeline):
            def __call__(self,prompt,prompt_2,image,ip_adapter_image,cross_attention_kwargs,controlnet_conditioning_scale,strength=.8):
                raise AssertionError("Must not run")
        with self.assertRaisesRegex(CharacterInputError,"text_to_image"):
            dispatch_conditioning_once(WrongPipeline(),self.call(),execution_authorized=True)

    def test_token_overflow_is_not_silently_truncated(self):
        pipeline=RecordingPipeline();pipeline.tokenizer_2.model_max_length=2
        with self.assertRaisesRegex(CharacterInputError,"would_truncate"):
            dispatch_conditioning_once(pipeline,self.call(),execution_authorized=True)
        self.assertEqual(pipeline.calls,[])

    def test_dispatch_uses_prepared_inputs_once_and_never_claims_quality(self):
        pipeline=RecordingPipeline();call=self.call()
        output,evidence=dispatch_conditioning_once(pipeline,call,execution_authorized=True)
        self.assertEqual(output.getpixel((0,0)),(255,165,0))
        self.assertEqual(pipeline.scales,call.adapter_scales)
        self.assertEqual(len(pipeline.calls),1)
        self.assertIsNone(evidence["quality_approval"])
        self.assertEqual(evidence["character_preservation"],"review_needed")
        with self.assertRaisesRegex(CharacterInputError,"already_dispatched"):
            dispatch_conditioning_once(pipeline,call,execution_authorized=True)

    def test_failure_propagates_without_retry_or_original_image_fallback(self):
        pipeline=RecordingPipeline();pipeline.fail=True;call=self.call()
        with self.assertRaisesRegex(RuntimeError,"sentinel"):
            dispatch_conditioning_once(pipeline,call,execution_authorized=True)
        self.assertEqual(len(pipeline.calls),1)
        self.assertEqual(call.evidence["state"],"failed")
        self.assertIn("sentinel",call.evidence["error"])

    def test_cpu_preparation_needs_no_network_or_cuda(self):
        import socket,torch
        def forbidden(*args,**kwargs):raise AssertionError("network/CUDA forbidden")
        self.prepare()
        with patch.object(socket.socket,"connect",forbidden),patch.object(torch.cuda,"_lazy_init",forbidden):
            call=build_conditioning_call(self.bundle,self.options)
        self.assertEqual(call.evidence["model_calls"],0)


    def test_pose_map_does_not_inherit_changed_source_colors(self):
        first=self.prepare()
        with Image.open(self.bundle/"structure-pose.png") as image:
            pose_before=image.tobytes()
        Image.new("RGB",(64,96),(240,10,170)).save(self.source)
        self.profile["source_sha256"]=sha256_bytes(self.source.read_bytes())
        other=self.root/"second-bundle"
        prepare_character_style_bundle(self.source,self.style,self.profile,other)
        with Image.open(other/"structure-pose.png") as image:
            self.assertEqual(pose_before,image.tobytes())
        with Image.open(other/"identity-face.png") as image:
            self.assertEqual(image.getpixel((0,0)),(240,10,170))
