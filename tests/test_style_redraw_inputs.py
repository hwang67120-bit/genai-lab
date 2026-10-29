"""모드 B CPU 준비(style_redraw_inputs) 시험. 모델·GPU·네트워크 불필요."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from genai_lab.character_style_inputs import CharacterInputError, prepare_character_style_bundle, sha256_bytes
from genai_lab.character_style_conditioning import ConditioningOptions, build_conditioning_call, build_feature_prompts
from genai_lab.style_redraw_inputs import (
    CANVAS, HAIR_SCALE, HEAD_BOX, OUTFIT_BACKGROUND, OUTFIT_BOX, OUTFIT_SPLIT_BOXES, PROPORTION_DEFAULTS, SCALES, STANDARD_POSE,
    StyleRedrawInputError, build_annotations, face_box, fitted_pose, framing_for, garment_words, hair_reference, measured_proportions,
    outfit_reference, outfit_references, trait_rows,
)


def features(eye=None, skin_l=None, length=None, colours=None, styles=None):
    def item(ok, **values):
        return dict(state="measured" if ok else "unknown", **values)
    return {"items": {
        "eye_color": item(eye, tags=eye or []),
        "skin": item(skin_l is not None, color=None if skin_l is None else {"lab": [skin_l, 10, 15]}),
        "hair_length": item(length, value=length),
        "hair_color": item(colours, tags=colours or []),
        "hair_style": item(styles, values=styles or []),
    }}


class TestStyleRedrawInputs(unittest.TestCase):
    def test_face_box_square_around_confident_points(self):
        points = [[100, 100], [140, 100], [120, 130], [110, 150], [130, 150], [0, 0]]
        scores = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1]          # 마지막 점은 점수가 낮아 뺀다
        # 중심 (120,126), 한 변 1.6 x 50 = 80
        self.assertEqual(face_box(points, scores, (768, 1024)), [80, 86, 160, 166])
        with self.assertRaises(StyleRedrawInputError):
            face_box(points[:3], scores[:3], (768, 1024))

    def test_outfit_reference_cuts_above_neck_on_grey(self):
        image = Image.new("RGB", (100, 200), (200, 30, 30))
        alpha = np.zeros((200, 100), np.uint8)
        alpha[20:180, 30:70] = 255                       # 사람 영역
        out, box = outfit_reference(image, alpha, top=56)
        self.assertEqual(box, [22, 48, 77, 187])         # 56에서 자르고, 마지막 픽셀(69,179)에서 8px 여유
        arr = np.asarray(out)
        self.assertEqual(tuple(arr[0, 0]), OUTFIT_BACKGROUND)
        self.assertEqual(tuple(arr[20, 20]), (200, 30, 30))
        with self.assertRaises(StyleRedrawInputError):
            outfit_reference(image, alpha, top=199)

    def test_proportions_measured_clipped_and_defaulted(self):
        joints = {"neck": [400, 300], "left_shoulder": [500, 310], "right_shoulder": [300, 310],
                  "left_hip": [470, 640], "right_hip": [330, 640]}
        props, source = measured_proportions(joints, [330, 100, 470, 240])     # 얼굴 140 / 어깨 200
        self.assertEqual((props["head"], props["torso"], props["hip"]), (0.7, 1.7, 0.7))
        self.assertEqual((source["leg"], props["leg"]), ("default", PROPORTION_DEFAULTS["leg"]))
        wide = dict(joints, left_hip=[900, 640], right_hip=[-100, 640])
        self.assertEqual(measured_proportions(wide, [330, 100, 470, 240])[0]["hip"], 0.9)   # 범위 위로 자름
        self.assertEqual(measured_proportions({}, [0, 0, 10, 10])[0], PROPORTION_DEFAULTS)

    def test_fitted_pose_follows_proportions_and_fits_canvas(self):
        small, _ = fitted_pose(dict(PROPORTION_DEFAULTS, head=0.6))
        big, boxes = fitted_pose(dict(PROPORTION_DEFAULTS, head=1.2))
        width = lambda j: j["left_shoulder"][0] - j["right_shoulder"][0]
        self.assertGreater(width(small), width(big))          # 머리가 크면 같은 키에서 어깨가 좁다
        for j in (small, big):
            self.assertAlmostEqual((j["left_ankle"][1] + j["right_ankle"][1]) / 2, 960, delta=1)
            self.assertTrue(all(0 <= x < 768 and 0 <= y < 1024 for x, y in j.values()))
        self.assertTrue(set(STANDARD_POSE) == set(big))
        face = boxes["face"]
        self.assertTrue(face[0] < big["nose"][0] < face[2] and face[1] < big["nose"][1] < face[3])
        self.assertLess(boxes["upper"][1], big["right_shoulder"][1])
        self.assertLessEqual(boxes["head"][3], big["neck"][1])     # 머리 참조는 어깨에 걸치지 않는다
        self.assertGreater(boxes["lower"][3], big["right_ankle"][1])

    def test_knee_up_framing_when_original_has_no_ankles(self):
        self.assertEqual(framing_for({"left_knee": [0, 0]}), "knee_up")
        self.assertEqual(framing_for({"right_ankle": [0, 0]}), "full")
        joints, boxes = fitted_pose(PROPORTION_DEFAULTS, framing="knee_up")
        self.assertNotIn("left_ankle", joints)
        self.assertAlmostEqual((joints["left_knee"][1] + joints["right_knee"][1]) / 2, 1000, delta=1)
        self.assertTrue(all(0 <= x < 768 and 0 <= y < 1024 for x, y in joints.values()))
        self.assertEqual(boxes["lower"][3], 1024)
        full, _ = fitted_pose(PROPORTION_DEFAULTS)
        width = lambda j: j["left_shoulder"][0] - j["right_shoulder"][0]
        self.assertGreater(width(joints), width(full))          # 무릎 위 구도는 사람이 더 크게 보인다
        with self.assertRaises(StyleRedrawInputError):
            fitted_pose(PROPORTION_DEFAULTS, framing="waist_up")

    def test_face_mask_goes_to_the_generated_face_place(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, style = root / "s.png", root / "t.png"
            Image.new("RGB", CANVAS, (1, 1, 1)).save(source); Image.new("RGB", CANVAS, (2, 2, 2)).save(style)
            joints, boxes = fitted_pose(PROPORTION_DEFAULTS)
            ann = build_annotations(sha256_bytes(source.read_bytes()), sha256_bytes(style.read_bytes()), [], [10, 10, 110, 110],
                                    trait_rows(features(), {}, None), None, joints, boxes["face"])
            prepare_character_style_bundle(source, style, ann, root / "bundle")
            with Image.open(root / "bundle" / "identity-mask.png") as mask:
                self.assertEqual(mask.getbbox(), tuple(boxes["face"]))
            with Image.open(root / "bundle" / "identity-face.png") as crop:
                self.assertEqual(crop.size, (100, 100))      # 자르기는 원본 얼굴 상자

    def test_hair_reference_crops_hair_and_face_with_margin(self):
        image = Image.new("RGB", (400, 1000), (20, 20, 20))
        alpha = np.zeros((1000, 400), np.uint8); alpha[50:900, 100:300] = 255
        hair = np.zeros((1000, 400), bool); hair[60:200, 120:280] = True
        face = np.zeros((1000, 400), bool); face[150:260, 150:250] = True
        out, box = hair_reference(image, alpha, hair, face)
        self.assertEqual(box, [80, 20, 319, 299])          # 머리+얼굴 (120..279, 60..259) 둘레 1000*0.04 = 40
        self.assertEqual(tuple(np.asarray(out)[5, 5]), OUTFIT_BACKGROUND)   # 사람 밖은 회색 바탕
        with self.assertRaises(StyleRedrawInputError):
            hair_reference(image, alpha, np.zeros_like(hair), np.zeros_like(face))

    def test_outfit_split_into_upper_and_lower_at_the_hips(self):
        image = Image.new("RGB", (400, 1000), (200, 30, 30))
        alpha = np.zeros((1000, 400), np.uint8)
        alpha[100:950, 150:250] = 255
        pieces = outfit_references(image, alpha, neck_y=200, hip_y=500)
        self.assertEqual([p[0] for p in pieces], ["upper", "lower"])
        self.assertEqual(pieces[0][2], [142, 172, 257, 557])   # 목 200 - 20 = 180 ~ 엉덩이 500 + 50 = 550(마지막 행 549)
        self.assertEqual(pieces[1][2], [142, 462, 257, 957])   # 엉덩이 500 - 30 = 470부터
        self.assertEqual([p[3] for p in pieces], [OUTFIT_SPLIT_BOXES["upper"], OUTFIT_SPLIT_BOXES["lower"]])
        self.assertEqual([p[0] for p in outfit_references(image, alpha, neck_y=200)], ["body"])
        short = alpha.copy(); short[480:] = 0                  # 허벅지 위에서 잘린 그림: 아래옷 조각 없음 → 몸 전체 한 장
        only = outfit_references(image, short, neck_y=200, hip_y=500)
        self.assertEqual([(p[0], p[3]) for p in only], [("body", OUTFIT_BOX)])

    def test_traits_from_measured_features_and_unknown_left_empty(self):
        rows = {r["name"]: r for r in trait_rows(
            features(eye=["purple_eyes"], skin_l=80, length="long_hair", colours=["brown_hair"], styles=["twin_braids"]),
            {"purple_shirt": 0.9, "belt": 0.6, "blue_pants": 0.7}, "female")}
        self.assertEqual(rows["eyes"]["text"], "purple eyes")
        self.assertEqual(rows["skin"]["text"], "fair skin")
        self.assertEqual(rows["hair"]["text"], "long brown hair, twin braids")
        self.assertEqual(rows["clothes"]["text"], "purple shirt, blue pants, belt")
        dreads = {r["name"]: r for r in trait_rows(features(length="long_hair", colours=["black_hair"]),
                                                   {"dreadlocks": 0.8, "undercut": 0.3}, "male")}
        self.assertEqual(dreads["hair"]["text"], "long black hair, dreadlocks")
        self.assertEqual(rows["build"]["text"], "adult woman")
        empty = {r["name"]: r for r in trait_rows(features(), {}, "unknown")}
        self.assertTrue(all(r["text"] == "" and r["status"] == "unmeasured" for r in empty.values()))

    def test_skin_words_by_lightness(self):
        for l, word in ((30, "dark brown skin"), (50, "brown skin"), (65, ""), (90, "fair skin")):
            with self.subTest(l=l):
                rows = {r["name"]: r for r in trait_rows(features(skin_l=l), {}, "male")}
                self.assertEqual(rows["skin"]["text"], word)

    def test_garment_words_skip_pose_and_head_accessories_and_cap_count(self):
        scores = {"hands_in_pockets": 0.9, "hair_ornament": 0.9, "shirt": 0.8, "vest": 0.85, "pants": 0.5,
                  "jacket": 0.45, "belt": 0.44, "boots": 0.43, "necktie": 0.42, "gloves": 0.41, "scarf": 0.1}
        words = garment_words(scores)
        self.assertNotIn("hands in pockets", words)
        self.assertNotIn("hair ornament", words)
        self.assertNotIn("scarf", words)
        self.assertLessEqual(len(words), 6)
        self.assertEqual(words[:2], ["vest", "shirt"])
        coloured = garment_words({"pants": 0.95, "shirt": 0.9, "vest": 0.85, "long_sleeves": 0.8, "collared_shirt": 0.7,
                                  "white_shirt": 0.6, "green_vest": 0.5, "grey_pants": 0.45})
        self.assertEqual(coloured, ["white shirt", "green vest", "grey pants", "long sleeves", "collared shirt"])

    def test_annotations_build_a_valid_three_reference_bundle(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, style, outfit = root / "source.png", root / "style.png", root / "outfit.png"
            Image.new("RGB", CANVAS, (90, 60, 40)).save(source)
            Image.new("RGB", CANVAS, (200, 200, 200)).save(style)
            Image.new("RGB", (300, 700), (10, 120, 90)).save(outfit)
            traits = trait_rows(features(eye=["green_eyes"]), {"shirt": 0.9}, "female")
            digest = sha256_bytes(outfit.read_bytes())
            ann = build_annotations(sha256_bytes(source.read_bytes()), sha256_bytes(style.read_bytes()),
                                    [("upper", digest, OUTFIT_SPLIT_BOXES["upper"]), ("lower", digest, OUTFIT_SPLIT_BOXES["lower"]),
                                     ("hair", digest, HEAD_BOX, HAIR_SCALE)],
                                    [300, 100, 460, 260], traits, "female")
            prepare_character_style_bundle(source, style, ann, root / "bundle", {"upper": outfit, "lower": outfit, "hair": outfit})
            call = build_conditioning_call(root / "bundle", ConditioningOptions(**SCALES))
            self.assertEqual(len(call.kwargs["ip_adapter_image"]), 5)
            self.assertEqual(call.adapter_scales[2:], [0.8, 0.8, 0.7])
            self.assertEqual(call.evidence["adapter_order"][2:], ["outfit_upper", "outfit_lower", "outfit_hair"])
            self.assertEqual(call.kwargs["prompt"],
                             "Full body, standing by a window. green eyes; shirt; adult woman. Painted illustration, blended shadows, nuanced colors")
            self.assertTrue(call.kwargs["negative_prompt"].endswith(", man, male, masculine"))

    def test_unmeasured_trait_cannot_carry_text_and_empty_traits_leave_prompt_clean(self):
        profile = dict(scene_prompt="standing", style_prompt="painted",
                       traits=[dict(name="eyes", text="", status="unmeasured")])
        self.assertEqual(build_feature_prompts(profile)[0], "standing. painted")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, style, outfit = root / "s.png", root / "t.png", root / "o.png"
            for p, c in ((source, 1), (style, 2), (outfit, 3)):
                Image.new("RGB", CANVAS if p != outfit else (10, 10), (c, c, c)).save(p)
            traits = trait_rows(features(), {}, "unknown")
            traits[0]["text"] = "blue eyes"                  # unmeasured인데 문장이 있음
            ann = build_annotations(sha256_bytes(source.read_bytes()), sha256_bytes(style.read_bytes()),
                                    [("body", sha256_bytes(outfit.read_bytes()), OUTFIT_BOX)], [300, 100, 460, 260], traits)
            with self.assertRaisesRegex(CharacterInputError, "unmeasured_trait_has_text"):
                prepare_character_style_bundle(source, style, ann, root / "bundle", {"body": outfit})


if __name__ == "__main__":
    unittest.main()
