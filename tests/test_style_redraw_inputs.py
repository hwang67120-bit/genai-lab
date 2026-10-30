"""모드 B 준비 도우미(style_redraw_inputs: 얼굴 상자·특징 문장) 시험. 모델·GPU·네트워크 불필요."""
import unittest

from genai_lab.style_redraw_inputs import StyleRedrawInputError, face_box, garment_words, trait_rows


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


if __name__ == "__main__":
    unittest.main()
