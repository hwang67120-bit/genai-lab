"""참조 색 맞춤(D-187 다음 단계): 원본 색으로 옮기되 참조의 명암 변화는 유지, 피부는 옷에서 빠진다."""
import numpy as np

from genai_lab.style_reference_colors import (clusters, lab_to_rgb, match_eyes, match_part, recolor_reference, rgb_to_lab,
                                              without_colours)


def test_lab_round_trip():
    rgb = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert np.abs(lab_to_rgb(rgb_to_lab(rgb)).astype(int) - rgb).max() <= 1


def test_match_moves_colour_and_keeps_shading():
    ref = np.zeros((40, 40, 3), np.uint8)
    ref[:, :20] = (200, 200, 205)                 # light silver with two shades
    ref[:, 20:] = (170, 170, 175)
    src = np.full((40, 40, 3), (25, 22, 20), np.uint8)   # black hair
    mask = np.ones((40, 40), bool)
    lab, record = match_part(rgb_to_lab(ref), mask, rgb_to_lab(src), mask)
    assert record["state"] == "matched"
    out = lab_to_rgb(lab)
    assert out.mean() < 80                           # now dark
    assert out[:, :20].mean() > out[:, 20:].mean()   # light/dark pattern of the reference kept


def test_two_colours_pair_by_lightness():
    lab = rgb_to_lab(np.array([[[30, 60, 30]] * 50 + [[240, 235, 220]] * 50], np.uint8)).reshape(-1, 3)
    label, centers = clusters(lab, 2)
    assert centers[0, 0] < centers[1, 0] and set(label[:50]) == {0} and set(label[50:]) == {1}


def test_skin_is_excluded_from_clothes_and_background_is_plain():
    h, w = 60, 60
    skin = (225, 170, 140)
    ref = np.full((h, w, 3), skin, np.uint8)
    ref[30:, :] = (60, 90, 160)                      # denim below
    src = np.full((h, w, 3), (120, 85, 60), np.uint8)
    src[30:, :] = (70, 70, 72)                       # charcoal trousers
    face = np.zeros((h, w), bool); face[:10, :10] = True
    bottom = np.zeros((h, w), bool); bottom[20:, :] = True   # parser also covers bare skin rows 20..29
    parts = dict(face=face, bottomwear=bottom)
    alpha = np.ones((h, w)); alpha[:, :5] = 0
    plan = {"lower": dict(reference=["bottomwear"], source=["bottomwear"])}
    out, record = recolor_reference(ref, parts, src, dict(face=face, bottomwear=bottom), plan, alpha)
    assert record["lower"]["state"] == "matched"
    assert np.abs(out[22, 30].astype(int) - skin).max() <= 3          # bare skin inside the parser mask untouched
    assert out[45, 30, 2] < 110                                      # denim blue gone
    assert tuple(out[45, 2]) == (214, 210, 202)                       # background plain


def test_shades_of_one_colour_merge_and_largest_picks_the_wide_colour():
    lab = np.array([[18.0, -7.2, 9.6]] * 60 + [[47.8, -11.1, 14.7]] * 60)    # T014 vest, dark and light side (measured)
    assert len(clusters(lab, 2)[1]) == 1                               # one olive vest in two shades
    assert len(clusters(lab, 2, merge_ab=0.0)[1]) == 2                 # the reference side keeps them apart
    ref = np.full((20, 20, 3), (60, 90, 160), np.uint8)                # denim
    src = np.full((20, 20, 3), (80, 78, 78), np.uint8)                 # wide grey trousers
    src[:4] = (64, 80, 50)                                             # small olive vest hem
    mask = np.ones((20, 20), bool)
    out, record = match_part(rgb_to_lab(ref), mask, rgb_to_lab(src), mask, source="largest")
    colour = lab_to_rgb(out)[10, 10].astype(int)
    assert record["source_colours"] == 2 and abs(colour[0] - colour[2]) < 12 and abs(colour[1] - colour[2]) < 12


def test_reference_light_and_dark_garments_pair_with_shirt_and_vest():
    ref = np.zeros((20, 40, 3), np.uint8)
    ref[:, :20] = (235, 235, 238)                                      # white crop top
    ref[:, 20:] = (40, 40, 48)                                         # dark jacket / print
    src = np.zeros((20, 40, 3), np.uint8)
    src[:, :20] = (60, 80, 45)                                         # olive vest
    src[:, 20:] = (240, 235, 220)                                      # ivory shirt
    mask = np.ones((20, 40), bool)
    out = lab_to_rgb(match_part(rgb_to_lab(ref), mask, rgb_to_lab(src), mask, k=2)[0]).astype(int)
    assert out[10, 5, 0] > 180 and out[10, 5, 2] < out[10, 5, 0]       # white top -> ivory
    assert out[10, 30, 1] > out[10, 30, 2] and out[10, 30].mean() < 120  # dark garment -> olive


def test_lower_skips_the_upper_colour_unless_nothing_else_is_left():
    lab = rgb_to_lab(np.array([[[120, 30, 60]] * 30 + [[110, 25, 55]] * 30 + [[90, 100, 120]] * 10], np.uint8))  # coat, coat shade, trousers
    coat = rgb_to_lab(np.array([[[120, 30, 60]]], np.uint8)).reshape(3)
    keep = without_colours(lab, np.ones(lab.shape[:2], bool), [coat])
    assert keep.sum() == 10                                             # only the trousers stay
    grey, ivory = rgb_to_lab(np.array([[[80, 78, 78], [240, 235, 220]]], np.uint8))[0]
    assert without_colours(grey[None, None], np.ones((1, 1), bool), [ivory]).all()   # grey trousers are not "ivory"


def test_outline_pixels_are_not_a_garment_colour():
    ref = np.full((20, 20, 3), (60, 90, 160), np.uint8)
    src = np.full((20, 20, 3), (5, 5, 6), np.uint8)                    # mostly black outlines
    src[:6] = (95, 105, 125)                                           # narrow blue-grey trousers
    mask = np.ones((20, 20), bool)
    out = lab_to_rgb(match_part(rgb_to_lab(ref), mask, rgb_to_lab(src), mask, source="largest")[0])
    assert out.mean() > 60                                             # trousers colour, not black


def test_uncertain_lower_colour_becomes_neutral_grey():
    ref = np.full((50, 50, 3), (60, 90, 160), np.uint8)                # denim shorts
    ref[25:] = (40, 60, 120)                                           # shaded half
    src = np.full((50, 50, 3), (230, 228, 225), np.uint8)              # background (outside the mask)
    src[:2, :10] = (95, 105, 125)                                      # a sliver of trousers: 0.8% of the image
    mask = np.zeros((50, 50), bool); mask[:2, :10] = True
    full = np.ones((50, 50), bool)
    out, record = match_part(rgb_to_lab(ref), full, rgb_to_lab(src), mask, source="largest", min_fraction=0.02)
    rgb = lab_to_rgb(out).astype(int)
    assert record["state"] == "neutral_uncertain"
    assert np.abs(rgb[10, 10] - rgb[10, 10].mean()).max() <= 2         # grey, no denim blue
    assert rgb[10, 10].mean() > rgb[40, 10].mean()                    # reference light/dark kept


def test_eyes_take_each_side_colour_and_keep_white_and_pupil():
    def two_eyes(left, right):
        img = np.full((40, 80, 3), 235, np.uint8)                       # skin-ish white background
        for x0, colour in ((10, left), (50, right)):
            img[10:30, x0:x0 + 20] = (245, 245, 245)                    # sclera
            img[14:26, x0 + 4:x0 + 16] = colour                         # iris
            img[18:22, x0 + 8:x0 + 12] = (15, 15, 20)                   # pupil
        return img
    ref = two_eyes((60, 120, 220), (60, 120, 220))                      # blue / blue
    src = two_eyes((220, 160, 40), (60, 120, 220))                      # gold / blue (odd eyes)
    mask = np.zeros((40, 80), bool); mask[10:30, 10:30] = True; mask[10:30, 50:70] = True
    lab, record = match_eyes(rgb_to_lab(ref), mask, rgb_to_lab(src), mask)
    out = lab_to_rgb(lab).astype(int)
    assert record["state"] == "matched"
    assert out[15, 15, 0] > out[15, 15, 2] + 60                         # image-left iris now gold
    assert out[15, 55, 2] > out[15, 55, 0] + 60                         # image-right iris stays blue
    assert out[11, 11].min() > 230 and out[20, 20].max() < 40           # sclera and pupil untouched
