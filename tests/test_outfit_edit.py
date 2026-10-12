"""제품 공통 모듈: 고정 지시문과 캔버스 규칙(GPU 없이)."""
import numpy as np
import pytest
from PIL import Image

from genai_lab import outfit_edit


def test_product_prompt_has_no_garment_description_and_only_accessory_choice_differs():
    keep, remove = outfit_edit.outfit_prompt("keep"), outfit_edit.outfit_prompt("remove")
    assert "coat" not in keep and "coat" not in remove          # 의상은 그림 2를 따른다(시험 지시문의 의상 묘사 제거)
    assert "head accessories" in keep and "same person" in remove
    assert keep.startswith(outfit_edit.OUTFIT_BASE) and remove.startswith(outfit_edit.OUTFIT_BASE)
    with pytest.raises(ValueError):
        outfit_edit.outfit_prompt("maybe")


@pytest.mark.parametrize("size", [(400, 1200), (1200, 400), (768, 1024)])
def test_canvas_keeps_whole_figure_on_white(size):
    image = Image.new("RGB", size, (10, 20, 30))
    out = np.asarray(outfit_edit.canvas(image))
    assert out.shape == (1024, 768, 3)
    painted = (out != 255).any(axis=2)
    ys, xs = np.nonzero(painted)
    w, h = size
    assert abs((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) - w / h) < 0.02   # 비율 유지, 자르지 않음


def test_garment_terms_keep_clothing_and_drop_pose_tags():
    # 제품 의상 분석이 이 정장에서 실제로 낸 태그(outputs/studio-runs/179d…/inputs/analysis.json)
    tags = ['shirt', 'suspenders', 'pants', 'hands in pockets', 'shoes', 'black ribbon', 'coat', 'Coat', 'black_socks']
    terms = outfit_edit.clean_garment_terms(tags)
    assert "hands in pockets" not in terms and terms.count("coat") == 1
    assert {"shoes", "black ribbon", "black socks"} <= set(terms)
    prompt = outfit_edit.outfit_prompt("keep", terms)
    assert "consists of: shirt, suspenders" in prompt and prompt.startswith(outfit_edit.OUTFIT_BASE)
    with pytest.raises(ValueError):
        outfit_edit.outfit_prompt("keep", ("shirt", "hands in pockets"))   # 정리되지 않은 목록은 거부


def test_base_prompt_asks_to_replace_all_current_clothing():
    # 요청 12 W3: "그림 2의 옷을 입혀라"만으로는 몸에 붙는 원래 옷이 남았다
    assert "Replace all of the character's current clothing" in outfit_edit.outfit_prompt("keep")


def figure(arms=True):
    image = np.full((1024, 768, 3), 255, np.uint8)
    image[250:700, 320:450] = (230, 150, 200)                      # 몸통
    if arms:
        import cv2
        cv2.line(image, (320, 270), (250, 420), (230, 150, 200), 18)  # 오른팔(그림 왼쪽)
        cv2.line(image, (250, 420), (240, 560), (240, 200, 190), 14)
        cv2.line(image, (450, 270), (520, 420), (230, 150, 200), 18)
        cv2.line(image, (520, 420), (530, 560), (240, 200, 190), 14)
    return image


def joints():
    def j(name, x, y):
        return {"joint_name": name, "x": x, "y": y, "confidence_score": 0.6, "detected": True}
    return [j("right_shoulder", 320, 270), j("right_elbow", 250, 420), j("right_wrist", 240, 560),
            j("left_shoulder", 450, 270), j("left_elbow", 520, 420), j("left_wrist", 530, 560)]


def test_limb_check_rejects_result_whose_arms_vanished():
    # 요청 12 W3: 재킷·총과 함께 팔이 지워졌는데 관절 인식은 팔 관절을 여전히 추정했다
    kept = outfit_edit.limb_check(joints(), figure(True), joints(), figure(True))
    assert kept["status"] == "ok" and kept["result"] == {"left": [1.0, 1.0], "right": [1.0, 1.0]}
    gone = outfit_edit.limb_check(joints(), figure(True), joints(), figure(False))
    assert gone["status"] == "rejected" and set(gone["missing"]) == {"left", "right"}


def test_limb_check_does_not_judge_arms_not_found_in_original():
    hidden = [dict(j, detected=False) if j["joint_name"].startswith("left") else j for j in joints()]
    report = outfit_edit.limb_check(hidden, figure(True), joints(), figure(False))
    assert report["checked"] == ["right"] and report["missing"] == ["right"]
    report = outfit_edit.limb_check([dict(j, detected=False) for j in joints()], figure(True), joints(), figure(False))
    assert report["status"] == "unmeasured"


def test_white_sleeves_inside_outline_count_as_arms():
    # 요청 13 B-W4: 흰 소매를 배경으로 보고 팔이 사라졌다고 오판했다
    import cv2
    image = figure(False)
    for (a, b) in (((320, 270), (250, 420)), ((250, 420), (240, 560)), ((450, 270), (520, 420)), ((520, 420), (530, 560))):
        cv2.line(image, a, b, (40, 40, 40), 22)          # 윤곽선
        cv2.line(image, a, b, (250, 250, 250), 14)       # 흰 소매 안쪽
    report = outfit_edit.limb_check(joints(), figure(True), joints(), image)
    assert report["status"] == "ok", report


def test_hidden_hand_is_unverified_not_missing():
    # 요청 13 A-W4·B-W3: 손이 주머니에 가려 손목을 못 찾으면 사라짐으로 단정하지 않는다
    pocket = [dict(j, detected=False) if j["joint_name"].endswith("wrist") else j for j in joints()]
    report = outfit_edit.limb_check(joints(), figure(True), pocket, figure(True))
    assert report["status"] == "ok" and report["result"] == {"left": [1.0], "right": [1.0]}
    nothing = [dict(j, detected=False) for j in joints()]
    report = outfit_edit.limb_check(joints(), figure(True), nothing, figure(False))
    assert report["status"] == "unmeasured" and set(report["unverified"]) == {"left", "right"}


def test_low_confidence_joint_segments_are_skipped():
    # 요청 13 A-W3: 손목이 머리카락 위에 낮은 신뢰도로 찍혔다
    wrong = [dict(j, x=700, y=900, confidence_score=0.2) if j["joint_name"].endswith("wrist") else j for j in joints()]
    report = outfit_edit.limb_check(joints(), figure(True), wrong, figure(True))
    assert report["status"] == "ok"


def test_prop_removal_sentence_can_be_left_out_only_explicitly():
    assert outfit_edit.OUTFIT_PROPS in outfit_edit.outfit_prompt("keep")
    assert outfit_edit.OUTFIT_PROPS not in outfit_edit.outfit_prompt("keep", remove_props=False)
