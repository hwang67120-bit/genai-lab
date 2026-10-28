"""제품 공통 모듈: 고정 지시문과 캔버스 규칙(GPU 없이)."""
import numpy as np
import pytest
from PIL import Image

from genai_lab import outfit_edit


def test_product_prompt_has_no_garment_description_and_only_accessory_choice_differs():
    keep, remove = outfit_edit.outfit_prompt("keep"), outfit_edit.outfit_prompt("remove")
    assert "consists of" not in keep and "consists of" not in remove   # 의상은 그림 2를 따른다(시험 지시문의 의상 묘사 제거)
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


def test_cream_sleeves_open_to_an_off_white_background_are_still_arms():
    # 요청 15 S04: Qwen 배경은 253 근처인데 "235 이상 = 배경" 규칙이 크림색 소매(227~231)를 배경으로 읽어 양팔 소실로 오판했다
    import cv2
    image = np.full((1024, 768, 3), 253, np.uint8)
    image[250:700, 320:450] = (230, 150, 200)
    for (a, b) in (((320, 270), (250, 420)), ((250, 420), (240, 560)), ((450, 270), (520, 420)), ((520, 420), (530, 560))):
        cv2.line(image, a, b, (231, 227, 223), 18)       # 윤곽선 없는 크림색 소매, 배경과 바로 닿음
    report = outfit_edit.limb_check(joints(), figure(True), joints(), image)
    assert report["status"] == "ok", report
    assert outfit_edit.background_color(image).tolist() == [253, 253, 253]
    gone = outfit_edit.limb_check(joints(), figure(True), joints(), np.full((1024, 768, 3), 253, np.uint8))
    assert gone["status"] == "rejected"                   # 배경이 순백이 아니어도 팔이 없으면 거부


def test_single_missed_sample_is_a_reading_error_not_a_vanished_arm():
    # 요청 14 G19/S07 첫 시도: 12개 표본 중 1개가 흰 소매 픽셀을 배경으로 읽혀 0.917로 거부됐다(팔은 멀쩡함)
    import cv2
    image = figure(True)
    cv2.line(image, (519, 491), (767, 491), (255, 255, 255), 9)     # 왼쪽 아래팔을 가로지르는 흰 틈(테두리까지 이어짐)
    report = outfit_edit.limb_check(joints(), figure(True), joints(), image)
    assert report["status"] == "ok" and report["result"]["left"][1] < 1.0 and report["gaps"]["left"] == [0, 1]
    assert outfit_edit.longest_gap([True, False, False, True, False]) == 2


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


def test_prompt_names_remaining_garments_body_parts_and_pose():
    # 요청 14(D-108): 걸친 코트·넥타이·장갑 잔존, 기계 꼬리·여러 꼬리 소실, 앉은 자세가 선 자세로 바뀜
    for prompt in (outfit_edit.outfit_prompt("keep"), outfit_edit.outfit_prompt("remove")):
        assert "coat draped over the shoulders, tie, gloves, choker" in prompt and "Nothing from the original outfit may remain" in prompt
        assert "If the character has animal ears, a tail or wings" in prompt and "not props" in prompt   # 확인 안 함: 조건문
        assert "Keep the character's pose exactly as in image 1" in prompt
    assert outfit_edit.outfit_prompt("keep").endswith(outfit_edit.OUTFIT_FRAMING)


def test_conditions_carry_source_and_reason_and_rebuild_the_same_prompt():
    # D-112: 조건마다 출처(policy/analysis/user)와 채택 이유, 제외한 조건의 이유가 남아야 "왜 그 문장을 넣었나"에 근거로 답할 수 있다
    assembled = outfit_edit.outfit_conditions("keep", gender="male", body_parts=())
    assert "".join(c["text"] for c in assembled["conditions"]) == outfit_edit.outfit_prompt("keep", gender="male", body_parts=())
    by_key = {c["key"]: c for c in assembled["conditions"]}
    assert by_key["gender"]["source"] == "user" and by_key["gender"]["basis"] == "male"
    assert by_key["body_parts"]["source"] == "user" and by_key["body_parts"]["basis"] == []
    assert by_key["replace_outfit"]["source"] == "policy" and all(c["reason"] for c in assembled["conditions"])
    assert [e["key"] for e in assembled["excluded"]] == ["garment_terms"]
    unknown = outfit_edit.outfit_conditions("keep", remove_props=False)
    keys = {c["key"]: c for c in unknown["conditions"]}
    assert keys["body_parts"]["source"] == "policy" and keys["body_parts"]["basis"] is None   # 확인 안 함 → 조건문
    assert {e["key"] for e in unknown["excluded"]} == {"remove_props", "garment_terms", "gender"}
    facts = outfit_edit.input_facts("female", ("tail",), {"body_parts": {"source": "user_confirmed", "prefill": "tags"}})
    assert facts["body_parts"]["tail"]["state"] == "present" and facts["body_parts"]["wings"]["state"] == "absent"
    assert facts["body_parts"]["ears"]["prefill"] == "tags" and facts["gender"]["source"] == "default"
    assert outfit_edit.input_facts("unspecified", None)["body_parts"]["tail"]["state"] == "unknown"   # 검출 못 함 ≠ 없음


def test_body_parts_sentence_names_only_confirmed_parts_and_forbids_the_rest():
    # 요청 15(D-111): 모든 캐릭터에 "꼬리를 유지하라"고 하자 귀·꼬리·날개가 없는 K에 셋 다 생겼다
    none = outfit_edit.outfit_prompt("keep", body_parts=())
    assert "has no animal ears, no tail and no wings; do not add any" in none and "If the character has" not in none
    tail = outfit_edit.outfit_prompt("keep", body_parts=("tail",))
    assert "The character has a tail or tails; keep them" in tail and "has no animal ears and no wings; do not add any" in tail
    both = outfit_edit.outfit_prompt("keep", body_parts=("tail", "ears"))
    assert "has animal ears and a tail or tails;" in both and "has no wings; do not add any" in both
    assert outfit_edit.outfit_prompt("keep", body_parts=("ears", "tail", "wings")).count("do not add") == 0
    for bad in (("horns",), ("tail", "tail")):
        with pytest.raises(ValueError):
            outfit_edit.outfit_prompt("keep", body_parts=bad)


def test_prop_removal_sentence_can_be_left_out_only_explicitly():
    assert outfit_edit.OUTFIT_PROPS in outfit_edit.outfit_prompt("keep")
    assert outfit_edit.OUTFIT_PROPS not in outfit_edit.outfit_prompt("keep", remove_props=False)


def test_gender_sentence_keeps_male_body_with_womens_outfit():
    # D-106: 남성 캐릭터에 크롭탑·치마를 입히자 여성 체형이 됐다
    male = outfit_edit.outfit_prompt("keep", gender="male")
    assert "male body with a flat chest" in male and "women's clothing" in male
    assert "female" in outfit_edit.outfit_prompt("keep", gender="female")
    assert outfit_edit.outfit_prompt("keep") == outfit_edit.outfit_prompt("keep", gender="unspecified")
    with pytest.raises(ValueError):
        outfit_edit.outfit_prompt("keep", gender="boy")
