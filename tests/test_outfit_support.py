"""지원 범위 경고(D-113, GPU 없이): 알려진 실패 유형의 특징만, 근거 사례와 함께."""
from genai_lab import outfit_support


def test_known_failure_features_raise_warnings_with_evidence():
    # 요청 14~16에서 실제로 측정한 태그 점수(S08 앉음·꼬리 여럿, K 걸친 재킷 0.39, S06 기계 꼬리 0.72)
    s08 = outfit_support.support_warnings({"sitting": 0.69, "multiple_tails": 0.91, "tail": 0.97}, ("ears", "tail"))
    assert [w["key"] for w in s08] == ["pose_not_standing", "multiple_tails"] and s08[0]["level"] == "outside"
    assert s08[0]["evidence"] == {"sitting": 0.69} and "S08" in s08[0]["basis"]
    k = outfit_support.support_warnings({"jacket": 0.75, "jacket_on_shoulders": 0.39}, ())
    assert [w["key"] for w in k] == ["draped_outerwear"]
    s06 = outfit_support.support_warnings({"mechanical_tail": 0.72, "bodysuit": 0.69}, ("tail",))
    assert [w["key"] for w in s06] == ["mechanical_body", "tail_shape"] and s06[1]["level"] == "notice"


def test_ordinary_standing_character_gets_no_warning():
    # W2·K(요청 14 통과분)에 해당하는 점수: 서 있고 걸친 겉옷·기계 몸 없음
    assert outfit_support.support_warnings({"standing": 0.68, "1girl": 1.0, "jacket": 0.3}, ()) == []
    assert outfit_support.support_warnings({}, None) == []
    assert outfit_support.support_summary([]) == ""
    text = outfit_support.support_summary(outfit_support.support_warnings({"sitting": 0.9}, ("tail",)))
    assert text.startswith("지원 범위 밖 특징 1개") and "참고:" in text and "실패하면 알려 드립니다" in text
