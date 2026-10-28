"""자유 생성 자세 유도(D-131, 설계 docs/FREE_GENERATION_POSE_GUIDANCE.md): 지원 자세 목록 + 캐릭터 비율에 맞춘 골격 그림.

- 자세 틀: 뼈마다 (각도, 길이 배율). 각도는 화면 아래 방향이 0°, 화면 오른쪽으로 돌면 +. 길이 배율은 앞뒤로 줄어 보이는 정도.
  캐릭터의 right_* 관절은 정면 그림에서 화면 왼쪽에 있다(DWPose/OpenPose 관례).
- 비율 맞춤: 기준 그림 관절에서 뼈 길이를 재고 틀의 각도로 다시 잇는다(자세는 틀, 길이는 캐릭터). 발 높이를 기준 그림과 맞춘다.
- 그림: OpenPose 18관절 색 규칙으로 코드가 그린다(다운로드 없음). 같은 관절이면 같은 그림 — 작업 프로세스가 다시 그려 대조한다.
- 결과 검사: 결과 관절의 뼈 각도와 틀 각도 차이. 다리·팔 개수 이상은 여전히 못 잡는다(관절 검출이 한 사람분).
"""
import math
import re

import numpy as np

JOINTS = ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",
          "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye",
          "right_ear", "left_ear")
LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17))
COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85))
ARM_LEG_BONES = (("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
                 ("left_elbow", "left_wrist"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                 ("left_hip", "left_knee"), ("left_knee", "left_ankle"))
# 자세 틀: 위팔·아래팔·허벅지·정강이 (각도, 길이 배율). 정면 기준. 화면 왼쪽 = 캐릭터 오른쪽.
POSE_TEMPLATES = {
    # 캐릭터 오른쪽(화면 왼쪽) 팔다리가 바깥으로 벌어지면 각도 −, 안쪽(몸 중앙)으로 오면 +. 왼쪽은 반대.
    "standing": dict(r_upper=(-12, 1), r_fore=(-6, 1), l_upper=(12, 1), l_fore=(6, 1),
                     r_thigh=(-4, 1), r_shin=(-2, 1), l_thigh=(4, 1), l_shin=(2, 1)),
    # 앉기·무릎 꿇기는 3/4 시점(화면 왼쪽을 향함): 정면에서는 허벅지가 짧은 선이라 선 자세와 구분되지 않는다.
    "sitting_stool": dict(r_upper=(-15, 1), r_fore=(-55, 0.9), l_upper=(-5, 1), l_fore=(-50, 0.9),         # 손은 무릎 위
                          r_thigh=(-80, 0.8), r_shin=(-5, 1), l_thigh=(-75, 0.75), l_shin=(-2, 1)),     # 허벅지 수평
    "kneeling_one_knee": dict(r_upper=(-5, 1), r_fore=(-5, 1), l_upper=(-20, 1), l_fore=(-60, 0.9),       # 한 손은 세운 무릎 위
                              r_thigh=(-5, 0.95), r_shin=(80, 0.7), l_thigh=(-80, 0.8), l_shin=(0, 1)),  # 오른 무릎 바닥·정강이 뒤로, 왼 무릎 세움
    "walking": dict(r_upper=(-14, 1), r_fore=(-6, 1), l_upper=(10, 0.95), l_fore=(16, 0.9),
                    r_thigh=(-6, 1), r_shin=(-2, 0.95), l_thigh=(8, 0.9), l_shin=(4, 1)),
    "arms_crossed": dict(r_upper=(-10, 0.9), r_fore=(100, 0.75), l_upper=(10, 0.9), l_fore=(-100, 0.75),
                         r_thigh=(-4, 1), r_shin=(-2, 1), l_thigh=(4, 1), l_shin=(2, 1)),
}
# 요청 문장 → 지원 자세(낱말 규칙). 순서 중요: 앞의 것이 먼저 맞으면 그것.
POSE_WORDS = (("kneeling_one_knee", r"\bkneel(?:s|ing)?\b|무릎(?:을)?\s*꿇"),
              ("sitting_stool", r"\bsit(?:s|ting)?\b|\bseated\b|\bstool\b|\bchair\b|앉"),
              ("walking", r"\bwalk(?:s|ing)?\b|걷|걸어"),
              ("arms_crossed", r"\barms?\s+crossed\b|\bcrossed\s+arms\b|\bcross(?:es|ing)?\s+(?:her|his|their|the)\s+arms\b|팔짱"),
              ("standing", r"\bstand(?:s|ing)?\b|서\s*있|서기|서서"))
DEFAULT_RATIO = dict(upper=0.115, fore=0.105, thigh=0.16, shin=0.16, torso=0.22, shoulder=0.07, hip=0.045)   # 캔버스 높이 비
SKELETON_PROMPT = "Follow the body pose shown by the colored skeleton in image 3, with exactly two arms and two legs. "
MATCH_REVIEW = 35            # 결과 뼈 각도가 틀과 이만큼(도) 넘게 다르면 확인 필요


def classify_pose(text):
    """요청 문장(실제 연출 문장)을 지원 자세로. 못 고르면 None(지원 범위 밖 → 글 지시만, 보장 없음)."""
    for key, pattern in POSE_WORDS:
        if re.search(pattern, text or "", re.I):
            return key
    return None


def _points(joints):
    found = {}
    for j in joints or ():
        if j.get("detected") and j.get("joint_name") in JOINTS:
            found[j["joint_name"]] = np.array([float(j["x"]), float(j["y"])])
    return found


def _length(points, a, b):
    return float(np.linalg.norm(points[a] - points[b])) if a in points and b in points else None


def body_lengths(joints, height):
    """기준 그림 관절에서 뼈 길이(픽셀). 못 잰 뼈는 반대쪽 → 표준 비율. 출처를 함께 돌려준다."""
    p = _points(joints)
    measured = {}
    for key, pairs in (("upper", (("right_shoulder", "right_elbow"), ("left_shoulder", "left_elbow"))),
                       ("fore", (("right_elbow", "right_wrist"), ("left_elbow", "left_wrist"))),
                       ("thigh", (("right_hip", "right_knee"), ("left_hip", "left_knee"))),
                       ("shin", (("right_knee", "right_ankle"), ("left_knee", "left_ankle")))):
        values = [v for v in (_length(p, a, b) for a, b in pairs) if v]
        if values:
            measured[key] = float(np.mean(values))
    if "neck" in p and ("right_hip" in p or "left_hip" in p):
        hips = [p[k] for k in ("right_hip", "left_hip") if k in p]
        measured["torso"] = float(np.linalg.norm(np.mean(hips, 0) - p["neck"]))
    for key, (a, b) in (("shoulder", ("right_shoulder", "left_shoulder")), ("hip", ("right_hip", "left_hip"))):
        value = _length(p, a, b)
        if value:
            measured[key] = value / 2
    lengths = {k: measured.get(k, DEFAULT_RATIO[k] * height) for k in DEFAULT_RATIO}
    return lengths, dict(measured=sorted(measured), defaulted=sorted(set(DEFAULT_RATIO) - set(measured)))


def _step(origin, angle, length):
    rad = math.radians(angle)
    return origin + np.array([math.sin(rad), math.cos(rad)]) * length


def retarget(pose_key, joints, size):
    """자세 틀을 기준 캐릭터 비율로. 반환: 18관절 기록(x, y, detected) 목록과 근거."""
    width, height = size
    template = POSE_TEMPLATES[pose_key]
    lengths, provenance = body_lengths(joints, height)
    base = _points(joints)
    neck = base.get("neck", np.array([width / 2, height * 0.22]))
    out = {"neck": neck}
    out["right_shoulder"] = neck + np.array([-lengths["shoulder"], lengths["shoulder"] * 0.15])
    out["left_shoulder"] = neck + np.array([lengths["shoulder"], lengths["shoulder"] * 0.15])
    hip_center = neck + np.array([0, lengths["torso"]])
    out["right_hip"] = hip_center + np.array([-lengths["hip"], 0])
    out["left_hip"] = hip_center + np.array([lengths["hip"], 0])
    for side, s in (("right", "r"), ("left", "l")):
        out[f"{side}_elbow"] = _step(out[f"{side}_shoulder"], template[f"{s}_upper"][0], lengths["upper"] * template[f"{s}_upper"][1])
        out[f"{side}_wrist"] = _step(out[f"{side}_elbow"], template[f"{s}_fore"][0], lengths["fore"] * template[f"{s}_fore"][1])
        out[f"{side}_knee"] = _step(out[f"{side}_hip"], template[f"{s}_thigh"][0], lengths["thigh"] * template[f"{s}_thigh"][1])
        out[f"{side}_ankle"] = _step(out[f"{side}_knee"], template[f"{s}_shin"][0], lengths["shin"] * template[f"{s}_shin"][1])
    for name in ("nose", "right_eye", "left_eye", "right_ear", "left_ear"):        # 머리는 기준 그림 그대로(목 기준 상대 위치)
        if name in base:
            out[name] = neck + (base[name] - base.get("neck", neck))
    feet = [base[k][1] for k in ("right_ankle", "left_ankle") if k in base]          # 발 높이를 기준 그림에 맞춘다
    lowest = max(out["right_ankle"][1], out["left_ankle"][1])
    shift = (float(np.mean(feet)) - lowest) if feet else 0.0
    pts = {k: v + np.array([0, shift]) for k, v in out.items()}
    top = min(v[1] for v in pts.values())
    if top < 4:                                                                       # 위로 넘치면 내린다
        pts = {k: v + np.array([0, 4 - top]) for k, v in pts.items()}
    records = []
    for name in JOINTS:
        if name in pts:
            x, y = (float(np.clip(pts[name][0], 0, width - 1)), float(np.clip(pts[name][1], 0, height - 1)))
            records.append(dict(joint_name=name, x=round(x, 1), y=round(y, 1), detected=True))
        else:
            records.append(dict(joint_name=name, x=0.0, y=0.0, detected=False))
    return records, dict(pose_key=pose_key, lengths={k: round(v, 1) for k, v in lengths.items()}, **provenance,
                         foot_shift=round(float(shift), 1))


def draw_skeleton(records, size, stick=4):
    """OpenPose 18관절 색 규칙의 RGB 골격(검은 바탕). 같은 기록이면 같은 픽셀."""
    import cv2
    width, height = size
    canvas = np.zeros((height, width, 3), np.uint8)
    pts = {i: (r["x"], r["y"]) for i, r in enumerate(records) if r["detected"]}
    for index, (a, b) in enumerate(LIMBS):
        if a not in pts or b not in pts:
            continue
        (x1, y1), (x2, y2) = pts[a], pts[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x1 - x2, y1 - y2)
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        polygon = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
        layer = canvas.copy()
        cv2.fillConvexPoly(layer, polygon, COLORS[index])
        canvas = (canvas * 0.4 + layer * 0.6).astype(np.uint8)
    for index, (x, y) in pts.items():
        cv2.circle(canvas, (int(x), int(y)), 4, COLORS[index], thickness=-1)
    return canvas


def bone_angle(points, a, b):
    if a not in points or b not in points:
        return None
    d = points[b] - points[a]
    return math.degrees(math.atan2(d[0], d[1]))


def compare_pose(skeleton_records, result_joints, size=None):
    """결과 관절의 팔·다리 뼈 각도 vs 골격. 확인 필요(차이 큼) / 일치 / 확인 불가(관절 부족)."""
    target, found = _points(skeleton_records), _points(result_joints)
    rows = []
    for a, b in ARM_LEG_BONES:
        want, got = bone_angle(target, a, b), bone_angle(found, a, b)
        if want is None or got is None:
            rows.append(dict(bone=f"{a}-{b}", status="unmeasurable"))
            continue
        gap = abs((got - want + 180) % 360 - 180)
        rows.append(dict(bone=f"{a}-{b}", gap=round(gap, 1), status="mismatch" if gap > MATCH_REVIEW else "match"))
    measured = [r for r in rows if r["status"] != "unmeasurable"]
    if len(measured) < 4:
        status = "unmeasurable"
    else:
        status = "mismatch" if any(r["status"] == "mismatch" for r in measured) else "match"
    return dict(status=status, bones=rows, threshold=MATCH_REVIEW,
                note="관절 검출은 한 사람분이라 다리·팔 개수 이상은 이 비교로 잡히지 않는다(사람 확인)")
