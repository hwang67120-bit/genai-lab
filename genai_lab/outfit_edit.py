"""원본 편집형 의상 교체의 공통 부분(D-090~D-093, D-094 제품 연결): Qwen 편집 결과를 원본 좌표로 되돌리고,
머리(머리카락·얼굴·선택한 머리 장식)만 원본 픽셀로 붙인다. 시험 실행기(scripts/preserve_qwen_edit.py, Qwen 환경)와
제품(메인 환경 GUI·Qwen 작업 프로세스)이 같은 코드를 쓴다. torch를 가져오지 않는다.

머리 정렬: 원본 머리 주변과 Qwen 결과 전체에서 특징점(SIFT)을 맞춰 위치·크기·기울기를 잰다. 얼굴 경계 25px 안 일치점이
8개 이상일 때만 믿는다. Qwen 결과 전체를 원본 좌표로 되돌린 뒤(필요하면 흰 여백으로 화면을 넓힘) 다시 재서 거의 일치할 때만
붙인다. 위치가 맞아도 머리 형태·연결(지지율·떠 있는 조각)이 다르면 붙이지 않는다. 붙이지 않은 장은 raw만 남긴다.
"""
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = (768, 1024)
FEATHER = 2
HEAD_EDGE = 2
MATCH_RATIO = 0.8               # D-093: 0.75에서 0.8로. 대신 얼굴 일치점·되돌린 뒤 재검사·형태 검사로 거른다
FACE_NEAR = 25                  # 얼굴 경계에서 이 거리 안의 일치점을 얼굴 일치점으로 센다
MIN_FACE_INLIERS = 8
HAIR_FAR = 40                   # 얼굴에서 이보다 먼 머리로 형태를 본다
HAIR_COLOR_DE = 15              # 원본 머리색(Lab k-means 4개 중심)과 이 거리 안이면 Qwen 머리로 본다
MIN_HAIR_SUPPORT = 0.5
MAX_FLOATING_HAIR = 0.2
RANSAC_PX = 3.0
MIN_INLIERS = 12
SCALE_RANGE = (0.75, 1.35)      # 되돌릴 수 있는 크기 차이
ROTATE_LIMIT = 10.0             # 되돌릴 수 있는 기울기(도)
RESIDUAL_SHIFT = 2.0            # 되돌린 뒤 남아도 되는 위치 차이(px)
RESIDUAL_SCALE = 0.02
RESIDUAL_ROTATE = 1.0
CONTENT_DIFF = 25               # 흰색과 채널 차이가 이보다 크면 그림 내용으로 본다
HEAD_WINDOW = 15
BORDER = 32
HEAD_ACCESSORIES = ("headwear", "earwear", "eyewear")  # 사용자 선택 대상은 머리 장식뿐이다
HEAD_KEEP = ("hair", "face", "eyes", "nose", "mouth", "ears", "tail", "wings")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def match_head(person, edited, window, face=None):
    """원본 머리 주변(window)의 특징점을 Qwen 결과 전체에서 찾아 원본→편집 닮음 변환(위치·크기·기울기)을 잰다.
    face가 있으면 얼굴 경계 FACE_NEAR px 안에 있는 일치점 수(face_inliers)를 센다."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray(person), window.astype(np.uint8) * 255)
    kp2, des2 = sift.detectAndCompute(gray(edited), None)
    record = dict(source_features=len(kp1), target_features=len(kp2), matches=0, inliers=0)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return dict(record, status="unmeasured")
    pairs = [m[0] for m in cv2.BFMatcher().knnMatch(des1, des2, k=2) if len(m) == 2 and m[0].distance < MATCH_RATIO * m[1].distance]
    record["matches"] = len(pairs)
    if len(pairs) < MIN_INLIERS:
        return dict(record, status="unmeasured")
    src = np.float32([kp1[m.queryIdx].pt for m in pairs])
    dst = np.float32([kp2[m.trainIdx].pt for m in pairs])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    record["inliers"] = int(inliers.sum()) if inliers is not None else 0
    if matrix is None or record["inliers"] < MIN_INLIERS:
        return dict(record, status="unmeasured")
    if face is not None:
        near = cv2.distanceTransform((~face).astype(np.uint8), cv2.DIST_L2, 5) <= FACE_NEAR
        points = src[inliers.ravel() == 1].astype(int)
        record["face_inliers"] = int(near[points[:, 1], points[:, 0]].sum())
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    ys, xs = np.nonzero(window)
    center = np.array([xs.mean(), ys.mean(), 1.0])
    moved = matrix @ center
    return dict(record, status="measured", matrix=matrix.tolist(), scale=float(np.hypot(a, b)),
                rotate_deg=float(np.degrees(np.arctan2(b, a))), head_shift_px=[float(moved[0] - center[0]), float(moved[1] - center[1])])


def near_identity(match):
    return (match["status"] == "measured" and abs(match["scale"] - 1) <= RESIDUAL_SCALE
            and abs(match["rotate_deg"]) <= RESIDUAL_ROTATE and max(map(abs, match["head_shift_px"])) <= RESIDUAL_SHIFT)


def content_mask(image):
    """흰 바탕이 아닌 그림 내용. 점 잡음은 3×3 열림으로 지운다."""
    content = (np.abs(255 - image.astype(np.int16)).max(axis=2) > CONTENT_DIFF).astype(np.uint8)
    return cv2.morphologyEx(content, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def outside_frame(points, width, height):
    x, y = points
    return (x < 0) | (y < 0) | (x > width - 1) | (y > height - 1)


def pad_image(image, pad, value):
    left, top, right, bottom = pad
    widths = ((top, bottom), (left, right)) + (((0, 0),) if image.ndim == 3 else ())
    return np.pad(image, widths, constant_values=value)


def realign(person, edited, window, expand=False, face=None):
    """머리가 원본과 같은 자리·크기가 되도록 Qwen 결과 전체를 되돌린다. 확인되지 않으면 None(붙이지 않음).
    expand=True면 되돌린 그림 내용이 잘리지 않게 화면을 흰 여백으로 넓힌다(원본 좌표는 pad만큼 옮겨진다)."""
    first = match_head(person, edited, window, face)
    report = dict(first=first, canvas_pad=[0, 0, 0, 0])
    if first["status"] != "measured":
        return None, dict(report, decision="unmeasured")
    if face is not None and first["face_inliers"] < MIN_FACE_INLIERS:
        return None, dict(report, decision="unmeasured", reason="얼굴 일치점 부족")
    if not (SCALE_RANGE[0] <= first["scale"] <= SCALE_RANGE[1] and abs(first["rotate_deg"]) <= ROTATE_LIMIT):
        return None, dict(report, decision="rejected", reason="크기·기울기 차이가 범위를 넘음")
    height, width = person.shape[:2]
    inverse = cv2.invertAffineTransform(np.float32(first["matrix"])).astype(np.float64)
    ys, xs = np.nonzero(content_mask(edited))
    mapped = inverse @ np.stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    pad = [0, 0, 0, 0]
    if expand and xs.size:
        pad = [max(0, int(np.ceil(-mapped[0].min()))), max(0, int(np.ceil(-mapped[1].min()))),
               max(0, int(np.ceil(mapped[0].max() - (width - 1)))), max(0, int(np.ceil(mapped[1].max() - (height - 1))))]
    inverse[0, 2] += pad[0]
    inverse[1, 2] += pad[1]
    size = (width + pad[0] + pad[2], height + pad[1] + pad[3])
    lost = int(outside_frame(mapped + np.array([[pad[0]], [pad[1]]]), *size).sum()) if xs.size else 0
    back = cv2.warpAffine(edited, inverse, size, flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    second = match_head(pad_image(person, pad, 255), back, pad_image(window, pad, False))
    report.update(second=second, canvas_pad=pad, content_outside_px=lost)
    if not near_identity(second):
        return None, dict(report, decision="rejected", reason="되돌린 뒤에도 머리가 맞지 않음")
    return back, dict(report, decision="aligned")


def fit_white(image, size):
    """비교 그림용: 넓어진 화면을 비율 유지로 size 안에 줄여 흰 바탕 가운데 놓는다(결과 파일은 줄이지 않는다)."""
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    scale = min(width / image.shape[1], height / image.shape[0])
    small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out = np.full((height, width, 3), 255, np.uint8)
    y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) * np.float32([100 / 255, 1, 1])


def hair_shape(person, aligned, hair, face, pasted):
    """위치가 맞은 뒤 머리 형태·연결 검사. 모두 같은 (넓힌) 화면 좌표여야 한다."""
    if not hair.any() or not face.any():
        return dict(status="unmeasured", reason="머리 또는 얼굴 분할 없음")
    samples = lab(person)[hair]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(samples, min(4, len(samples)), None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    distance = np.min(np.linalg.norm(lab(aligned)[..., None, :] - centers[None, None], axis=-1), axis=-1)
    qwen_hair = (distance < HAIR_COLOR_DE) & content_mask(aligned)
    far = hair & (cv2.distanceTransform((~face).astype(np.uint8), cv2.DIST_L2, 5) > HAIR_FAR)
    near_qwen = cv2.dilate(qwen_hair.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    support = float((far & near_qwen).sum() / far.sum()) if far.any() else 1.0
    _, labels = cv2.connectedComponents((pasted | qwen_hair).astype(np.uint8), connectivity=8)
    attached = np.setdiff1d(np.unique(labels[face & (pasted | qwen_hair)]), [0])
    floating = float((pasted & ~np.isin(labels, attached)).sum() / max(1, pasted.sum()))
    ok = support >= MIN_HAIR_SUPPORT and floating <= MAX_FLOATING_HAIR
    return dict(status="ok" if ok else "rejected", far_hair_px=int(far.sum()), support=round(support, 3),
                floating=round(floating, 3))


def composite(person, edited, change):
    weight = cv2.GaussianBlur(change.astype(np.float32), (0, 0), FEATHER)
    weight[~change] = 0
    return np.clip(np.rint(person * (1 - weight[..., None]) + edited * weight[..., None]), 0, 255).astype(np.uint8)


def head_protect(parts, protect_png, accessories):
    """머리 붙이기 영역. 확인된 protect.png(머리 장식 포함)와 1단계 분할이 일치하는지 확인한 뒤,
    remove면 머리 장식만 뺀다. 머리 장식 외의 영역은 바꾸지 않는다."""
    base = np.logical_or.reduce([parts[k] for k in HEAD_KEEP])
    worn = np.logical_or.reduce([parts[k] for k in HEAD_ACCESSORIES])
    require(np.array_equal(base | worn, protect_png), "1단계 분할과 protect.png가 다릅니다.")
    return base | worn if accessories == "keep" else base & ~worn


def head_weight(protect):
    """머리 붙이기 가중치: 보존 영역을 2px 줄인 뒤 가장자리만 부드럽게 한다. 보존 영역 밖 가중치는 0(원본 배경이 따라오지 않음).
    core는 가중치가 사실상 1인 안쪽(8σ 줄임)으로, 원본 픽셀과 정확히 같아야 한다."""
    inner = cv2.erode(protect.astype(np.uint8), np.ones((2 * HEAD_EDGE + 1,) * 2, np.uint8)) > 0
    weight = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), FEATHER)
    weight[~inner] = 0
    core = cv2.erode(inner.astype(np.uint8), np.ones((8 * FEATHER + 1,) * 2, np.uint8)) > 0
    return weight, core


def head_composite(person, edited, protect):
    weight, core = head_weight(protect)
    result = np.clip(np.rint(person * weight[..., None] + edited * (1 - weight[..., None])), 0, 255).astype(np.uint8)
    return result, core


def head_window(protect):
    return cv2.dilate(protect.astype(np.uint8), np.ones((2 * HEAD_WINDOW + 1,) * 2, np.uint8)) > 0


def border_white_ratio(image):
    """가장자리 32px 띠에서 거의 흰색(세 채널 모두 235 이상)인 비율. 배경이 남았는지 보는 기록용 값."""
    frame = np.ones(image.shape[:2], bool)
    frame[BORDER:-BORDER, BORDER:-BORDER] = False
    return float((image[frame].min(axis=1) >= 235).mean())


class MemoryGuard:
    """단계별 PyTorch 메모리 기록. 각 단계 최고값은 직전 기록 뒤부터 잰다(단계마다 최고값을 초기화).
    한도는 apply_cap()으로 할당 상한을 걸어 지키고, check()는 넘었는지 다시 확인해 멈춘다."""

    def __init__(self, cuda, limit_gib, log_path=None):
        self.cuda, self.limit, self.log_path, self.stages, self.stopped = cuda, limit_gib, log_path, [], None
        self.current = "시작"

    def begin(self, stage):
        """다음 작업 이름을 먼저 남긴다. 그 작업 중 할당이 실패해도 어느 단계였는지 남는다."""
        self.current = stage
        self.write()

    def apply_cap(self, device=0):
        if self.limit is None:
            return None
        total = self.cuda.get_device_properties(device).total_memory
        fraction = min(1.0, self.limit * 2 ** 30 / total)
        self.cuda.set_per_process_memory_fraction(fraction, device)
        return fraction

    def write(self):
        if self.log_path is not None:  # 한도로 멈춰도 어느 단계까지 썼는지 남는다
            write_json(self.log_path, dict(limit_gib=self.limit, current=self.current, stages=self.stages,
                                           stopped=self.stopped))

    def check(self, stage):
        self.current = stage
        peak = self.cuda.max_memory_reserved() / 2 ** 30
        self.stages.append(dict(stage=stage, allocated_gib=round(self.cuda.memory_allocated() / 2 ** 30, 3),
                                reserved_gib=round(self.cuda.memory_reserved() / 2 ** 30, 3),
                                stage_peak_reserved_gib=round(peak, 3)))
        self.cuda.reset_peak_memory_stats()
        self.write()
        if self.limit is not None and peak > self.limit:
            self.stopped = f"{stage}: 단계 최고 reserved {peak:.2f}GiB > 한도 {self.limit}GiB"
            self.write()
            raise MemoryError(self.stopped)

    def fail(self, error):
        try:
            at_failure = dict(allocated_gib=round(self.cuda.memory_allocated() / 2 ** 30, 3),
                              reserved_gib=round(self.cuda.memory_reserved() / 2 ** 30, 3),
                              stage_peak_reserved_gib=round(self.cuda.max_memory_reserved() / 2 ** 30, 3))
        except Exception as reason:  # 수집 실패는 미측정으로 남긴다
            at_failure = dict(unmeasured=str(reason))
        self.stages.append(dict(stage=f"{self.current}(실패)", **at_failure))
        self.stopped = f"'{self.current}' 중 할당 실패: {error}"
        self.write()

    def step_callback(self, name):
        def callback(pipe, step, timestep, kwargs):
            self.check(f"{name}/step{step}" + ("(조건 그림 VAE 인코딩 포함)" if step == 0 else ""))
            last = step + 1 == pipe.num_timesteps if hasattr(pipe, "num_timesteps") else False
            self.current = f"{name}/decode" if last else f"{name}/step{step + 1}"
            return kwargs
        return callback


def text_encoder_device_map():
    """문장 인코딩은 마지막 은닉 상태만 쓰고 lm_head(어휘 152,064 × 3,584, bf16 1.02GiB) 출력은 쓰지 않는다.
    그래서 lm_head는 GPU에 올리지 않고 CPU에 둔 뒤 적재 후 Identity로 바꿔 계산도 하지 않는다. 인코딩 결과는 같다."""
    return {"model": 0, "lm_head": "cpu"}


def canvas(image):
    """비율 유지, 흰 여백으로 768×1024에 넣는다(머리·발을 자르지 않음). CatVTON resize_and_padding과 같은 규칙."""
    w, h = image.size
    tw, th = SIZE
    if w / h < tw / th:
        nh, nw = th, w * th // h
    else:
        nw, nh = tw, h * tw // w
    resized = image.convert("RGB").resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", SIZE, (255, 255, 255))
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


# 제품 지시문: 캐릭터별 의상 묘사를 손으로 쓰지 않는다(시험 지시문의 "long dark brown coat"가 의상을 바꾼 정황, D-092).
# 요청 12: "그림 2의 옷을 입혀라"만으로는 W3의 몸에 붙는 원래 옷이 그대로 남았다 → 지금 옷 전체를 바꾸라고 명시한다(D-095).
OUTFIT_REPLACE = ("Replace all of the character's current clothing in image 1, including any bodysuit, jacket, socks and shoes, "
                  "with the complete outfit shown in image 2, matching its shape, length, colors and details. ")
OUTFIT_PROPS = "Remove any weapon and every carried or held prop. "
OUTFIT_BACKGROUND = "Place the character alone on a plain pure white background with nothing else. "
OUTFIT_BASE = OUTFIT_REPLACE + OUTFIT_PROPS + OUTFIT_BACKGROUND
OUTFIT_KEEP = ("Keep the character's face, hairstyle, hair color, head accessories and body proportions exactly the same as "
               "in image 1. ")
OUTFIT_REMOVE = ("The character must stay the same person as in image 1: same gender, age, face, hair color and hairstyle. "
                 "Only take off the head accessories: remove every head accessory (headphones, headbands, hair ornaments, "
                 "hats, earrings and glasses) and show the character's own hair where they were. Keep body proportions "
                 "exactly the same as in image 1. ")
OUTFIT_FRAMING = "Do not change the camera angle, framing or the character's size or position."
GARMENT_TERM = re.compile(r"[a-z][a-z0-9 '-]{0,39}")
NOT_GARMENT = ("hand", "arm", "leg", "pose", "holding", "pocket", "standing", "sitting", "looking")
MAX_GARMENT_TERMS = 20


def clean_garment_terms(terms):
    """의상 그림 자동 분석(사용자 확인)의 옷 이름만 남긴다. 자세·손 동작 태그(예: hands in pockets)는 뺀다."""
    from genai_lab.onepass_garment_vocabulary import garment_nouns
    from genai_lab.garment_detail_analysis import detail_group
    kept = []
    for term in terms:
        text = str(term).replace("_", " ").strip().lower()
        words = text.split()
        # 제품 입력 분석(studio_analysis.features)과 같은 옷 판정 규칙
        garment = garment_nouns((text,)) or detail_group(text) in (
            "component_candidate", "construction", "shape", "surface", "style_context")
        if (GARMENT_TERM.fullmatch(text) and garment
                and not any(w in words or w + "s" in words for w in NOT_GARMENT) and text not in kept):
            kept.append(text)
    return tuple(kept[:MAX_GARMENT_TERMS])


def outfit_prompt(head_accessories, garment_terms=(), remove_props=True):
    """제품용 고정 지시문. 머리 장식 선택(keep/remove)과, 있으면 의상 그림에서 자동으로 읽은 옷 이름 목록만 바뀐다.
    remove_props=False는 원인 확인 시험 전용이다(요청 12 W3: 소품 제거 지시와 함께 팔까지 지워졌는지 확인, D-096)."""
    if head_accessories not in ("keep", "remove"):
        raise ValueError("머리 장식은 keep 또는 remove만 고를 수 있습니다.")
    terms = tuple(garment_terms)
    if terms != clean_garment_terms(terms):
        raise ValueError("옷 이름 목록은 정리된 의상 용어만 쓸 수 있습니다.")
    listed = f"The outfit in image 2 consists of: {', '.join(terms)}. " if terms else ""
    body = OUTFIT_KEEP if head_accessories == "keep" else OUTFIT_REMOVE
    base = OUTFIT_BASE if remove_props else OUTFIT_REPLACE + OUTFIT_BACKGROUND
    return base + listed + body + OUTFIT_FRAMING


ARM_SEGMENTS = (("shoulder", "elbow"), ("elbow", "wrist"))
ARM_LINE_MIN = 0.95     # 팔 구간 위가 인물 영역이어야 하는 비율(요청 12 W3 팔 없음: 0.58, 팔 있는 13장: 모두 1.00)
ARM_JOINT_CONF = 0.4    # 이보다 낮은 관절이 낀 구간은 판정하지 않는다(가려짐·잘못 찍힘)
ARM_SAMPLES = 12
NEAR_WHITE = 235


def figure_mask(image):
    """테두리와 이어진 거의 흰색만 배경이다. 윤곽선 안의 흰 소매·셔츠는 인물로 남는다(요청 13 B-W4: 흰 소매를 배경으로 오판)."""
    near_white = (image.min(axis=2) >= NEAR_WHITE).astype(np.uint8)
    _, labels = cv2.connectedComponents(near_white, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, border[border > 0]) & (near_white > 0)
    return ~background


def arm_lines(joints, image):
    """관절 인식이 추정한 어깨→팔꿈치, 팔꿈치→손목 구간마다 인물 영역 위에 있는 비율을 잰다.
    두 관절이 모두 찾아졌고 신뢰도 ARM_JOINT_CONF 이상인 구간만 잰다. 손이 주머니에 가려 손목을 못 찾으면 그 구간은 빠진다.
    관절 신뢰도만으로 팔 유무를 정하지 않는다: 팔이 없는 그림에서도 팔 관절을 0.6 신뢰도로 추정했다(요청 12 W3)."""
    named = {j["joint_name"]: j for j in joints}
    figure = figure_mask(image)
    height, width = figure.shape
    lines = {}
    for side in ("left", "right"):
        segments = []
        for first, second in ARM_SEGMENTS:
            a, b = named.get(f"{side}_{first}"), named.get(f"{side}_{second}")
            if not (a and b and a.get("detected") and b.get("detected")
                    and min(a.get("confidence_score", 0), b.get("confidence_score", 0)) >= ARM_JOINT_CONF):
                continue
            hits = []
            for t in np.linspace(0.15, 0.95, ARM_SAMPLES):
                x = int(np.clip(a["x"] + (b["x"] - a["x"]) * t, 0, width - 1))
                y = int(np.clip(a["y"] + (b["y"] - a["y"]) * t, 0, height - 1))
                hits.append(bool(figure[max(0, y - 2):y + 3, max(0, x - 2):x + 3].any()))
            segments.append(round(float(np.mean(hits)), 3))
        lines[side] = segments
    return lines


def limb_check(original_joints, original_image, result_joints, result_image):
    """원본에 있던 팔이 Qwen 결과에서 사라졌으면 거부한다.
    - 원본에서 그 팔의 구간을 하나도 확인하지 못했으면 판정하지 않는다.
    - 결과에서 잰 구간 중 하나라도 인물 영역 밖(ARM_LINE_MIN 미만)이면 그 팔은 사라진 것으로 본다.
    - 결과에서 그 팔의 구간을 하나도 재지 못했으면(가려짐) 사라짐으로 단정하지 않고 확인 불가로 남긴다."""
    before, after = arm_lines(original_joints, original_image), arm_lines(result_joints, result_image)
    checked = [side for side in ("left", "right") if any(v >= ARM_LINE_MIN for v in before[side])]
    missing = [side for side in checked if after[side] and min(after[side]) < ARM_LINE_MIN]
    unverified = [side for side in checked if not after[side]]
    measured = [side for side in checked if after[side]]
    status = "rejected" if missing else ("ok" if measured else "unmeasured")
    return dict(status=status, original=before, result=after, checked=checked, missing=missing, unverified=unverified)
