"""옷 참조 그림의 공통 전처리(D-117 검토): 옷 픽셀은 손대지 않고, 배경과 머리만 뺀다.

요청 18(2026-10-14)에서 같은 옷 G11의 모자·마네킹 머리·배경·받침대를 뺀 참조만으로 세 캐릭터 모두 옷 교체가 개선됐다.
여기서는 그 정리를 어떤 옷 그림에도 같은 규칙으로 적용한다(그림별 조정 없음, D-104).

1. 배경 흰색화: 테두리 중앙값이 흰색이 아니면 GrabCut(내용 상자 초기화)으로 배경을 나눠 흰색으로. 옷 쪽으로 2px 물러선다.
   내용이 테두리 전체에 닿거나 전경 비율이 비정상이면 건너뛴다(그림 그대로).
2. 머리 제거: 태그 분석기가 "머리 있음"(head_out_of_frame < 0.10, no_humans < 0.5)으로 보고 골격의 목·어깨가 있으면
   목선에서 어깨 폭의 1/4만큼 위를 흰색으로 칠한다. 머리 장식은 원본 캐릭터 것을 쓰므로(머리는 원본 픽셀) 참조의 머리·모자는 필요 없다.
   참조 폴더 23장(2026-10-14 CPU): 머리 있는 8장 모두 적용, 머리 없는 15장 적용 없음(`rules18-cpu/clean-proto-report.json`).
3. 여백 다듬기: 1·2가 적용된 그림만, 흰 여백(≥250)을 잘라 옷이 작아지지 않게 한다(12px 여유).
받침대·글자·마네킹 팔은 다루지 않는다(일반 규칙 없음).
"""
from pathlib import Path

import numpy as np

WHITE_BORDER = 240          # 테두리 중앙값이 이 이상이면 배경은 이미 흰색
BACKGROUND_STEP = 45        # 내용 상자: 테두리 색과 이만큼 다른 픽셀
GUARD = 2                   # 배경을 옷 쪽으로 이만큼 물러서서 칠한다
FOREGROUND_RANGE = (0.08, 0.9)
HEAD_OUT_MAX = 0.10         # 이 이상이면 머리가 잘린 그림
NO_HUMANS_MAX = 0.5         # 이 이상이면 사람·마네킹이 없는 그림(평면 착장)
NECK_UP = 0.25              # 목선에서 어깨 폭 × 이 값만큼 위가 머리 경계
MIN_KEEP = 0.6              # 머리 제거 뒤 남는 높이는 그림의 이 비율 이상
TRIM_WHITE = 250            # 여백 다듬기: 모든 채널이 이 이상이면 흰 여백
TRIM_MARGIN = 12


def border_median(image):
    rows = np.concatenate([image[:6].reshape(-1, 3), image[-6:].reshape(-1, 3), image[:, :6].reshape(-1, 3),
                           image[:, -6:].reshape(-1, 3)])
    return np.median(rows, 0)


def whiten_background(image):
    """배경을 흰색으로. 반환: (그림, 보고). 옷 픽셀은 바꾸지 않는다."""
    import cv2
    h, w = image.shape[:2]
    bg = border_median(image)
    if (bg >= WHITE_BORDER).all():
        return image, dict(status="skipped", reason="white border", border=[int(v) for v in bg])
    coarse = np.abs(image.astype(int) - bg.astype(int)).max(2) > BACKGROUND_STEP
    ys, xs = np.where(coarse)
    if len(xs) == 0:
        return image, dict(status="skipped", reason="no content", border=[int(v) for v in bg])
    x0, x1 = max(0, int(xs.min()) - 10), min(w, int(xs.max()) + 10)
    y0, y1 = max(0, int(ys.min()) - 10), min(h, int(ys.max()) + 10)
    if (x1 - x0) * (y1 - y0) >= 0.97 * h * w:
        return image, dict(status="skipped", reason="content touches every border", border=[int(v) for v in bg])
    mask = np.zeros((h, w), np.uint8)
    models = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(image), mask, (x0, y0, x1 - x0, y1 - y0), *models, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as error:
        return image, dict(status="skipped", reason="grabcut: " + str(error).splitlines()[-1][:80], border=[int(v) for v in bg])
    background = np.isin(mask, (cv2.GC_BGD, cv2.GC_PR_BGD)).astype(np.uint8)
    background = cv2.erode(background, np.ones((2 * GUARD + 1,) * 2, np.uint8)).astype(bool)
    fraction = float(1 - background.mean())
    if not FOREGROUND_RANGE[0] <= fraction <= FOREGROUND_RANGE[1]:
        return image, dict(status="skipped", reason=f"foreground fraction {fraction:.2f}", border=[int(v) for v in bg])
    out = image.copy()
    out[background] = 255
    return out, dict(status="applied", border=[int(v) for v in bg], foreground_fraction=round(fraction, 3),
                     changed_px=int(background.sum()))


def head_present(tag_scores):
    return (float(tag_scores.get("head_out_of_frame", 0.0)) < HEAD_OUT_MAX
            and float(tag_scores.get("no_humans", 0.0)) < NO_HUMANS_MAX)


def head_line(joints, height):
    """머리 경계 행. 목·어깨가 없거나 선이 범위 밖이면 None."""
    found = {j["joint_name"]: j for j in joints if j.get("detected")}
    if not all(k in found for k in ("neck", "left_shoulder", "right_shoulder")):
        return None
    width = abs(found["left_shoulder"]["x"] - found["right_shoulder"]["x"])
    line = int(found["neck"]["y"] - NECK_UP * width)
    if line <= 0 or line >= (1 - MIN_KEEP) * height:
        return None
    return line


def remove_head(image, tag_scores, joints):
    """태그가 머리 있음이고 골격 목선이 있으면 목선 위를 흰색으로. 반환: (그림, 보고)."""
    facts = dict(head_out_of_frame=round(float(tag_scores.get("head_out_of_frame", 0.0)), 3),
                 no_humans=round(float(tag_scores.get("no_humans", 0.0)), 3))
    if not head_present(tag_scores):
        return image, dict(status="skipped", reason="no head by tags", **facts)
    line = head_line(joints, image.shape[0])
    if line is None:
        return image, dict(status="skipped", reason="no neck line", **facts)
    out = image.copy()
    out[:line] = 255
    return out, dict(status="applied", line=line, changed_px=int(line * image.shape[1]), **facts)


def trim_white(image):
    """흰 여백을 잘라낸다(옷 픽셀 무변경). 반환: (그림, 보고)."""
    content = (image < TRIM_WHITE).any(2)
    ys, xs = np.where(content)
    if len(xs) == 0:
        return image, dict(status="skipped", reason="all white")
    h, w = image.shape[:2]
    y0, y1 = max(0, int(ys.min()) - TRIM_MARGIN), min(h, int(ys.max()) + 1 + TRIM_MARGIN)
    x0, x1 = max(0, int(xs.min()) - TRIM_MARGIN), min(w, int(xs.max()) + 1 + TRIM_MARGIN)
    if (y0, x0, y1, x1) == (0, 0, h, w):
        return image, dict(status="skipped", reason="no margin")
    return image[y0:y1, x0:x1], dict(status="applied", box=[x0, y0, x1, y1], size=[x1 - x0, y1 - y0])


def clean_garment_reference(image, tag_scores, joints=()):
    """세 단계를 차례로. 반환: (그림, 보고{background, head, trim, changed}). 아무것도 적용되지 않으면 원본 배열 그대로."""
    step1, background = whiten_background(image)
    step2, head = remove_head(step1, tag_scores, joints)
    changed = background["status"] == "applied" or head["status"] == "applied"
    if not changed:
        return image, dict(background=background, head=head, trim=dict(status="skipped", reason="nothing applied"), changed=False)
    step3, trim = trim_white(step2)
    return step3, dict(background=background, head=head, trim=trim, changed=True)


def prepare_garment_reference(garment_path, directory, runtime, *, cancelled=lambda: False):
    """옷 참조 파일을 정리해 실행 폴더에 쓴다. 바뀐 게 없으면 원래 경로를 그대로 돌려준다(재인코딩 없음).
    반환: dict(path, report). 태그·골격은 기존 CPU 분석기(태그 분석기, DWPose)를 쓴다."""
    from PIL import Image
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.outfit_edit import write_json
    from genai_lab.outfit_edit_pipeline import detect_joints
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with Image.open(garment_path) as source:
        image = np.asarray(source.convert("RGB"))
    path = Path(garment_path)
    try:
        settings = ClothingDesignAnalysisSettings(cache_dir=str(runtime.model_cache), local_files_only=True)
        with WdTagSession(settings) as tagger:
            scores = dict(tagger.analyze(Image.fromarray(image)).raw_general_scores)
        joints = detect_joints(image, directory / "pose", runtime, cancelled=cancelled) if head_present(scores) else ()
    except Exception as error:          # 분석기가 없거나 실패해도 편집은 막지 않는다: 원래 옷 그대로, 사실만 기록
        if "취소" in str(error):
            raise
        report = dict(status="failed", error=str(error)[:200], changed=False, source=str(garment_path), path=str(path))
        write_json(directory / "garment_reference.json", report)
        return dict(path=str(path), report=report)
    cleaned, report = clean_garment_reference(image, scores, joints)
    if report["changed"]:
        path = directory / "garment-clean.png"
        Image.fromarray(cleaned).save(path)
    report = dict(report, source=str(garment_path), path=str(path),
                  tags={k: round(float(scores.get(k, 0.0)), 3) for k in ("head_out_of_frame", "no_humans", "mannequin", "hat")})
    write_json(directory / "garment_reference.json", report)
    return dict(path=str(path), report=report)
