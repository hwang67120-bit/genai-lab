"""요청 33 시각 참조 고정(CPU, 2026-10-22). 요청 29 사용자 통과 결과(40단계) 중 캐릭터마다 기억 참조 하나를 규칙 R-a~R-d로 고르고,
그 그림 자신의 저장 분할(result-analysis/parts/parts.npz, 768x1024 캔버스 좌표)로 머리 크롭 + 머리 밖 흰색 참조를 만든다.
기존 평가 결과는 읽기만 한다. 사용: python -I make_refs.py <이 폴더>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from genai_lab.character_features import hair_mask_without_background, skin_color, to_lab  # noqa: E402
from genai_lab.free_generation import request_conflicts  # noqa: E402
from genai_lab.free_generation_pipeline import WIDE_REFERENCE, head_box  # noqa: E402

G = ROOT / "outputs/generalization-20261012"
OUT = Path(sys.argv[1])
SIZE = (768, 1024)
HEAD_KEYS = ("hair", "face", "eyes", "nose", "mouth", "headwear", "eyewear", "earwear", "ears")   # head_box와 같은 열쇠
ROWS = {"M3": ("S2", "S4"), "M6": ("S4", "S1"), "N1": ("S4",), "K": ("S2",)}                  # 요청 33 시험 줄
VARIANT_FILE = {"raw": "qwen/raw.png", "hair_auto": "result-analysis/hair-auto.png", "hair_restored": "result-analysis/hair-restored.png"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canvas_of(path):
    with Image.open(path) as im:
        return im.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)    # observe_result와 같은 변환


def parts_of(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].astype(bool) for k in data.files}


def lab_bands(image, mask, bands=3):
    """머리 마스크 위·가운데·끝 띠의 L*a*b* 중앙값(밝기 포함 — D-139: 밝기를 빼면 어두워짐을 놓친다)."""
    lab = to_lab(np.asarray(image))
    rows = np.where(mask.any(axis=1))[0]
    edges = np.linspace(rows[0], rows[-1] + 1, bands + 1).astype(int)
    out = []
    for top, bottom in zip(edges[:-1], edges[1:]):
        part = np.zeros_like(mask)
        part[top:bottom] = mask[top:bottom]
        out.append(np.median(lab[part], 0) if part.sum() >= 50 else None)
    return out


def face_connected(mask, parts):
    """머리 합집합에서 얼굴과 닿는 연결 요소만 남긴다(떨어진 분할 조각 제거)."""
    import cv2
    face = np.logical_or.reduce([parts[k] for k in ("face", "eyes") if k in parts])
    n, labels = cv2.connectedComponents(np.ascontiguousarray(mask).astype(np.uint8), connectivity=8)
    keep = set(np.unique(labels[face & mask])) - {0}
    return np.isin(labels, list(keep))


def band_error(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return float(np.mean([np.linalg.norm(x - y) for x, y in pairs])) if len(pairs) >= 2 else None


def main():
    ledger = json.loads((G / "judgement-29/ledger.json").read_text(encoding="utf-8"))
    manifest = dict(schema="memory_reference_fixed/1", created="2026-10-22", rules=dict(
        R_a="요청 29 사용자 통과(quality_pass true), 40단계, 같은 캐릭터, 판정 SHA와 파일 SHA 일치",
        R_b="후보 장면이 시험 줄 장면과 같지 않음(같은 장면 복사 방지·기억에 없는 장면 시험)",
        R_c="후보 그림 자기 분할로 얼굴 기준 창 크롭 성공(넓은 크롭 아님), 후보 장면이 눈 감기·뒷모습 아님",
        R_d="기준 원본 대비 머리 띠별 L*a*b* 오차 + 얼굴 피부 ΔE(L*a*b*) 합이 가장 작은 것, 같으면 먼저 판정한 것"),
        reference_build=dict(
            crop="free_generation_pipeline.head_box(후보 parts, 768x1024) — 지금 기준 머리 크롭과 같은 함수",
            mask="HEAD_KEYS 분할 합집합 중 얼굴(face·eyes)과 이어진 연결 요소만(8-연결, 팽창 없음), crop 안", background="마스크 밖은 RGB(255,255,255)",
            canvas="후보 그림을 768x1024 LANCZOS로 맞춤(observe_result와 같음). 보정본(hair_auto·hair_restored)은 같은 시도 raw의 분할을 쓴다 — 보정은 머리 색만 바꾸고 모양을 바꾸지 않는다"),
        characters={})
    for char, rows in ROWS.items():
        cases = [r for r in ledger if r["case"].startswith(char + "-")]
        base_run = G / "free-r29" / cases[0]["case"]
        base = canvas_of(base_run / "inputs/canvas.png")
        base_parts = parts_of(base_run / "base-parts/parts.npz")
        base_bands = lab_bands(base, hair_mask_without_background(np.asarray(base), base_parts["hair"]))
        base_skin = (skin_color(np.asarray(base), base_parts) or {}).get("lab")
        record = dict(base=dict(canvas=str(base_run / "inputs/canvas.png"), canvas_sha256=sha(base_run / "inputs/canvas.png"),
                                head_reference=str(base_run / "inputs/head-reference.png"),
                                head_reference_sha256=sha(base_run / "inputs/head-reference.png")),
                      rows=list(rows), candidates=[])
        for r in cases:
            if not r.get("quality_pass"):
                continue                        # 실패 판정은 후보가 아니다(R-a)
            scene = r["case"].split("-")[1]
            run = G / "free-r29" / r["case"]
            variant = r["target"].get("variant", "raw")
            path = run / VARIANT_FILE[variant]
            cand = dict(case=r["case"], variant=variant, attempt=r["target"]["attempt"], path=str(path), sha256=sha(path), excluded=[])
            if cand["sha256"] != r["target"]["sha256"]:
                cand["excluded"].append("R-a: 판정 대상 SHA와 파일 SHA 불일치")
            if scene in rows:
                cand["excluded"].append("R-b: 시험 줄과 같은 장면")
            staging = json.loads((run / "trace.json").read_text(encoding="utf-8"))["direction"]["staging_text"]
            if set(request_conflicts(staging)) & {"eye_color", "skin"}:
                cand["excluded"].append("R-c: 눈 감기·뒷모습 장면")
            parts_path = run / ("result-analysis" if cand["attempt"] == 1 else f"result-analysis-{cand['attempt']}") / "parts/parts.npz"
            observation = json.loads((parts_path.parents[1] / "observation.json").read_text(encoding="utf-8"))
            cand.update(parts=str(parts_path), parts_sha256=sha(parts_path))
            if cand["parts_sha256"] != observation["segmentation"]["parts_sha256"]:
                cand["excluded"].append("분할 파일 SHA가 관찰 기록과 다름")
            image = canvas_of(path)
            parts = parts_of(parts_path)
            box = head_box(parts, SIZE)
            if box is None:
                cand["excluded"].append("R-c: 얼굴 없음")
            else:
                ratio = (box[2] - box[0]) * (box[3] - box[1]) / (SIZE[0] * SIZE[1])
                cand.update(box=list(box), area_ratio=round(ratio, 3))
                if ratio > WIDE_REFERENCE:
                    cand["excluded"].append("R-c: 넓은 크롭(scope wide)")
            hair = hair_mask_without_background(np.asarray(image), parts["hair"]) if "hair" in parts else None
            error = band_error(base_bands, lab_bands(image, hair)) if hair is not None and hair.sum() else None
            skin = (skin_color(np.asarray(image), parts) or {}).get("lab")
            cand["hair_band_error_lab"] = None if error is None else round(error, 2)
            cand["skin_delta_e"] = round(float(np.linalg.norm(np.array(skin) - np.array(base_skin))), 2) if skin and base_skin else None
            if cand["hair_band_error_lab"] is None or cand["skin_delta_e"] is None:
                cand["excluded"].append("R-d: 색 측정 불가")
            else:
                cand["score"] = round(cand["hair_band_error_lab"] + cand["skin_delta_e"], 2)
            record["candidates"].append(cand)
        eligible = [c for c in record["candidates"] if not c["excluded"]]
        record["chosen"] = None
        if eligible:
            chosen = min(eligible, key=lambda c: c["score"])      # 같으면 앞(먼저 판정한 것)
            image = canvas_of(chosen["path"])
            parts = parts_of(chosen["parts"])
            x0, y0, x1, y1 = chosen["box"]
            mask = face_connected(np.logical_or.reduce([parts[k] for k in HEAD_KEYS if k in parts]), parts)[y0:y1, x0:x1]
            crop = np.asarray(image)[y0:y1, x0:x1].copy()
            crop[~mask] = 255
            directory = OUT / char
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(directory / "memory-reference.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(directory / "memory-reference-mask.png")
            record["chosen"] = dict(case=chosen["case"], variant=chosen["variant"], source=chosen["path"], source_sha256=chosen["sha256"],
                                    parts=chosen["parts"], parts_sha256=chosen["parts_sha256"], box=chosen["box"], size=[x1 - x0, y1 - y0],
                                    mask_keys=[k for k in HEAD_KEYS if k in parts], mask_px=int(mask.sum()),
                                    reference=str(directory / "memory-reference.png"), reference_sha256=sha(directory / "memory-reference.png"),
                                    mask=str(directory / "memory-reference-mask.png"), mask_sha256=sha(directory / "memory-reference-mask.png"))
        manifest["characters"][char] = record
        print(char, "->", record["chosen"] and record["chosen"]["case"] + "/" + record["chosen"]["variant"],
              [(c["case"], c["variant"], c.get("hair_band_error_lab"), c.get("skin_delta_e"), c["excluded"]) for c in record["candidates"]])
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
