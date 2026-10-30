"""모드 B 본체 = FLUX.2-klein-4B 편집(D-210, 사용자 10/30).

원본 그림을 참조로 넣고 S3b 그림체 문장으로 같은 캐릭터를 다시 그린다. RealVisXL + 스타일 LoRA는 본체가 선·칠을 정해 셀 그림체에
머물렀다(D-209). FLUX는 학습 자료를 만든 모델이라 S3b를 그대로 그리고, IP-Adapter(ViT-H)가 필요 없다(Apache-2.0).
- 자세: 원본과 같은 자세·팔·손을 지시한다("팔을 내리고 서 있음"을 넣으면 팔짱 원본에서 팔이 늘어났다).
- 색: 편집은 색을 밝고 선명한 쪽으로 옮긴다(짙은 피부가 밝아짐, 와인→빨강, 청록→파랑) → 사람 영역 안에서 부위별(피부·머리·윗옷·아래옷)
  Lab 평균·퍼짐을 원본 쪽으로 옮긴다. 명암(그림체)은 퍼짐 비율을 0.8~1.25로 묶어 유지하고, 한 번에 옮기는 양에 상한을 둔다.
- 손: 결과의 손 개수가 원본보다 많으면 표시한다(후보 고를 때 참고, 자동 탈락은 아님).
"""
import numpy as np

SEEDS = (42, 43, 44)
STEPS = 4
GUIDANCE = 1.0
MAX_SEQUENCE_LENGTH = 256
STYLE = ("anime illustration with semi-realistic painted shading, detailed digital painting: clean thin anime line art, anime facial "
         "proportions with expressive eyes, soft painted skin shading with gentle glossy highlights, fine individual hair strands with rim "
         "light, well-defined fabric materials with natural folds, strong directional sunlight with soft shadows, rich colours")
BACKGROUND = "Plain warm-gray studio background, soft warm key light from upper left."
COLOUR_SHIFT_CAP = np.array([15.0, 12.0, 12.0])     # L, a, b — 이보다 크게 옮기지 않는다(잘못 잡은 영역의 피해를 줄임)
SPREAD_CLIP = (0.8, 1.25)
MIN_PART_PIXELS = 200
PARTS = ("skin", "hair", "upper", "lower")


def flux_prompt(traits):
    """같은 캐릭터를 S3b 그림체로 다시 그리라는 문장. 특징 문장(자동 측정)과 피부색 유지·같은 자세 지시를 넣는다."""
    described = "; ".join(row["text"].strip() for row in traits if row.get("text", "").strip())
    keep = f" ({described})" if described else ""
    return (f"Redraw the same adult character as in the reference image as a new {STYLE}. Keep the same face, hairstyle, hair colour, "
            f"eye colour, skin tone and outfit exactly as in the reference{keep}. Keep the exact same skin tone as the reference; do not "
            f"lighten or darken the skin. Keep the same pose, arms, hands and framing as in the reference. {BACKGROUND} "
            f"No lettering, logo, watermark or multiple people.")


def part_masks(rgb, parts, person):
    """분할 결과(figure_parts)와 사람 영역(anime-seg)에서 색 맞춤할 네 부위. 모두 사람 영역 안으로 자른다."""
    from genai_lab.style_reference_colors import skin_like
    shape = np.asarray(rgb).shape[:2]
    empty = np.zeros(shape, bool)
    def any_of(*keys):
        return np.logical_or.reduce([np.asarray(parts.get(k, empty), bool) for k in keys])
    person = np.asarray(person, bool)
    clothes = any_of("topwear", "handwear", "bottomwear", "legwear", "footwear")
    features = any_of("hair", "eyes", "mouth", "eyewear")
    skin = ((skin_like(rgb, parts["face"]) & ~clothes) | any_of("face", "neck")) & ~features & person
    return dict(skin=skin, hair=any_of("hair") & person, upper=any_of("topwear", "handwear") & ~skin & person,
                lower=any_of("bottomwear", "legwear") & ~skin & person)


def colour_match(result_rgb, original_rgb, result_masks, original_masks, feather=2.0):
    """부위마다 결과 색을 원본 쪽으로(Lab 평균 이동 + 퍼짐 비율). 반환: (RGB uint8, {부위: 옮긴 양 또는 None})."""
    from scipy.ndimage import gaussian_filter
    from genai_lab.style_reference_colors import lab_to_rgb, rgb_to_lab
    lab, source = rgb_to_lab(np.asarray(result_rgb)), rgb_to_lab(np.asarray(original_rgb))
    shifts = {}
    for part in PARTS:
        mr, ms = np.asarray(result_masks[part], bool), np.asarray(original_masks[part], bool)
        if mr.sum() < MIN_PART_PIXELS or ms.sum() < MIN_PART_PIXELS:
            shifts[part] = None
            continue
        a, b = lab[mr], source[ms]
        mu_r, sd_r, mu_s, sd_s = a.mean(0), a.std(0) + 1e-3, b.mean(0), b.std(0) + 1e-3
        shift = np.clip(mu_s - mu_r, -COLOUR_SHIFT_CAP, COLOUR_SHIFT_CAP)
        fixed = (lab - mu_r) * np.clip(sd_s / sd_r, *SPREAD_CLIP) + mu_r + shift
        w = gaussian_filter(mr.astype(np.float32), feather)[..., None] if feather else mr[..., None].astype(np.float32)
        lab = lab * (1 - w) + fixed * w
        shifts[part] = [round(float(x), 1) for x in shift]
    return np.clip(lab_to_rgb(lab), 0, 255).astype(np.uint8), shifts


def hand_flag(original_count, result_count):
    """손이 원본보다 많으면 팔·손이 늘어난 후보로 표시. 셀 수 없으면 None."""
    if original_count is None or result_count is None:
        return None
    return result_count > original_count
