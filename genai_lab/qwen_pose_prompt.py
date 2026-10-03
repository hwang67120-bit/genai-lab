"""B-condition natural-language assembly; no model or SDXL tokenizer imports."""
from dataclasses import dataclass
import hashlib

TEMPLATE_VERSION = "qwen_preservation_b_v1"
PREFIX = ("Picture 1 is the approved character and outfit to preserve. "
          "Picture 2 is an OpenPose skeleton used only for the target body pose. ")
PRESERVE = ("Keep the same character identity, hairstyle, body proportions, outfit design, "
            "garment colors, patterns, existing accessories, and illustration style as Picture 1. "
            "Adapt folds and occlusion naturally to the new pose. ")
SUFFIX = ("Keep a plain white background and the entire character in frame. "
          "Do not add or remove garments, accessories, or body parts, "
          "and do not increase exposure beyond the outfit in Picture 1.")


@dataclass(frozen=True)
class PoseEditInstructions:
    viewpoint: str = ""
    hands: str = ""
    confirmation: str = ""

    def __post_init__(self):
        if (self.viewpoint or self.hands) and not self.confirmation.strip():
            raise ValueError("선택 자세 문장은 사용자 확인이 필요합니다.")
        if any(v != v.strip() or "\n" in v for v in (self.viewpoint, self.hands)):
            raise ValueError("선택 문장은 앞뒤 공백 없는 한 줄이어야 합니다.")


def assemble_pose_edit_prompt(spec, instructions=PoseEditInstructions()):
    spec.verify_image()
    kept = [x for x in spec.items if x.status == "confirmed"]
    if not any(x.kind != "style" for x in kept):
        raise ValueError("B 조건은 사용자가 확인한 구체적 보존 항목이 필요합니다.")
    direction = f"{instructions.viewpoint} " if instructions.viewpoint else ""
    pose = f"Redraw the character from Picture 1 in the {direction}body pose of Picture 2. "
    hands = instructions.hands + " " if instructions.hands else ""
    positive = PREFIX + pose + PRESERVE + hands + SUFFIX + " " + " ".join(x.english for x in kept)
    return {"positive": positive, "negative": " ", "template_version": TEMPLATE_VERSION,
            "positive_sha256": hashlib.sha256(positive.encode("utf-8")).hexdigest(),
            "spec_sha256": spec.sha256, "included": [x.id for x in kept],
            "excluded": [{"id": x.id, "reason": x.status} for x in spec.items if x.status != "confirmed"]}
