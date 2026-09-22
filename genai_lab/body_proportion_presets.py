"""Project-authored body-proportion anchors for the Animagine Base stage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps


ASSET_VERSION = "body_proportion_presets_v3"
ASSET_ROOT = Path("inputs/body_proportion_presets/v3")
CANVAS_SIZE = (512, 768)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BodyProportionPreset:
    preset_id: str
    family: str
    label_ko: str
    head_count: float
    life_stage: str
    body_form: str
    shoulder_width: float
    waist_width: float
    pelvis_width: float
    limb_thickness: float

    def __post_init__(self) -> None:
        if not self.preset_id or not self.family or not self.label_ko:
            raise ValueError("체형 프리셋 식별자와 이름이 필요합니다.")
        if not 2.0 <= float(self.head_count) <= 9.0:
            raise ValueError("체형 프리셋 등신 값은 2~9 범위여야 합니다.")
        for name in (
            "shoulder_width",
            "waist_width",
            "pelvis_width",
            "limb_thickness",
        ):
            value = getattr(self, name)
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"체형 프리셋 {name}은 0~1 범위여야 합니다.")

    @property
    def control_path(self) -> Path:
        return ASSET_ROOT / "control" / f"{self.preset_id}.png"

    @property
    def preview_path(self) -> Path:
        return ASSET_ROOT / "preview" / f"{self.preset_id}.png"

    def record(self) -> dict[str, Any]:
        return {
            "version": ASSET_VERSION,
            "preset_id": self.preset_id,
            "family": self.family,
            "label_ko": self.label_ko,
            "head_count": self.head_count,
            "life_stage": self.life_stage,
            "body_form": self.body_form,
            "geometry": {
                "shoulder_width": self.shoulder_width,
                "waist_width": self.waist_width,
                "pelvis_width": self.pelvis_width,
                "limb_thickness": self.limb_thickness,
            },
            "control_path": self.control_path.as_posix(),
            "preview_path": self.preview_path.as_posix(),
            "gender_conditioning_applied": False,
            "gender_identity_source": "separate_character_request",
            "openpose_used": False,
            "application_stage": "animagine_base_only",
            "model_conditioning_applied": True,
        }


BODY_PROPORTION_PRESETS = (
    BodyProportionPreset(
        "sd_3h_neutral",
        "super_deformed",
        "SD 3등신 중립",
        3.0,
        "stylized_age_neutral",
        "neutral",
        .54,
        .42,
        .50,
        .18,
    ),
    BodyProportionPreset(
        "child_5h_neutral",
        "child_five_head",
        "아동 5등신 중립",
        5.0,
        "child",
        "neutral",
        .42,
        .34,
        .39,
        .12,
    ),
    BodyProportionPreset(
        "standard_7_5h_shoulder",
        "standard_adult",
        "표준 7.5등신 어깨 중심",
        7.5,
        "adult",
        "shoulder_dominant",
        .48,
        .30,
        .36,
        .10,
    ),
    BodyProportionPreset(
        "standard_7_5h_hip",
        "standard_adult",
        "표준 7.5등신 골반 중심",
        7.5,
        "adult",
        "hip_dominant",
        .38,
        .28,
        .44,
        .095,
    ),
    BodyProportionPreset(
        "fashion_8h_shoulder",
        "fashion_adult",
        "패션 8등신 어깨 중심",
        8.0,
        "adult",
        "shoulder_dominant",
        .44,
        .27,
        .33,
        .082,
    ),
    BodyProportionPreset(
        "fashion_8h_hip",
        "fashion_adult",
        "패션 8등신 골반 중심",
        8.0,
        "adult",
        "hip_dominant",
        .35,
        .25,
        .40,
        .078,
    ),
)


_PRESETS_BY_ID = {item.preset_id: item for item in BODY_PROPORTION_PRESETS}


def get_body_proportion_preset(preset_id: str) -> BodyProportionPreset:
    try:
        return _PRESETS_BY_ID[str(preset_id)]
    except KeyError as error:
        raise ValueError(f"지원하지 않는 체형 프리셋입니다: {preset_id}") from error


def load_body_proportion_control(
    project_root: str | Path,
    preset_id: str,
) -> Image.Image:
    preset = get_body_proportion_preset(preset_id)
    path = Path(project_root) / preset.control_path
    if not path.is_file():
        raise ValueError(f"체형 프리셋 제어 이미지가 없습니다: {path}")
    with Image.open(path) as opened:
        opened.load()
        image = opened.convert("L")
    if image.size != CANVAS_SIZE or image.getbbox() is None:
        image.close()
        raise ValueError(f"체형 프리셋 제어 이미지가 손상됐습니다: {path}")
    return image

@dataclass(frozen=True)
class BodyProportionControlSettings:
    enabled: bool
    status: str
    model_id: str
    conditioning_scale: float
    guidance_start: float
    guidance_end: float
    local_files_only: bool
    user_selection_required: bool

    def record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "model_id": self.model_id,
            "conditioning_scale": self.conditioning_scale,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
            "local_files_only": self.local_files_only,
            "user_selection_required": self.user_selection_required,
            "application_stage": "animagine_base_only",
            "openpose_used": False,
        }


def resolve_body_proportion_control(
    config: Mapping[str, Any] | None,
) -> BodyProportionControlSettings:
    raw = (config or {}).get("body_proportion_presets", {})
    if not isinstance(raw, Mapping):
        raise ValueError("body_proportion_presets는 항목 묶음이어야 합니다.")
    enabled = raw.get("enabled", False)
    if enabled not in (True, False):
        raise ValueError("body_proportion_presets.enabled는 true 또는 false여야 합니다.")
    settings = BodyProportionControlSettings(
        enabled=enabled is True,
        status=str(raw.get(
            "status",
            "animagine_base_connected" if enabled is True else "disabled",
        )),
        model_id=str(raw.get(
            "model_id", "xinsir/controlnet-canny-sdxl-1.0"
        )),
        conditioning_scale=float(raw.get("conditioning_scale", .42)),
        guidance_start=float(raw.get("guidance_start", 0.0)),
        guidance_end=float(raw.get("guidance_end", .72)),
        local_files_only=bool(raw.get("local_files_only", True)),
        user_selection_required=bool(
            raw.get("user_selection_required", True)
        ),
    )
    if settings.enabled:
        if settings.status != "animagine_base_connected":
            raise ValueError("활성 체형 프리셋 상태가 Animagine Base 연결 상태가 아닙니다.")
        if (config or {}).get("model", {}).get("family") != "sdxl":
            raise ValueError("체형 프리셋 ControlNet에는 SDXL 모델이 필요합니다.")
        if not settings.model_id.strip():
            raise ValueError("체형 프리셋 ControlNet 모델 ID가 비어 있습니다.")
        values = (
            settings.conditioning_scale,
            settings.guidance_start,
            settings.guidance_end,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("체형 프리셋 강도와 적용 구간은 0~1 유한값이어야 합니다.")
        if settings.guidance_start >= settings.guidance_end:
            raise ValueError("체형 프리셋 적용 시작은 종료보다 작아야 합니다.")
    return settings


def _manifest_record(project_root: str | Path, preset_id: str) -> dict[str, Any]:
    path = Path(project_root) / ASSET_ROOT / "manifest.json"
    if not path.is_file():
        raise ValueError(f"체형 프리셋 manifest가 없습니다: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != ASSET_VERSION
        or manifest.get("status") != "approved_asset"
    ):
        raise ValueError("검증되지 않은 체형 프리셋 manifest입니다.")
    by_id = {
        str(record.get("preset_id")): record
        for record in manifest.get("presets", ())
    }
    record = by_id.get(preset_id)
    if record is None:
        raise ValueError(f"manifest에 체형 프리셋이 없습니다: {preset_id}")
    control_path = Path(project_root) / str(record["control_path"])
    actual = hashlib.sha256(control_path.read_bytes()).hexdigest()
    if actual != record.get("control_sha256"):
        raise ValueError(f"체형 프리셋 해시가 다릅니다: {preset_id}")
    return record


def body_proportion_contract(
    config: Mapping[str, Any],
    request: Any,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    settings = resolve_body_proportion_control(config)
    preset_id = getattr(request, "body_proportion_preset_id", None)
    if not settings.enabled:
        return {
            "version": ASSET_VERSION,
            "status": "disabled",
            "preset_id": preset_id,
            "settings": settings.record(),
        }
    if preset_id is None:
        return {
            "version": ASSET_VERSION,
            "status": "not_selected",
            "preset_id": None,
            "settings": settings.record(),
        }
    preset = get_body_proportion_preset(str(preset_id))
    source = _manifest_record(project_root, preset.preset_id)
    framing = getattr(
        getattr(request, "framing_type", None),
        "value",
        getattr(request, "framing_type", None),
    )
    status = "active" if framing == "full_body" else "skipped_non_full_body"
    return {
        "version": ASSET_VERSION,
        "status": status,
        "preset_id": preset.preset_id,
        "preset": preset.record(),
        "control_sha256": source["control_sha256"],
        "render_profile": source.get("render_profile"),
        "settings": settings.record(),
    }


def active_body_proportion_preset_id(request: Any) -> str | None:
    """Return the preset that may condition an Animagine full-body pass."""
    preset_id = getattr(request, "body_proportion_preset_id", None)
    framing = getattr(
        getattr(request, "framing_type", None),
        "value",
        getattr(request, "framing_type", None),
    )
    if preset_id is None or framing != "full_body":
        return None
    return str(preset_id)


def prepare_body_proportion_control(
    config: Mapping[str, Any],
    request: Any,
    project_root: str | Path,
) -> tuple[Image.Image | None, dict[str, Any]]:
    contract = body_proportion_contract(config, request, project_root)
    if contract["status"] != "active":
        return None, contract
    control = load_body_proportion_control(
        project_root, str(contract["preset_id"])
    )
    try:
        fitted = ImageOps.pad(
            control,
            (int(request.width), int(request.height)),
            method=Image.Resampling.LANCZOS,
            color=0,
            centering=(.5, .5),
        ).convert("RGB")
    finally:
        control.close()
    contract = {
        **contract,
        "runtime_size": [fitted.width, fitted.height],
        "coordinate_source": "project_preset_output_canvas",
    }
    return fitted, contract
