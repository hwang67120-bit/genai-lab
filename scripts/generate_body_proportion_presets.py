"""Split the approved human-form reference sheet into deterministic presets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.body_proportion_presets import (
    ASSET_ROOT,
    ASSET_VERSION,
    BODY_PROPORTION_PRESETS,
    CANVAS_SIZE,
)

SOURCE_NAME = "human_form_reference_sheet.png"
SOURCE_PATH = ASSET_ROOT / "source" / SOURCE_NAME
PANEL_BOXES = (
    (2, 1, 397, 597),
    (399, 1, 820, 597),
    (822, 1, 1215, 597),
    (2, 600, 397, 1292),
    (399, 600, 820, 1292),
    (822, 600, 1215, 1292),
)
RENDER_PROFILE = "project_generated_human_form_sheet_v3"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_panel(source, box):
    panel = source.crop(box)
    fitted = ImageOps.pad(
        panel,
        CANVAS_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )
    panel.close()
    return fitted


def _line_control(preview):
    gray = ImageOps.grayscale(preview)
    # Preserve the dark anatomy contour and construction details while removing
    # the pale blue proportion grid and neutral fill.
    control = gray.point(lambda value: 255 if value < 170 else 0, mode="L")
    # Panel separators from the source sheet must never become body controls.
    ImageDraw.Draw(control).rectangle(
        (0, 0, control.width - 1, control.height - 1),
        outline=0,
        width=12,
    )
    return control


def main():
    destination = PROJECT_ROOT / ASSET_ROOT
    source_path = PROJECT_ROOT / SOURCE_PATH
    if not source_path.is_file():
        raise ValueError(f"체형 원본 시트가 없습니다: {source_path}")

    control_dir = destination / "control"
    preview_dir = destination / "preview"
    control_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as opened:
        opened.load()
        source = opened.convert("RGB")
    if source.size != (1216, 1293):
        source.close()
        raise ValueError(f"승인된 체형 원본 시트 크기가 다릅니다: {source.size}")

    records = []
    previews = []
    for preset, box in zip(BODY_PROPORTION_PRESETS, PANEL_BOXES, strict=True):
        preview = _fit_panel(source, box)
        control = _line_control(preview)
        preview_path = PROJECT_ROOT / preset.preview_path
        control_path = PROJECT_ROOT / preset.control_path
        preview.save(preview_path, optimize=True)
        control.save(control_path, optimize=True)
        previews.append((preset, preview.copy()))
        preview.close()
        control.close()

        record = preset.record()
        record["render_profile"] = RENDER_PROFILE
        record["panel_box"] = list(box)
        record["source_path"] = SOURCE_PATH.as_posix()
        record["control_sha256"] = _sha256(control_path)
        record["preview_sha256"] = _sha256(preview_path)
        records.append(record)
    source.close()

    card_w, card_h = CANVAS_SIZE
    sheet = Image.new("RGB", (card_w * 3, (card_h + 48) * 2), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    for index, (preset, preview) in enumerate(previews):
        x = index % 3 * card_w
        y = index // 3 * (card_h + 48)
        sheet.paste(preview, (x, y))
        draw.text(
            (x + 16, y + card_h + 14),
            preset.preset_id.replace("_", " ").upper(),
            fill=(240, 240, 240),
        )
        preview.close()
    sheet.save(destination / "contact_sheet.png", optimize=True)
    sheet.close()

    manifest = {
        "version": ASSET_VERSION,
        "status": "approved_asset",
        "render_profile": RENDER_PROFILE,
        "source_type": "project_generated_original",
        "source_path": SOURCE_PATH.as_posix(),
        "source_sha256": _sha256(source_path),
        "reference_usage": "structure_only",
        "copied_character_design": False,
        "canvas_size": list(CANVAS_SIZE),
        "family_count": 4,
        "preset_count": len(records),
        "gender_conditioning_applied": False,
        "openpose_used": False,
        "application_stage": "animagine_base_only",
        "presets": records,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()