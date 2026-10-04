from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import yaml

from genai_lab.approved_base_replay import load_approved_base_batch


VISUAL_FINGERPRINT = (
    "bcfc90ef08980cb2e82b1b244cbca19e094a1dc42d3aa10c5d3f0d5350bef8f4"
)


class Approval:
    def __init__(self, visual_fingerprint=VISUAL_FINGERPRINT):
        self.fingerprint = "c" * 64
        self._record = {
            "visual_fingerprint": visual_fingerprint,
            "gender": "male",
        }

    def record(self):
        return dict(self._record)


def request():
    return SimpleNamespace(
        model_id="cagliostrolab/animagine-xl-3.1",
        width=768,
        height=1344,
        framing_type=SimpleNamespace(value="full_body"),
        reference_image_name="character.png",
        reference_enhancement_applied=False,
        reference_enhancement_model_id=None,
        reference_quality_status="approved",
        reference_adapter_id="h94/IP-Adapter",
        original_image_change_strength=.25,
        reference_image_strength=.8,
    )


def inputs():
    return SimpleNamespace(
        source=Image.new("RGB", (8, 8), "white"),
        garment=Image.new("RGB", (8, 8), "blue"),
    )


def test_unreviewed_configured_base_replay_is_disabled():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "animagine.yaml").read_text(
            encoding="utf-8"
        )
    )
    visual_inputs = inputs()
    try:
        assert config["approved_base_replay"]["enabled"] is False
        assert load_approved_base_batch(
            config,
            request(),
            project_root,
            visual_inputs,
            Approval(),
        ) is None
    finally:
        visual_inputs.source.close()
        visual_inputs.garment.close()


def test_different_visual_fingerprint_does_not_replay_approved_base():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "animagine.yaml").read_text(
            encoding="utf-8"
        )
    )
    visual_inputs = inputs()
    try:
        assert load_approved_base_batch(
            config,
            request(),
            project_root,
            visual_inputs,
            Approval("d" * 64),
        ) is None
    finally:
        visual_inputs.source.close()
        visual_inputs.garment.close()
