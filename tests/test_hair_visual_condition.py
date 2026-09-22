import numpy as np
import pytest
from PIL import Image

from genai_lab.hair_visual_condition import (
    HairVisualConditionSettings,
    hair_identity_separation_available,
    prepare_hair_visual_condition,
    resolve_hair_visual_condition,
)


def mask(box, size=(32, 32)):
    result = Image.new("L", size)
    result.paste(255, box)
    return result


def test_hair_condition_splits_identity_and_uses_neutral_square_crop():
    settings = HairVisualConditionSettings(
        enabled=True, reference_scale=.61, padding_ratio=.10)
    with Image.new("RGB", (32, 32), "white") as source:
        source.paste((20, 80, 220), (8, 2, 24, 14))
        with mask((6, 0, 26, 20)) as identity, mask((6, 20, 26, 32)) as garment:
            with mask((8, 2, 24, 14)) as hair:
                face, reference, record = prepare_hair_visual_condition(
                    source, identity, garment, hair, (), settings)
                hair_pixels = np.asarray(hair) == 255
    try:
        face_pixels = np.asarray(face) == 255
        assert not np.any(face_pixels & hair_pixels)
        assert reference.mode == "RGB"
        assert reference.width == reference.height
        assert reference.getpixel((0, 0)) == (128, 128, 128)
        assert record["status"] == "prepared"
        assert record["reference_scale"] == .61
        assert record["identity_hair_overlap_removed"] > 0
        assert record["other_conditions_mutated"] is False
    finally:
        face.close()
        reference.close()


def test_hair_condition_rejects_unreconciled_other_reference_regions():
    settings = HairVisualConditionSettings(enabled=True)
    with Image.new("RGB", (32, 32), "white") as source:
        with mask((6, 0, 26, 20)) as identity, mask((6, 20, 26, 32)) as garment:
            with mask((8, 2, 24, 14)) as hair, mask((8, 2, 12, 8)) as ears:
                ears_before = ears.tobytes()
                with pytest.raises(ValueError, match="다른 부위 조건과 겹칩니다"):
                    prepare_hair_visual_condition(
                        source, identity, garment, hair, (("ears", ears),), settings)
                assert ears.tobytes() == ears_before


def test_hair_visual_settings_are_bounded():
    config = {"reference_analysis": {"hair_visual_reference": {
        "enabled": True, "reference_scale": .6,
        "padding_ratio": .15, "neutral_background": [128, 128, 128],
    }}}
    assert resolve_hair_visual_condition(config).enabled is True
    config["reference_analysis"]["hair_visual_reference"]["reference_scale"] = 1.1
    with pytest.raises(ValueError, match="0~1"):
        resolve_hair_visual_condition(config)

def test_hair_identity_separation_preflight_rejects_empty_face_region():
    with mask((4, 4, 20, 20)) as identity:
        with mask((4, 4, 20, 20)) as hair:
            assert hair_identity_separation_available(identity, hair) is False
        with mask((4, 4, 12, 12)) as partial_hair:
            assert hair_identity_separation_available(
                identity, partial_hair) is True


