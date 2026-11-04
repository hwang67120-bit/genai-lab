"""옷 교체의 원본 구도 전달 회귀검사. GPU 불필요."""
import pytest
from genai_lab.style_redraw_flux import flux_prompt, FULL_BODY_FRAMING

@pytest.mark.parametrize("shot", [
    "Upper-body shot: head to waist; do not show the legs.",
    "Cowboy shot: head to thighs; do not show the lower legs or feet.",
])
def test_garment_only_delivers_explicit_framing(shot):
    prompt = flux_prompt([], with_garment=True, shot=shot, gender="male",
                         garment_words=["white shirt", "black vest"])
    assert shot in prompt
    assert "white shirt" in prompt
    assert "black vest" in prompt
    assert "adult man" in prompt

def test_default_garment_framing_keeps_original_instruction():
    prompt = flux_prompt([], with_garment=True)
    assert FULL_BODY_FRAMING not in prompt
    assert "Keep the same pose and framing as image 1. Render as a " in prompt

def test_garment_only_excludes_original_clothes():
    prompt = flux_prompt([{"name": "clothes", "text": "old red jacket"}],
                         with_garment=True, shot="Upper-body shot: do not show the legs.")
    assert "old red jacket" not in prompt
