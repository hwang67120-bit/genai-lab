"""Diagnostic text preserves exact punctuation and identifies actual vs sealed text."""
import hashlib
import pytest
from genai_lab.selected_garment_correction import _diagnostic_prompt


def test_exact_override_preserves_whitespace_and_records_both_prompts():
    assembled = "full body, original topology sentence."
    override = "full body, crop top, midriff, navel  "
    actual, evidence = _diagnostic_prompt(assembled, override)
    assert actual == override
    assert evidence["assembled_prompt"] == assembled
    assert evidence["actual_prompt"] == override
    assert evidence["actual_sha256"] == hashlib.sha256(override.encode()).hexdigest()
    assert evidence["assembled_sha256"] == hashlib.sha256(assembled.encode()).hexdigest()
    assert evidence["scope"] == "refinement_positive_prompt_only"


@pytest.mark.parametrize("invalid", ["", "  ", ["crop top"], 42])
def test_invalid_diagnostic_text_is_not_silently_normalized(invalid):
    with pytest.raises(ValueError, match="nonempty string"):
        _diagnostic_prompt("original", invalid)
