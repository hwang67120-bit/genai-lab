from PIL import Image

from genai_lab.person_count_diagnostic import evaluate_person_count


def config(**overrides):
    section = {
        "enabled": True,
        "expected_count": 1,
        "minimum_area_ratio": 0.01,
        "duplicate_iou": 0.65,
        "blocking": False,
    }
    section.update(overrides)
    return {"person_count_diagnostic": section}


def test_person_count_deduplicates_overlapping_boxes(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    try:
        report = evaluate_person_count(
            image,
            config(),
            detector=lambda *_: (
                [(10, 10, 80, 90), (12, 12, 79, 88)],
                [0.95, 0.80],
            ),
            output_directory=tmp_path,
        )
    finally:
        image.close()

    assert report["status"] == "PASS"
    assert report["detected_count"] == 1
    assert report["raw_detection_count"] == 2
    assert report["return_blocked"] is False
    assert (tmp_path / "person-count-overlay.png").is_file()


def test_person_count_mismatch_is_diagnostic_only():
    image = Image.new("RGB", (100, 100), "white")
    try:
        report = evaluate_person_count(
            image,
            config(),
            detector=lambda *_: (
                [(0, 0, 40, 90), (55, 0, 99, 90)],
                [0.9, 0.85],
            ),
        )
    finally:
        image.close()

    assert report["status"] == "REVIEW"
    assert report["detected_count"] == 2
    assert report["matches_expected"] is False
    assert report["return_blocked"] is False


def test_person_count_disabled_does_not_call_detector():
    image = Image.new("RGB", (32, 32), "white")
    called = []
    try:
        report = evaluate_person_count(
            image,
            {"person_count_diagnostic": {"enabled": False}},
            detector=lambda *_: called.append(True),
        )
    finally:
        image.close()

    assert report["status"] == "DISABLED"
    assert called == []
