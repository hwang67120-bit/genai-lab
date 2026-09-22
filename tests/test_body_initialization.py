from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from genai_lab.body_initialization import binary, create_body_initialization
from genai_lab.image_digest import calculate_image_pixel_sha256
from genai_lab.original_body_pose import OriginalBodyPose
from genai_lab.pose_estimation import PoseEstimationApprovedInput, PoseJointCoordinateCandidate


@pytest.fixture
def body_inputs():
    size = (256, 384)
    skin = (234, 199, 183)
    source = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((85, 20, 170, 83), fill=skin)
    draw.rectangle((50, 90, 205, 365), fill=(85, 38, 133))
    # Hands deliberately differ: their color must not contaminate face sampling.
    draw.rectangle((43, 182, 67, 209), fill=(20, 30, 50))
    draw.rectangle((189, 182, 213, 209), fill=(20, 30, 50))
    foreground = Image.new("L", size, 0)
    fg = ImageDraw.Draw(foreground)
    fg.rectangle((85, 20, 170, 83), fill=255)
    fg.rectangle((40, 90, 215, 367), fill=255)
    protected = Image.new("L", size, 0)
    ImageDraw.Draw(protected).rectangle((185, 240, 215, 300), fill=255)
    mask = np.zeros((384, 256), dtype=np.uint8)
    mask[90:366, 50:206] = 255
    mask[182:210, 43:68] = 0
    mask[182:210, 189:214] = 0
    mask[np.asarray(protected) > 0] = 0
    removal = Image.fromarray(mask)
    points = {
        "neck": (128, 95), "left_ear": (95, 48), "right_ear": (160, 48),
        "left_eye": (110, 48), "right_eye": (145, 48), "nose": (128, 58),
        "left_shoulder": (95, 100), "right_shoulder": (161, 100),
        "left_elbow": (75, 143), "right_elbow": (181, 143),
        "left_wrist": (58, 185), "right_wrist": (198, 185),
        "left_hip": (108, 195), "right_hip": (148, 195),
        "left_knee": (105, 270), "right_knee": (151, 270),
        "left_ankle": (103, 350), "right_ankle": (153, 350),
    }
    joints = tuple(PoseJointCoordinateCandidate(name, x, y, .99, True)
                   for name, (x, y) in points.items())
    control = Image.new("RGB", size, "black")
    ImageDraw.Draw(control).line((128, 58, 128, 350), fill="green", width=3)
    approved = PoseEstimationApprovedInput(control, joints, len(joints), 0, .3, ("test",))
    pose = OriginalBodyPose(calculate_image_pixel_sha256(source, "RGB"),
                            *size, approved, 1, 1, 100, 0.0)
    yield source, removal, protected, foreground, pose
    for image in (source, removal, protected, foreground):
        image.close()
    pose.close()


def test_partition_colors_and_pixel_protection(body_inputs):
    source, removal, protected, foreground, pose = body_inputs
    original_mask = np.array(removal)
    candidate = create_body_initialization(*body_inputs)
    try:
        skin, wear, background = (binary(candidate.images[name]) for name in (
            "body_skin_region", "body_basewear_region", "body_background_region"))
        assert skin.any() and wear.any() and background.any()
        assert not np.any((skin & wear) | (skin & background) | (wear & background))
        assert np.array_equal(skin | wear | background, binary(removal))
        assert np.array_equal(original_mask, np.asarray(removal))
        original, output = np.asarray(source), np.asarray(candidate.initial_image)
        unchanged = ~binary(removal) | binary(protected)
        assert np.array_equal(original[unchanged], output[unchanged])
        assert np.all(output[skin] == (234, 199, 183))
        assert np.all(output[wear] == (48, 72, 104))
        assert np.all(output[background] == 255)
        assert candidate.metadata["hand_comparison"] == "different_not_blended"
    finally:
        candidate.close()


def test_missing_joint_does_not_invent_position(body_inputs):
    source, removal, protected, foreground, pose = body_inputs
    approved = replace(pose.approved_pose, joint_coordinates=tuple(
        j for j in pose.approved_pose.joint_coordinates if j.joint_name != "left_eye"))
    with pytest.raises(ValueError, match="left_eye"):
        create_body_initialization(source, removal, protected, foreground,
                                   replace(pose, approved_pose=approved))


def test_mismatched_source_is_rejected(body_inputs):
    source, removal, protected, foreground, pose = body_inputs
    with source.copy() as different:
        different.putpixel((0, 0), (1, 2, 3))
        with pytest.raises(ValueError, match="DWPose"):
            create_body_initialization(different, removal, protected, foreground, pose)


@pytest.mark.parametrize("invalid", ["empty", "protection", "outside", "size"])
def test_invalid_masks_fail_before_generation(body_inputs, invalid):
    source, removal, protected, foreground, pose = body_inputs
    with removal.copy() as changed:
        if invalid == "empty":
            changed.paste(0, (0, 0, *source.size))
        elif invalid == "protection":
            changed.putpixel((195, 260), 255)
        elif invalid == "outside":
            changed.putpixel((0, 0), 255)
        else:
            with changed.resize((20, 20)) as small:
                with pytest.raises(ValueError, match="크기"):
                    create_body_initialization(source, small, protected, foreground, pose)
            return
        with pytest.raises(ValueError):
            create_body_initialization(source, changed, protected, foreground, pose)


def test_unusual_skin_color_is_sampled_not_replaced_by_default(body_inputs):
    source, removal, protected, foreground, pose = body_inputs
    with source.copy() as blue:
        ImageDraw.Draw(blue).rectangle((85, 20, 170, 83), fill=(70, 115, 195))
        candidate = create_body_initialization(
            blue, removal, protected, foreground,
            replace(pose, source_image_sha256=calculate_image_pixel_sha256(blue, "RGB")))
        try:
            assert candidate.metadata["skin_rgb"] == [70, 115, 195]
        finally:
            candidate.close()


def test_copy_has_independent_image_ownership(body_inputs):
    candidate = create_body_initialization(*body_inputs)
    copied = candidate.copy()
    candidate.close()
    try:
        assert copied.initial_image.getpixel((128, 140)) == (48, 72, 104)
    finally:
        copied.close()


@pytest.mark.parametrize("fails", [False, True])
def test_initial_and_artifacts_reach_runner_even_on_failure(
    body_inputs, tmp_path, monkeypatch, fails,
):
    import json
    import subprocess
    from pathlib import Path
    from test_garment_inpaint import _settings, _success
    from genai_lab.garment_inpaint import execute_garment_inpaint, GarmentInpaintError
    source, removal, protected, foreground, pose = body_inputs
    initialization = create_body_initialization(*body_inputs)
    initial_array = np.array(initialization.initial_image)
    recorded_paths = []

    def run(command, **kwargs):
        initial_path = Path(command[command.index("--initial-image") + 1])
        recorded_paths.append(initial_path.parent)
        with Image.open(initial_path) as initial:
            assert np.array_equal(initial_array, np.asarray(initial))
        assert (initial_path.parent / "body_proxy.png").is_file()
        if fails:
            return subprocess.CompletedProcess(command, 1, "", "test failure")
        return _success(command, **kwargs)

    monkeypatch.setattr("genai_lab.garment_inpaint.subprocess.run", run)
    try:
        with Image.new("RGBA", (32, 32), "white") as dummy:
            args = (source, initialization.initial_image, removal, dummy, "opaque basic one-piece",
                    "gray body", 7, replace(_settings(tmp_path), operation="body_restoration"))
            if fails:
                with pytest.raises(GarmentInpaintError, match="test failure"):
                    execute_garment_inpaint(
                        *args, body_pose_control_image=pose.approved_pose.control_map_image,
                        body_initialization=initialization)
            else:
                result = execute_garment_inpaint(
                    *args, body_pose_control_image=pose.approved_pose.control_map_image,
                    body_initialization=initialization)
                assert result.protected_changed_outside_mask_pixels == 0
                result.close()
        metadata = json.loads((recorded_paths[0] / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == ("failed" if fails else "completed")
        assert metadata["body_initialization"]["skin_rgb"] == [234, 199, 183]
        assert len(metadata["body_initialization"]["artifacts"]) == 7
    finally:
        initialization.close()


def test_gui_worker_owns_and_forwards_initialization(body_inputs, tmp_path, monkeypatch):
    from test_garment_inpaint import _settings
    from gui_main import GarmentInpaintWorker
    source, removal, protected, foreground, pose = body_inputs
    initialization = create_body_initialization(*body_inputs)
    expected = np.array(initialization.initial_image)
    with Image.new("RGBA", (32, 32), "white") as dummy:
        worker = GarmentInpaintWorker(
            source, initialization.initial_image, removal, removal, dummy,
            "opaque basic one-piece", "", 7,
            replace(_settings(tmp_path), operation="body_restoration"),
            body_pose_control_image=pose.approved_pose.control_map_image,
            body_initialization=initialization,
        )
    initialization.close()
    calls = []
    failures = []
    worker.failed.connect(lambda message, details: failures.append((message, details)))

    def execute(*args, **kwargs):
        assert np.array_equal(np.asarray(args[1]), expected)
        assert kwargs["body_initialization"].metadata["skin_rgb"] == [234, 199, 183]
        assert np.array_equal(np.asarray(kwargs["body_initialization"].initial_image), expected)
        calls.append(True)
        return None

    monkeypatch.setattr("gui_main.execute_garment_inpaint", execute)
    worker.run()
    assert not failures
    assert calls == [True]


def test_initialization_review_cancel_never_starts_gpu(body_inputs, monkeypatch):
    monkeypatch.setattr('gui_main.GenerationResolutionDialog.exec', lambda self: 1)
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication, QDialog
    from gui_main import GenAILabWindow
    application = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    source, removal, protected, foreground, pose = body_inputs
    window.pending_clothing_base_candidate = SimpleNamespace(image=source)
    window.confirmed_character_body_comparison = SimpleNamespace(
        approved_change_mask=removal, approved_foreground_mask=foreground,
        approved_protection_mask=protected)
    window.original_body_pose = pose
    seen = []
    paused = []

    def reject(dialog):
        seen.append(dialog.windowTitle())
        return int(QDialog.DialogCode.Rejected)

    monkeypatch.setattr(window, "execute_approval_dialog", reject)
    monkeypatch.setattr(window, "pause_generation_workflow", lambda *args: paused.append(args))
    monkeypatch.setattr(window, "release_step5_pipeline", lambda: pytest.fail("GPU reached"))
    try:
        window._start_inpaint(restoration_only=True)
        assert seen and "4단계" in seen[0]
        assert paused
        assert window.garment_inpaint_worker is None
    finally:
        window.pending_clothing_base_candidate = None
        window.confirmed_character_body_comparison = None
        window.original_body_pose = None
        window.close()


def test_approved_initialization_is_used_when_gui_constructs_worker(
    body_inputs, monkeypatch, tmp_path,
):
    monkeypatch.setattr('gui_main.GenerationResolutionDialog.exec', lambda self: 1)
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication, QDialog
    from gui_main import GenAILabWindow
    from test_garment_inpaint import _settings
    application = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    source, removal, protected, foreground, pose = body_inputs
    initial = create_body_initialization(*body_inputs)
    expected = np.array(initial.initial_image)
    initial.close()
    window.pending_clothing_base_candidate = SimpleNamespace(
        image=source, prompt="old clothes", negative_prompt="", seed=7)
    window.confirmed_character_body_comparison = SimpleNamespace(
        approved_change_mask=removal, approved_composite_mask=removal,
        approved_foreground_mask=foreground, approved_protection_mask=protected,
        neutral_rgb=(127, 127, 127))
    window.original_body_pose = pose
    monkeypatch.setattr(window, "execute_approval_dialog",
                        lambda dialog: int(QDialog.DialogCode.Accepted))
    monkeypatch.setattr(window, "release_step5_pipeline", lambda: {
        "before_allocated_mib": 0, "after_allocated_mib": 0, "after_reserved_mib": 0})
    monkeypatch.setattr(window, "create_garment_inpaint_settings", lambda: _settings(tmp_path))
    calls = []

    class CapturedBeforeThreadStart(RuntimeError):
        pass

    def worker(*args, **kwargs):
        assert np.array_equal(np.asarray(args[1]), expected)
        assert np.array_equal(np.asarray(args[2]), np.asarray(removal))
        assert kwargs["body_initialization"] is not None
        assert kwargs["body_pose_control_image"] is not None
        calls.append(True)
        raise CapturedBeforeThreadStart

    monkeypatch.setattr("gui_main.GarmentInpaintWorker", worker)
    try:
        with pytest.raises(CapturedBeforeThreadStart):
            window._start_inpaint(restoration_only=True)
        assert calls == [True]
    finally:
        window.pending_clothing_base_candidate = None
        window.confirmed_character_body_comparison = None
        window.original_body_pose = None
        if window.garment_inpaint_worker_thread is not None:
            window.garment_inpaint_worker_thread.deleteLater()
        window.garment_inpaint_worker_thread = None
        window.close()
