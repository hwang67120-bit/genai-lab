import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog

import gui_main
from genai_lab.external_candidate import load_external_character_candidate
from genai_lab.visual_reference import CandidateBatch
from gui_main import GenAILabWindow


def test_gui_routes_selected_base_to_native_refinement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    base_path = tmp_path / "candidate_3.png"
    garment_path = tmp_path / "garment.png"
    garment_board_path = tmp_path / "input_garment.png"
    Image.new("RGB", (48, 80), "navy").save(base_path)
    Image.new("RGB", (48, 80), "green").save(garment_path)
    Image.new("RGB", (48, 80), "green").save(garment_board_path)
    candidate_record = base_path.with_suffix(".json")
    candidate_record.write_text("{}", encoding="utf-8")
    approved_run = tmp_path / "approved_generation.json"
    approved_run.write_text("{}", encoding="utf-8")

    loaded = load_external_character_candidate(
        base_path,
        reference_image_name="character.png",
        clothing_reference_name="garment.png",
        framing_type="full_body",
    )
    thumbnail = loaded.image.copy()
    thumbnail.thumbnail((24, 40))
    loaded.image.close()
    candidate = replace(loaded, image=thumbnail)
    batch = CandidateBatch(
        candidates=[candidate],
        paths=[base_path],
        directory=tmp_path,
        review_stage="native_base",
    )
    dialog = SimpleNamespace(
        selection=SimpleNamespace(currentData=lambda: 0)
    )
    monkeypatch.setattr(
        gui_main,
        "VisualCandidateReview",
        lambda *args, **kwargs: dialog,
    )

    window = GenAILabWindow()
    window.config = {
        "native_pipeline_v2": {
            "enabled": True,
            "base_profile": "character_only",
            "base_garment_prompt_enabled": False,
            "base_garment_adapter_enabled": False,
            "native_garment_source": "isolated_garment_board",
            "reuse_animagine_prompt_in_native_stage": False,
            "raw_garment_person_image_allowed": False,
        },
        "refinement_execution": {"enabled": True, "mode": "sdxl_local"},
    }
    window.selected_outfit_path = garment_path
    selection = SimpleNamespace(
        base_image=base_path,
        candidate_record=candidate_record,
        approved_run=approved_run,
        garment_reference=garment_board_path,
    )

    class FakeOrchestrator:
        def select_base_candidate(self, selected_batch, index):
            assert selected_batch is batch
            assert index == 0
            return selection

    orchestrator = FakeOrchestrator()
    window.worker = SimpleNamespace(orchestrator=orchestrator)
    routed: list[tuple] = []
    def accept_after_worker_thread_finishes(value):
        # QDialog.exec() runs a nested event loop. The worker thread may finish
        # and clear window.worker before the dialog returns.
        window.clear_worker()
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        window,
        "execute_approval_dialog",
        accept_after_worker_thread_finishes,
    )
    monkeypatch.setattr(
        window,
        "start_native_refinement",
        lambda candidate, **kwargs: routed.append((candidate, kwargs)),
    )
    try:
        window.generation_completed(batch, None)

        assert len(routed) == 1
        routed_candidate, paths = routed[0]
        assert routed_candidate.image.size == (48, 80)
        assert paths == {
            "orchestrator": orchestrator,
            "selection": selection,
        }
        assert window.pending_character_candidate is None
    finally:
        for routed_candidate, _ in routed:
            routed_candidate.image.close()
        window.close()
        application.processEvents()


def test_gui_shows_local_pipeline_as_primary_action() -> None:
    application = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    try:
        assert "전체 로컬 파이프라인" in window.generate_button.text()
        assert window.refinement_mode_combo.count() == 1
        assert "의상 조건이 없는 Animagine 캐릭터 Base" in window.framing_help.text()
        assert "선택 기능" in window.external_candidate_button.text()
        assert "SDXL" in window.local_engine_status_label.text()
        assert "OmniGen" not in window.local_engine_status_label.text()
    finally:
        window.close()
        application.processEvents()



def test_native_tag_review_resolves_activation_in_outer_scope(monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    source = Image.new("RGB", (32, 48), "blue")
    completed: list[bool] = []
    captured: dict[str, object] = {}
    worker = SimpleNamespace(
        cancel_requested=SimpleNamespace(is_set=lambda: False),
        config={
            "native_pipeline_v2": {
                "enabled": True,
                "base_profile": "character_only",
                "base_garment_prompt_enabled": False,
                "base_garment_adapter_enabled": False,
                "native_garment_source": "isolated_garment_board",
                "reuse_animagine_prompt_in_native_stage": False,
                "raw_garment_person_image_allowed": False,
            },
            "clothing_reference_generation": {
                "require_prompt_approval": True,
                "approved_tags": ("skirt",),
                "character_gender": "male",
                "part_color_descriptions": (),
            },
        },
        generation_request=SimpleNamespace(reference_image=source),
        reference_tokenizers=(),
        character_tags_done=SimpleNamespace(
            set=lambda: completed.append(True)
        ),
        character_tags_approved=False,
    )

    def fake_review(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(gui_main, "CharacterTagReview", fake_review)
    window = GenAILabWindow()
    window.worker = worker
    monkeypatch.setattr(
        window,
        "execute_approval_dialog",
        lambda value: int(QDialog.DialogCode.Rejected),
    )
    try:
        window.review_character_generation_tags(SimpleNamespace())
        assert captured["outfit_tags"] == ()
        assert callable(captured["prompt_builder"])
        assert completed == [True]
    finally:
        source.close()
        window.close()
        application.processEvents()

def test_gui_exposes_refinement_mask_and_person_diagnostics(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    hard = tmp_path / "hard_edit_domain.png"
    soft = tmp_path / "soft_guidance.png"
    person = tmp_path / "person-count-overlay.png"
    for path, color in ((hard, 255), (soft, 128), (person, "red")):
        mode = "L" if isinstance(color, int) else "RGB"
        Image.new(mode, (16, 16), color).save(path)
    result = SimpleNamespace(
        status="PASS",
        report_path=tmp_path / "run.json",
        report={
            "refinement_mode": "sdxl_local",
            "diagnostic_images": {
                "hard_edit_domain": str(hard),
                "soft_guidance": str(soft),
            },
            "local_refinement": {
                "outside_change": {"outside_changed_ratio": 0.0}
            },
            "person_count_diagnostic": {
                "detected_count": 1,
                "overlay_path": str(person),
            },
        },
    )
    window = GenAILabWindow()
    try:
        window.set_refinement_diagnostics(result)
        assert window.refinement_diagnostics_button.isEnabled()
        assert set(window.refinement_diagnostic_paths) == {
            "hard_edit_domain",
            "soft_guidance",
            "person_count_overlay",
        }
        assert "모드=sdxl_local" in window.refinement_diagnostic_summary
        assert "검출 인물=1" in window.refinement_diagnostic_summary
        assert "최대 유도 범위 외 RGB 변화=0.0000%" in window.refinement_diagnostic_summary
    finally:
        window.close()
        application.processEvents()
