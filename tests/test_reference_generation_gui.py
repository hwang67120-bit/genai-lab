from pathlib import Path
from types import SimpleNamespace
import pytest

from PySide6.QtWidgets import QApplication

from gui_main import GenAILabWindow
from genai_lab.workflow import GenerationWorkflowContext, GenerationWorkflowStage
from genai_lab.clothing_reference_generation_review import ClothingReferenceGenerationDialog


def test_reference_result_goes_directly_to_review(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.outfit_path = 'outfit.png'
    window.workflow_context = GenerationWorkflowContext(Path('character.png'), Path('outfit.png'), None,
                                                        reference_generation=True)
    candidate = SimpleNamespace(original_generated_image=None, before_clothing_image=None,
                                clothing_try_on_status='not_requested')
    shown = []
    monkeypatch.setattr(window, 'show_character_candidate', shown.append)
    def forbidden():
        raise AssertionError('legacy stage must not start')
    monkeypatch.setattr(window, 'start_character_body_comparison', forbidden)
    window.generation_completed(candidate, None)
    assert shown == [candidate]
    assert window.pending_clothing_base_candidate is None
    assert window.pending_character_candidate is candidate
    assert window.workflow_context.current_stage is GenerationWorkflowStage.FINAL_REVIEW
    assert window.workflow_context.progress == (5, 5)
    assert not window.save_candidate_button.isEnabled()
    window.pending_character_candidate = None
    window.workflow_context = None
    window.close()


def test_reference_dialog_resolution_strength_and_no_body_instruction():
    app = QApplication.instance() or QApplication([])
    dialog = ClothingReferenceGenerationDialog((768, 1344), ('blue jacket', 'gold buttons'))
    dialog.width_input.setValue(512)
    assert '512×896' in dialog.preview.text()
    assert not hasattr(dialog, 'strength_input')
    assert dialog.identity_scale_input.value() == .7
    assert dialog.garment_scale_input.value() == .45
    assert not hasattr(dialog, 'no_initial_input')
    assert dialog.candidate_count_input.value() == 2
    assert dialog.candidate_count_input.minimum() == 2
    assert not hasattr(dialog, 'garment_ab_input')
    assert dialog.garment_scale_input.minimum() > 0
    dialog.candidate_count_input.setValue(1)
    assert dialog.candidate_count_input.value() == 2
    assert '생성 결과는 아래 연산 해상도로 저장' in dialog.layout().itemAt(0).widget().text()
    dialog.close()


def test_character_tag_review_is_editable_and_does_not_auto_select_gender():
    from PIL import Image
    from genai_lab.character_tag_review import CharacterTagReview
    from genai_lab.clothing_reference import ClothingDesignTagCandidate
    app = QApplication.instance() or QApplication([])
    image = Image.new('RGB', (64, 96), 'white')
    result = SimpleNamespace(tag_candidates=tuple(ClothingDesignTagCandidate(tag, tag, .9)
        for tag in ('1boy', 'blue_hair', 'blue_eyes', 'jacket')))
    dialog = CharacterTagReview(image, result)
    assert dialog.approved_tags == ('blue_hair', 'blue_eyes')
    assert not dialog.result()
    for name, box in dialog.tag_checkboxes:
        box.setChecked(name == '1boy')
    assert dialog.approved_tags == ('1boy',)
    dialog.close()
    image.close()


def test_pose_is_explicitly_disabled_for_minimal_reference_mode():
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.set_workflow_input_buttons_enabled(True)
    assert not window.pose_button.isEnabled()
    assert not window.body_comparison_button.isVisible()
    window.close()


def test_reference_mode_requires_registered_outfit_to_enable_start():
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.style_path = 'character.png'
    window.update_input_ready_status()
    assert not window.generate_button.isEnabled()
    assert '의상 이미지를 모두 등록' in window.status_label.text()
    window.selected_outfit_path = Path('outfit.png')
    window.update_input_ready_status()
    assert window.generate_button.isEnabled()
    window.close()


def test_category_selector_removed_and_framing_keeps_start_enabled():
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.style_path = 'character.png'
    window.selected_outfit_path = Path('outfit.png')
    window.update_input_ready_status()
    assert window.generate_button.isEnabled()
    assert window.approved_reference_image is None
    assert window.confirmed_clothing_design is None
    assert not hasattr(window, 'clothing_category_combo')
    assert not hasattr(window, 'invalidate_character_body_comparison')
    assert window.framing_combo.count() == 3
    for index in range(window.framing_combo.count()):
        window.framing_combo.setCurrentIndex(index)
        assert window.generate_button.isEnabled()
        assert not window.body_comparison_button.isEnabled()
    window.set_workflow_input_buttons_enabled(False)
    assert not window.framing_combo.isEnabled()
    window.set_workflow_input_buttons_enabled(True)
    assert window.framing_combo.isEnabled()
    window.close()


@pytest.mark.parametrize('blocked', ['no_source', 'worker', 'analysis', 'workflow', 'candidate', 'approval'])
def test_framing_changes_cannot_bypass_busy_or_missing_inputs(blocked):
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.style_path = None if blocked == 'no_source' else 'character.png'
    if blocked == 'worker':
        window.worker_thread = SimpleNamespace(isRunning=lambda: True)
    if blocked == 'analysis':
        window.mask_worker_thread = SimpleNamespace(isRunning=lambda: True)
    if blocked == 'workflow':
        window.workflow_context = SimpleNamespace(active=True)
    if blocked == 'candidate':
        window.pending_character_candidate = object()
    if blocked == 'approval':
        window.approval_dialog_open = True
    window.status_label.setText('busy sentinel')
    window.framing_combo.setCurrentIndex(1)
    window.update_input_ready_status()
    assert not window.generate_button.isEnabled()
    if blocked != 'no_source':
        assert window.status_label.text() == 'busy sentinel'
    window.worker_thread = None
    window.mask_worker_thread = None
    window.workflow_context = None
    window.pending_character_candidate = None
    window.approval_dialog_open = False
    window.close()


def test_legacy_mode_cannot_restore_category_selector_or_guess_a_category(monkeypatch):
    import gui_main
    monkeypatch.setattr(gui_main, 'CLOTHING_REFERENCE_GENERATION_MODE', False)
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    window.style_path = 'character.png'
    window.update_input_ready_status()
    was_enabled = window.generate_button.isEnabled()
    assert not hasattr(window, 'clothing_category_combo')
    with pytest.raises(ValueError, match='승인된 의상 태그'):
        window.resolve_body_comparison_clothing_category()
    assert window.generate_button.isEnabled() == was_enabled
    window.close()


@pytest.mark.parametrize('framing_index', [0, 1, 2])
def test_visual_request_setup_reaches_worker_without_local_torch(monkeypatch, tmp_path, framing_index):
    import gui_main
    from PIL import Image
    from PySide6.QtWidgets import QDialog
    app = QApplication.instance() or QApplication([])
    window = GenAILabWindow()
    image = Image.new('RGB', (64, 112), 'white')
    window.framing_combo.setCurrentIndex(framing_index)
    image.save(tmp_path / 'character.png')
    window.style_path = str(tmp_path / 'character.png')
    window.outfit_path = 'garment.png'
    window.approved_reference_image = SimpleNamespace(image=image,
        enhancement_applied=False, enhancement_model_id=None,
        quality_status=SimpleNamespace(value='good'))
    window.confirmed_clothing_design = SimpleNamespace(design_tags=('blue jacket',))
    window.pending_clothing_extraction = SimpleNamespace(extracted_image=image)
    failures, started, cache_calls = [], [], []
    log = SimpleNamespace(write_stage=lambda *a: None, write_failure=lambda *a: failures.append(a),
                          close=lambda: None, file_path=tmp_path / 'run.log')
    monkeypatch.setattr(gui_main, 'create_generation_run_log', lambda *a: log)
    def approve(dialog):
        dialog.candidate_count_input.setValue(5)
        dialog.identity_scale_input.setValue(.85)
        dialog.garment_scale_input.setValue(.3)
        dialog.character_gender_input.setCurrentIndex(dialog.character_gender_input.findData('male'))
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(gui_main, 'load_character_gender', lambda path: None)
    monkeypatch.setattr(gui_main, 'save_character_gender', lambda *args: None)
    monkeypatch.setattr(window, 'execute_approval_dialog', approve)
    monkeypatch.setattr(gui_main.QMessageBox, 'critical', lambda *a: failures.append(a))
    monkeypatch.setattr(gui_main.torch.cuda, 'empty_cache', lambda: cache_calls.append(True))
    monkeypatch.setattr(gui_main.QThread, 'start', lambda thread: started.append(thread))
    try:
        window._start_model_generation()
        assert not failures, failures
        assert cache_calls == [True]
        assert len(started) == 1
        assert window.worker.config['clothing_reference_generation']['visual_enabled']
        prepared = window.worker.config['clothing_reference_generation']
        assert 'clothing_category' not in prepared
        assert not hasattr(window, 'clothing_category_combo')
        assert window.worker.generation_request.framing_type.value == window.framing_combo.currentData()
        assert 'garment_ab_test' not in prepared
        assert prepared['character_gender'] == 'male'
        assert prepared['without_initial_image'] is False
        assert prepared['character_tag_review'] is True
        assert prepared['identity_reference_scale'] == .85
        # Native v2의 Animagine Base는 캐릭터 전용이다. 대화창에 남아
        # 있는 의상 강도 값이 Base IP-Adapter로 전달되면 안 된다.
        assert prepared['garment_reference_scale'] == 0.0
        assert prepared['native_base_profile'] == 'character_only'
        assert prepared['candidate_count'] == 5
    finally:
        if hasattr(window, 'visual_progress'):
            window.visual_progress.close()
        if window.worker is not None:
            window.worker.generation_request.reference_image.close()
            window.worker.deleteLater()
            window.worker = None
        if window.worker_thread is not None:
            window.worker_thread.deleteLater()
            window.worker_thread = None
        if window.config:
            garment = window.config.get('clothing_reference_generation', {}).pop('garment_image', None)
            if garment is not None:
                garment.close()
        window.approved_reference_image = None
        window.pending_clothing_extraction = None
        window.confirmed_clothing_design = None
        window.close()
        image.close()


def test_gui_methods_do_not_shadow_global_torch():
    import ast
    import gui_main
    tree = ast.parse(Path(gui_main.__file__).read_text(encoding='utf-8'))
    local_bindings = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Import):
                if any((alias.asname or alias.name.split('.')[0]) == 'torch' for alias in node.names):
                    local_bindings.append((function.name, node.lineno))
            if isinstance(node, ast.Name) and node.id == 'torch' and isinstance(node.ctx, ast.Store):
                local_bindings.append((function.name, node.lineno))
    assert not local_bindings


def test_prompt_approval_and_generation_use_same_garment_policy():
    """승인 빌더에서 생성 단계의 의상 정책 전달이 빠지는 회귀를 막는다."""
    import ast
    import gui_main

    tree = ast.parse(Path(gui_main.__file__).read_text(encoding='utf-8'))
    review_method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == 'review_character_generation_tags'
    )
    calls = [
        node for node in ast.walk(review_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'prepare_design_reference_request'
    ]
    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert 'garment_prompt_policy' in keyword_names
