"""작업실 이미지 생성의 읽기 시작점.

gui_main.GenAILabWindow.start_generation → StudioController.start
주 흐름: start → inputs_ready → generated → select → approve → save.
분석·생성은 launch로 작업 스레드에 맡기고 finished가 GUI에서 다음 단계로 연결한다.
사용자 확인에서 취소하면 생성하지 않고, 실패하면 완성 후보를 제공하지 않는다.

이 파일은 주 흐름 → 실행 보조 → 확인창 순서로 읽는다.
모델·태그 계산의 세부 구현은 studio_generation, 전체 지도는 docs/CODE_FLOW.md.
"""

from pathlib import Path
from dataclasses import replace
import threading
import logging
import traceback
import uuid
from PySide6.QtCore import QThread, Qt, Signal, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QComboBox, QPlainTextEdit, QLineEdit, QWidget, QPushButton, QMessageBox, QFileDialog, QScrollArea)
from genai_lab.studio_generation import (StudioRuntime, StudioResults, analyze_inputs,
    build_request, generate_onepass_request, record)
from genai_lab.onepass_generation import OnePassCancelled, validate_local_models
from genai_lab.generation_cleanup import CancelledGeneration, GenerationCleanupError, detach_error_frames, release_cuda_cache
from genai_lab.studio_background import prepare_backgrounds
from genai_lab.studio_optional_finishing import finish_selected
from genai_lab.studio_finishing_gui import choose_finished, NOTICE as FINISHING_NOTICE
from genai_lab.studio_proportion import prepare_studio_pose, prepare_studio_head, confirmed_options
from genai_lab.studio_proportion_gui import select_head_region, review_head_outline
from genai_lab.studio_garment_warnings import garment_warnings, garment_review
from genai_lab.character_preferences import load_character_gender
from genai_lab.onepass_prompt import AppearanceOverrides, PartAppearance, appendage_tags

class StudioController(QObject):
    """한 요청의 입력 확인부터 승인·저장까지 연결한다. 자동 승인은 하지 않는다."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.task = None
        self.cleanup_blocked = False
        self.last_cleanup_report = None
        self.analysis = None
        self.results = None
        self.runtime = None
        self.proportion_enabled = False
        self.shoulder_enabled = False
        self.proportion_pending = None
        self.prepared_proportion_pose = None
        self.run_directory = None
        self.confirming = False
        self.pending_action = None
        self.pending_error = None
        self.tail_context = None
        self.tail_settings_path = ""
        self.pending_tail_recognition = None
        self.finishing_context = None
        self.last_saved = None
        self.output_root = Path(__file__).resolve().parents[1] / "outputs" / "studio-runs"
        window.studio_candidate_combo.currentIndexChanged.connect(self.select)
        window.studio_cancel_button.clicked.connect(self.cancel)
        window.tail_edit_button.clicked.connect(self.edit_tail)
        window.studio_finish_button.clicked.connect(self.finish_selected_image)
        window.studio_raw_button.clicked.connect(lambda: self.original(raw=True))
        window.studio_before_head_button.clicked.connect(self.toggle_before_head)
        from genai_lab.studio_identity_report import IdentityReportController
        self.identity_report_controller = IdentityReportController(self)
        from genai_lab.studio_head_paste_gui import HeadPasteController
        self.head_paste_controller = HeadPasteController(self)

    @property
    def occupied(self):
        return self.task is not None or self.confirming or self.results is not None or self.cleanup_blocked

    # 1. 입력 준비: CPU 분석 완료 후 inputs_ready로 이어진다.

    def start(self):
        w = self.window
        if self.occupied or not w.can_start_registered_generation():
            return
        try:
            # 새 사전 검사 실패로 이전 실행 상태를 덮어쓰지 않는다.
            self.run_directory = self.output_root / uuid.uuid4().hex
            self.run_directory.mkdir(parents=True)
            self.runtime = StudioRuntime.from_environment()
            self.proportion_enabled = w.studio_proportion_checkbox.isChecked()
            self.shoulder_enabled = self.proportion_enabled and w.studio_shoulder_checkbox.isChecked()
            self.proportion_pending = None
            self.prepared_proportion_pose = None
            self.head_paste_controller.generation_context = None
            validate_local_models(self.runtime.generation)
            w.candidate_preview.clear()
            w.candidate_preview.setText("선택한 캐릭터와 옷을 준비하고 있습니다.")
            w.status_label.setText("상태: 캐릭터와 옷 확인 중")
            character, garment = Path(w.style_path), Path(w.selected_outfit_path)
            self.launch(lambda cancel, progress: analyze_inputs(character, garment,
                self.run_directory / "inputs", self.runtime, cancelled=cancel), self.inputs_ready)
        except Exception as error:
            self.fail(error, traceback.format_exc())
            self.controls()

    # 2. 사용자 확인: 승인한 조건만 생성 요청으로 만들고, 완료 후 generated로 이어진다.

    def inputs_ready(self, analysis):
        self.analysis = analysis
        self.confirming = True
        self.controls()
        try:
            decision = confirm_inputs(self.window, analysis,
                self.runtime.prompt.enable_ear_override if self.runtime else False, self.runtime)
        finally:
            self.confirming = False
        if decision is None:
            self.window.status_label.setText("상태: 입력 확인을 취소했습니다. 다른 이미지를 선택할 수 있습니다.")
            self.window.candidate_preview.clear()
            self.window.candidate_preview.setText("아직 생성하지 않았습니다.")
            return
        if self.proportion_enabled:
            self.proportion_pending = decision
            self.window.status_label.setText("상태: 원본 자세 준비 중 · CPU 분석")
            self.launch(lambda cancel, progress: prepare_studio_pose(analysis, self.runtime,
                self.run_directory / "proportion-inputs/pose", cancelled=cancel), self.proportion_pose_ready)
        else:
            self.begin_generation(decision)

    def proportion_pose_ready(self, pose):
        self.prepared_proportion_pose = pose
        self.confirming = True
        self.controls()
        try:
            selection = select_head_region(self.window, pose)
        finally:
            self.confirming = False
            self.controls()
        if selection is None:
            self.cancel_proportion_setup()
            return
        self.window.status_label.setText("상태: 머리 윤곽 준비 중 · CPU 분석 · 아직 생성하지 않았습니다.")
        directory = self.run_directory / "proportion-inputs" / ("head-" + uuid.uuid4().hex)
        self.launch(lambda cancel, progress: prepare_studio_head(pose, selection, self.runtime,
            directory, cancelled=cancel), self.proportion_head_ready)

    def proportion_head_ready(self, draft):
        pose = self.prepared_proportion_pose
        if self.shoulder_enabled:
            from genai_lab.shoulder_control import prepare_shoulder_control
            from genai_lab.studio_proportion import restore_head
            draft["shoulder_correction"] = prepare_shoulder_control(pose["control_file"], pose["control_sha256"],
                pose["joints"], restore_head(draft), Path(draft["head"]["contour_file"]).parent / "shoulder")
        self.confirming = True
        self.controls()
        try:
            choice = review_head_outline(self.window, pose, draft)
        finally:
            self.confirming = False
            self.controls()
        if choice == "retry":
            self.proportion_pose_ready(pose)
        elif choice == "confirm":
            options = confirmed_options(pose, draft, self.runtime)
            if self.shoulder_enabled:
                from genai_lab.proportion_inputs import sha
                path = Path(draft["head"]["contour_file"]).parent / "shoulder/shoulder.json"
                options = replace(options, shoulder_pull="0.85", shoulder_record_file=path, shoulder_record_sha256=sha(path))
            self.head_paste_controller.prepare_generation(self.proportion_pending, pose, options)
        else:
            self.cancel_proportion_setup()

    def cancel_proportion_setup(self):
        self.proportion_pending = None
        self.prepared_proportion_pose = None
        record(self.run_directory / "gui-status.json", {"status":"cancelled", "stage":"proportion_confirmation"})
        self.window.candidate_preview.setText("머리 확인을 취소했습니다. 이미지를 생성하지 않았습니다.")
        self.window.status_label.setText("상태: 비율 확인 취소 · 설정을 바꾸거나 다시 시작할 수 있습니다.")
        self.controls()

    def begin_generation(self, decision, *, pose=None, options=None):
        gender, tags, appearance = decision
        kwargs = {"proportion_pose": pose} if pose is not None else {}
        # 선택 저장소는 화면 스레드에서 접근한다. 토크나이저는 로컬 파일만 읽는다.
        request = build_request(self.analysis, tags, gender, confirmed=True, runtime=self.runtime,
                                appearance=appearance, **kwargs)
        self.window.status_label.setText("상태: 2단계 비율 생성 중 · 먼저 1단계 4장을 만듭니다." if options
                                         else "상태: 이미지 만드는 중 · 0/4")
        cancellation = CancelledGeneration(self.run_directory)
        def generate(cancel, progress):
            count = [0]
            def on_image(candidate):
                record(candidate.path.parent / "garment-review.json",
                       garment_review(tags, request.inputs.prompt.negative, self.runtime.garment_warnings))
                count[0] += 1
                progress(f"상태: {'2단계 결과 준비' if options else '이미지 만드는 중'} · {count[0]}/4 · 아직 결과 확인 전")
            generation_options = {"proportion": options} if options is not None else {}
            batch = generate_onepass_request(request, cancellation.directory,
                                             cancelled=cancel, on_image=on_image,
                                             on_directory_created=cancellation.claim, **generation_options)
            if options is not None:
                return self.head_paste_controller.apply_automatically(batch, cancel, progress)
            if self.runtime.finishing:
                from genai_lab.studio_finishing import finish_batch
                finish_batch(batch, request.inputs.prompt, replace(self.runtime, finishing_face_reference=True), cancelled=cancel, progress=progress)
                return prepare_backgrounds(batch, self.runtime.model_cache, cancelled=cancel, progress=progress, finishing=True)
            # CPU 외곽 모델을 로드하기 전에 생성 실행기를 닫는다.
            return prepare_backgrounds(batch, self.runtime.model_cache, cancelled=cancel, progress=progress)
        self.launch(generate, self.generated, on_cancel=cancellation.cleanup)

    # 3. 생성 완료: 원시 결과를 검토 대기로 표시한다. 아직 승인·저장 상태가 아니다.

    def generated(self, batch):
        self.results = StudioResults(batch, appendage_review_required=bool(
            self.analysis and any(appendage_tags(self.analysis["groups"]["fixed"]).values())))
        combo = self.window.studio_candidate_combo
        combo.blockSignals(True)
        combo.clear()
        for i in range(len(batch.candidates)):
            combo.addItem(f"결과 {i+1}")
        combo.blockSignals(False)
        self.select(0)
        self.window.status_label.setText("상태: 결과 4장 준비됨 · 마음에 드는 결과를 골라 확인해 주세요. " + self.results.head_notice)
        self.controls()

    # 4. 결과 선택: 후보를 바꾸면 기존 승인은 해제된다.

    def select(self, index):
        if self.results is None or index < 0:
            return
        self.results.select(index)
        self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
        self.window.status_label.setText(f"상태: 결과 {index+1} 확인 전 · 다른 결과와 비교할 수 있습니다. " + self.results.head_notice)
        self.controls()

    # 5. 사용자 승인: 캐릭터·의상·노출 확인을 저장한다.

    def approve(self):
        if self.results is None:
            return
        checks = confirm_result(self.window, replace(self.results.candidate, path=self.results.current_path), self.analysis)
        if checks is not None:
            try:
                self.results.approve(checks)
                self.window.status_label.setText("상태: 결과 확인 완료 · 저장 위치를 선택해 주세요.")
            except Exception as error:
                QMessageBox.warning(self.window, "확인 필요", str(error))
        self.controls()

    # 6. 내보내기: 승인한 이미지와 동일한 파일만 지정 위치에 저장한다.

    def save(self):
        if self.results is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.window, "이미지 저장", "character.png", "PNG (*.png)")
        if not path:
            return
        try:
            self.last_saved = self.results.export(path)
            self.window.status_label.setText(f"상태: 저장 완료 · {path}")
            self.results = None
            self.controls()
        except Exception as error:
            QMessageBox.warning(self.window, "저장하지 못했습니다", str(error))

    # 다른 선택: 재검토·폐기·취소·원본 보기는 자동 재생성을 요청하지 않는다.

    def reject(self):
        if self.results is not None:
            index = self.results.reject_and_next()
            combo = self.window.studio_candidate_combo
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
            self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
            self.window.status_label.setText(f"상태: 이전 결과 승인 안 함 · 결과 {index+1}을 표시합니다. " + self.results.head_notice)
        self.controls()

    def toggle_before_head(self):
        if self.results is None or self.task is not None or self.confirming:
            return
        self.results.toggle_head_view()
        self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
        self.window.status_label.setText("상태: " + self.results.head_notice + " · 저장 전 다시 확인해 주세요.")
        self.controls()

    def discard(self):
        if self.results is not None:
            self.results.status = "discarded"
            self.results.persist()
        self.results = None
        self.window.candidate_preview.clear()
        self.window.candidate_preview.setText("결과를 저장하지 않았습니다.")
        self.window.status_label.setText("상태: 새 이미지를 만들 수 있습니다.")
        self.controls()

    def cancel(self):
        if self.task and self.task.request_cancel():
            self.window.status_label.setText("상태: 취소 중 · 현재 단계를 안전하게 마칠 때까지 기다려 주세요.")

    def original(self, *, raw=False):
        if self.results is None:
            return
        dialog = QDialog(self.window)
        dialog.setWindowTitle("생성 원본(raw) · 배경 정리 전" if raw else "결과 원본 크기")
        dialog.resize(850, 800)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        image = QLabel()
        image.setPixmap(QPixmap(str(self.results.raw_path if raw else self.results.current_path)))
        scroll.setWidget(image)
        layout.addWidget(scroll)
        dialog.exec()

    def finish_selected_image(self):
        if self.results is None or self.task is not None or self.confirming:
            return
        if self.results.finishing_unavailable_reason:
            return
        if any(value is not None and hasattr(value, "isRunning") and value.isRunning()
               for name, value in vars(self.window).items() if name.endswith("thread")):
            QMessageBox.information(self.window, "실행 중", "현재 작업이 끝난 뒤 마무리해 주세요.")
            return
        try:
            before_sha = self.results.verify_current()
            destination = self.results.batch.directory / "optional-finishing" / uuid.uuid4().hex
            context = {"selected": self.results.selected, "before": str(self.results.current_path),
                       "before_sha256": before_sha, "directory": str(destination)}
            self.finishing_context = context
            candidate = self.results.candidate
            runtime = self.runtime or StudioRuntime.from_environment()
            self.release_tail_predecessor()
            self.window.status_label.setText("상태: 선택한 한 장 고화질 마무리 중 · 원본은 그대로 보관합니다.")
            self.launch(lambda cancel, progress: finish_selected(candidate, context["before"], before_sha,
                destination, runtime, cancelled=cancel, progress=progress), self.finishing_ready,
                on_error=self.finishing_failed)
        except Exception as error:
            self.finishing_failed(error, traceback.format_exc())

    def finishing_ready(self, info):
        context = self.finishing_context
        if (not context or self.results is None or self.results.selected != context["selected"]
                or self.results.verify_current() != context["before_sha256"]):
            raise ValueError("마무리 중 선택한 기준 이미지가 변경됐습니다.")
        self.confirming = True
        self.controls()
        try:
            # 파생 결과를 보여주기 전에 화면 스레드에서 검증한다.
            from genai_lab.studio_optional_finishing import finishing_info
            verified = finishing_info(context["directory"], self.results.verify_original(), context["before_sha256"])
            if verified != info:
                raise ValueError("마무리 결과 기록이 변경됐습니다.")
            adopted = choose_finished(self.window, context["before"], info)
            record(Path(context["directory"]) / "selection.json",
                   {"selection": "finished" if adopted else "original", "reviewer": "user"})
            if adopted:
                self.results.adopt_finishing(context["directory"])
            self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
            self.window.status_label.setText("상태: 마무리본 선택 · 저장 전 결과를 확인해 주세요." if adopted
                                            else "상태: 마무리 전 원본을 유지합니다.")
            self.finishing_context = None
        finally:
            self.confirming = False
            self.controls()

    def finishing_failed(self, error, details=""):
        cancelled = isinstance(error, OnePassCancelled)
        context, self.finishing_context = self.finishing_context, None
        if context:
            try:
                record(Path(context["directory"]) / "gui-status.json",
                       {"status": "cancelled" if cancelled else "failed", "error": str(error), "detail": details})
            except Exception:
                logging.getLogger(__name__).exception("마무리 실패 기록 저장 실패")
        if self.results is not None:
            self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
        self.window.status_label.setText("상태: 마무리 취소 · 원본 유지" if cancelled else "상태: 마무리 실패 · 원본 유지")
        if not cancelled:
            QMessageBox.warning(self.window, "원본을 유지합니다", str(error))
        self.controls()

    # 선택적 꼬리 보정: 정상 완료 후보만 사용하며, 실패해도 후보 묶음을 버리지 않는다.

    def edit_tail(self):
        if (self.results is None or self.task is not None or self.confirming or not self.analysis
                or not appendage_tags(self.analysis["groups"]["fixed"])["tail"]):
            return
        if any(value is not None and hasattr(value, "isRunning") and value.isRunning()
               for name, value in vars(self.window).items() if name.endswith("thread")):
            QMessageBox.information(self.window, "실행 중", "현재 작업이 끝난 뒤 꼬리를 고쳐 주세요.")
            return
        from genai_lab.qwen_tail_gui import TailInputDialog
        from genai_lab.qwen_tail_edit import prepare_tail_spec
        from genai_lab.qwen_preservation import file_sha
        self.confirming = True
        self.controls()
        try:
            basis_sha = self.results.verify_basis()
            source = self.analysis["source"]
            source_sha = self.analysis["references"].get("character", {}).get("sha256") or file_sha(source)
            dialog = TailInputDialog(source, source_sha, self.window, settings_path=self.tail_settings_path)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            self.tail_settings_path = dialog.settings_path
            root = self.results.batch.directory / "tail-edits" / uuid.uuid4().hex
            spec = prepare_tail_spec(self.results.basis_path, source, dialog.canvas.box,
                pattern=dialog.pattern.currentData(), tip=dialog.tip.text(), confirmed=dialog.confirm.isChecked(),
                directory=root / "inputs", source_sha256=source_sha, image_sha256=basis_sha)
            from genai_lab.tail_complexity import persist_advisory
            advisory = persist_advisory(root / "complexity", spec, getattr(dialog, "complexity_result", None))
            self.tail_context = {"directory": root / "edit", "source": spec.source_path,
                                 "tail_complexity": advisory,
                                 "before": spec.image_path, "basis_sha256": basis_sha,
                                 "selected": self.results.selected}
            self.release_tail_predecessor()
            recognition = getattr(dialog, "recognition_enabled", None)
            if recognition is not None and recognition.isChecked():
                from genai_lab.tail_recognition import RecognitionSettings, recognize_tail
                config = RecognitionSettings.load(dialog.recognition_settings_path)
                self.pending_tail_recognition = (spec, dialog.settings)
                def analyze(cancel, progress):
                    return recognize_tail(spec, config, root / "recognition", cancelled=cancel, progress=progress)
                self.launch(analyze, self.tail_recognized, on_error=self.tail_failed)
            else:
                self.launch_tail_edit(spec, dialog.settings)
        except Exception as error:
            self.tail_failed(error, traceback.format_exc())
        finally:
            self.confirming = False
            self.controls()

    def release_tail_predecessor(self):
        """인식이나 편집 시작 전에 이전 생성 모델을 해제한다."""
        pipeline = getattr(self.window, "pipeline", None)
        self.window.pipeline = None
        if pipeline is not None:
            remove = getattr(pipeline, "remove_all_hooks", None)
            if callable(remove):
                remove()
            del remove
        del pipeline

    def tail_recognized(self, report):
        from genai_lab.qwen_tail_gui import review_tail_recognition
        from genai_lab.qwen_record_io import write_json
        spec, settings = self.pending_tail_recognition
        self.pending_tail_recognition = None
        self.confirming = True
        self.controls()
        try:
            if self.results.selected != self.tail_context["selected"]:
                raise ValueError("인식 중 선택 이미지가 변경됐습니다.")
            approval = review_tail_recognition(self.window, spec, report)
            root = self.tail_context["directory"].parent
            write_json(root / "recognition-review.json", {"adopted": approval is not None, "approval": approval})
            if approval is None:
                self.tail_context = None
                self.window.status_label.setText("상태: 이미지 인식 내용 미채택 · 편집하지 않았습니다.")
                return
            self.launch_tail_edit(replace(spec, recognition=approval), settings)
        finally:
            self.confirming = False
            self.controls()

    def launch_tail_edit(self, spec, settings):
        from genai_lab.qwen_tail_edit import make_tail_request, TailEditWorkflow, run_tail_edit
        request = make_tail_request(spec, settings=settings)
        flow = TailEditWorkflow(spec)
        directory = self.tail_context["directory"]
        advisory = self.tail_context.get("tail_complexity")
        def operation(cancel, progress):
            def report(data):
                progress(f"상태: 꼬리 고치는 중 · {len(data.get('steps', []))}/40 · 고치기 전 결과는 보관 중")
            try:
                return run_tail_edit(request, directory, flow, cancelled=cancel, on_progress=report)
            finally:
                if advisory is not None:
                    from genai_lab.tail_complexity import annotate_finished_run
                    try:
                        annotate_finished_run(directory, advisory)
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "꼬리 분석 부가 기록 저장 실패: %s (편집 결과는 유지)", directory, exc_info=True)
        self.window.status_label.setText("상태: 꼬리 편집 준비 중 · 약 30분 걸릴 수 있습니다.")
        self.launch(operation, self.tail_edited, on_error=self.tail_failed)

    def tail_edited(self, product):
        from genai_lab.qwen_tail_gui import review_tail_result
        from genai_lab.qwen_tail_edit import tail_product_info
        from genai_lab.qwen_record_io import write_json
        context = self.tail_context
        if self.results is None or self.results.selected != context["selected"]:
            raise ValueError("편집을 시작한 후보가 현재 선택과 다릅니다.")
        info = tail_product_info(context["directory"], self.results.verify_basis())
        if Path(product).resolve() != (context["directory"] / "product.png").resolve():
            raise ValueError("실행 기록과 다른 편집 미리보기입니다.")
        self.confirming = True
        self.controls()
        try:
            checks = review_tail_result(self.window, context["source"], context["before"], product)
            write_json(context["directory"] / "tail-review.json", {
                **info, "adopted": checks is not None, "checks": checks or {},
                "reviewer": "user", "automatic_verdict": False})
            if checks is not None:
                self.results.adopt_tail_edit(context["directory"])
                self.results.approve(checks)
                self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
                self.window.status_label.setText("상태: 꼬리 편집본 확인 완료 · 저장 위치를 선택해 주세요.")
            else:
                self.window.status_label.setText("상태: 편집본을 사용하지 않았습니다. 이전 결과를 유지합니다.")
        finally:
            self.confirming = False
            self.tail_context = None
            self.controls()

    def tail_failed(self, error, details=""):
        self.pending_tail_recognition = None
        # 선택적 편집 실패는 후보 묶음을 지우는 fail()로 전달하지 않는다.
        context = self.tail_context
        if context is not None:
            from genai_lab.qwen_record_io import write_json
            try:
                write_json(context["directory"].parent / "gui-error.json", {"error": str(error), "detail": details})
            except OSError as record_error:
                details += f"\n편집 화면 오류 기록 실패: {record_error}"
                error = RuntimeError(f"{error}\n편집 화면 오류 기록도 저장하지 못했습니다: {record_error}")
        self.tail_context = None
        self.window.status_label.setText("상태: 꼬리 편집을 마치지 못했습니다. 이전 결과는 그대로 남아 있습니다.")
        if self.results is not None:
            self.window.candidate_preview.setPixmap(QPixmap(str(self.results.current_path)))
        if not isinstance(error, OnePassCancelled):
            QMessageBox.warning(self.window, "꼬리 편집 중단", str(error))

    # 실행 보조: 아래 코드는 버튼 상태와 비동기 실행·실패 처리를 담당한다.

    def controls(self):
        w = self.window
        active = self.task is not None or self.confirming
        for control in (w.style_button, w.outfit_button, w.clear_outfit_button,
                        w.qwen_pose_button, w.external_candidate_button):
            control.setEnabled(not self.occupied)
        w.studio_proportion_checkbox.setEnabled(not self.occupied)
        w.studio_shoulder_checkbox.setEnabled(not self.occupied and w.studio_proportion_checkbox.isChecked())
        w.generate_button.setEnabled(not self.occupied and bool(w.style_path and w.selected_outfit_path))
        w.studio_cancel_button.setVisible(self.task is not None)
        w.studio_candidate_combo.setVisible(self.results is not None)
        w.studio_candidate_combo.setEnabled(not active)
        ready = self.results is not None and not active
        w.approve_candidate_button.setEnabled(ready)
        w.reject_candidate_button.setEnabled(ready)
        w.discard_candidate_button.setEnabled(ready)
        w.save_candidate_button.setEnabled(ready and self.results.status == "user_approved")
        w.open_original_size_button.setEnabled(ready)
        w.studio_raw_button.setEnabled(ready)
        w.studio_raw_button.setVisible(self.results is not None)
        w.studio_before_head_button.setVisible(bool(self.results and self.results.auto_head))
        w.studio_before_head_button.setEnabled(ready and bool(self.results.active_head))
        w.studio_before_head_button.setText("붙인 결과 보기" if self.results and self.results.show_before_head else "붙이기 전 보기")
        w.studio_background_notice.setVisible(self.results is not None)
        if self.results is not None:
            w.studio_background_notice.setText(self.results.background_notice)
        has_tail = bool(self.analysis and appendage_tags(self.analysis["groups"]["fixed"])["tail"])
        w.tail_edit_button.setVisible(self.results is not None and has_tail)
        w.tail_edit_button.setEnabled(ready and has_tail)
        reason = self.results.finishing_unavailable_reason if self.results else ""
        w.studio_finish_button.setVisible(self.results is not None)
        w.studio_finish_button.setEnabled(ready and not reason)
        w.studio_finish_button.setToolTip(reason or FINISHING_NOTICE)
        w.studio_finish_notice.setText(reason or FINISHING_NOTICE)
        w.studio_finish_notice.setVisible(self.results is not None)
        self.head_paste_controller.refresh(ready)
        self.identity_report_controller.refresh()

    def launch(self, action, complete, *, on_error=None, on_cancel=None):
        if self.task is not None or self.cleanup_blocked:
            raise RuntimeError("실행 중이거나 GPU 정리를 마치지 못했습니다.")
        task = StudioTask(action, self.window, on_cancel=on_cancel)
        self.task = task
        # 기존 앱 종료·GPU 동시 사용 방지 검사에도 이 작업을 포함한다.
        self.window.worker_thread = task
        task.progress.connect(self.window.status_label.setText)
        self.pending_action = complete
        self.pending_error = on_error or self.fail
        task.finished.connect(self.finished)
        self.controls()
        task.start()

    # 작업 스레드 종료 신호가 도착하면 GUI 스레드에서 다음 단계 함수를 호출한다.

    def finished(self):
        task = self.task
        complete = self.pending_action
        failed = self.pending_error or self.fail
        self.last_cleanup_report = task.cleanup_report
        self.task = None
        self.window.worker_thread = None
        self.pending_action = None
        self.pending_error = None
        try:
            if self.run_directory and task.end_memory is not None:
                record(self.run_directory / "task-memory.json", task.end_memory)
            if task.error:
                failed(task.error, task.detail)
            else:
                complete(task.result)
        except Exception as error:
            failed(error, traceback.format_exc())
        finally:
            task.result = None
            task.deleteLater()
            self.controls()

    def fail(self, error, details=""):
        cancelled = isinstance(error, OnePassCancelled)
        self.cleanup_blocked = self.cleanup_blocked or isinstance(error, GenerationCleanupError)
        self.results = None  # 취소한 생성 결과는 폐기하며 완성 후보로 제시하지 않는다.
        self.window.candidate_preview.clear()
        self.window.candidate_preview.setText("취소되었습니다." if cancelled else "완성된 결과가 없습니다.")
        message = "상태: 취소됨 · 생성 결과와 GPU 자원 정리 완료" if cancelled and self.last_cleanup_report else "상태: 취소됨"
        if not cancelled:
            message = "상태: 정리 실패 · 앱을 다시 시작해 주세요." if self.cleanup_blocked else "상태: 생성하지 못했습니다. 입력과 실행 정보를 확인해 주세요."
        self.window.status_label.setText(message)
        if self.run_directory:
            self.run_directory.mkdir(parents=True, exist_ok=True)
            record(self.run_directory / "gui-status.json", {"status": "cancel_cleanup_failed" if self.cleanup_blocked else "cancelled" if cancelled else "failed",
                   "error": str(error), "detail": details, "cleanup": self.last_cleanup_report})
        if not cancelled:
            box = QMessageBox(self.window)
            box.setWindowTitle("작업을 마치지 못했습니다")
            box.setText(str(error))
            box.setDetailedText(details)
            box.exec()

# 작업 실행기: 화면을 직접 바꾸지 않고 결과·오류·진행 신호를 돌려준다.

class StudioTask(QThread):
    progress = Signal(str)

    def __init__(self, action, parent, *, on_cancel=None):
        super().__init__(parent)
        self.action = action
        self.on_cancel = on_cancel
        self.cleanup_report = None
        self.end_memory = None
        self.stop = threading.Event()
        self.cancel_lock = threading.Lock()
        self.accepting_cancel = True
        self.result = None
        self.error = None
        self.detail = ""

    def run(self):
        try:
            self.result = self.action(self.stop.is_set, self.progress.emit)
            if self.stop.is_set():
                raise OnePassCancelled("요청을 취소했습니다.")
        except Exception as error:
            self.error = error
            self.detail = traceback.format_exc()
            detach_error_frames(error)
        finally:
            self.action = None  # 완료된 작업의 클로저에 모델·중간 결과가 남지 않게 한다.
        with self.cancel_lock:
            self.accepting_cancel = False
            cancelled = isinstance(self.error, OnePassCancelled) or self.stop.is_set()
        if cancelled:
            self.discard_cancelled_result()
        self.on_cancel = None
        # 종료 신호 전에 정리해 완료 콜백에서 시작할 다음 GPU 작업과 겹치지 않는다.
        try:
            import sys
            self.end_memory = release_cuda_cache(sys.modules.get("torch"))
        except Exception as error:
            self.error = GenerationCleanupError("작업 종료 GPU 정리 실패: " + str(error))
            detach_error_frames(error)
            self.result = None

    def request_cancel(self):
        """완료 판정과 같은 잠금으로 취소를 받는다. 이미 완료된 작업에는 취소를 표시하지 않는다."""
        with self.cancel_lock:
            if not self.accepting_cancel:
                return False
            self.stop.set()
            return True

    def discard_cancelled_result(self):
        """작업과 참조를 해제한 뒤 결과를 폐기한다. 정리 실패는 정상 취소로 바꾸지 않는다."""
        self.result = None
        if self.on_cancel is None:
            return
        self.progress.emit("상태: 취소 정리 중 · GPU 자원과 생성 결과를 폐기합니다.")
        try:
            self.cleanup_report = self.on_cancel()
        except Exception as error:
            self.error = error
            self.detail += "\n취소 정리 오류:\n" + traceback.format_exc()
            detach_error_frames(error)
            return
        if isinstance(self.error, GenerationCleanupError):
            self.cleanup_report["status"] = "cancel_cleanup_failed"
        else:
            self.error = OnePassCancelled("생성을 취소하고 결과와 GPU 자원을 정리했습니다.")

# 확인창의 배치 세부사항: 주 흐름에서는 confirm_inputs / confirm_result만 읽으면 된다.

def picture(path, label, width=200, height=220):
    box = QVBoxLayout()
    box.addWidget(QLabel(label))
    image = QLabel()
    image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    image.setPixmap(QPixmap(str(path)).scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
    box.addWidget(image)
    return box

# 화면 표시용 번역. 생성에는 승인된 원문 태그를 그대로 전달한다.

NAMES = {"shirt": "셔츠", "camisole": "캐미솔", "tank top": "민소매 상의", "crop top": "짧은 상의",
    "shorts": "반바지", "pants": "바지", "skirt": "치마", "dress": "원피스", "jacket": "재킷",
    "coat": "코트", "bodysuit": "바디슈트", "gloves": "장갑", "sweater": "스웨터",
    "necktie": "넥타이", "long sleeves": "긴 소매", "short sleeves": "짧은 소매",
    "thighhighs": "허벅지 길이 양말", "elbow gloves": "팔꿈치 길이 장갑",
    "jeans": "청바지", "leggings": "레깅스", "pantyhose": "스타킹", "long skirt": "긴 치마",
    "hoodie": "후드 상의", "blazer": "블레이저", "t-shirt": "티셔츠", "sweatshirt": "맨투맨",
    "cardigan": "카디건", "suspenders": "멜빵", "vest": "조끼", "shrug": "짧은 걸침옷"}

def appearance_field(layout, part, detected, enabled):
    """선택적인 사용자 설명 하나를 받는다. 문구를 바꾸면 확인은 무효가 된다."""
    title = "꼬리" if part == "tail" else "귀"
    label = QLabel(f"{title} · 자동으로 읽은 내용: {', '.join(detected)}")
    label.setWordWrap(True)
    layout.addWidget(label)
    text = QLineEdit()
    text.setObjectName(f"{part}_appearance_text")
    text.setPlaceholderText("예: light blue tail, striped tail" if part == "tail" else "영어 외형 문구 (현재 적용 안 함)")
    text.setEnabled(enabled)
    layout.addWidget(text)
    confirmed = QCheckBox(f"이 {title} 설명을 확인했고 생성에 적용합니다.")
    confirmed.setObjectName(f"{part}_appearance_confirmed")
    confirmed.setEnabled(False)
    layout.addWidget(confirmed)
    hint = QLabel("입력 후 확인한 문구만 적용합니다." if enabled else "귀 외형 변경은 현재 적용하지 않습니다.")
    hint.setWordWrap(True)
    layout.addWidget(hint)
    def changed():
        confirmed.setChecked(False)
        try:
            PartAppearance(text.text(), True)
        except (TypeError, ValueError) as error:
            confirmed.setEnabled(False)
            hint.setText(str(error) if text.text().strip() else "문구 없음 · 기존 태그를 그대로 사용합니다.")
        else:
            confirmed.setEnabled(enabled)
            hint.setText("아직 적용하지 않습니다. 내용을 확인한 뒤 체크해 주세요.")
    text.textChanged.connect(changed)
    confirmed.toggled.connect(lambda checked: hint.setText(
        "확인한 문구를 적용합니다." if checked else "문구 미확인 · 기존 태그를 그대로 사용합니다."))
    return text, confirmed


def confirm_inputs(window, analysis, enable_ear_override=False, runtime=None):
    from genai_lab.onepass_garment_vocabulary import garment_nouns
    dialog = QDialog(window)
    dialog.setWindowTitle("만들 이미지 확인")
    dialog.resize(780, 650)
    outer = QVBoxLayout(dialog)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    scroll.setWidget(content)
    outer.addWidget(scroll)
    mode = bool(window and window.studio_proportion_checkbox.isChecked())
    note = QLabel("이 캐릭터에 선택한 옷을 입힌 이미지 4장을 만듭니다. " +
                  ("이후 원본 자세와 머리 윤곽을 확인해 2단계 생성합니다." if mode
                   else "이번 생성에서는 자세를 지정하지 않습니다."))
    note.setWordWrap(True)
    layout.addWidget(note)
    pictures = QHBoxLayout()
    directory = Path(analysis["directory"])
    for filename, title in (("character.png", "캐릭터"), ("face.png", "생성에 사용할 얼굴"), ("garment.png", "입힐 옷")):
        pictures.addLayout(picture(directory / filename, title))
    layout.addLayout(pictures)
    nouns = garment_nouns(analysis["garment_tags"])
    summary = ", ".join(NAMES.get(n.name, n.name) for n in nouns)
    layout.addWidget(QLabel("읽어낸 옷: " + (summary or "아래 상세 설명을 확인해 주세요.")))
    gender = QComboBox()
    gender.addItem("캐릭터 성별을 선택해 주세요", None)
    for label, value in (("남성", "male"), ("여성", "female"), ("지정 안 함", "unspecified")):
        gender.addItem(label, value)
    saved = load_character_gender(analysis["source"])
    if saved is not None:
        gender.setCurrentIndex(gender.findData(saved))
    layout.addWidget(gender)
    details = QCheckBox("옷 설명 자세히 보기·수정 (선택)")
    layout.addWidget(details)
    tags = QPlainTextEdit(", ".join(analysis["garment_tags"]))
    tags.setObjectName("garment_tags_editor")
    tags.setMaximumHeight(85)
    tags.hide()
    details.toggled.connect(tags.setVisible)
    layout.addWidget(tags)
    runtime = runtime or StudioRuntime()
    warning_label = QLabel()
    warning_label.setObjectName("garment_warnings")
    warning_label.setWordWrap(True)
    warning_label.setTextFormat(Qt.TextFormat.PlainText)
    layout.addWidget(warning_label)
    def show_warnings():
        current = tuple(t.strip() for t in tags.toPlainText().split(",") if t.strip())
        warnings = garment_warnings(current, runtime.prompt.negative_template, runtime.garment_warnings)
        warning_label.setText("\n".join(w["message"] + " (" + ", ".join(w["tags"]) + ")" for w in warnings))
        warning_label.setVisible(bool(warnings))
    tags.textChanged.connect(show_warnings)
    show_warnings()
    from genai_lab.studio_skin_tone import skin_tone_field
    skin_choice = skin_tone_field(layout, analysis, runtime.skin_recommendation)
    fields = {}
    for part, detected in appendage_tags(analysis["groups"]["fixed"]).items():
        if detected:
            fields[part] = appearance_field(layout, part, detected,
                part == "tail" or enable_ear_override)
    if "tail" in fields:
        notice = QLabel("꼬리 색·모양을 적지 않으면 종 대표 색이나 머리색으로 그려질 수 있습니다.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
    checked = QCheckBox("얼굴에 팔·몸이 남지 않았고, 읽어낸 옷이 선택한 옷과 맞습니다.")
    layout.addWidget(checked)
    buttons = QHBoxLayout()
    back = QPushButton("다른 이미지 선택")
    back.clicked.connect(dialog.reject)
    create = QPushButton("이 조건으로 만들기")
    create.setEnabled(False)
    update = lambda: create.setEnabled(checked.isChecked() and gender.currentData() is not None and bool(tags.toPlainText().strip()))
    checked.toggled.connect(update)
    gender.currentIndexChanged.connect(update)
    tags.textChanged.connect(lambda: checked.setChecked(False))
    tags.textChanged.connect(update)
    create.clicked.connect(dialog.accept)
    buttons.addWidget(back)
    buttons.addWidget(create)
    outer.addLayout(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    appearance = AppearanceOverrides(skin_tone=skin_choice(), **{
        part: PartAppearance(text.text(), confirmed.isChecked())
        for part, (text, confirmed) in fields.items()
    })
    return gender.currentData(), tuple(t.strip() for t in tags.toPlainText().split(",") if t.strip()), appearance

def confirm_result(window, candidate, analysis):
    dialog = QDialog(window)
    dialog.setWindowTitle("저장할 결과 확인")
    dialog.resize(800, 670)
    layout = QVBoxLayout(dialog)
    row = QHBoxLayout()
    for path, name in ((Path(analysis["directory"])/"character.png", "원래 캐릭터"),
                       (Path(analysis["directory"])/"garment.png", "선택한 옷"), (candidate.path, "저장할 결과")):
        row.addLayout(picture(path, name, 235, 380))
    layout.addLayout(row)
    checks = {}
    for key, text in (("character", "원하는 캐릭터로 보입니다."), ("garment", "옷의 모양과 색이 마음에 듭니다."),
                      ("exposure", "원치 않는 노출이나 잘못 그려진 부분이 없습니다.")):
        checks[key] = QCheckBox(text)
        layout.addWidget(checks[key])
    if any(appendage_tags(analysis["groups"]["fixed"]).values()):
        checks["appendages"] = QCheckBox("꼬리·귀의 개수·종류·색이 원래 캐릭터와 같습니다.")
        layout.addWidget(checks["appendages"])
    accept = QPushButton("이 결과 사용")
    accept.setEnabled(False)
    for check in checks.values():
        check.toggled.connect(lambda: accept.setEnabled(all(c.isChecked() for c in checks.values())))
    accept.clicked.connect(dialog.accept)
    back = QPushButton("다시 비교하기")
    back.clicked.connect(dialog.reject)
    layout.addWidget(accept)
    layout.addWidget(back)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return {key: check.isChecked() for key, check in checks.items()}
    return None
