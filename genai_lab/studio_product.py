"""사용자 화면(D-151, 2026-10-22 사용자 "사용자한테 필요한 건 참조 이미지와 이미지 생성, 결과 미리보기에는 생성되는 과정만").

화면: 참조 이미지(캐릭터 · 의상은 선택) → '이미지 생성' 한 번 → 미리보기에 생성 과정 → 후보 3장 중 고르기 → '저장'.
2026-10-31 FLUX 한 경로(D-222, 사용자 "Qwen 제외 확정"): 옷 바꾸기·자세·장면·목표 그림체(S4w)를 FLUX.2-klein 한 단계로
(style_redraw_pipeline.run_style_redraw). 예전 Qwen ①→② 흐름은 화면에서 뺐다(코드·모델 파일은 D-218대로 남김).
나머지는 자동이다:
- 성별·부위: 저장된 캐릭터 기억·파일 저장값이 있으면 확정값으로 넘기고, 없으면 준비 단계가 태그로 추정한다(사람 확인 창 없음).
- 장면(D-153): 입력 칸이 비면 장면 자동 조합(scene_composer — 장소·동작·빛·날씨·카메라 무작위, 전신, 얼굴 보이는 자세만),
  쓰면 그 문장(배경 말이 없으면 장소·빛을 덧붙임). 외형·의상을 바꾸는 말은 막는다.
- 의상을 골랐으면 그 옷으로 갈아입히고, 안 골랐으면 캐릭터 그림의 옷 그대로. 안전 태그 자동 차단(캐릭터·옷 그림)은 그대로.
개발·평가 화면(확인 창·체크 판정·진단)은 GENAI_STUDIO_DEVELOPER=1 또는 --developer로 연다.
"""
import hashlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QLineEdit

GENERATE = "이미지 생성"
GENERATE_TIP = "캐릭터 그림(과 의상 그림)으로 새 장면의 그림 후보 3장을 만듭니다. 약 3분 걸립니다."
READY = "상태: '이미지 생성'을 누르세요. 의상 이미지를 고르면 그 옷으로 바꿔 입히고, 안 고르면 지금 옷 그대로 새 자세·장면을 만듭니다."
NEED_CHARACTER = "상태: 캐릭터 이미지를 골라 주세요."
EMPTY_PREVIEW = "아직 만든 이미지가 없습니다.\n\n왼쪽에서 캐릭터(와 의상)를 고른 뒤\n'이미지 생성'을 눌러 주세요."
DONE = "상태: 후보 3장을 만들었습니다. 결과 목록에서 골라 보고, 마음에 들면 '저장'을 누르세요. 다시 만들려면 '이미지 생성'을 누르세요."
STEPS = "참조 이미지 고르기  →  이미지 생성  →  저장"
SCENE_HINT = "비워 두면 매번 새 장면(장소·자세·빛)을 자동으로 만듭니다. 예: 비 오는 밤 네온 거리에서 우산을 들고 걷는"
DEVELOPER_LABELS = ("결과는 검토·승인한 뒤 저장할 수 있습니다.", "결과를 비교·승인하면 저장 버튼이 활성화됩니다.")


def developer_mode():
    return os.environ.get("GENAI_STUDIO_DEVELOPER") == "1" or "--developer" in sys.argv


def auto_identity(source, analysis, memory):
    """확인 창 없이 쓸 성별·부위·특징표. 저장된 캐릭터 기억·파일 저장값이 있으면 그 값, 없으면 자동 분석."""
    from genai_lab.character_features import DISTINCTIVE, TAG_ITEMS, tag_features
    from genai_lab.studio_controller import guessed_body_parts, guessed_gender
    from genai_lab.studio_free_generation import character_link_info
    info, identity, error, _ = character_link_info(Path(source), memory)
    guessed = tag_features(analysis.get("character_scores", {}))
    features = dict(identity.get("features") or {k: guessed[k]["values"] for k in TAG_ITEMS})
    features = {**{k: None for k in TAG_ITEMS}, DISTINCTIVE: None, **features}
    parts = identity.get("body_parts")
    return dict(gender=identity.get("gender") or guessed_gender(analysis),
                body_parts=tuple(guessed_body_parts(analysis) if parts is None else parts), features=features,
                prefill=(info or {}).get("source") or "automatic", memory_error=error or None,
                confirmed_gender=identity.get("gender"), confirmed_body_parts=None if parts is None else tuple(parts))


def automatic_edit_base(results, directory):
    """① 자동 결과를 ② 기준으로: 같은 바이트 + 승인 기록. reviewer automatic_chain — 사람 승인이 아님을 그대로 적는다."""
    data = Path(results.current_path).read_bytes()
    target = Path(directory) / "edit-result.png"
    target.write_bytes(data)
    record = dict(mode="outfit_edit", reviewer="automatic_chain", sha256=hashlib.sha256(data).hexdigest(),
                  source_run=str(results.directory), trace=results.outcome.get("trace"), composited=results.composited,
                  reason=results.outcome.get("reason"), checks=dict(automatic=True),
                  note="사용자 화면 자동 흐름(D-151): 사람 승인 없이 ② 자세·장면으로 이어 감")
    target.with_suffix(".review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


class ProductFlow:
    """작업실 제어(StudioController)에 얹는 사용자 화면. 개발 화면 위젯은 지우지 않고 숨긴다."""

    def __init__(self, studio):
        self.studio, self.window = studio, studio.window
        w = self.window
        self.identity = self.character = self.garment = None
        self.preview_seen = None
        self.timer = QTimer(w)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll_preview)
        w.generate_button.clicked.disconnect()
        w.generate_button.clicked.connect(self.start)
        w.save_candidate_button.clicked.disconnect()
        w.save_candidate_button.clicked.connect(self.save)
        w.save_candidate_button.setText("저장")
        self.hidden = [w.free_generation_button.parentWidget(), w.diagnostics_toggle, w.diagnostics_panel,
                       w.approve_candidate_button, w.reject_candidate_button, w.discard_candidate_button,
                       w.studio_raw_button, w.studio_before_head_button, w.identity_report_button,
                       w.identity_report_label, w.studio_background_notice, w.studio_finish_button, w.studio_finish_notice,
                       w.tail_edit_button, w.studio_head_paste_button]
        self.hidden += [label for label in w.findChildren(QLabel) if label.text() in DEVELOPER_LABELS]
        self.scene_input, self.scene_note = self.build_scene_card()
        self.scene = None
        w.candidate_preview.setText(EMPTY_PREVIEW)
        w.studio_steps_label.setText(STEPS)
        self.refresh()

    def build_scene_card(self):
        """참조 이미지 아래 '장면(선택)' 칸. 연출(자세·배경·상황)만 — 외형·의상을 바꾸는 말은 기존 규칙으로 막는다."""
        from genai_lab.studio_ui import _card, _label
        card, box = _card("장면 (선택)")
        field = QLineEdit()
        field.setObjectName("product_scene")
        field.setPlaceholderText("비워 두면 자동으로 새 장면을 만듭니다")
        field.setAccessibleName("장면 설명")
        box.addWidget(field)
        note = _label(SCENE_HINT, "muted")
        note.setWordWrap(True)
        box.addWidget(note)
        layout = self.window.input_scroll.widget().layout()
        layout.insertWidget(1, card)            # 참조 이미지 카드 바로 아래
        field.textChanged.connect(lambda *_: self.inputs_changed())
        return field, note

    def scene_problem(self):
        """장면 칸 문장이 연출 범위를 벗어나면 이유, 아니면 빈 문자열."""
        from genai_lab.free_generation import FreeGenerationRejected, check_request
        text = self.scene_input.text().strip()
        if not text:
            return ""
        try:
            checked = check_request(text)
        except FreeGenerationRejected as error:
            return str(error)
        return "" if checked["ok"] else "장면에는 자세·배경·상황만 적어 주세요(외형·의상을 바꾸는 말은 쓸 수 없습니다): " + ", ".join(checked["matched"])

    def choose_scene(self):
        """이번 실행의 장면: 칸이 비면 자동 조합, 쓰면 그 문장(+ 필요하면 장소·빛)."""
        import secrets
        from genai_lab.scene_composer import complete_user_scene, compose_scene
        seed = secrets.randbelow(2**31)
        text = self.scene_input.text().strip()
        return complete_user_scene(text, seed) if text else compose_scene(seed)

    # 화면

    def refresh(self):
        s, w = self.studio, self.window
        for widget in self.hidden:
            widget.hide()
        active = s.task is not None or s.confirming
        from genai_lab.style_redraw_results import StyleRedrawResults
        choosing = isinstance(s.results, StyleRedrawResults) and len(s.results.candidates) > 1
        w.studio_candidate_combo.setVisible(choosing)
        w.studio_candidate_combo.setEnabled(choosing and not active)
        w.generate_button.setText(GENERATE)
        w.generate_button.setToolTip(GENERATE_TIP)
        problem = self.scene_problem()
        self.scene_note.setText(problem or SCENE_HINT)
        self.scene_note.setStyleSheet("color: #b00020;" if problem else "")
        w.generate_button.setEnabled(not active and not s.cleanup_blocked and bool(w.style_path)
                                     and w.pending_character_candidate is None and not problem)
        w.save_candidate_button.setEnabled(not active and s.results is not None and getattr(s.results, "free_mode", False))

    def inputs_changed(self):
        self.refresh()
        s = self.studio
        if s.task is None and not s.confirming and s.results is None:
            self.window.status_label.setText(READY if self.window.style_path else NEED_CHARACTER)

    def poll_preview(self):
        """FLUX 작업 프로세스가 후보마다 쓰는 result-raw-seed*.png 중 가장 새것을 미리보기에, progress.json을 상태 줄에."""
        directory = self.studio.run_directory
        if directory is None:
            return
        flux = Path(directory) / "flux"
        try:
            progress = json.loads((flux / "progress.json").read_text(encoding="utf-8"))
            self.window.status_label.setText(f"상태: 이미지 만드는 중 · 후보 {progress['candidate']}/{progress['candidates']}")
        except (OSError, ValueError, KeyError):
            pass
        raws = sorted(flux.glob("result-raw-seed*.png"), key=lambda p: p.stat().st_mtime) if flux.is_dir() else []
        if not raws:
            return
        latest = raws[-1]
        stamp = (str(latest), latest.stat().st_mtime)
        if stamp != self.preview_seen:
            pixmap = QPixmap(str(latest))
            if not pixmap.isNull():
                self.preview_seen = stamp
                self.window.candidate_preview.setPixmap(pixmap)

    # 흐름

    def start(self):
        s, w = self.studio, self.window
        if s.task is not None or s.confirming or s.cleanup_blocked or not w.style_path:
            return
        from genai_lab import model_check, studio_free_generation, studio_generation
        try:
            s.runtime = studio_generation.StudioRuntime.from_environment()
            # D-172 필수 모델 공통 검사(D-222 FLUX 목록): 빠진 모델을 한 번에 알리고, 없거나 다르면 분석·생성을 시작하지 않는다
            model_check.require_flux_models(s.runtime)
            s.results = None
            s.run_directory = s.output_root / uuid.uuid4().hex
            s.run_directory.mkdir(parents=True)
        except Exception as error:  # noqa: BLE001 — 설정 문제는 작업실 오류 안내로
            s.fail(error, traceback.format_exc())
            return
        self.character = Path(w.style_path)
        self.garment = Path(w.selected_outfit_path) if w.selected_outfit_path else None
        self.scene = self.choose_scene()
        (s.run_directory / "scene.json").write_text(json.dumps(self.scene, ensure_ascii=False, indent=2), encoding="utf-8")
        self.preview_seen = None
        w.candidate_preview.setText("캐릭터를 읽는 중입니다…")
        w.status_label.setText("상태: 캐릭터를 읽는 중")
        self.timer.start()
        s.launch(lambda cancel, progress: studio_free_generation.analyze_base(self.character, s.runtime),
                 self.analysis_ready, on_error=self.failed)

    def analysis_ready(self, analysis):
        s = self.studio
        s.analysis = analysis
        self.identity = ident = auto_identity(self.character, analysis, s.memory())
        character, garment, scene, runtime = self.character, self.garment, self.scene, s.runtime
        directory = s.run_directory / "flux"
        def operation(cancel, progress):
            from genai_lab.style_redraw_pipeline import run_style_redraw
            progress("상태: 이미지 만드는 중 · 후보 3장, 약 2~3분")
            # 확정값(캐릭터 기억·파일 저장값)만 넘긴다. 없으면 준비 단계가 태그로 추정한다(자동 성별 낱말은 문장에 쓰지 않음, D-198).
            return run_style_redraw(character, directory, runtime, confirmed_gender=ident["confirmed_gender"],
                                    confirmed_body_parts=ident["confirmed_body_parts"], garment_path=garment, scene=scene["text"],
                                    cancelled=cancel)
        s.launch(operation, self.done, on_error=self.failed)

    def done(self, outcome):
        from genai_lab.style_redraw_results import StyleRedrawResults
        self.timer.stop()
        s, w = self.studio, self.window
        if outcome.get("state") != "review_pending":
            self.failed(RuntimeError(outcome.get("message") or "이미지를 만들지 못했습니다."), json.dumps(outcome, ensure_ascii=False)[:2000])
            return
        s.results = StyleRedrawResults(outcome)
        combo = w.studio_candidate_combo
        combo.blockSignals(True)
        combo.clear()
        for i in range(len(s.results.candidates)):
            combo.addItem(f"후보 {i + 1}")
        combo.setCurrentIndex(s.results.selected)
        combo.blockSignals(False)
        w.candidate_preview.setPixmap(QPixmap(str(s.results.current_path)))
        w.status_label.setText(DONE + " " + s.results.head_notice)
        s.controls()

    def failed(self, error, details=""):
        self.timer.stop()
        self.studio.fail(error, details)

    def save(self):
        """저장 = 이 결과를 쓰겠다는 사용자 결정. 품질 항목 판정은 아니므로 그렇게 기록한다(approve_saved_by_user)."""
        results = self.studio.results
        if results is None or not getattr(results, "free_mode", False):
            return
        if results.status != "user_approved":
            results.approve_saved_by_user()
        self.studio.save()
