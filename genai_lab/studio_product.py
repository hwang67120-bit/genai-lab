"""사용자 화면(D-151, 2026-10-22 사용자 "사용자한테 필요한 건 참조 이미지와 이미지 생성, 결과 미리보기에는 생성되는 과정만").

화면: 참조 이미지(캐릭터 · 의상은 선택) → '이미지 생성' 한 번 → 미리보기에 생성 과정 → 완성 그림 → '저장'.
나머지는 자동이다(요청 26~31에서 고정한 조합, D-140):
- 성별·부위·특징표: 저장된 캐릭터 기억 → 이 파일의 저장값 → 자동 분석 순(사람 확인 창 없음). 자동값은 기억에 쓰지 않는다.
- 의상을 골랐으면 ① 옷 바꾸기 → 그 결과로 ② 자세·장면, 안 골랐으면 캐릭터 그림의 옷 그대로 ② 자세·장면.
- ② 연출은 요청 문장 없음 = 고정 자세 목록에서 무작위 + 흰 배경(D-098), 40단계 제품 설정, 재시도·머리색 자동 보정 그대로.
- ①→② 연결은 사람 승인이 아니라 자동(reviewer automatic_chain)이고 기록에도 그렇게 남긴다. 안전 태그 자동 차단은 그대로.
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
from PySide6.QtWidgets import QLabel

GENERATE = "이미지 생성"
GENERATE_TIP = "캐릭터 그림(과 의상 그림)으로 새 그림을 만듭니다. 의상을 고르면 1~2시간, 안 고르면 약 40분 걸립니다."
READY = "상태: '이미지 생성'을 누르세요. 의상 이미지를 고르면 그 옷으로 바꿔 입히고, 안 고르면 지금 옷 그대로 새 자세·장면을 만듭니다."
NEED_CHARACTER = "상태: 캐릭터 이미지를 골라 주세요."
EMPTY_PREVIEW = "아직 만든 이미지가 없습니다.\n\n왼쪽에서 캐릭터(와 의상)를 고른 뒤\n'이미지 생성'을 눌러 주세요."
DONE = "상태: 완성했습니다. 마음에 들면 '저장'을 누르세요. 다시 만들려면 '이미지 생성'을 누르세요."
STEPS = "참조 이미지 고르기  →  이미지 생성  →  저장"
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
                prefill=(info or {}).get("source") or "automatic", memory_error=error or None)


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
        self.identity = self.settings = self.character = None
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
                       w.studio_candidate_combo, w.studio_raw_button, w.studio_before_head_button, w.identity_report_button,
                       w.identity_report_label, w.studio_background_notice, w.studio_finish_button, w.studio_finish_notice,
                       w.tail_edit_button, w.studio_head_paste_button]
        self.hidden += [label for label in w.findChildren(QLabel) if label.text() in DEVELOPER_LABELS]
        w.candidate_preview.setText(EMPTY_PREVIEW)
        w.studio_steps_label.setText(STEPS)
        self.refresh()

    # 화면

    def refresh(self):
        s, w = self.studio, self.window
        for widget in self.hidden:
            widget.hide()
        active = s.task is not None or s.confirming
        w.generate_button.setText(GENERATE)
        w.generate_button.setToolTip(GENERATE_TIP)
        w.generate_button.setEnabled(not active and not s.cleanup_blocked and bool(w.style_path)
                                     and w.pending_character_candidate is None)
        w.save_candidate_button.setEnabled(not active and s.results is not None and getattr(s.results, "free_mode", False))

    def inputs_changed(self):
        self.refresh()
        s = self.studio
        if s.task is None and not s.confirming and s.results is None:
            self.window.status_label.setText(READY if self.window.style_path else NEED_CHARACTER)

    def poll_preview(self):
        """작업 프로세스가 단계마다 쓰는 preview.png 중 가장 새것을 미리보기에 띄운다."""
        directory = self.studio.run_directory
        if directory is None:
            return
        paths = [p for sub in ("outfit-edit", "free-generation") for p in (Path(directory) / sub).glob("qwen*/preview.png")]
        if not paths:
            return
        try:
            latest = max(paths, key=lambda p: p.stat().st_mtime)
            stamp = (str(latest), latest.stat().st_mtime)
        except OSError:
            return
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
        from genai_lab import studio_free_generation, studio_generation
        from genai_lab.qwen_outfit_edit import load_qwen_settings
        try:
            s.runtime = studio_generation.StudioRuntime.from_environment()
            self.settings = load_qwen_settings(s.runtime.qwen_settings)
            s.results = None
            s.run_directory = s.output_root / uuid.uuid4().hex
            s.run_directory.mkdir(parents=True)
        except Exception as error:  # noqa: BLE001 — 설정 문제는 작업실 오류 안내로
            s.fail(error, traceback.format_exc())
            return
        self.character = Path(w.style_path)
        garment = w.selected_outfit_path
        self.preview_seen = None
        w.candidate_preview.setText("캐릭터를 읽는 중입니다…")
        w.status_label.setText("상태: 캐릭터" + ("와 의상" if garment else "") + "을 읽는 중")
        self.timer.start()
        if garment:
            s.outfit_edit_enabled = True
            s.launch(lambda cancel, progress: studio_generation.analyze_inputs(
                self.character, Path(garment), s.run_directory / "inputs", s.runtime, cancelled=cancel, require_face=False),
                self.edit_ready, on_error=self.failed)
        else:
            s.outfit_edit_enabled = False
            s.launch(lambda cancel, progress: studio_free_generation.analyze_base(self.character, s.runtime),
                     self.staging_ready, on_error=self.failed)

    def edit_ready(self, analysis):
        from genai_lab.studio_controller import character_source, confirmation_provenance
        s = self.studio
        s.analysis = analysis
        self.identity = ident = auto_identity(character_source(analysis) or self.character, analysis, s.memory())
        provenance = confirmation_provenance(analysis, None, None, ident["body_parts"])
        for key in ("gender", "body_parts"):
            provenance[key]["source"] = "automatic_product"
        provenance["character_features"] = dict(source="automatic_product", prefill=ident["prefill"])
        inputs, directory, settings, runtime = Path(analysis["directory"]), s.run_directory / "outfit-edit", self.settings, s.runtime
        def operation(cancel, progress):
            from genai_lab.outfit_edit_pipeline import run_outfit_pipeline
            return run_outfit_pipeline(inputs / "character.png", inputs / "garment.png", settings=settings, head_accessories="keep",
                                       runtime=runtime, directory=directory, gender=ident["gender"], body_parts=ident["body_parts"],
                                       provenance=provenance, character_features=ident["features"], cancelled=cancel,
                                       progress=lambda text: progress("상태: ① 옷 바꾸는 중 · " + text.removeprefix("상태: ")))
        s.launch(operation, self.edit_done, on_error=self.failed)

    def edit_done(self, outcome):
        from genai_lab.outfit_edit_results import OutfitEditResults
        results = OutfitEditResults(outcome)
        self.window.candidate_preview.setPixmap(QPixmap(str(results.current_path)))
        self.run_staging(automatic_edit_base(results, self.studio.run_directory), "edited", chain=True)

    def staging_ready(self, analysis):
        s = self.studio
        s.analysis = analysis
        self.identity = auto_identity(self.character, analysis, s.memory())
        self.run_staging(self.character, "original", chain=False)

    def run_staging(self, base, base_kind, chain):
        s, ident, settings, runtime = self.studio, self.identity, self.settings, self.studio.runtime
        directory = s.run_directory / "free-generation"
        label = "상태: ② 자세·장면 만드는 중 · " if chain else "상태: 자세·장면 만드는 중 · "
        def operation(cancel, progress):
            from genai_lab.free_generation_pipeline import run_free_generation
            return run_free_generation(base, ident["features"], "", base_kind=base_kind, gender=ident["gender"],
                                       body_parts=list(ident["body_parts"]), confirmed=True, adult_confirmed=True,
                                       settings=settings, runtime=runtime, directory=directory, cancelled=cancel,
                                       progress=lambda text: progress(label + text.removeprefix("상태: ")),
                                       automatic_edit_base=chain, confirmation_source="automatic_product")
        s.launch(operation, self.done, on_error=self.failed)

    def done(self, outcome):
        from genai_lab import free_generation_results
        self.timer.stop()
        s = self.studio
        s.results = free_generation_results.FreeGenerationResults(outcome)
        self.window.candidate_preview.setPixmap(QPixmap(str(s.results.current_path)))
        self.window.status_label.setText(DONE)

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
