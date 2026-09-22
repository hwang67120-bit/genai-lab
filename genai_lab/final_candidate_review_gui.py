"""PySide6 Stage 8 comparison and approval dialog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from genai_lab.final_candidate_review import (
    APPROVED,
    APPROVED_WITH_REFINEMENT,
    REJECTED,
    FinalCandidateReviewError,
    FinalReviewDecision,
    FinalReviewEvidence,
    REFINEMENT_TARGET_LABELS,
    REJECTION_POLICIES,
    create_final_review_decision,
)


def _pixmap_from_image(image: Image.Image, width: int = 240, height: int = 380) -> QPixmap:
    rgb = image.convert("RGB")
    rgb.thumbnail((width, height))
    qimage = QImage(
        rgb.tobytes(),
        rgb.width,
        rgb.height,
        rgb.width * 3,
        QImage.Format.Format_RGB888,
    ).copy()
    rgb.close()
    return QPixmap.fromImage(qimage)


def _preview(source: Any, empty_text: str) -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(180, 260)
    label.setStyleSheet("border: 1px solid #777; background: #202020; color: #dddddd;")
    try:
        if isinstance(source, Image.Image):
            pixmap = _pixmap_from_image(source)
        else:
            path = Path(str(source))
            if not path.is_file():
                label.setText(empty_text)
                return label
            with Image.open(path) as opened:
                pixmap = _pixmap_from_image(opened)
        label.setPixmap(pixmap)
    except (OSError, ValueError):
        label.setText(empty_text)
    return label


class FinalCandidateReviewDialog(QDialog):
    """Show reference, Base, result, diagnostics, and one explicit decision."""

    def __init__(
        self,
        evidence: FinalReviewEvidence,
        final_image: Image.Image,
        *,
        character_reference: Any = None,
        garment_reference: Any = None,
        initial_decision: str = APPROVED,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.evidence = evidence
        self.selected_decision: FinalReviewDecision | None = None
        self.setWindowTitle("8단계 — 결과·유도 과정·진단 비교와 사용자 승인")
        self.resize(1180, 860)

        shell = QVBoxLayout(self)
        notice = QLabel(
            "사용자는 점수가 아니라 최종 이미지의 사용 여부를 결정합니다. "
            "유사도 수치는 확률이나 정확도가 아니며, 거절 결과는 최종 선택과 "
            "학습 자료에서 제외됩니다."
        )
        notice.setWordWrap(True)
        shell.addWidget(notice)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget(scroll)
        content = QVBoxLayout(container)
        scroll.setWidget(container)
        shell.addWidget(scroll)

        images = QHBoxLayout()
        sources = evidence.sources
        preview_specs = (
            ("캐릭터 참조", character_reference or sources.get("character_reference")),
            ("의상 참조", garment_reference or sources.get("garment_reference")),
            ("승인 Base", sources.get("approved_base")),
            ("최종 후보", final_image),
        )
        for title, source in preview_specs:
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            layout.addWidget(_preview(source, "이미지 경로 없음"))
            if source is not None and not isinstance(source, Image.Image):
                path_label = QLabel(str(source))
                path_label.setWordWrap(True)
                layout.addWidget(path_label)
            images.addWidget(group)
        content.addLayout(images)

        technical = evidence.technical_safety
        technical_text = QLabel(
            "기술적 안전: {status} / Native: {native} / 인물 수: {people} "
            "({person_status})".format(
                status=technical.get("status", "UNKNOWN"),
                native=technical.get("native_status", "UNKNOWN"),
                people=technical.get("person_count", "계산 불가"),
                person_status=technical.get("person_count_status", "NOT_REPORTED"),
            )
        )
        technical_text.setWordWrap(True)
        content.addWidget(technical_text)

        comparison_group = QGroupBox("항목별 비교 — 각 항목은 지정된 참조와 비교")
        comparison_layout = QVBoxLayout(comparison_group)
        reference_labels = {
            "character_reference": "캐릭터 참조",
            "garment_reference": "의상 참조",
            "combined_references": "두 참조",
        }
        for item in evidence.comparisons:
            value = item.get("value")
            value_text = "계산 불가" if value is None else f"{float(value):.2f}"
            comparison_layout.addWidget(QLabel(
                f"{item.get('label_ko')}: {value_text} / "
                f"기준={reference_labels.get(item.get('reference'), item.get('reference'))}"
            ))
        comparison_layout.addWidget(QLabel(
            "위 수치는 유사도 지표이며 합격 확률이 아닙니다."
        ))
        content.addWidget(comparison_group)

        guidance = evidence.guidance
        guidance_group = QGroupBox("유도 과정 적용 기록")
        guidance_layout = QVBoxLayout(guidance_group)
        for key, title in (
            ("requested", "요청"),
            ("applied", "적용"),
            ("unsupported", "미지원"),
            ("not_reported", "기록 없음"),
        ):
            values = guidance.get(key, ())
            guidance_layout.addWidget(QLabel(
                f"{title}: {', '.join(values) if values else '없음'}"
            ))
        if guidance.get("note_ko"):
            note = QLabel(str(guidance["note_ko"]))
            note.setWordWrap(True)
            guidance_layout.addWidget(note)
        content.addWidget(guidance_group)

        decision_group = QGroupBox("사용자 결정")
        decision_layout = QVBoxLayout(decision_group)
        self.decision_combo = QComboBox()
        self.decision_combo.addItem("최종 승인", APPROVED)
        self.decision_combo.addItem("조건부 승인 — 결과 보존 후 미세조정", APPROVED_WITH_REFINEMENT)
        self.decision_combo.addItem("거절 — 사유에 따라 다음 행동 기록", REJECTED)
        index = self.decision_combo.findData(initial_decision)
        self.decision_combo.setCurrentIndex(max(0, index))
        decision_layout.addWidget(self.decision_combo)

        self.reason_combo = QComboBox()
        self.reason_combo.addItem("거절 사유를 선택하세요", None)
        for code, policy in REJECTION_POLICIES.items():
            self.reason_combo.addItem(policy[0], code)
        decision_layout.addWidget(self.reason_combo)

        target_group = QGroupBox("미세조정 대상")
        target_layout = QHBoxLayout(target_group)
        self.target_checks: dict[str, QCheckBox] = {}
        for key, label in REFINEMENT_TARGET_LABELS.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in evidence.refinement_targets)
            self.target_checks[key] = checkbox
            target_layout.addWidget(checkbox)
        decision_layout.addWidget(target_group)
        self.next_action_label = QLabel()
        self.next_action_label.setWordWrap(True)
        decision_layout.addWidget(self.next_action_label)
        content.addWidget(decision_group)

        buttons = QHBoxLayout()
        confirm = QPushButton("결정 기록")
        cancel = QPushButton("취소")
        confirm.clicked.connect(self.confirm)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(confirm)
        buttons.addWidget(cancel)
        shell.addLayout(buttons)

        self.decision_combo.currentIndexChanged.connect(self._update_controls)
        self.reason_combo.currentIndexChanged.connect(self._update_controls)
        self._update_controls()

    def _update_controls(self) -> None:
        decision = self.decision_combo.currentData()
        rejected = decision == REJECTED
        conditional = decision == APPROVED_WITH_REFINEMENT
        self.reason_combo.setEnabled(rejected)
        for checkbox in self.target_checks.values():
            checkbox.setEnabled(rejected or conditional)

        if rejected:
            reason = self.reason_combo.currentData()
            policy = REJECTION_POLICIES.get(reason)
            if policy is None:
                message = "거절 사유를 선택하면 다음 행동이 표시됩니다."
            elif policy[1] == "retry_same_seed_guidance":
                message = (
                    "동일 Base·Seed를 유지하고 선택 부위 유도만 조정합니다. "
                    "1~7단계 유도 실행 연결 전에는 자동 GPU 재시도를 시작하지 않습니다."
                )
            elif policy[1] == "retry_new_seed":
                message = "거절 후보를 제외하고 새 Seed 재생성을 준비합니다."
            elif policy[1] == "return_to_input_review":
                message = "자동 재시도를 중단하고 참조 분석·태그·마스크 확인으로 돌아갑니다."
            else:
                message = "현재 실행을 종료합니다."
        elif conditional:
            message = "현재 결과를 보존하고 선택한 항목을 후속 미세조정 대상으로 기록합니다."
        else:
            message = "현재 최종 이미지를 승인하고 저장 결정을 진행합니다."
        self.next_action_label.setText(message)

    def confirm(self) -> None:
        targets = [
            key for key, checkbox in self.target_checks.items()
            if checkbox.isChecked()
        ]
        try:
            self.selected_decision = create_final_review_decision(
                self.evidence,
                self.decision_combo.currentData(),
                reason_code=self.reason_combo.currentData(),
                refinement_targets=targets,
            )
        except FinalCandidateReviewError as error:
            QMessageBox.warning(self, "결정 확인", str(error))
            return
        self.accept()
