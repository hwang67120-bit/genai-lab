"""Explicit approval of source-character tags; no inferred identity is enforced."""
from dataclasses import fields
from pathlib import Path
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QCheckBox, QScrollArea,
                              QWidget, QPushButton, QHBoxLayout)
from PIL import Image
from genai_lab.eye_color_analysis import is_eye_color_tag, review_eye_candidates, MULTICOLOR
from genai_lab.reference_features import automatic_feature_selection, tag_scope
from genai_lab.reference_prompt_budget import PromptBudgetError
from genai_lab.reference_tag_policy import (
    validate_character_gender, gender_tag_kind, GENDER_LABELS, normalize_tag,
)

_EYE_REASON_LABELS = {
    'low_score': '눈색 점수가 낮음',
    'ambiguous_colors': '서로 다른 색의 점수 차이가 작음',
    'crop_disagreement': '두 얼굴 크롭의 눈색 판단이 다름',
    'multicolor_signal': '복합색 가능성 또는 색상 혼동',
    'full_view_multicolor_signal': '전체 이미지에서 복합색 가능성 감지',
    'missing_head_mask': '얼굴·헤어 마스크 없음',
    'empty_head_mask': '얼굴·헤어 영역이 비어 있음',
    'head_roi_too_small': '얼굴·헤어 영역 해상도가 너무 작음',
    'no_supported_eye_labels': '모델에 지원되는 눈색 후보 없음',
    'consistent_across_crops': '두 크롭의 눈색 후보와 점수 기준이 일치함',
}


def _eye_reason_text(reason):
    prefix, separator, code = reason.partition(':')
    key = code if separator else prefix
    label = _EYE_REASON_LABELS.get(key, reason)
    if separator and prefix.startswith('view_'):
        label = f'크롭 {int(prefix[5:]) + 1}: {label}'
    return label


def analyze_character_tags(image, config, *, face_hair_mask=None,
                           hair_mask=None, face_mask=None,
                           cancelled=lambda: False):
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.eye_color_analysis import analyze_with_eye_review
    allowed = {field.name for field in fields(ClothingDesignAnalysisSettings)}
    values = {key: value for key, value in config.get('clothing_design_analysis', {}).items() if key in allowed}
    if 'cache_dir' in values:
        values['cache_dir'] = Path(values['cache_dir'])
    values['execution_provider'] = 'CPUExecutionProvider'
    eye_settings = config.get('character_eye_analysis', {})
    if cancelled():
        raise InterruptedError('캐릭터 태그 분석을 취소했습니다.')
    with WdTagSession(ClothingDesignAnalysisSettings(**values)) as session:
        result = analyze_with_eye_review(
            session, image, face_hair_mask, cancelled=cancelled,
            threshold=eye_settings.get('threshold', .35),
            minimum_margin=eye_settings.get('minimum_margin', .10))
        if hair_mask is None or face_mask is None:
            return result
        from genai_lab.hair_detail_analysis import (
            analyze_hair_details, resolve_hair_detail_settings,
        )
        return analyze_hair_details(
            session, image, hair_mask, face_mask, result,
            resolve_hair_detail_settings(config), cancelled=cancelled)


class CharacterTagReview(QDialog):
    def __init__(self, image, result, parent=None, *, character_gender='unspecified',
                 prompt_builder=None, outfit_tags=(), part_color_descriptions=()):
        super().__init__(parent)
        validate_character_gender(character_gender)
        from genai_lab.visual_reference_review import preview
        self.setWindowTitle('유지할 캐릭터 특징 확인')
        self.resize(620, 760)
        layout = QVBoxLayout(self)
        guide = QLabel('기준 이미지의 캐릭터 특징만 자동으로 골랐습니다. 분석은 추정이며 정답이 아닙니다.\n'
                       '기존 의상과 배경·인원수는 자동 반영하지 않습니다. '
                       '요약을 확인하고 승인하세요. 바꾸려면 상세 수정을 열어주세요.')
        guide.setWordWrap(True)
        layout.addWidget(guide)
        if character_gender != 'unspecified':
            layout.addWidget(QLabel(
                f'사용자 지정 성별: {GENDER_LABELS[character_gender]} — 분석 성별 태그는 제외됩니다. '
                '헤어·눈·체형 등 외형 태그는 그대로 선택할 수 있습니다.'))
        layout.addWidget(preview(image))
        self.prompt_builder = prompt_builder
        self.prompt_record = None
        self.prepared_request = None
        self.feature_route = automatic_feature_selection(result.tag_candidates, 'character')
        hair_report = getattr(result, 'hair_detail_report', None)
        optional_hair = {
            normalize_tag(tag)
            for tag in (hair_report or {}).get('optional_detail_tags', ())
        }
        if optional_hair:
            selected = tuple(
                tag for tag in self.feature_route['selected']
                if normalize_tag(tag) not in optional_hair
            )
            demoted = tuple(
                tag for tag in self.feature_route['selected']
                if normalize_tag(tag) in optional_hair
            )
            self.feature_route = dict(
                self.feature_route,
                selected=selected,
                unresolved=tuple(dict.fromkeys(
                    (*self.feature_route['unresolved'], *demoted))),
            )
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        from genai_lab.hair_detail_analysis import hair_detail_review_text
        hair_summary = QLabel(hair_detail_review_text(hair_report))
        hair_summary.setWordWrap(True)
        hair_summary.setVisible(bool(hair_summary.text()))
        layout.addWidget(hair_summary)
        if hair_report and hair_report.get('status') == 'analyzed':
            hair_row = QHBoxLayout()
            for view in hair_report.get('views', ()):
                path = Path(view['texture_path'])
                if path.is_file():
                    with Image.open(path) as texture:
                        thumbnail = texture.copy()
                    try:
                        thumbnail.thumbnail((120, 160))
                        hair_row.addWidget(preview(thumbnail))
                    finally:
                        thumbnail.close()
            layout.addLayout(hair_row)
        if outfit_tags:
            outfit_label = QLabel(
                '별도 의상 참조 이미지에서 승인한 생성 조건: '
                + ', '.join(normalize_tag(t) for t in outfit_tags))
            outfit_label.setWordWrap(True)
            layout.addWidget(outfit_label)
        from genai_lab.part_color_descriptions import color_description_summary
        self.part_color_summary = QLabel(color_description_summary(part_color_descriptions))
        self.part_color_summary.setWordWrap(True)
        self.part_color_summary.setVisible(bool(part_color_descriptions))
        layout.addWidget(self.part_color_summary)
        self.part_color_options = []
        from genai_lab.part_color_descriptions import PART_LABELS
        for description in part_color_descriptions:
            if description.status != 'proposed':
                continue
            option = QCheckBox(
                f"{PART_LABELS[description.part_name][1]} 색상 생성 조건 승인: "
                f"{description.prompt_text}"
            )
            option.setChecked(False)
            self.part_color_options.append((description.part_name, option))
            layout.addWidget(option)
        self.eye_omit_confirmation = QCheckBox('눈색을 확정하지 않고 진행 (눈색 태그 생략)')
        report = getattr(result, 'eye_color_report', None)
        self.eye_needs_review = report is not None and report.get('status') != 'suggested'
        self.eye_omit_confirmation.setVisible(self.eye_needs_review)
        layout.addWidget(self.eye_omit_confirmation)
        self.budget_label = QLabel()
        self.budget_label.setWordWrap(True)
        layout.addWidget(self.budget_label)
        self.details_button = QPushButton('상세 수정 / 눈색 확인')
        self.details_button.setCheckable(True)
        layout.addWidget(self.details_button)
        scroll = QScrollArea()
        self.details_panel = scroll
        scroll.setWidgetResizable(True)
        content = QWidget()
        tag_layout = QVBoxLayout(content)
        self.tag_checkboxes = []
        self.eye_options = []
        if report is not None:
            suggested = report.get('suggested_tag') if report.get('status') == 'suggested' else None
            description = (f'눈색 추천: {suggested} — 두 크롭 결과 일치, 확정 아님'
                           if suggested else '눈색 판단 보류 — 자동 선택하지 않았습니다.')
            status = QLabel(description + '\n원본 눈색과 비교하여 승인하세요. '
                            '점수는 전체/크롭 중 최대 원점수이며 정확도 확률이 아닙니다.\n'
                            '보류/추천 사유: ' + ', '.join(
                                _eye_reason_text(reason) for reason in report.get('reasons', [])))
            status.setWordWrap(True)
            tag_layout.addWidget(status)
            crop_row = QHBoxLayout()
            directory = Path(report['debug_dir']) if report.get('debug_dir') else None
            for index in range(2):
                path = directory / f'head_input_{index}.png' if directory else None
                if path is not None and path.is_file():
                    with Image.open(path) as crop:
                        crop_row.addWidget(preview(crop))
            tag_layout.addLayout(crop_row)
            eye_guide = QLabel(
                '눈색은 여러 개 선택할 수 있습니다. 모두 해제하면 눈색을 지정하지 않습니다.\n'
                '여러 색 선택은 오드아이 지정이 아닙니다. 색의 위치·비율·그라데이션까지 고정하지는 못합니다.')
            eye_guide.setWordWrap(True)
            tag_layout.addWidget(eye_guide)
            self.clear_eye_selection_button = QPushButton('눈색 모두 해제')
            self.clear_eye_selection_button.clicked.connect(self.clear_eye_selection)
            tag_layout.addWidget(self.clear_eye_selection_button)
            for name, score in review_eye_candidates(report):
                option = QCheckBox(f'{name.replace("_", " ")} — 최대 원점수 {score:.3f}')
                self.eye_options.append((name, option))
                tag_layout.addWidget(option)
                if name == suggested:
                    option.setChecked(True)
            if directory:
                path_label = QLabel(f'진단 파일: {directory / "eye_color_report.json"}')
                path_label.setWordWrap(True)
                tag_layout.addWidget(path_label)
        for tag in result.tag_candidates:
            if report is not None and is_eye_color_tag(tag.tag_name):
                continue
            check = QCheckBox(f'{tag.display_name} ({tag.score:.1%})')
            check.setChecked(tag.tag_name in self.feature_route['selected'])
            if tag_scope(tag.tag_name) == 'garment':
                check.setChecked(False)
                check.setEnabled(False)
                check.setText(check.text() + ' — 기준 이미지의 의상은 제외')
            if report is not None and normalize_tag(tag.tag_name) in MULTICOLOR:
                check.setChecked(False)  # Conflicting colors do not diagnose heterochromia.
            if character_gender != 'unspecified' and gender_tag_kind(tag.tag_name) is not None:
                check.setChecked(False)
                check.setEnabled(False)
                check.setText(check.text() + ' — 사용자 성별 지정으로 제외')
            self.tag_checkboxes.append((tag.tag_name, check))
            tag_layout.addWidget(check)
        if not self.tag_checkboxes:
            tag_layout.addWidget(QLabel('분석 기준을 통과한 태그가 없습니다. 태그 없이 승인할 수 있습니다.'))
        scroll.setWidget(content)
        layout.addWidget(scroll)
        scroll.setVisible(False)
        self.details_button.toggled.connect(scroll.setVisible)
        self.prompt_details = QLabel()
        self.prompt_details.setWordWrap(True)
        tag_layout.addWidget(self.prompt_details)
        self.approve_button = QPushButton('이 특징과 생성 조건 승인')
        self.approve_button.clicked.connect(self._approve_features)
        layout.addWidget(self.approve_button)
        cancel = QPushButton('취소 — 생성 중단')
        cancel.clicked.connect(self.reject)
        layout.addWidget(cancel)
        for _, checkbox in (
                self.tag_checkboxes + self.eye_options + self.part_color_options):
            checkbox.toggled.connect(self.refresh_summary)
        self.eye_omit_confirmation.toggled.connect(self._eye_omit_changed)
        self.refresh_summary()

    def _eye_omit_changed(self, checked):
        if checked:
            for _, option in self.eye_options:
                option.blockSignals(True)
                option.setChecked(False)
                option.blockSignals(False)
        self.refresh_summary()

    def refresh_summary(self, *_):
        tags = self.approved_tags
        uncertain = [tag for tag in self.feature_route['unresolved'] if tag not in tags]
        message = '유지할 캐릭터 특징: ' + (', '.join(normalize_tag(t) for t in tags) or '이미지 참조만 사용')
        if uncertain:
            message += '\n자동 확정하지 않은 특징: ' + ', '.join(normalize_tag(t) for t in uncertain)
        chosen_eyes = any(check.isChecked() for _, check in self.eye_options)
        if chosen_eyes and self.eye_omit_confirmation.isChecked():
            self.eye_omit_confirmation.blockSignals(True)
            self.eye_omit_confirmation.setChecked(False)
            self.eye_omit_confirmation.blockSignals(False)
        eye_ok = not self.eye_needs_review or chosen_eyes or self.eye_omit_confirmation.isChecked()
        if not eye_ok:
            message += '\n눈색 판단이 애매합니다. 상세 수정에서 색을 고르거나 눈색 생략을 확인하세요.'
        self.summary_label.setText(message)
        self.prompt_record = None
        self.prepared_request = None
        budget_ok = True
        if self.prompt_builder is not None:
            try:
                if self.part_color_options:
                    prepared, record = self.prompt_builder(
                        tags, self.approved_part_color_names)
                else:
                    prepared, record = self.prompt_builder(tags)
                self.prepared_request, self.prompt_record = prepared, record
                positive, negative = record['positive'], record['negative']
                counts = ' / '.join(f'{n}/{limit}' for n, limit in zip(
                    positive['effective_token_counts'], positive['tokenizer_limits']))
                omitted = positive.get('omitted_optional', [])
                text = f'생성 조건 길이: {counts} 토큰 — 승인한 캐릭터·의상 조건 모두 포함'
                if omitted:
                    text += '\n공간 부족으로 제외할 선택 표현: ' + ', '.join(omitted)
                long_plan = positive.get('long_prompt', {})
                if long_plan.get('chunk_count', 1) > 1:
                    text += (
                        f"\n승인 필요: SDXL 2청크 임베딩 "
                        f"({long_plan['chunk_count']}개, 추론 단계 변경 없음)"
                    )
                if negative['truncated']:
                    text += '\n부정 조건도 한도에 맞춰 일부 제외됩니다. 상세 내용에서 확인하세요.'
                self.budget_label.setText(text)
                chunk_details = ''
                if long_plan.get('chunk_count', 1) > 1:
                    chunk_details = '\n\n청크별 실제 문구:\n' + '\n'.join(
                        f"{item['index']}: {item['text']}"
                        for item in long_plan['chunks'])
                self.prompt_details.setText(
                    '실제 생성 프롬프트:\n' + prepared.prompt + chunk_details +
                    '\n\n실제 부정 프롬프트:\n' + prepared.negative_prompt +
                    '\n\n원래 부정 조건:\n' + negative.get('source_before_conflict_removal', negative['original']))
            except (ValueError, RuntimeError) as error:
                budget_ok = False
                self.budget_label.setText(str(error))
                self.prompt_details.setText(
                    error.report['required_prompt'] if isinstance(error, PromptBudgetError) else str(error))
        self.approve_button.setEnabled(eye_ok and budget_ok)

    def _approve_features(self):
        self.refresh_summary()
        if self.approve_button.isEnabled():
            self.accept()

    def clear_eye_selection(self):
        for _, option in self.eye_options:
            option.setChecked(False)

    @property
    def approved_tags(self):
        return tuple(name for name, check in self.tag_checkboxes + self.eye_options
                     if check.isChecked() and check.isEnabled())

    @property
    def approved_part_color_names(self):
        return tuple(
            name for name, check in self.part_color_options if check.isChecked())
