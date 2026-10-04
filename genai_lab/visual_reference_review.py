"""Explicit input approval and advisory-score candidate selection."""
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget, QComboBox, QToolButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PIL import Image


def preview(image, max_size=(220, 360)):
    rgb = image.convert('RGB')
    rgb.thumbnail(max_size)
    qimage = QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
    label = QLabel()
    label.setPixmap(QPixmap.fromImage(qimage))
    rgb.close()
    return label


class VisualInputReview(QDialog):
    def __init__(self, inputs, parent=None):
        super().__init__(parent)
        self._reference_inputs = inputs
        self.setWindowTitle('얼굴·의상 참조 입력 검토 — 초기 이미지 없음')
        self.resize(980, 760)
        shell = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)
        shell.addWidget(scroll)
        layout.addWidget(QLabel(
            '얼굴·헤어·의상 참조와 영향 영역만 전달합니다. 자세·구도·픽셀 보존은 보장되지 않습니다.\n'
            '얼굴, 헤어 또는 의상 영향 영역이 틀리면 취소하세요.'))
        row = QHBoxLayout()
        for name, image in (('기준', inputs.source),
                            ('얼굴 전용 참조', inputs.identity),
                            ('헤어 전용 참조', inputs.hair_reference),
                            ('헤어 영향 영역', inputs.hair_mask),
                            ('의상 참조', inputs.garment),
                            ('의상 영향 영역', inputs.garment_mask)):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            column.addWidget(preview(image) if image is not None else QLabel('미사용 (생성기에 전달하지 않음)'))
            row.addLayout(column)
        layout.addLayout(row)
        from genai_lab.reference_order import adapter_reference_names
        labels = {'identity': '얼굴', 'hair': '헤어',
                  'human_ears': '사람 귀', 'animal_ears': '동물 귀',
                  'ears': '동물 귀',
                  'tail': '꼬리', 'garment': '의상'}
        order = adapter_reference_names(inputs)
        layout.addWidget(QLabel('참조 전달 순서: ' + ' → '.join(labels.get(name, name) for name in order)))
        record = getattr(inputs, 'analysis_record', None)
        if record is not None:
            part_status = {
                'analyzer_not_connected': '자동 검출기 미연결',
                'measured': '검출 결과 있음 — 원본과 직접 확인',
                'unresolved': '판단 보류',
                'detected_not_conditioned': '원본 검출 완료 / 생성 위치 미확정 — 귀·꼬리 참조 미전달',
                'source_regions_experiment': '귀·꼬리 RGB 전달 — 원본 위치를 영향 영역으로 사용한 실험 (자세 고정 아님)',
            }.get(record.get('additional_parts_status'), '기록 없음')
            warning = QLabel(
                '참조 영역만 추출했습니다. 중립색 이미지 생성·의상 제거 검증은 실행하지 않습니다.\n'
                '의상 태그의 성별 제외는 의상 이미지의 체형 영향 제거를 보장하지 않습니다.\n'
                '홍채·귀·꼬리 등 미지원/미검출 부위는 원본과 자동으로 일치한다고 판단하지 않습니다.\n'
                f"추가 부위 분석: {part_status}")
            warning.setWordWrap(True)
            layout.addWidget(warning)
            policy = record.get("part_detection", {}).get("filter_policy", {})
            if policy.get("requires_input_review"):
                note = QLabel("추가 부위는 실험 규칙으로 고른 제안입니다. "
                              "승인은 아래 입력의 사용 허용이며 자동 검출의 정답 인증이 아닙니다.")
                note.setWordWrap(True)
                layout.addWidget(note)
            separated = record.get("condition_separation")
            if separated:
                note = QLabel("부위 원본 조건 분리 적용: 얼굴·의상·추가 부위가 서로의 원본 조건을 변경하지 않습니다.\n"
                              "아래 영향 영역은 기존 배치 규칙으로 조립한 결과입니다. 공통 생성 모델과 자세는 그대로입니다.")
                note.setWordWrap(True)
                layout.addWidget(note)
            hair_visual = record.get("hair_visual_condition", {})
            if hair_visual.get("status") == "prepared":
                note = QLabel(
                    "헤어 시각 참조 분리 적용: 얼굴 전용 identity에서 헤어를 제외하고, "
                    "중성 회색 배경의 헤어 전용 참조를 별도 어댑터로 전달합니다.\n"
                    f"헤어 강도={hair_visual.get('reference_scale')}, "
                    f"얼굴에서 제외된 헤어 픽셀="
                    f"{hair_visual.get('identity_hair_overlap_removed', 0)}. "
                    "귀·꼬리·의상 조건은 변경하지 않습니다.")
                note.setWordWrap(True)
                layout.addWidget(note)
            region = record.get("part_region_experiment", {})
            if region:
                note = QLabel(
                    f"원본 위치 배정 제안: 얼굴 영향 {region.get('identity_overlap_removed', 0)}픽셀, "
                    f"의상 영향 {region.get('garment_overlap_removed', 0)}픽셀 제외. "
                    "추출 RGB는 바꾸지 않지만 생성 영향 영역은 달라집니다.")
                note.setWordWrap(True)
                layout.addWidget(note)
                for name, audit in region.get("generation_region_strengthening", {}).items():
                    if audit.get("changed"):
                        cells = audit.get("attention_cells", {})
                        detail = ", ".join(
                            f"{queries}={value.get('strong_cells')}/{value.get('minimum_cells')}셀"
                            for queries, value in cells.items())
                        strengthened = QLabel(
                            f"{name} 생성 영역 보강: 반경 {audit.get('radius_pixels')}px, "
                            f"면적 {audit.get('area_growth', 1):.2f}배, {detail}. "
                            "원본 RGB·검출 마스크·참조 강도는 변경하지 않습니다.")
                        strengthened.setWordWrap(True)
                        layout.addWidget(strengthened)
            for name, part in record.get("part_detection", {}).get("parts", {}).items():
                layout.addWidget(QLabel(f"{name}: 원본 검출={part['status']}, 생성 위치={part['target_status']}"))
                if any(item.get("requires_review") and (item.get("garment_overlap", 0) > 0 or item.get("hair_overlap", 0) > 0)
                       for item in part.get("source_filter", [])):
                    notice = QLabel("부분 겹침 후보 유지: 의상·헤어 마스크와 판단이 다릅니다. "
                                    "아래 원본 크롭에 소매나 머리 전체가 섞였으면 취소하세요.")
                    notice.setWordWrap(True)
                    layout.addWidget(notice)
                if part.get("crop_path"):
                    with Image.open(part["crop_path"]) as crop:
                        layout.addWidget(preview(crop))
        for name, evidence in (getattr(inputs, 'color_evidence', None) or {}).items():
            palette_row = QHBoxLayout()
            palette_row.addWidget(QLabel(f'{name} 색 분포 (패턴 판단 보류)'))
            if not evidence.palette_rgb:
                palette_row.addWidget(QLabel('유효 영역 미검출 — 색을 추정하지 않음'))
            for rgb, ratio in zip(evidence.palette_rgb, evidence.area_ratios):
                chip = QLabel(f'RGB {tuple(rgb)} / {ratio:.1%}')
                chip.setStyleSheet(
                    f'background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); '
                    f'color: {"black" if sum(rgb) > 382 else "white"}; padding: 4px;')
                palette_row.addWidget(chip)
            layout.addLayout(palette_row)
        extras = getattr(inputs, 'extra_references', ())
        if extras:
            extra_row = QHBoxLayout()
            for part in extras:
                column = QVBoxLayout()
                column.addWidget(QLabel(f'{part.name} 참조 / 강도 {part.scale}'))
                column.addWidget(preview(part.rgb))
                column.addWidget(preview(part.region))
                extra_row.addLayout(column)
            layout.addLayout(extra_row)
        from genai_lab.part_color_descriptions import color_description_summary
        color_summary = color_description_summary(getattr(inputs, 'part_color_descriptions', ()))
        self.part_color_summary = QLabel(color_summary or
            '색 분포는 원본 확인용입니다. 부위 색상 설명 연결은 사용하지 않습니다.')
        self.part_color_summary.setWordWrap(True)
        layout.addWidget(self.part_color_summary)
        scene = getattr(inputs, 'scene_condition', None)
        if scene is not None:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            container = QWidget()
            scene_row = QHBoxLayout(container)
            for name, image in [('구조 선화 (초기 RGB 아님)', scene.hint)] + [
                    (part.name, part.rgb) for part in scene.references]:
                column = QVBoxLayout()
                column.addWidget(QLabel(name))
                column.addWidget(preview(image))
                scene_row.addLayout(column)
            scroll.setWidget(container)
            layout.addWidget(scroll)
        approve = QPushButton('입력 승인 후 후보 생성')
        approve.clicked.connect(self.accept)
        cancel = QPushButton('취소')
        cancel.clicked.connect(self.reject)
        shell.addWidget(approve)
        shell.addWidget(cancel)


    def accept(self):
        from genai_lab.reference_experiment_approval import approve_reference_experiment
        approve_reference_experiment(self._reference_inputs)
        super().accept()

    def reject(self):
        self._reference_inputs.approved_generation = None
        record = self._reference_inputs.analysis_record
        if record is not None:
            record.pop("experimental_input_approval", None)
        super().reject()


class VisualCandidateReview(QDialog):
    def __init__(self, batch, parent=None):
        super().__init__(parent)
        native_base_review = batch.review_stage == 'native_base'
        self.setWindowTitle(
            '의상 적용 전 · 캐릭터 후보 선택'
            if native_base_review
            else '후보 비교 — 선택 후 기존 승인·저장 단계로 이동'
        )
        self.resize(950, 650)
        layout = QVBoxLayout(self)
        self.stage_banner = QLabel(
            '의상 적용 전 — 아직 최종 결과가 아닙니다.\n'
            '원본 캐릭터와 닮은 후보를 고르세요. 선택한 뒤 목표 의상을 적용합니다.'
            if native_base_review else
            '생성 후보 비교 — 아직 승인·저장되지 않았습니다.\n'
            '후보를 선택하면 최종 검토 화면으로 이동합니다.'
        )
        self.stage_banner.setWordWrap(True)
        self.stage_banner.setStyleSheet(
            'background: #fff2cf; color: #59400c; border: 1px solid #d3b466; '
            'border-radius: 8px; padding: 12px; font-size: 15px; font-weight: 600;')
        layout.addWidget(self.stage_banner)
        self.stage_progress = QLabel(
            '현재: 캐릭터 후보 선택  →  다음: 목표 의상 적용  →  최종 결과 검토·저장'
            if native_base_review else '현재: 후보 선택  →  다음: 최종 결과 검토·저장')
        self.stage_progress.setWordWrap(True)
        layout.addWidget(self.stage_progress)
        self.diagnostics_toggle = QToolButton()
        self.diagnostics_toggle.setText('기술 진단 펼치기')
        self.diagnostics_toggle.setCheckable(True)
        self.diagnostics_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.diagnostics_toggle.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(self.diagnostics_toggle)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        self.diagnostics_label = QLabel(
            ('캐릭터 후보는 의상 조건 없이 생성되었습니다. 목표 의상은 다음 단계에 사용합니다.\n'
             if native_base_review else '')
            + '유사도는 참고 지표이며 확률이나 정확도가 아닙니다. '
            '검사 통과는 최종 이미지 승인이나 의상 적용 완료를 뜻하지 않습니다.\n'
            + batch.warning)
        self.diagnostics_label.setWordWrap(True)
        self.diagnostics_label.hide()
        body_layout.addWidget(self.diagnostics_label)
        def toggle_diagnostics(checked):
            self.diagnostics_label.setVisible(checked)
            self.diagnostics_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
            self.diagnostics_toggle.setText('기술 진단 접기' if checked else '기술 진단 펼치기')
        self.diagnostics_toggle.toggled.connect(toggle_diagnostics)
        reference_row = QHBoxLayout()
        for name, filename in (
            ('원본 캐릭터 · 비교 기준', 'input_source.png'),
            (
                '다음 단계에 적용할 목표 의상 · 아직 미적용'
                if native_base_review
                else '참조 의상',
                'input_garment.png',
            ),
        ):
            path = batch.directory / filename
            if path.exists():
                column = QVBoxLayout()
                column.addWidget(QLabel(name))
                with Image.open(path) as reference:
                    column.addWidget(preview(reference, (140, 180)))
                reference_row.addLayout(column)
        body_layout.addLayout(reference_row)
        container = QWidget()
        row = QHBoxLayout(container)
        self.selection = QComboBox()
        self.batch = batch
        self.part_reviews = []
        self.candidate_stage_labels = []
        for index, candidate in enumerate(batch.candidates):
            column = QVBoxLayout()
            stage_label = QLabel(
                f'캐릭터 후보 {index + 1} · 의상 적용 전'
                if native_base_review else f'생성 후보 {index + 1} · 최종 검토 전')
            stage_label.setStyleSheet('background: #fff2cf; color: #59400c; padding: 8px; font-weight: 600;')
            column.addWidget(stage_label)
            self.candidate_stage_labels.append(stage_label)
            scores = candidate.design_reference_record['similarity']
            from genai_lab.reference_tag_policy import GENDER_LABELS
            gender = candidate.design_reference_record.get('character_gender', 'unspecified')
            column.addWidget(QLabel('사용자 지정 성별: ' + GENDER_LABELS.get(gender, '기록 없음')))
            column.addWidget(QLabel(candidate.design_reference_record.get('reference_variant', '의상 이미지 참조 ON')))
            column.addWidget(QLabel('참조 강도: 얼굴 {} / 헤어 {} / 의상 {}'.format(
                candidate.design_reference_record.get('identity_scale', '기록 없음'),
                candidate.design_reference_record.get(
                    'hair_reference_scale', '미사용'),
                candidate.design_reference_record.get('garment_scale', '기록 없음'))))
            def value(key):
                raw = scores.get(key)
                if raw is None:
                    return '계산 불가'
                percentage = max(0.0, min(1.0, float(raw))) * 100.0
                return f'{percentage:.1f}%'
            score_text = (
                f'후보 {index + 1} / 시드 {candidate.seed}\n'
                f'얼굴 {value("character")} / 헤어 {value("hair")}\n'
                '의상은 Base 승인 항목 아님'
                if native_base_review
                else (
                    f'후보 {index + 1} / 시드 {candidate.seed}\n'
                    f'얼굴 {value("character")} / 의상 {value("garment")}\n'
                    f'전신 인상 {value("vibe")}'
                )
            )
            column.addWidget(QLabel(score_text))
            decision = candidate.design_reference_record.get(
                'candidate_decision', {})
            refinement_targets = decision.get('refinement_targets', [])
            quality_text = (
                '미세 조정 필요: ' + ', '.join(refinement_targets)
                if decision.get('refinement_required')
                else '참고 유사도 기준 충족 · 최종 승인 아님'
            )
            column.addWidget(QLabel(quality_text))
            column.addWidget(preview(candidate.image))
            column.addWidget(QLabel(
                '인물 구조·성별·얼굴·헤어·동물귀·꼬리를 확인하세요.'
                if native_base_review
                else '비율·캐릭터 특징·의상 디자인을 직접 확인 후 선택하세요.'
            ))
            review_fields = {}
            review_panel = QWidget()
            review_layout = QVBoxLayout(review_panel)
            review_panel.setVisible(False)
            review_toggle = QPushButton('부위별 확인 기록 (선택)')
            review_toggle.setCheckable(True)
            review_toggle.toggled.connect(review_panel.setVisible)
            column.addWidget(review_toggle)
            for key, title in (
                ('gender_condition', '지정 성별 조건'), ('eyes', '눈 디자인'),
                ('hair', '헤어 디자인'),
                ('human_ears', '사람 귀 형태·개수·위치'),
                ('animal_ears', '동물 귀 형태·종·개수·위치'),
                ('tail', '꼬리 형태·배색'), ('garment', '현재 옷 관찰 (목표 의상 미적용)' if native_base_review else '의상 디자인'), ('anatomy', '인체 비율')):
                selector = QComboBox()
                for value, label in (
                    ('not_reviewed', '미확인'), ('similar', '유사'),
                    ('different', '차이 있음'), ('unavailable', '해당 없음/확인 불가')):
                    selector.addItem(f'{title}: {label}', value)
                review_fields[key] = selector
                review_layout.addWidget(selector)
            column.addWidget(review_panel)
            self.part_reviews.append(review_fields)
            if scores['warnings']:
                column.addWidget(QLabel('\n'.join(scores['warnings'])))
            view = QPushButton('원본 해상도로 보기')
            view.clicked.connect(lambda checked=False, p=batch.paths[index]: self.show_full(p))
            column.addWidget(view)
            row.addLayout(column)
            self.selection.addItem(f'후보 {index + 1}', index)
        body_layout.addWidget(container)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.next_action_hint = QLabel(
            "선택해도 완성·저장되지 않습니다. 다음 단계에서 목표 의상을 적용합니다."
            if native_base_review else "선택 후 최종 검토에서 승인·저장을 결정합니다.")
        self.next_action_hint.setWordWrap(True)
        layout.addWidget(self.next_action_hint)
        layout.addWidget(self.selection)
        choose = QPushButton(
            '이 캐릭터를 선택하고 의상 적용하기'
            if native_base_review
            else '선택한 후보를 최종 검토로 전달'
        )
        self.choose_button = choose
        choose.setMinimumHeight(40)
        choose.clicked.connect(self.confirm_selection)
        reject = QPushButton('선택하지 않고 돌아가기')
        reject.clicked.connect(self.reject)
        layout.addWidget(choose)
        layout.addWidget(reject)
        self.diagnostics_label.setText(self.diagnostics_label.text() + f'\n미승인 후보 임시 보관: {batch.directory}')

    def confirm_selection(self):
        # 체크를 강요하거나 점수로 자동 합격시키지 않는다.
        import json
        index = self.selection.currentData()
        review = {key: box.currentData() for key, box in self.part_reviews[index].items()}
        record = self.batch.candidates[index].design_reference_record
        record['part_review'] = review
        # 격리된 시도가 있으면 UI 순번과 실제 후보 번호가 다를 수 있다.
        path = self.batch.paths[index].with_suffix('.json')
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                saved['part_review'] = review
                path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
            except (OSError, ValueError) as error:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, '검토 기록 저장 실패', str(error))
                return
        self.accept()

    def show_full(self, path):
        dialog = QDialog(self)
        dialog.resize(800, 850)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        label = QLabel()
        label.setPixmap(QPixmap(str(path)))
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.exec()
