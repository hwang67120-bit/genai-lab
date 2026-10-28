"""기존 분석 결과를 연결한다. 모든 후보는 미확정 상태로 유지한다."""
from dataclasses import replace
from genai_lab.qwen_preservation import file_sha


def report_candidates(source, image_sha256, report):
    """실패·미검출의 불확실성을 보존한다. 개수·무늬 위치·성별은 추정하지 않는다."""
    candidates = []
    def add(fact, english="", location="미확정", issues=()):
        candidates.append({"fact_ko": str(fact), "english": english, "check_at": location,
                           "issues": list(issues)})
    if source == "part_color_descriptions":
        for row in report:
            text = row.get("prompt_text", "")
            add(text or "부위 색 미확정", f"Preserve the {text} shown in Picture 1." if text else "",
                row.get("part_name", "미확정"), ("verify_color_not_pattern",))
    elif source == "eye_color_analysis":
        tag = report.get("suggested_tag") if report.get("status") == "suggested" else None
        add(tag or "눈 색 미확정 — 사용자 입력 필요",
            f"The character has the same face with {tag}." if tag else "", "눈",
            () if tag else ("analysis_unresolved",))
    elif source in ("hair_detail_analysis", "garment_detail_analysis"):
        for row in report.get("evidence", ()):
            if source == "hair_detail_analysis" and not row.get("eligible_for_prompt", False):
                continue
            tag = row.get("tag", "")
            add(tag, f"Preserve the {tag} shown in Picture 1.",
                str(row.get("semantic_location", row.get("location", "미확정"))),
                ("model_candidate_not_fact",))
    elif source in ("extra_parts_analysis", "accessory_analysis"):
        # 검출만으로 개수·모양을 확정할 수 없으므로 전체 보고서를 검토용으로 보존한다.
        add("귀·꼬리/장신구 검출 근거를 확인하고 보이는 형태를 입력하세요", "", "검출 영역",
            ("count_and_shape_need_user_input",))
    if not candidates:
        add(f"{source}: 분석 미확정 — 사용자 입력 필요", issues=("analysis_unresolved",))
    return {"source": source, "image_sha256": image_sha256, "kind": source,
            "candidates": candidates, "raw_report": report}


class FinishedImageAnalyzer:
    """승인된 완성 이미지를 CPU WD로 분석하며 저장 마스크를 선택적으로 쓴다. 새 분할은 실행하지 않는다. 마스크가 없으면 수동 입력 후보를 만들며 이미지에
    연결된 기존 보고서는 report_candidates로 별도 전달할 수 있다.
    """
    def __init__(self, settings, *, head_mask=None, hair_mask=None, face_mask=None):
        self.settings = replace(settings, execution_provider="CPUExecutionProvider", local_files_only=True)
        self.head_mask, self.hair_mask, self.face_mask = head_mask, hair_mask, face_mask
        self.session = None

    def analyze(self, image, image_sha256):
        from genai_lab.clothing_analysis import WdTagSession
        from genai_lab.garment_detail_analysis import analyze_garment_details
        self.session = WdTagSession(self.settings)
        reports = []
        # 여러 시점의 의상 관찰까지 하나의 세션을 재사용한다.
        class Borrow:
            def __enter__(inner): return self.session
            def __exit__(inner, *args): return False
        result = analyze_garment_details(image, self.settings, lambda _: Borrow())
        reports.append(report_candidates("garment_detail_analysis", image_sha256, result.garment_detail_report))
        if self.head_mask is not None:
            from genai_lab.eye_color_analysis import analyze_with_eye_review
            observed = analyze_with_eye_review(self.session, image, self.head_mask)
            eye_report = observed.eye_color_report
        else:
            eye_report = {"status": "unresolved", "reason": "missing_finished_image_head_mask"}
        reports.append(report_candidates("eye_color_analysis", image_sha256, eye_report))
        # 전체 이미지 외형 제안은 부분 머리 분석을 대신하지 못한다.
        hair_candidates = [{"fact_ko": x.tag_name, "english": f"Preserve the {x.tag_name} shown in Picture 1.",
                            "check_at": "머리", "issues": ["whole_image_candidate"]}
                           for x in result.tag_candidates if x.tag_name.replace("_", " ").endswith(" hair")]
        reports.append({"source": "wd_finished_image_hair", "image_sha256": image_sha256,
                        "kind": "hair", "candidates": hair_candidates})
        if self.hair_mask is not None and self.face_mask is not None:
            from genai_lab.hair_detail_analysis import analyze_hair_details, HairDetailAnalysisSettings
            try:
                observed = analyze_hair_details(self.session, image, self.hair_mask, self.face_mask,
                                               result, HairDetailAnalysisSettings())
                hair_report = observed.hair_detail_report
            except Exception as error:
                hair_report = {"status": "unresolved", "reason": str(error)}
        else:
            hair_report = {"status": "unresolved", "reason": "missing_finished_image_hair_face_masks"}
        reports.append(report_candidates("hair_detail_analysis", image_sha256, hair_report))
        for source in ("extra_parts_analysis", "accessory_analysis"):
            reports.append(report_candidates(source, image_sha256, {"status": "unresolved",
                "reason": "localized_finished_image_evidence_required"}))
        return reports

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
