"""FLUX 한 경로(D-222) 결과의 고르기·저장. 사용자 화면이 쓰는 결과 객체(자유 생성 결과와 같은 저장 규칙).

후보 = run_style_redraw의 후보 3장(+ 부위별 고르기로 만든 그림). 처음 보여 주는 후보는 자동 표시(손 늘어남·부위 사라짐)가
없는 첫 후보이고, 품질 판정이 아니다. '저장' = 사용자가 이 그림을 쓰겠다는 결정(품질 항목 판정 아님, approve_saved_by_user).
"""
import hashlib
import json
from pathlib import Path
from genai_lab.style_redraw_anatomy import candidate_anatomy, require_reviewable


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StyleRedrawResults:
    free_mode = True                 # 사용자 화면의 '저장' 대상(자유 생성 결과와 같은 자리)
    edit_mode = False
    auto_head = False
    active_head = None
    show_before_head = False
    head_paste_unavailable_reason = "FLUX 결과는 원본 머리 픽셀을 합성하지 않습니다."
    finishing_unavailable_reason = "FLUX 결과 후처리는 이번 범위에 없습니다."
    background_notice = "배경·자세는 장면 문장으로 FLUX가 그립니다. 실제 반영을 확인해 주세요."

    def __init__(self, outcome):
        if outcome.get("state") != "review_pending":
            raise ValueError(outcome.get("message") or "후보가 없습니다.")
        self.outcome = outcome
        self.directory = Path(outcome["candidates"][0]).parent
        generated = json.loads((self.directory / "generation.json").read_text(encoding="utf-8"))
        by_file = {c["file"]: c for c in generated.get("candidates", [])}
        self.candidates = []
        for path in outcome["candidates"]:
            info = by_file.get(Path(path).name, {})
            if not candidate_anatomy(info)["eligible_for_review"]:
                continue
            self.candidates.append(dict(hands=info.get("hands"), anatomy=candidate_anatomy(info), path=str(path), seed=info.get("seed"), sha256=info.get("sha256") or _sha(path),
                                        extra_hands=info.get("extra_hands"), missing_body_parts=info.get("missing_body_parts") or [],
                                        identity_anchors=info.get("identity_anchors"), autofix=info.get("autofix"),
                                        style_lora_weight=info.get("style_lora_weight"), quality_diagnostic=info.get("quality_diagnostic"), variant="raw"))
        if not self.candidates:
            raise ValueError("손 과다 의심 또는 검사 미확인으로 선택 가능한 후보가 없습니다.")
        flagged = lambda c: ((c.get("anatomy") or {}).get("state") == "anatomy_review_required" or bool(c["extra_hands"]) or bool(c["missing_body_parts"])
                             or (c.get("quality_diagnostic") or {}).get("state") in ("issues_detected", "inconclusive"))
        self.selected = next((i for i, c in enumerate(self.candidates) if not flagged(c)), 0)
        self.status, self.approved_sha, self.checks = "awaiting_user_review", None, {}
        self.verify_current()

    @property
    def candidate(self):
        return self.candidates[self.selected]

    @property
    def current_path(self):
        return Path(self.candidate["path"])

    def verify_current(self):
        require_reviewable(self.candidate)
        digest = _sha(self.current_path)
        if digest != self.candidate["sha256"]:
            raise ValueError("후보 그림이 바뀌었습니다.")
        return digest

    def select(self, index):
        if not 0 <= index < len(self.candidates):
            raise ValueError("없는 후보입니다.")
        if index != self.selected:
            self.selected, self.approved_sha, self.status = index, None, "awaiting_user_review"
            self.persist()
        self.verify_current()

    def add_candidate(self, path, variant, **record):
        """부위별 고르기 등으로 만든 그림을 후보로 더하고 그것을 보여 준다."""
        require_reviewable(record)
        self.candidates.append(dict(path=str(path), seed=None, sha256=_sha(path), extra_hands=None, missing_body_parts=[],
                                    variant=variant, **record))
        self.select(len(self.candidates) - 1)

    @property
    def head_notice(self):
        c = self.candidate
        label = f"후보 {self.selected + 1}/{len(self.candidates)}" + (f" (seed {c['seed']})" if c["seed"] is not None else f" ({c['variant']})")
        notes = []
        anatomy = candidate_anatomy(c)
        if anatomy.get("schema_version") == 2 and not anatomy.get("automatic_correction_eligible"):
            notes.append("팔 연결·손 위치 미확인: 자동 보정 채택 보류, 원본 결과 직접 검토 필요")
        if "user_anatomy_rejection_not_cleared" in anatomy.get("arm_evidence", {}).get("issues", []):
            notes.append("사용자가 지적한 인체 결함 해소 미확인: 보정본 자동 채택 안 함")
        if "correction_geometry_review_required" in anatomy.get("arm_evidence", {}).get("issues", []):
            notes.append("보정 전후 팔·손 배치 변경 또는 기존 결함 미해소: 자동 채택 보류")
        notes.append("손가락 모양·여분 팔의 부재는 자동 검사로 확인되지 않음")
        frame = anatomy.get("arm_evidence", {}).get("framing", {})
        if frame.get("state") == "boundary_risk":
            notes.append("팔·손이 화면 경계에 닿음: 잘림 여부 확인 필요")
        elif frame.get("state") == "unassessed":
            notes.append("팔·손 잘림 여부 미확인: 가림과 검출 실패를 구별하지 못함")
        if c["extra_hands"]:
            notes.append("손이 원본보다 많게 보입니다")
        if c["missing_body_parts"]:
            notes.append("사라진 부위: " + ", ".join(c["missing_body_parts"]))
        diagnostic = c.get("quality_diagnostic") or {}
        for group, issue_label in (("identity", "특징 누락 의심"), ("garment", "의상 누락 의심")):
            tags = diagnostic.get("missing", {}).get(group, [])
            if tags:
                notes.append(issue_label + ": " + ", ".join(t.replace("_", " ") for t in tags))
        if diagnostic.get("contamination_suspected"):
            notes.append("문자·서명 오염 의심: " + ", ".join(diagnostic["contamination_suspected"]))
        if diagnostic.get("state") == "inconclusive":
            notes.append("자동 검사로 확인하지 못한 특징이 있습니다")
        from genai_lab.identity_anchors import summary_ko
        anchors = f" {summary_ko(c['identity_anchors'])}." if c.get("identity_anchors") else ""
        if c.get("style_lora_weight") == 0.0:                                              # D-246 털 수인 마지막 장치
            anchors += " 털 수인 특징을 지키려고 그림체 없이 다시 그렸습니다."
        elif c.get("style_lora_weight") is not None and c["style_lora_weight"] < 1.0:     # D-245 털 수인
            anchors += " 털 수인으로 판별해 그림체를 약하게 적용했습니다."
        fixed = c.get("autofix") or {}
        if fixed.get("blocked_reason") == "prompt_truncated":
            notes.append("보정 지시가 입력 한도를 넘어 자동 보정을 생략했습니다")
        if fixed.get("round"):        # D-241 자동 고치기로 다시 그린 후보
            anchors += " 자동 보정: 빠진 특징(" + ", ".join(t.replace("_", " ") for t in fixed.get("asked", [])) + ")을 다시 요청해 그렸습니다."
        return (f"FLUX 결과 {label}." + anchors + (" 자동 표시: " + " · ".join(notes) + "." if notes else "")
                + " 자동 표시는 품질 판정이 아닙니다.")

    def user_rejected(self):
        return any(self.directory.glob(f"rejected-{self.candidate['sha256']}-*.json"))

    def approve_saved_by_user(self):
        """'저장' = 이 그림을 쓰겠다는 사용자 결정(같은 캐릭터 등 품질 항목 판정은 아님)."""
        if self.user_rejected():
            raise ValueError("사용자가 거절한 후보입니다.")
        self.approved_sha = self.verify_current()
        self.checks = dict(saved_by_user=True)
        self.status = "user_approved"
        self.persist()

    def persist(self, **extra):
        record = dict(mode="style_redraw", status=self.status, selected=self.selected, candidate=self.candidate,
                      approved_sha256=self.approved_sha, checks=self.checks, **extra)
        (self.directory / "user-review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def export(self, destination):
        if self.user_rejected() or self.status != "user_approved" or self.verify_current() != self.approved_sha:
            raise ValueError("현재 결과를 먼저 승인해 주세요.")
        destination = Path(destination)
        if destination.suffix.lower() != ".png":
            raise ValueError("PNG로 저장해 주세요.")
        sidecar = destination.with_suffix(".review.json")
        if destination.exists() or sidecar.exists():
            raise FileExistsError("이미 있는 파일입니다.")
        from genai_lab.ai_label import label_record, labeled_png   # D-152: 저장 파일에 AI 생성 표시(픽셀은 그대로)
        label = label_record("generated")
        data = labeled_png(self.current_path.read_bytes(), label)
        metadata = dict(mode="style_redraw", engine="flux", reviewer="user", checks=self.checks, source_run=str(self.directory),
                        candidate=self.candidate, approved_sha256=self.approved_sha, sha256=hashlib.sha256(data).hexdigest(),
                        ai_label=label, outcome=str(self.directory / "outcome.json"), approval_record=str(self.directory / "user-review.json"))
        created = []
        try:
            with sidecar.open("x", encoding="utf-8") as stream:
                created.append(sidecar)
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
            with destination.open("xb") as stream:
                created.append(destination)
                stream.write(data)
        except Exception:
            for path in created:
                path.unlink()
            raise
        self.status = "saved"
        self.persist(destination=str(destination))
        return destination
