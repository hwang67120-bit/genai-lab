"""FLUX 한 경로(D-222) 결과의 고르기·저장. 사용자 화면이 쓰는 결과 객체(자유 생성 결과와 같은 저장 규칙).

후보 = run_style_redraw의 후보 3장(+ 부위별 고르기로 만든 그림). 처음 보여 주는 후보는 자동 표시(손 늘어남·부위 사라짐)가
없는 첫 후보이고, 품질 판정이 아니다. '저장' = 사용자가 이 그림을 쓰겠다는 결정(품질 항목 판정 아님, approve_saved_by_user).
"""
import hashlib
import json
from pathlib import Path


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
            self.candidates.append(dict(path=str(path), seed=info.get("seed"), sha256=info.get("sha256") or _sha(path),
                                        extra_hands=info.get("extra_hands"), missing_body_parts=info.get("missing_body_parts") or [],
                                        identity_anchors=info.get("identity_anchors"), autofix=info.get("autofix"),
                                        style_lora_weight=info.get("style_lora_weight"), variant="raw"))
        flagged = lambda c: bool(c["extra_hands"]) or bool(c["missing_body_parts"])
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
        self.candidates.append(dict(path=str(path), seed=None, sha256=_sha(path), extra_hands=None, missing_body_parts=[],
                                    variant=variant, **record))
        self.select(len(self.candidates) - 1)

    @property
    def head_notice(self):
        c = self.candidate
        label = f"후보 {self.selected + 1}/{len(self.candidates)}" + (f" (seed {c['seed']})" if c["seed"] is not None else f" ({c['variant']})")
        notes = []
        if c["extra_hands"]:
            notes.append("손이 원본보다 많게 보입니다")
        if c["missing_body_parts"]:
            notes.append("사라진 부위: " + ", ".join(c["missing_body_parts"]))
        from genai_lab.identity_anchors import summary_ko
        anchors = f" {summary_ko(c['identity_anchors'])}." if c.get("identity_anchors") else ""
        if c.get("style_lora_weight") is not None and c["style_lora_weight"] < 1.0:       # D-245 털 수인
            anchors += " 털 수인으로 판별해 그림체를 약하게 적용했습니다."
        fixed = c.get("autofix") or {}
        if fixed.get("round"):        # D-241 자동 고치기로 다시 그린 후보
            anchors += " 자동 보정: 빠진 특징(" + ", ".join(t.replace("_", " ") for t in fixed.get("asked", [])) + ")을 다시 요청해 그렸습니다."
        return (f"FLUX 결과 {label}." + anchors + (" 자동 표시: " + " · ".join(notes) + "." if notes else "")
                + " 자동 표시는 품질 판정이 아닙니다.")

    def approve_saved_by_user(self):
        """'저장' = 이 그림을 쓰겠다는 사용자 결정(같은 캐릭터 등 품질 항목 판정은 아님)."""
        self.approved_sha = self.verify_current()
        self.checks = dict(saved_by_user=True)
        self.status = "user_approved"
        self.persist()

    def persist(self, **extra):
        record = dict(mode="style_redraw", status=self.status, selected=self.selected, candidate=self.candidate,
                      approved_sha256=self.approved_sha, checks=self.checks, **extra)
        (self.directory / "user-review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def export(self, destination):
        if self.status != "user_approved" or self.verify_current() != self.approved_sha:
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
