"""자유 생성 결과의 사람 승인·SHA 연결. 편집 단계 quality_pass/머리 합성 기준과 혼용하지 않는다."""
import json
from pathlib import Path
from genai_lab.outfit_edit import sha, write_json
from genai_lab.outfit_edit_results import OutfitEditResults
from genai_lab.outfit_judgement import validate_judgement


class FreeGenerationResults(OutfitEditResults):
    free_mode = True
    head_paste_unavailable_reason = "자유 생성은 원본 머리 픽셀을 합성하지 않습니다."
    finishing_unavailable_reason = "자유 생성 후처리는 이번 범위에 없습니다."
    background_notice = "요청한 배경 또는 기본 흰 배경을 Qwen이 생성합니다. 실제 반영을 확인해 주세요."

    def __init__(self, outcome):
        if outcome["status"] == "blocked":
            raise ValueError("금지된 결과 관찰로 사용을 차단했습니다. trace.json을 확인해 주세요.")
        # 고를 수 있는 시도(D-126 ⑤): 안전 차단된 시도는 빼고, 처음 보여 주는 것은 자동 위반이 적은 시도
        self.candidates = [c for c in (outcome.get("candidates") or
                                       [dict(attempt=outcome.get("selected_attempt") or len(outcome["attempts"]),
                                             raw=outcome["result"], sha256=outcome["result_sha256"])]) if not c.get("blocked")]
        if not self.candidates:
            raise ValueError("사용할 수 있는 시도가 없습니다.")
        first = next((i for i, c in enumerate(self.candidates) if c["attempt"] == outcome.get("selected_attempt")), 0)
        self.selected = first
        super().__init__(dict(outcome, qwen_view=outcome["result"]))
        self.verify_current()

    @property
    def candidate(self):
        return self.candidates[self.selected]

    @property
    def current_path(self):
        return Path(self.candidate["raw"])

    @property
    def expected_sha(self):
        return self.candidate["sha256"]

    @expected_sha.setter
    def expected_sha(self, value):          # 부모 생성자가 쓰는 자리(자유 생성은 시도별 SHA를 쓴다)
        pass

    def select(self, index):
        if not 0 <= index < len(self.candidates):
            raise ValueError("없는 시도입니다.")
        if index != self.selected:
            self.selected = index
            self.approved_sha = None        # 다른 시도를 고르면 다시 확인한다
            self.status = "awaiting_user_review"
            self.persist()
        self.verify_current()

    @property
    def head_notice(self):
        text = "Qwen 생성 결과입니다. 원본 머리 합성 없음 · 자동 특징 관찰과 사람 품질 판정은 별도입니다."
        if len(self.candidates) > 1:
            text += f" 시도 {len(self.candidates)}개 중 {self.candidate['attempt']}번째를 보고 있습니다 — 결과 목록에서 바꿔 보고 골라 주세요(자동 관찰 기준 순서, 품질 보장 아님)."
        return text

    def approve(self, checks):
        required = {"character", "garment", "request", "exposure"} | ({"distinctive"} if self.outcome.get("distinctive_review") else set())
        if set(checks) != required or not all(v is True for v in checks.values()):
            raise ValueError("같은 캐릭터·같은 의상·연출 반영·심한 이상/노출 없음(그리고 고유 형태 메모)을 확인해 주세요.")
        self.approved_sha = self.verify_current()
        self.checks = dict(checks)
        self.status = "user_approved"
        self.persist()
        self.record_human(dict(same_character="yes", outfit="same", request_followed="yes", severe_defect="none",
                               quality_pass=True), "gui_approval")

    def record_human(self, values, source):
        target = dict(image=str(self.current_path), sha256=self.verify_current(), attempt=self.candidate["attempt"])
        # 편집 단계와 같은 판정 형식(D-114: 항목 검증·recorded_at)으로 남겨 운영 집계(D-118)에 들어가게 한다(검토 4번)
        entry = validate_judgement(dict(judge="user", source=source, target=target, mode="free_generation", **values))
        path = self.directory / "judgement.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        previous.append(entry)
        write_json(path, previous)
        trace_path = self.directory / "trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["human_judgements"] = previous
        trace["judgement"] = dict(values, judge="user", target=target)
        trace["status"] = self.status
        # 같은 기준/프롬프트/최종 시도에 대한 사람 관찰이다. 자동 관찰은 그대로 보존.
        trace["human_condition_judgements"] = [dict(key=k, status="fulfilled", judge="user", target=target,
            evidence=source) for k in ("keep_character", "keep_outfit", "request", "safety")] if values.get("quality_pass") else []
        write_json(trace_path, trace)

    def reject_and_next(self):
        self.status = "rejected_by_user"
        self.approved_sha = None
        self.persist()
        self.record_human(dict(quality_pass=False, same_character="not_judged", outfit="not_judged",
                              request_followed="not_judged", severe_defect="not_judged"), "gui_reject")
        return 0

    def export(self, destination):
        if self.status != "user_approved" or self.verify_current() != self.approved_sha:
            raise ValueError("현재 결과를 먼저 승인해 주세요.")
        destination = Path(destination)
        if destination.suffix.lower() != ".png":
            raise ValueError("PNG로 저장해 주세요.")
        sidecar = destination.with_suffix(".review.json")
        if destination.exists() or sidecar.exists():
            raise FileExistsError("이미 있는 파일입니다.")
        data = self.current_path.read_bytes()
        import hashlib
        if hashlib.sha256(data).hexdigest() != self.approved_sha:
            raise ValueError("승인 이미지 변경")
        metadata = dict(mode="free_generation", sha256=self.approved_sha, reviewer="user", checks=self.checks,
                        source_run=str(self.directory), base_kind=self.outcome["base_kind"], attempt=self.candidate["attempt"],
                        head_composited=False, trace=self.outcome["trace"], approval_record=str(self.directory / "user-review.json"))
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
