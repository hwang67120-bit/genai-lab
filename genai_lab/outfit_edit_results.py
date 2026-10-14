"""원본 편집형 의상 교체 결과 1장의 검토·승인·저장(D-094 4단계). 작업실이 쓰는 StudioResults와 같은 이름의 속성·함수를 둔다.
자동 승인은 없다. 머리를 붙이지 못한 경우에도 Qwen 결과를 보여 주고 그 사실을 알린다(사용자가 판단).
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from genai_lab.outfit_edit import sha, write_json
from genai_lab.outfit_judgement import record_judgement


class OutfitEditResults:
    edit_mode = True
    two_pass = False
    auto_head = None
    active_head = None
    show_before_head = False
    head_paste_unavailable_reason = "원본 편집 결과에는 머리가 이미 원본 픽셀로 붙어 있습니다."
    finishing_unavailable_reason = "원본 편집 결과에는 고화질 마무리를 아직 쓰지 않습니다."
    background_notice = "배경은 Qwen이 흰 배경으로 그렸습니다."
    selected = 0

    def __init__(self, outcome, *, appendage_review_required=False):
        self.outcome = dict(outcome)
        self.directory = Path(outcome["directory"])
        self.batch = SimpleNamespace(directory=self.directory, candidates=(None,))
        self.composited = bool(outcome["composited"])
        self.result_path = (Path(outcome["result"]) if self.composited
                            else Path(outcome.get("qwen_view", self.directory / "compose" / "qwen_view.png")))
        self.raw_path = Path(outcome.get("raw", self.directory / "qwen" / "raw.png"))
        self.appendage_review_required = appendage_review_required
        self.expected_sha = outcome.get("result_sha256") if self.composited else None
        if self.expected_sha and sha(self.result_path) != self.expected_sha:
            raise ValueError("의상 편집 결과가 기록과 다릅니다.")
        self.approved_sha = None
        self.checks = {}
        self.status = "awaiting_user_review"
        self.persist()

    @property
    def current_path(self):
        return self.result_path

    @property
    def head_notice(self):
        retried = len(self.outcome.get("attempts", ())) > 1
        if self.composited:
            text = "머리(얼굴·머리카락·장식)는 원본 그대로 붙였습니다." + (" 첫 결과가 검사에 걸려 한 번 더 그렸습니다." if retried else "")
        else:
            text = ("머리를 원본으로 붙이지 못했습니다" + (" (두 번 그렸지만 모두 검사에 걸림)" if retried else "") + ": "
                    + self.outcome.get("reason", "") + " Qwen이 그린 그대로입니다.")
        outside = [w for w in (self.outcome.get("support") or {}).get("warnings", []) if w.get("level") == "outside"]
        if outside:  # D-113: 시작 전 경고가 있던 그림은 결과에서도 그 사실을 다시 알린다
            text += " 시작 전 경고(지원 범위 밖 " + str(len(outside)) + "개)가 있던 그림입니다 — 옷·자세·꼬리를 꼭 확인해 주세요."
        text += garment_reference_notice(self.outcome.get("garment_reference"))
        return text

    def verify_current(self):
        data = self.current_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if self.expected_sha and digest != self.expected_sha:
            raise ValueError("의상 편집 결과가 변경됐습니다.")
        return digest

    def select(self, index):
        if index != 0:
            raise ValueError("의상 편집 결과는 1장입니다.")

    def reject_and_next(self):
        self.status = "rejected_by_user"
        self.persist()
        # 거절 이유는 묻지 않으므로 항목은 not_judged로 둔다(추측해서 채우지 않음, D-114)
        record_judgement(self.directory, dict(judge="user", source="gui_reject", note="사용자가 결과를 쓰지 않음"))
        return 0

    def toggle_head_view(self):
        return None

    def approve(self, checks):
        required = ({"character", "garment", "exposure", "original_outfit_gone"}
                    | ({"appendages"} if self.appendage_review_required else set()))
        if set(checks) != required or not all(checks.values()):
            raise ValueError("캐릭터·의상·노출·원래 옷 잔존 및 해당하는 꼬리·귀 확인을 완료해 주세요.")
        self.approved_sha = self.verify_current()
        self.checks = dict(checks)
        self.status = "user_approved"
        self.persist()
        # D-114: 확인 체크를 항목별 판정으로 기록한다. 체크로 묻지 않은 항목(자세)은 not_judged
        record_judgement(self.directory, dict(
            judge="user", source="gui_approval", same_character="yes", outfit="accepted", original_garment_remaining="none",
            severe_defect="none", body_parts="kept" if self.appendage_review_required else "not_applicable"))

    def export(self, destination):
        if self.status != "user_approved" or self.verify_current() != self.approved_sha:
            raise ValueError("현재 결과를 먼저 확인해 주세요.")
        destination = Path(destination)
        if destination.suffix.lower() != ".png":
            raise ValueError("PNG 파일로 저장해 주세요.")
        sidecar = destination.with_suffix(".review.json")
        if destination.exists() or sidecar.exists():
            raise FileExistsError("같은 이름의 파일이 있습니다. 다른 이름을 선택하세요.")
        data = self.current_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.approved_sha:
            raise ValueError("확인한 이미지가 변경됐습니다.")
        metadata = {"mode": "outfit_edit", "source_run": str(self.directory), "sha256": self.approved_sha,
                    "head_composited": self.composited, "reason": self.outcome.get("reason"),
                    "head_accessories": self.outcome.get("head_accessories"), "alignment": self.outcome.get("alignment"),
                    "hair_shape": self.outcome.get("hair_shape"), "limbs": self.outcome.get("limbs"),
                    "raw_sha256": self.outcome.get("raw_sha256"), "attempts": self.outcome.get("attempts"),
                    "trace": self.outcome.get("trace"), "conditions": self.outcome.get("conditions"),
                    "support": self.outcome.get("support"), "judgement": str(self.directory / "judgement.json"),
                    "garment_reference": self.outcome.get("garment_reference"),
                    "approval_record": str(self.directory / "user-review.json"),
                    "reviewer": "user", "checks": self.checks}
        created = []
        try:
            with sidecar.open("x", encoding="utf-8") as f:
                created.append(sidecar)
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            with destination.open("xb") as f:
                created.append(destination)
                f.write(data)
        except Exception:
            for path in created:
                path.unlink()
            raise
        self.status = "saved"
        self.persist(destination=str(destination))
        return destination

    def persist(self, **extra):
        write_json(self.directory / "user-review.json", dict(status=self.status, checks=self.checks,
                   approved_sha256=self.approved_sha, composited=self.composited, result=str(self.result_path),
                   updated=datetime.now().isoformat(timespec="seconds"), **extra))


def garment_reference_notice(report):
    """옷 참조에 공통 전처리(D-117)가 적용됐으면 무엇을 뺐는지 알린다. 옷 자체는 바뀌지 않는다."""
    if not report or not report.get("changed"):
        return ""
    done = [name for key, name in (("background", "배경을 흰색으로"), ("head", "머리 부분을"))
            if str((report.get(key) or {}).get("status", "")).startswith("applied")]
    return " 옷 참조의 " + "·".join(done) + " 자동으로 뺐습니다(옷 자체는 그대로)."
