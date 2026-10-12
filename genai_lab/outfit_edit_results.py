"""원본 편집형 의상 교체 결과 1장의 검토·승인·저장(D-094 4단계). 작업실이 쓰는 StudioResults와 같은 이름의 속성·함수를 둔다.
자동 승인은 없다. 머리를 붙이지 못한 경우에도 Qwen 결과를 보여 주고 그 사실을 알린다(사용자가 판단).
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from genai_lab.outfit_edit import sha, write_json


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
        self.result_path = Path(outcome["result"]) if self.composited else self.directory / "compose" / "qwen_view.png"
        self.raw_path = self.directory / "qwen" / "raw.png"
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
        if self.composited:
            return "머리(얼굴·머리카락·선택한 장식)는 원본 그대로 붙였습니다."
        return "머리를 원본으로 붙이지 못했습니다: " + self.outcome.get("reason", "") + " Qwen이 그린 그대로입니다."

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
        return 0

    def toggle_head_view(self):
        return None

    def approve(self, checks):
        required = {"character", "garment", "exposure"} | ({"appendages"} if self.appendage_review_required else set())
        if set(checks) != required or not all(checks.values()):
            raise ValueError("캐릭터·의상·노출 및 해당하는 꼬리·귀 확인을 완료해 주세요.")
        self.approved_sha = self.verify_current()
        self.checks = dict(checks)
        self.status = "user_approved"
        self.persist()

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
                    "raw_sha256": self.outcome.get("raw_sha256"), "approval_record": str(self.directory / "user-review.json"),
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
