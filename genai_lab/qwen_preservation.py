"""사용자가 확인한 보존 사실만 사용한다. 분석기는 제안하며 승인하지 않는다."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def require_sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("전체 SHA-256이 필요합니다.")


@dataclass(frozen=True)
class PreservationItem:
    id: str
    kind: str
    fact_ko: str
    english: str
    source: str
    evidence_sha256: str
    check_at: str
    status: str = "draft"
    confirmation: str = ""
    issues: tuple[str, ...] = ()
    resolution: str = ""
    approved_gender: bool = False

    def __post_init__(self):
        require_sha(self.evidence_sha256)
        if type(self.approved_gender) is not bool:
            raise ValueError("성별 근거 확인은 명시적 boolean이어야 합니다.")
        if not self.id or self.status not in ("draft", "confirmed", "excluded"):
            raise ValueError("보존 항목 ID/상태 오류")
        if not self.source:
            raise ValueError("후보 출처가 필요합니다.")
        if self.status == "confirmed":
            if not all((self.fact_ko.strip(), self.english.strip(), self.confirmation.strip())):
                raise ValueError("한국어·영어 문장 및 사용자 확인 근거가 필요합니다.")
            if self.issues and not self.resolution.strip():
                raise ValueError("불일치/미확정 항목의 사용자 해결 기록이 필요합니다.")
            problems = instruction_issues(self)
            if problems:
                raise ValueError("보존 지시 수정 필요: " + ", ".join(problems))


def instruction_issues(item):
    """알려진 단어 규칙 위반만 검사한다. 의미상 정확성 분류기는 아니다."""
    text = item.english.lower().replace("_", " ")
    issues = []
    if re.search(r"\b(raccoon|fox|wolf|cat|dog|rabbit)\b", text):
        issues.append("종 이름 대신 보이는 색·개수·형태를 적으세요")
    if re.search(r"\b(hands? in pockets?|arms? up|raised arms?|standing|kneeling|sitting|walking)\b", text):
        issues.append("자세는 별도 선택 입력으로 분리하세요")
    if re.search(r"\b(petite|loli|childlike|flat chest|small breasts|large breasts|slender|skinny)\b", text):
        issues.append("새 체형 지시는 허용하지 않습니다")
    if re.search(r"\b(male|female|masculine|feminine|1boy|1girl)\b", text) and not item.approved_gender:
        issues.append("성별 표현의 승인 근거가 필요합니다")
    if item.kind == "style" and item.english != "Keep the illustration style of Picture 1.":
        issues.append("그림체는 기준 이미지 참조 문장만 사용합니다")
    return issues


@dataclass(frozen=True)
class PreservationSpec:
    image_path: str
    image_sha256: str
    image_approval: str
    items: tuple[PreservationItem, ...]
    revision: int = 1
    schema_version: int = 1

    def __post_init__(self):
        require_sha(self.image_sha256)
        if not self.image_approval.strip() or self.revision < 1 or self.schema_version != 1:
            raise ValueError("기준 이미지 승인과 유효한 명세 버전이 필요합니다.")
        if len({x.id for x in self.items}) != len(self.items):
            raise ValueError("항목 ID 중복")
        for item in self.items:
            if item.status == "confirmed" and item.evidence_sha256 != self.image_sha256 and not item.resolution:
                raise ValueError("다른 이미지 근거는 현재 이미지에서 재확인해야 합니다.")

    def record(self):
        return asdict(self)

    @property
    def sha256(self):
        return json_sha(self.record())

    @classmethod
    def from_record(cls, value):
        data = dict(value)
        data["items"] = tuple(PreservationItem(**dict(x, issues=tuple(x.get("issues", ()))))
                              for x in data["items"])
        return cls(**data)

    def verify_image(self):
        if file_sha(self.image_path) != self.image_sha256:
            raise ValueError("승인 기준 이미지가 변경됐습니다. 명세 재확인이 필요합니다.")

    def revise_item(self, item_id, **changes):
        if not any(x.id == item_id for x in self.items):
            raise KeyError(item_id)
        if set(changes) - {"kind", "fact_ko", "english", "check_at", "source", "approved_gender"}:
            raise ValueError("수정 가능한 항목 필드가 아닙니다.")
        items = tuple(replace(x, **changes, status="draft", confirmation="", resolution="")
                      if x.id == item_id else x for x in self.items)
        return replace(self, items=items, revision=self.revision + 1)

    def rebind(self, path, approval):
        return PreservationSpec(str(path), file_sha(path), approval,
            tuple(replace(x, status="draft", confirmation="", resolution="",
                          issues=tuple(dict.fromkeys((*x.issues, "basis_image_changed")))) for x in self.items),
            self.revision + 1)


def confirm_item(spec, item_id, *, confirmation, resolution="", exclude=False):
    if not any(x.id == item_id for x in spec.items):
        raise KeyError(item_id)
    spec.verify_image()
    items = []
    for item in spec.items:
        if item.id == item_id:
            issues = item.issues
            if item.evidence_sha256 != spec.image_sha256:
                issues = tuple(dict.fromkeys((*issues, "different_evidence_image")))
            item = replace(item, status="excluded" if exclude else "confirmed",
                           confirmation=confirmation, resolution=resolution, issues=issues)
        items.append(item)
    return replace(spec, items=tuple(items), revision=spec.revision + 1)


def draft_from_reports(image_path, approval, reports=(), generation_tags=()):
    """이미지에 연결된 저장 보고서를 읽는다. 근거가 없는 관찰은 미확정으로 둔다. reports 항목은 source, image_sha256, kind,
    candidates[{fact_ko,english,check_at,issues}]다. 보고서는 근거일 뿐 사용자 승인이나 부위 없음의 확증이 아니다.
    """
    sha = file_sha(image_path)
    items = []
    for report in reports:
        source_sha = report["image_sha256"]
        for candidate in report.get("candidates", ()):
            issues = tuple(candidate.get("issues", ()))
            if source_sha != sha:
                issues += ("different_evidence_image",)
            if generation_tags:
                issues += ("compare_with_generation_tags",)
            items.append(PreservationItem(str(len(items) + 1), report["kind"],
                candidate.get("fact_ko", ""), candidate.get("english", ""), report["source"],
                source_sha, candidate.get("check_at", "미지정"), issues=issues))
    for tag in generation_tags:
        items.append(PreservationItem(str(len(items) + 1), "generation_hint", str(tag), "",
            "generation_tags", sha, "완성 이미지와 대조", issues=("requested_not_observed",)))
    items.append(PreservationItem("style", "style", "기준 이미지의 그림체 유지",
        "Keep the illustration style of Picture 1.", "reference_image", sha, "전체"))
    return PreservationSpec(str(image_path), sha, approval, tuple(items))
