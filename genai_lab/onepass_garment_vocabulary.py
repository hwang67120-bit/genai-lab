"""1회 생성의 의상 명사와 덮는 범위·주머니 역할을 한 표에서 관리한다. 근거: integration-20260929/make_plan.py,
e2e-20261001/plan.py, garment-fidelity-20261001/make_plan_rules.py. 셔츠와 민소매 상의는 자동 배 노출도
막되 승인 태그는 제거하지 않는다. 태그 기반 규칙이며 실제 의상 픽셀의 판정은 아니다.
"""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class GarmentNoun:
    name: str
    legs: bool = False
    sleeves: bool = False
    torso: bool = False
    pockets: bool = False
    sleeveless_top: bool = False

GARMENT_VOCABULARY = (
    GarmentNoun('pants', legs=True, pockets=True),
    GarmentNoun('jeans', legs=True, pockets=True),
    GarmentNoun('leggings', legs=True), GarmentNoun('pantyhose', legs=True),
    GarmentNoun('thighhighs', legs=True), GarmentNoun('long skirt', legs=True),
    GarmentNoun('bodysuit', legs=True, torso=True),
    GarmentNoun('long sleeves', sleeves=True), GarmentNoun('short sleeves', sleeves=True),
    GarmentNoun('jacket', sleeves=True, torso=True, pockets=True),
    GarmentNoun('hoodie', sleeves=True, torso=True, pockets=True),
    GarmentNoun('coat', sleeves=True, torso=True, pockets=True),
    GarmentNoun('blazer', sleeves=True, torso=True, pockets=True),
    GarmentNoun('shirt', sleeves=True, torso=True), GarmentNoun('t-shirt', sleeves=True, torso=True),
    GarmentNoun('sweater', sleeves=True, torso=True), GarmentNoun('sweatshirt', sleeves=True, torso=True),
    GarmentNoun('cardigan', sleeves=True, torso=True),
    GarmentNoun('camisole', torso=True, sleeveless_top=True),
    GarmentNoun('tank top', torso=True, sleeveless_top=True), GarmentNoun('dress', torso=True),
    GarmentNoun('shorts', pockets=True), GarmentNoun('skirt'), GarmentNoun('crop top'),
    GarmentNoun('gloves'), GarmentNoun('elbow gloves'), GarmentNoun('necktie'),
    GarmentNoun('suspenders'), GarmentNoun('vest'), GarmentNoun('shrug'),
)
_LOOKUP = tuple(sorted(GARMENT_VOCABULARY, key=lambda n: -len(n.name)))


def garment_nouns(tags):
    """가장 긴 접미사가 우선한다. 뒤 괄호 설명은 매칭에만 쓴다. white shrug (clothing)은 shrug로 매칭하되 승인 문장은 유지한다. 알 수
    없는 명사는 추측하지 않으며 마네킹 오분류도 승인 자료에 남긴다.
    """
    found = {}
    for tag in tags:
        text = re.sub(r'(?:\s*\([^()]*\))+$', '', tag.replace('_', ' ').strip().lower()).strip()
        for noun in _LOOKUP:
            if text == noun.name or text.endswith(' '+noun.name):
                found[noun.name] = noun
                break
    return tuple(found.values())


def has_pockets(tags):
    return any(n.pockets for n in garment_nouns(tags))


def filter_pocket_tags(pose_tags, garment_tags):
    if has_pockets(garment_tags):
        return tuple(pose_tags), ()
    removed = tuple(t for t in pose_tags if t.replace('_', ' ').strip().lower() in ('hands in pockets', 'hand in pocket'))
    return tuple(t for t in pose_tags if t not in removed), removed


def uncovered_parts(tags):
    nouns = garment_nouns(tags)
    names = {n.name for n in nouns}
    added = []
    if not any(n.legs for n in nouns):
        added.append('bare legs')
    if any(n.sleeveless_top for n in nouns) and not any(n.sleeves for n in nouns):
        added.extend(('bare shoulders', 'bare arms'))
    if 'crop top' in names and not any(n.torso for n in nouns):
        added.append('midriff')
    approved = {t.replace('_', ' ').strip().lower() for t in tags}
    return tuple(t for t in added if t not in approved)
