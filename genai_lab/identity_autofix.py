"""자동 동일성 고치기(S32, 사용자 11/1 "수동 조정이 아닌 AI가 스스로 고칠 수 있는 계산법").

사람이 고른 특징 목록(고정 특징표) 대신, 원본 그림의 WD 태그 전체에서 결과 후보들이 잃은 태그를 찾고, 태그 종류(tag_kinds: AI가 나눈
몸·입은 것·장면)로 장면 태그를 뺀 뒤, 남은 것을 '빠진 특징' 문장으로 생성 문장에 붙여 다시 그린다. 털 수인(F1)이 사람으로 바뀐 것을
한 바퀴에 furry 0.03→0.84~0.88로 되돌렸다. 옷 그림을 넣었으면 원래 옷 태그는 잃는 것이 맞으므로 몸 특징만 본다.
"""
PRESENT = 0.5           # 원본에서 이만큼 확실한 태그만
GONE = 0.35             # 결과에서 이 미만이면 잃음(고정 특징표 KEPT_MIN과 같음)
MAJORITY = 0.5          # 후보 중 이 비율 이상에서 잃었으면 고칠 대상(한 장만 흔들린 것은 seed 탓)
# 고치기는 한 바퀴(S32 순환 시험): 두 번째 바퀴는 F1에서 더 나아지지 않고 두 seed가 털을 다시 잃었다. 원래 후보와 고친 후보 중
# seed마다 덜 잃은 쪽을 고른다.
MAX_ROUNDS = 1
# 태그 종류 표(configs/tag_kinds_wd_v3.json, Qwen3-14B가 WD v3 일반 태그 8,106개를 나눔)의 확신 하한. '입은 것'은 애매한 태그가 모이는 칸이라
# (qr_code 0.66·handcuffs 0.70·signature 0.70) 더 높게. 표에 없는 태그는 몸으로 보고 고친다(놓치는 쪽보다 낫다).
BODY_MIN = 0.6
OUTFIT_MIN = 0.9
KINDS_TABLE = "configs/tag_kinds_wd_v3.json"


def lost_tags(original, results):
    """원본 태그 점수, 후보들의 태그 점수 목록 → 후보 과반에서 잃은 원본 태그(많이 잃은 순)."""
    counts = {}
    for tag, value in original.items():
        if value >= PRESENT:
            n = sum(1 for r in results if r.get(tag, 0.0) < GONE)
            if results and n / len(results) >= MAJORITY:
                counts[tag] = n
    return sorted(counts, key=lambda t: (-counts[t], -original[t], t))


def to_fix(lost, kinds, with_garment=False):
    """잃은 태그 중 고칠 것: 몸 특징, 옷을 바꾸지 않았으면 입은 것까지. 종류를 모르는 태그는 고친다(놓치는 쪽보다 낫다)."""
    allowed = ("body",) if with_garment else ("body", "outfit")

    def sure(tag):
        row = kinds.get(tag)
        if not row:
            return True
        kind = row.get("kind")
        if kind not in allowed:
            return False
        floor = BODY_MIN if kind == "body" else OUTFIT_MIN
        return row.get(kind, 1.0) >= floor
    return [t for t in lost if sure(t)]


def fix_sentence(tags):
    if not tags:
        return ""
    return ("These features of the character in image 1 are missing and must be shown clearly: "
            + ", ".join(t.replace("_", " ") for t in tags) + ". ")


def fixed_prompt(prompt, tags):
    """'빠진 특징' 문장을 동일성 지시('Keep the same face') 앞에 넣는다. 그 자리가 없으면 맨 끝."""
    sentence = fix_sentence(tags)
    if not sentence:
        return prompt
    anchor = "Keep the same face,"
    return prompt.replace(anchor, sentence + anchor, 1) if anchor in prompt else prompt + " " + sentence.strip()


def remaining(original, result, tags):
    """고친 뒤 후보 하나에서 여전히 잃은 태그."""
    return [t for t in tags if original.get(t, 0.0) >= PRESENT and result.get(t, 0.0) < GONE]


def next_round(asked, lost_now, kinds, with_garment=False):
    """다음 바퀴에 요청할 태그: 지금까지 요청한 것 전부 + 새로 잃은 것. 안 돌아온 태그도 빼지 않는다 — 함께 요청해야 효과가 있다
    (S32: 2바퀴째에 동물 코·몸 털을 빼자 이미 돌아온 털 수인까지 다시 사라짐)."""
    return list(dict.fromkeys(list(asked) + to_fix(lost_now, kinds, with_garment)))


def best_per_seed(original, rounds, watch):
    """바퀴마다 같은 seed의 후보 중 지킬 태그(watch = 첫 진단에서 고치기로 한 태그 — 뒤 바퀴에 더한 태그로 앞 바퀴를 깎지 않게)를
    가장 적게 잃은 것(같으면 앞 바퀴 — 원래 문장에 가까움). rounds: [후보 태그 점수 목록, ...] → seed 자리마다 바퀴 번호."""
    picks = []
    for i in range(len(rounds[0])):
        counts = [len(remaining(original, r[i], watch)) for r in rounds]
        picks.append(counts.index(min(counts)))
    return picks


def load_kinds(path):
    """태그 종류 표(tag_kinds 판정 결과 json의 table) → {태그: dict(kind, ...)}. 없으면 빈 표(모르는 태그는 고친다)."""
    import json
    from pathlib import Path
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")).get("table", {}) if path.is_file() else {}
