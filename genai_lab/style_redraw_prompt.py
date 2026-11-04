"""고정 그림체와 입력별 조건을 짧은 항목으로 조립한다. 임의 문장 절단·모델 설정 변경은 하지 않는다."""
from genai_lab.style_redraw_flux import STYLE, STYLE_LORA_TRIGGER, FULL_BODY_FRAMING, UPPER_FRAMING, COWBOY_FRAMING, BACKGROUND
from genai_lab.style_redraw_quality import outside_frame

BACKGROUND_DETAIL = ("The background is a richly detailed place with clear depth: a foreground, a middle ground and a distant "
                     "background, natural perspective and soft atmospheric haze, painted with the same polish as the character.")


def compact_scene(scene):
    """앱이 붙인 정형 장황 문구만 줄인다. 사용자의 자유 문장은 임의로 요약하지 않는다."""
    text = scene.strip()
    text = text.replace(FULL_BODY_FRAMING, "").replace(UPPER_FRAMING, "").replace(COWBOY_FRAMING, "")
    text = text.replace(BACKGROUND_DETAIL, "Detailed layered background, atmospheric perspective.")
    text = text.removeprefix("The character is ")
    text = text.replace(", with a ", ", ").replace(" in the morning.", " morning.")
    text = text.replace("Color design: a ", "Background/light palette: ")
    text = text.replace(" palette in the background and lighting only.", ".")
    text = text.replace(", natural body proportions.", ".")
    return " ".join(text.split())


def compact_parts(parts, kinds):
    """귀·꼬리·날개의 존재/부재와 알려진 종류를 모두 전달한다."""
    if parts is None:
        return "Keep any original animal ears/tail/wings."
    names = {"ears": "animal ears", "tail": "tail", "wings": "wings"}
    if set(parts) - set(names) or len(parts) != len(set(parts)):
        raise ValueError("확정 부위가 잘못되었습니다.")
    kinds = kinds or {}
    present = []
    for part in parts:
        name = str(kinds.get(part) or names[part]).replace("_", " ")
        if part == "ears" and kinds.get("ears_floppy"):
            name = "floppy " + name
        present.append(name)
    absent = [name for part, name in names.items() if part not in parts]
    return ("Keep original " + ", ".join(present) + " (shape/count/colour/position). " if present else "") + (
        "No " + "/".join(absent) + "." if absent else "")



def scene_arm_instruction(shot):
    """새 동작은 기존 팔을 이동시킨다. 상반신은 하체 확장 대신 팔의 배치로 화면 여유를 확보한다."""
    placement = ("Keep the head-to-waist crop, no legs; bend the elbows and place the hands within the torso crop. "
                 if shot.startswith("Upper-body") else
                 "Keep the head-to-mid-thigh crop, no lower legs or feet; bend the elbows to fit the arms. "
                 if shot.startswith("Cowboy") else
                 "Keep the full body and hands inside the canvas with space around them. ")
    return ("Move the original arms into the requested action as one coherent pose; replace their old positions, "
            "never retain an old arm beside a repositioned copy. " + placement +
            "Leave space above raised hands and beside elbows. Preserve natural occlusion; never add a limb to explain a hidden hand.")


def corrected_rejection_prompt(prepared, instruction, *, anatomy=False):
    """보존된 준비 입력에 수정 지시를 적용한다. 실패 그림은 모델 참조로 사용하지 않는다."""
    result = dict(prepared)
    prompt = prepared["prompt"]
    sections = dict(prepared.get("prompt_sections") or {})
    # 이전 제품이 붙인 정형 문구만 바꾼다. 장면·사용자 자유 문장은 그대로 유지한다.
    if anatomy and prepared.get("framing"):
        old = "Keep arms connected at the shoulders. Allow hidden or off-frame hands. No duplicate limbs, text, logos, watermarks or extra people."
        integrity = scene_arm_instruction(prepared["framing"]) + " No duplicate limbs, text, logos, watermarks or extra people."
        if sections and prompt == " ".join(sections.values()):
            sections["integrity"] = integrity
            prompt = " ".join(sections.values())
        elif old in prompt:
            prompt = prompt.replace(old, integrity)
            if sections.get("integrity") == old:
                sections["integrity"] = integrity
        else:
            prompt += " " + integrity
        result.update(prompt_sections=sections, prompt_format_version=2)
    result["prompt"] = prompt + " " + instruction
    result.pop("prompt_delivery", None)
    return result


def compact_flux_prompt(traits, with_garment=False, body_parts=None, scene=None, body_part_kinds=None,
                        shot=FULL_BODY_FRAMING, gender=None, garment_words=None):
    """보존 조건을 명시한 문장과 항목별 원문을 반환한다. 길이는 실제 토크나이저로 별도 검사한다."""
    described = "; ".join(row["text"].strip() for row in traits if row.get("text", "").strip()
                          and not (with_garment and row.get("name") == "clothes"))
    frame = ("Match image 1's upper-body crop; no legs." if shot.startswith("Upper-body") else
             "Head to mid-thigh; no lower legs/feet." if shot.startswith("Cowboy") else "Head to feet.")
    clothes = [word for word in garment_words or [] if not outside_frame(word.replace(" ", "_"), shot)]
    garment = ("Image 2 clothes only" + (": " + ", ".join(clothes) if clothes else "")
               + "; shape/colour/details, no mannequin." if with_garment else "Keep image 1 outfit.")
    gender_text = "Masculine adult man, no makeup. " if gender == "male" else "Adult woman. " if gender == "female" else "Adult. "
    sections = dict(
        style=f"{STYLE_LORA_TRIGGER} {STYLE}.",
        identity=f"{gender_text}Image 1: {described}. Same face/proportions/colours.",
        body=compact_parts(body_parts, body_part_kinds),
        garment=garment,
        framing=frame,
        scene=compact_scene(scene) if scene else "Same pose as image 1. " + BACKGROUND,
        integrity=(scene_arm_instruction(shot) if scene else "Keep the original pose and crop, including naturally hidden or off-frame limbs.")
                  + " No duplicate limbs, text, logos, watermarks or extra people.")
    return dict(prompt=" ".join(sections.values()), sections=sections, format_version=2)
