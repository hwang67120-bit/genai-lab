"""선택적 1회 생성의 프롬프트 기본값이다. 활성 설정은 변경하지 않는다."""
from dataclasses import dataclass, field
from pathlib import Path
from genai_lab.onepass_input_settings import CharacterInputSettings
from genai_lab.onepass_gender import validate_onepass_negative_template

# bg-prevent-20260930/plan.json의 첫 부정 문구에서 맨 앞 1boy만 제거한 값이다.
NEGATIVE_TEMPLATE = 'nsfw, panties, underwear, buruma, abstract background, speed lines, light rays, lowres, bad, text, worst quality, low quality, watermark, signature, different character, different hairstyle, different hair color, different eye color, missing character features, bad anatomy, bad hands, malformed hands, extra fingers, uneven eyes'

@dataclass(frozen=True)
class OnePassPromptSettings:
    negative_template: str = NEGATIVE_TEMPLATE
    tokenizer_root: Path = Path('D:/genai-cache/huggingface/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b')
    tail: tuple[str, ...] = ('solo', 'full body', 'white background', 'simple background', 'coherent anatomy', 'best quality')
    # ps 기본값은 CharacterInputSettings가 관리하며 중복 정의하지 않는다. 정책 결정 대기 상태다.
    character: CharacterInputSettings = field(default_factory=CharacterInputSettings)

    # 귀 태그 교체는 시험 근거가 없으므로 명시적으로 선택한 경우에만 적용한다.
    enable_ear_override: bool = False

    def __post_init__(self):
        if type(self.enable_ear_override) is not bool:
            raise ValueError('귀 외형 적용 설정은 bool이어야 합니다.')
        validate_onepass_negative_template(self.negative_template)
