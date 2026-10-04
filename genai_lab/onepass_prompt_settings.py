"""Opt-in one-pass prompt defaults; active configs are unchanged."""
from dataclasses import dataclass, field
from pathlib import Path
from genai_lab.onepass_input_settings import CharacterInputSettings
from genai_lab.onepass_gender import validate_onepass_negative_template

# bg-prevent-20260930/plan.json[0].negative, removing only leading '1boy, '.
NEGATIVE_TEMPLATE = 'nsfw, panties, underwear, buruma, abstract background, speed lines, light rays, lowres, bad, text, worst quality, low quality, watermark, signature, different character, different hairstyle, different hair color, different eye color, missing character features, bad anatomy, bad hands, malformed hands, extra fingers, uneven eyes'

@dataclass(frozen=True)
class OnePassPromptSettings:
    negative_template: str = NEGATIVE_TEMPLATE
    tokenizer_root: Path = Path('D:/genai-cache/huggingface/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b')
    tail: tuple[str, ...] = ('solo', 'full body', 'white background', 'simple background', 'coherent anatomy', 'best quality')
    # Decision pending: ps default is owned by CharacterInputSettings, not duplicated.
    character: CharacterInputSettings = field(default_factory=CharacterInputSettings)

    # Ear replacement has no experiment evidence; explicit opt-in only.
    enable_ear_override: bool = False

    def __post_init__(self):
        if type(self.enable_ear_override) is not bool:
            raise ValueError('귀 외형 적용 설정은 bool이어야 합니다.')
        validate_onepass_negative_template(self.negative_template)
