"""S44 그림체 자료 3판(사용자 11/2 "얼굴은 2D 같아야 하고, 머리카락 묘사를 높이고, 옷 섬유 묘사를 더 — 아직 진짜 사람을 미형으로 만든
것 같다", 고른 2판 16장: R029·030·031·032·033·037·038·048·051·062·067·068·069·071·076·078). Q4 문장은 '매끈한 피부'를 요구해
얼굴이 실사 쪽 → Q5: 2D 애니메이션 얼굴(큰 눈·단순한 코와 입·셀 음영·피부 질감 없음), 가닥 많은 머리카락, 옷감 결·바느질·주름.
48장: 얼굴 근접 16 · 허벅지 위 16 · 원근 배경 전신 16. 규칙은 S25·S40과 같다(D-203 성인 원본 캐릭터, 작가·작품 이름 없음).
사용: generate(클라우드) / check(이 PC) — S25와 같음.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qwen_image_dataset_s25 as s25  # noqa: E402
import qwen_image_dataset_s40 as s40  # noqa: E402

COUNT = 48
STYLE = ("Style: high-quality 2D anime illustration, a clean 2D anime face with large expressive anime eyes, detailed irises with bright "
         "highlights, a small simple nose and a small mouth, soft cel shading on the skin with gentle blush and no realistic skin "
         "texture; extremely detailed hair with many fine individual strands, layered locks, flyaway wisps and glossy highlights; "
         "detailed clothing with visible fabric texture, weave, stitching, seams and natural folds; crisp clean line art, soft "
         "lighting with gentle bloom, polished professional finish.")
FRAMES = (["Close-up portrait from the chest up: the face is large in the frame with detailed eyes, eyelashes and hair strands, "
           "looking at the viewer."] * 16
          + ["Cowboy shot from the head to the mid thigh, the clothing fabric clearly visible."] * 16
          + ["Full body shot from head to shoes: the whole figure is visible with space above the head and below the feet, both feet "
             "standing on the ground, and the background is a richly detailed place with clear depth."] * 16)


def describe(i, frame):
    rng = random.Random(14000 + i)
    accessory = rng.choice(s25.ACCESSORY)
    return (f"Original adult character: {rng.choice(s25.WHO)} with {rng.choice(s25.SKIN)} skin, {rng.choice(s25.HAIR)} and "
            f"{rng.choice(s25.EYES)} eyes, wearing {rng.choice(s25.OUTFIT)}" + (f" and {accessory}" if accessory else "") +
            f", {rng.choice(s25.ACTION)} {rng.choice(s40.RICH_PLACE)}. {frame}")


def plan():
    frames = list(FRAMES)
    random.Random(44).shuffle(frames)
    return [dict(id=f"T{i:03d}", seed=14000 + i, prompt=f"{describe(i, frames[i - 1])} {STYLE}{s25.TAIL}") for i in range(1, COUNT + 1)]


s25.COUNT = COUNT
s25.STYLE = STYLE
s25.plan = plan

if __name__ == "__main__":
    if sys.argv[1] == "generate":
        s25.generate(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        s25.check(Path(sys.argv[2]))
