"""S40 그림체 자료 2판(사용자 11/1 "남은 크레딧을 깔끔하게 다 쓰고 이미지 생성 마무리"): S25와 같은 Q4 그림체·규칙(D-203, 성인 원본 캐릭터,
작가·작품 이름 없음)으로 80장. 1판(사용자 선택 21장)에 모자랐던 쪽을 채운다: 얼굴·상반신 근접 25(전신에서 얼굴 세부가 뭉개짐),
소품·빛이 많은 원근 배경의 전신 30(배경 품질, 장면 4판과 같은 방향), 허벅지 위 15, 성인 수인·동물귀 10(처음 보는 수인 캐릭터에도 그림체가 붙게).
사용: generate(클라우드) / check(이 PC) — S25와 같음.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qwen_image_dataset_s25 as s25  # noqa: E402

COUNT = 80
RICH_PLACE = ("in a sunlit greenhouse with hanging plants, glass panes, watering cans and dappled light",
              "on a stone bridge over a calm river with willow trees, distant hills and soft morning mist",
              "in a cosy bookstore with tall wooden shelves, a rolling ladder, warm lamps and dust motes in the light",
              "on a seaside promenade with lampposts, small boats in the harbour and gulls in a pale blue sky",
              "in a flower garden with stone paths, a wooden arbour, butterflies and petals drifting in the breeze",
              "in a fantasy town square with pastel houses, a fountain, bunting and warm afternoon light",
              "in a quiet cafe with wooden tables, potted plants, a chalkboard without writing and sunlight through the window",
              "in a forest clearing with tall trees, ferns, light rays through the leaves and floating pollen",
              "on a rooftop garden at golden hour with string lights, planters and a city skyline far away",
              "in a museum hall with marble columns, framed paintings, a polished floor and soft skylight",
              "in a snowy mountain village with wooden houses, lanterns, falling snow and distant peaks",
              "at a lakeside dock with a small rowing boat, reeds, mountains reflected in the water and evening glow")
ANTHRO = ("an adult anthropomorphic fox woman with orange fur-covered body, a fox face with a black nose and a large fluffy tail",
          "an adult anthropomorphic wolf man with grey fur-covered body, a wolf muzzle and a bushy tail",
          "an adult anthropomorphic rabbit woman with white fur-covered body, a rabbit face and long upright ears",
          "an adult anthropomorphic cat man with black fur-covered body, a cat face with whiskers and a long tail",
          "an adult woman with cat ears and a cat tail", "an adult man with dog ears and a dog tail",
          "an adult anthropomorphic tiger woman with striped fur-covered body and a tiger face",
          "an adult anthropomorphic bear man with brown fur-covered body and a round bear face",
          "an adult woman with fox ears and a fluffy fox tail", "an adult anthropomorphic mouse woman with grey fur, round ears and a thin tail")
FRAMES = (["Close-up portrait from the chest up: the face is large in the frame with detailed eyes, eyelashes and hair strands, "
           "looking at the viewer."] * 25
          + ["Full body shot from head to shoes: the whole figure is visible with space above the head and below the feet, both feet "
             "standing on the ground, and the background is a richly detailed place with clear depth."] * 30
          + ["Cowboy shot from the head to the mid thigh."] * 15
          + ["ANTHRO"] * 10)


def describe(i, frame):
    rng = random.Random(12000 + i)
    accessory = rng.choice(s25.ACCESSORY)
    place = rng.choice(RICH_PLACE)
    action = rng.choice(s25.ACTION)
    if frame == "ANTHRO":
        who = rng.choice(ANTHRO)
        return (f"Original adult character: {who}, wearing {rng.choice(s25.OUTFIT)}, {action} {place}. "
                f"Full body shot from head to feet with the whole figure visible.")
    return (f"Original adult character: {rng.choice(s25.WHO)} with {rng.choice(s25.SKIN)} skin, {rng.choice(s25.HAIR)} and "
            f"{rng.choice(s25.EYES)} eyes, wearing {rng.choice(s25.OUTFIT)}" + (f" and {accessory}" if accessory else "") +
            f", {action} {place}. {frame}")


def plan():
    frames = list(FRAMES)
    random.Random(40).shuffle(frames)
    return [dict(id=f"R{i:03d}", seed=12000 + i, prompt=f"{describe(i, frames[i - 1])} {s25.STYLE}{s25.TAIL}") for i in range(1, COUNT + 1)]


s25.COUNT = COUNT
s25.plan = plan

if __name__ == "__main__":
    if sys.argv[1] == "generate":
        s25.generate(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        s25.check(Path(sys.argv[2]))
