"""모드 A → 모드 B 연결(D-211, 사용자 10/30): 옷을 바꾼 캐릭터를 목표 그림체로 다시 그린다.

① run_outfit_pipeline(Qwen 옷 바꾸기) → 결과 1장을 자동으로 넘김(사용자 화면 자동 흐름과 같은 automatic_edit_base 기록,
사람 승인이 아님을 그대로 적는다) → ② run_style_redraw(FLUX.2-klein 편집, 후보 3장). 두 단계는 각자 작업 프로세스라
GPU를 동시에 쓰지 않는다. 결과는 검토 대기로 돌려준다(확인 창 없음, 공통 진입점).
"""
from pathlib import Path
import json
import time

GENDERS = {"male": "male", "female": "female"}


def run_outfit_then_style(character_path, garment_path, directory, *, outfit_kwargs, runtime=None, paths=None,
                          outfit_runner=None, style_runner=None):
    """반환: dict(state, outfit=①요약, edit_base=①→② 넘긴 그림, style=② outcome). ①이 결과를 못 만들면 ②는 돌리지 않는다."""
    from genai_lab.outfit_edit_results import OutfitEditResults
    from genai_lab.studio_product import automatic_edit_base
    if outfit_runner is None:
        from genai_lab.outfit_edit_pipeline import run_outfit_pipeline as outfit_runner
    if style_runner is None:
        from genai_lab.style_redraw_pipeline import run_style_redraw as style_runner
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    chain = dict(mode="outfit_then_style", character=str(character_path), garment=str(garment_path), started=time.time())
    try:
        outcome = outfit_runner(character_path, garment_path, runtime=runtime, directory=directory / "outfit", **outfit_kwargs)
        results = OutfitEditResults(outcome)
        chain["outfit"] = dict(composited=results.composited, reason=outcome.get("reason"), result=str(results.current_path),
                               trace=outcome.get("trace"))
        if not Path(results.current_path).is_file():
            chain.update(state="failed", stage="outfit", message="옷 바꾸기 결과 그림이 없습니다.")
            return chain
        base = automatic_edit_base(results, directory)
        chain["edit_base"] = str(base)
        gender = GENDERS.get(outfit_kwargs.get("gender"))
        style = style_runner(base, directory / "style", runtime, confirmed_gender=gender, paths=paths)
        chain["style"] = style
        chain["state"] = style.get("state", "failed")
        return chain
    finally:
        chain["finished"] = time.time()
        (directory / "chain.json").write_text(json.dumps(chain, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
