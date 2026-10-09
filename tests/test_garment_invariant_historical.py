"""실제 검사 구문에 저장 측정값을 넣어 재현한다. 파이프라인·모델 호출은 없다."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from genai_lab.selected_garment_correction import GarmentCorrectionContractError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'genai_lab/selected_garment_correction.py'


def execute_guard(error_name, **values):
    # >= 조건을 다시 구현하지 않고 변경되지 않은 제품 조건문을 실행한다.
    tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'correct_selected_garment')
    guards = [n for n in ast.walk(function) if isinstance(n, ast.If)
              and any(isinstance(v, ast.Constant) and isinstance(v.value, str)
                      and v.value.startswith(error_name + ':')
                      for statement in n.body if isinstance(statement, ast.Raise)
                      for v in ast.walk(statement))]
    assert len(guards) == 1
    block = ast.fix_missing_locations(ast.Module(body=[guards[0]], type_ignores=[]))
    exec(compile(block, str(SOURCE), 'exec'),
         {'GarmentCorrectionContractError': GarmentCorrectionContractError, **values})


# 측정 출처: outputs/sdxl-local-garment-strength-v2-20260923/cases/<case>/run.json
# 또는 outputs/color-channel-probe-v2-20260923/cases/<case>/run.json을 사용한다.
# 각 JSON의 측정 경로는 /local_refinement/garment_edit_plan/metrics다.
@pytest.mark.parametrize('folder,case,protection,requested,blocked', [
    ('sdxl-local-garment-strength-v2-20260923', 'muscular-male', 514975, 490735, True),
    ('sdxl-local-garment-strength-v2-20260923', 'ordinary-female', 23845, 160496, False),
    ('sdxl-local-garment-strength-v2-20260923', 'muscular-female', 29214, 206549, False),
    ('sdxl-local-garment-strength-v2-20260923', 'raccoon', 76477, 590284, False),
    ('color-channel-probe-v2-20260923', 'ordinary-female', 23845, 182817, False),
    ('color-channel-probe-v2-20260923', 'raccoon', 76477, 627615, False),
])
def test_saved_protection_guard(folder, case, protection, requested, blocked):
    path = ROOT / 'outputs' / folder / 'cases' / case / 'run.json'
    if not path.exists():
        pytest.skip('local historical artifact not available: ' + str(path))
    run = json.loads(path.read_text(encoding='utf-8-sig'))
    metrics = run['local_refinement']['garment_edit_plan']['metrics']
    assert metrics['hard_protection_pixels'] == protection
    assert metrics['requested_pixels'] == requested
    assert run['local_refinement']['target_garment_coverage']['status'] == 'REVIEW'
    if blocked:
        with pytest.raises(GarmentCorrectionContractError,
                           match='protection_exceeds_requested_region'):
            execute_guard('protection_exceeds_requested_region', metrics=metrics)
    else:
        execute_guard('protection_exceeds_requested_region', metrics=metrics)


def test_synthetic_equal_protection_is_blocked():
    # 실측 쌍이 아닌 합성 경계값이다. 관찰한 요청 면적 하나를 재사용한다.
    metrics = {'hard_protection_pixels': 160496, 'requested_pixels': 160496}
    with pytest.raises(GarmentCorrectionContractError,
                       match='protection_exceeds_requested_region'):
        execute_guard('protection_exceeds_requested_region', metrics=metrics)


def test_current_coverage_status_vocabulary():
    tree = ast.parse((ROOT / 'genai_lab/garment_edit_plan.py').read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'project_target_garment_coverage')
    statuses = {v.value for n in ast.walk(function) if isinstance(n, ast.Dict)
                for k, v in zip(n.keys, n.values)
                if isinstance(k, ast.Constant) and k.value == 'status'
                and isinstance(v, ast.Constant)}
    assert statuses == {'UNRESOLVED', 'REVIEW'}


@pytest.mark.parametrize('status,blocked', [
    ('UNRESOLVED', True), ('REVIEW', False), ('OK', False), ('PASS', False),
])
def test_synthetic_coverage_status_guard(status, blocked):
    # 여기 기록은 모두 합성 자료다. 현재 실제 상태는 UNRESOLVED/REVIEW만 나오며,
    # OK/PASS는 추가한 합성 정상 예시이지 관찰된 상태가 아니다.
    values = {'target_coverage': SimpleNamespace(record={'status': status}),
              'approved_tags': ('jacket',)}
    if blocked:
        with pytest.raises(GarmentCorrectionContractError, match='garment_category_unresolved'):
            execute_guard('garment_category_unresolved', **values)
    else:
        execute_guard('garment_category_unresolved', **values)
