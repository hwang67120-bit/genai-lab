"""저장된 의상 입력과 실제 프롬프트를 CPU로 대조한다. 추론·생성·다운로드는 없다."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genai_lab.studio_garment_warnings import garment_warnings, contains_term
from genai_lab.onepass_garment_vocabulary import uncovered_parts
from genai_lab.onepass_prompt import plan_prompt_chunks

RUNS = (
    'c3b8d1dfff3c499191f63b73e3225e7d', '6086444053af43b3858e6f7619a9f68d',
    '4809668199014b30b13dacf7cfe7b2cd', '179d4705559f406b9a14e5fd6d9c92ae',
    'f991f402bddc4e0ea74f60c3dc9e32c1', '0daf9077afbd4ee1b59945331a493951',
    '265228c0e6be4cb1bd8f5c0a4bb5b341', '3fc2d0d3c1f2479396d3a5e998bb4ce2',
)
DETAIL_TERMS = {
    'zipper': ('zipper', 'zip'),
    'zipper_state': ('unzipped', 'zipped', 'open zipper', 'closed zipper', 'partially unzipped'),
    'zipper_parts': ('zipper slider', 'zipper teeth', 'zipper pull'),
    'outer_length': ('waist-length', 'hip-length', 'thigh-length', 'knee-length', 'ankle-length', 'short jacket', 'cropped jacket', 'short coat', 'long coat'),
    'waist_coverage': ('covered stomach', 'covered waist', 'no midriff', 'midriff', 'crop top', 'cropped shirt', 'cropped sweater'),
    'surface_details': ('wrinkles', 'folds', 'pleats', 'pleated', 'cable knit', 'ribbed', 'knitted', 'seams', 'stitching'),
}


def read_record(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_terms(text):
    """쉼표로 나눈 승인 항목을 비교한다. coat를 raincoat 안에서 찾지 않는다."""
    return tuple(' '.join(t.lower().replace('_', ' ').split()) for t in text.split(',') if t.strip())


def check_prompt_delivery(approved, prompt, tokenizers=None):
    """승인 항목 존재·중복·충돌·기록된 토큰 계획을 나누어 검사한다."""
    positive = normalized_terms(prompt['positive'])
    approved_terms = normalized_terms(','.join(approved))
    missing = [t for t in approved_terms if t not in positive]
    extras = [t for t in positive if t not in approved_terms]
    details = {key: [t for t in positive if any(contains_term(t, term) for term in terms)]
               for key, terms in DETAIL_TERMS.items()}
    rebuilt = None
    if tokenizers is not None:
        plans = plan_prompt_chunks(prompt['positive'], prompt['negative'], tokenizers)
        expected = json.loads(json.dumps([asdict(p) for p in plans]))
        rebuilt = expected == prompt.get('encoders')
    return dict(missing_approved_tags=missing, positive_terms=positive,
        non_garment_terms=extras, warnings=garment_warnings(approved, prompt['negative']),
        detail_terms=details, encoders_reproduced=rebuilt,
        chunks=[len(e.get('positive', [])) for e in prompt.get('encoders', [])],
        automatic_uncovered=uncovered_parts(approved),
        recorded_uncovered=prompt.get('rules', {}).get('uncovered_added'),
        interpretation='입력 일관성 검사이며 생성 결과의 성공 판정은 아님')


def generation_records(directory):
    """2단계 원본 기록을 선택하고 같은 결과를 저장한 제품 기록은 중복 집계하지 않는다."""
    records = []
    contour_seeds = set()
    for path in sorted(directory.glob('CONTOUR_*/run.json')):
        row = read_record(path)
        if isinstance(row.get('prompt'), dict):
            contour_seeds.add(row['seed']); records.append((path, row, 'CONTOUR'))
    for path in sorted(directory.glob('BASE_*/run.json')):
        row = read_record(path)
        if isinstance(row.get('prompt'), dict): records.append((path, row, 'BASE'))
    for path in sorted(directory.glob('seed-*/run.json')):
        row = read_record(path)
        if isinstance(row.get('prompt'), dict) and row['seed'] not in contour_seeds:
            records.append((path, row, 'single'))
    return records


def audit_run(folder, tokenizers):
    """저장된 분석 → 승인 → 실제 생성 기록을 따라가며 입력 파일은 읽기만 한다."""
    analysis_file = folder / 'inputs/analysis.json'
    approval_file = folder / 'inputs/approval.json'
    reference_file = folder / 'inputs/references.json'
    analysis, approval, references = map(read_record, (analysis_file, approval_file, reference_file))
    approved = approval['garment_tags']
    records = generation_records(folder / 'generation')
    files = [analysis_file, approval_file, reference_file] + [p for p, _, _ in records]
    locked = {str(p): digest(p) for p in files}
    result = dict(run=folder.name, garment_reference=references['garment'],
        analysis_tags=analysis['garment_tags'], approved_tags=approved,
        analysis_tags_removed=[t for t in analysis['garment_tags'] if t not in approved],
        analysis_tags_added=[t for t in approved if t not in analysis['garment_tags']],
        saved_warnings=approval.get('garment_review', {}).get('warnings'),
        records=[], input_sha256=locked)
    for path, row, stage in records:
        prompt = row['prompt']
        check = check_prompt_delivery(approved, prompt, tokenizers)
        result['records'].append(dict(file=str(path), seed=row['seed'], stage=stage,
            completed=row.get('completed'), prompt=prompt['positive'], negative=prompt['negative'],
            inputs=row.get('inputs'), checks=check,
            matches_approval_prompt=prompt['positive'] == approval.get('prompt')))
    result['files_unchanged'] = all(digest(p) == value for p, value in locked.items())
    if not result['files_unchanged']: raise ValueError('검사 중 입력 기록이 변경됐습니다: '+folder.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', default=RUNS)
    args = parser.parse_args()
    for name in args.runs:
        if not name or Path(name).name != name or '/' in name or '\\' in name:
            parser.error('실행 폴더 이름만 지정해야 합니다.')
    args.output.mkdir(parents=True, exist_ok=False)
    from genai_lab.onepass_prompt_tokenizers import load_onepass_tokenizers
    started = time.perf_counter()
    tokenizers = load_onepass_tokenizers()
    runs = [audit_run(ROOT/'outputs/studio-runs'/name, tokenizers) for name in args.runs]
    records = [row for run in runs for row in run['records']]
    summary = dict(runs=len(runs), records=len(records),
        prompt_delivery_pass=sum(not row['checks']['missing_approved_tags'] for row in records),
        token_plan_pass=sum(row['checks']['encoders_reproduced'] is True for row in records),
        runs_with_warnings=sum(any(row['checks']['warnings'] for row in run['records']) for run in runs),
        records_with_midriff=sum('midriff' in row['checks']['positive_terms'] for row in records),
        files_unchanged=all(run['files_unchanged'] for run in runs),
        gpu_executed=False, model_inference=False, download=False,
        seconds=time.perf_counter()-started)
    report = dict(summary=summary, runs=runs,
        limits=['문장에 조건이 있다는 사실은 모델이 이를 따랐다는 증거가 아님',
                '원본에 없는 디자인 판정에는 해당 참조 그림과 생성 결과의 별도 비교가 필요함',
                '길이·지퍼 구조의 단어 누락은 정보 부족의 증거이며 단독 원인 확정은 아님'])
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    for run in runs:
        if not run['records']: continue
        checks = run['records'][0]['checks']
        print(run['run'][:8], 'records', len(run['records']), 'warnings',
              [w['rule'] for w in checks['warnings']], 'details', checks['detail_terms'], flush=True)


if __name__ == '__main__':
    main()
