"""기존 6건을 일괄 머리 붙이기로 재현한다. 기본은 CPU 입력 검사이고 --gpu 승인 시만 생성한다."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genai_lab.studio_head_paste import read, run_worker, verify_manifest
from genai_lab.studio_generation import StudioRuntime
from genai_lab.head_paste_worker import prepare_paste
from genai_lab.head_paste_semantics import load_parts
from genai_lab.proportion_inputs import require, sha
from genai_lab.qwen_record_io import write_json


def reference_cases(source):
    """잠긴 기존 결과에서만 사례를 읽는다. 이미지 생성이나 입력 변경은 하지 않는다."""
    groups, manifest = {}, {}
    for filename in sorted(source.glob('*/seed-*/result.json')):
        result = read(filename)
        request = read(filename.parent / 'request.json')
        require(result['status'] == 'awaiting_user_review', '완료된 기준 결과가 아닙니다.')
        for data in (result, request):
            for path, digest in data['manifest'].items():
                require(path not in manifest or manifest[path] == digest, '기준 자료의 SHA 충돌')
                manifest[path] = digest
        group = filename.parent.parent.name
        groups.setdefault(group, []).append((filename.parent, request, result))
    require(len(groups) == 3 and all(len(cases) == 2 for cases in groups.values()), '캐릭터 3명 × seed 2개의 기준값이 필요합니다.')
    verify_manifest(manifest)
    return groups


def replay_cpu(groups, output):
    """GPU에서 저장한 분할을 고정 입력으로 사용해 붙이기 계산의 픽셀 재현만 검사한다."""
    rows = []
    for group, cases in groups.items():
        for source, request, result in cases:
            target = output / group / str(request['seed'])
            target.mkdir(parents=True)
            parts_file = source / 'generated-parts.npz'
            require(sha(parts_file) == result['manifest'][str(parts_file)], '생성 분할이 변경됐습니다.')
            class FixedSegmenter:
                def __init__(self, *args): pass
                def parse(self, image): return load_parts(parts_file)
                def metrics(self): return {'gpu_executed': False, 'fixed_parts': str(parts_file)}
                def close(self): pass
            replay = prepare_paste(request, target, FixedSegmenter)
            actual = replay['per_seed'][str(request['seed'])]
            expected = read(source / 'paste-inputs.json')['per_seed'][str(request['seed'])]
            checks = {key: actual[key + '_sha256'] == expected[key + '_sha256']
                      for key in ('init','mask','features','paste','sketch')}
            checks['H'] = replay['H_sha256'] == result['H_sha256']
            row = dict(case=group, seed=request['seed'], checks=checks)
            rows.append(row)
            write_json(output / 'results.json', dict(gpu_executed=False, rows=rows))
            require(all(checks.values()), f'CPU 재현 불일치: {row}')
            print(group, request['seed'], 'CPU SHA 일치', flush=True)
    return rows


def replay_gpu(groups, output):
    """캐릭터별 2장을 한 프로세스에서 처리해 모델 한 번 로드 경로를 대조한다."""
    rows = []
    for group, cases in groups.items():
        jobs = [request for _, request, _ in cases]
        manifest = {}
        for job in jobs: manifest.update(job['manifest'])
        request = dict(jobs=jobs, manifest=manifest, gui_memory_before_worker={'standalone_verification': True})
        info = run_worker('batch', request, output / group, StudioRuntime(), lambda:False, print)
        require(len(info['items']) == len(cases), '일괄 결과 수가 다릅니다.')
        for item, (_, job, expected) in zip(info['items'], cases):
            require(item['status'] == 'completed' and item['seed'] == job['seed'], str(item))
            actual = read(Path(item['directory']) / 'result.json')
            a = read(actual['redraw_record']); e = read(expected['redraw_record'])
            checks = dict(head_1024=a['head_sha256'] == e['head_sha256'],
                          raw_redraw=a['result_sha256'] == e['result_sha256'],
                          product=actual['product_sha256'] == expected['product_sha256'])
            rows.append(dict(case=group, seed=item['seed'], checks=checks))
            write_json(output / 'results.json', dict(gpu_executed=True, rows=rows))
            require(all(checks.values()), f'GPU 재현 불일치: {rows[-1]}')
            print(group, item['seed'], 'GPU SHA 일치', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='명시적 GPU 재현 승인. 지정하지 않으면 분할 고정 CPU 검사만 실행')
    parser.add_argument('--source', type=Path, default=Path('outputs/head-paste-product-verify-20261010/gpu-03'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups = reference_cases(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.gpu: replay_gpu(groups, args.output)
    else: replay_cpu(groups, args.output)


if __name__ == '__main__':
    main()
