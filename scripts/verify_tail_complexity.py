"""Replay 7 original and 22 generated tails on CPU without altering locked trials."""
import argparse
import json
import hashlib
from pathlib import Path
import time


def collect_cases(dataset):
    inputs = dataset.parent/'suin-tail-test-20261006'/'inputs'
    sources = {p.name[:3]: p for p in inputs.iterdir() if p.is_file()}
    cases = []
    for stage, boxes_file, measurements_file in (
        ('original', 'boxes.json', 'measure_v2/measurements.json'),
        ('generated', 'v3/boxes_v3.json', 'v3/measure/measurements.json')):
        boxes = json.loads((dataset/boxes_file).read_text(encoding='utf-8'))
        expected = json.loads((dataset/measurements_file).read_text(encoding='utf-8'))
        for key, box in boxes.items():
            if stage == 'original':
                source = sources[key]
            else:
                character, seed = key.split('_')
                source = dataset.parent/'suin-tail-test-20261006'/'cases'/character/f'seed-{209211000+int(seed[1:])}'/'raw.png'
            if '143960292_19' in str(source) or '0e0f64558c7c6f84b97b43b5b60e693e' in str(source):
                raise ValueError('금지 파일 입력')
            cases.append((stage, key, source, box, expected[key]))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from genai_lab.tail_complexity_worker import cpu_environment, CpuTailSegmenter, analyze_image, PeakRAM
    from genai_lab.qwen_record_io import write_json
    cpu_environment()
    start = time.monotonic()
    results = []
    with PeakRAM() as ram:
        segmenter = CpuTailSegmenter()
        for stage, key, source, box, expected in collect_cases(args.dataset):
            folder = args.output/stage/key
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            actual = analyze_image(segmenter, source, sha, box, folder)
            expected_deep = expected.get('deep_valleys', sum(p >= 20 for p in expected['valley_prominence']))
            comparisons = dict(color_count=len(actual['color_families']) == len(expected['color_families']),
                               deep_valleys=actual['deep_valleys'] == expected_deep,
                               turn=abs(actual['turn_deg']-expected['turn_deg']) <= 1,
                               mask=actual['mask_px'] == expected['mask_px'])
            row = dict(stage=stage, case=key, source=str(source), source_sha256=sha,
                       actual=actual, expected=expected, checks=comparisons, passed=all(comparisons.values()))
            write_json(folder/'comparison.json', row)
            results.append(row)
            print(stage, key, 'PASS' if row['passed'] else 'MISMATCH', actual['analysis_seconds'], flush=True)
    write_json(args.output/'results.json', dict(cases=results, passed=sum(r['passed'] for r in results),
               count=len(results), load_seconds=segmenter.load_seconds, total_seconds=time.monotonic()-start,
               peak_ram_bytes=ram.peak, model_revision=Path(segmenter.snapshot).name, device='cpu'))
    print('DONE', sum(r['passed'] for r in results), '/', len(results), 'RAM', ram.peak, flush=True)


if __name__ == '__main__':
    main()
