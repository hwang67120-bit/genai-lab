"""Advisory placeholder diagnostics and host-side elapsed timings, no quality gate."""
from contextlib import contextmanager
from time import perf_counter
from pathlib import Path
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def verify_and_save_ip_adapter_input(garment_image, output_dir, run_log=None,
                                     part_name="garment"):
    """Record one exact PIL adapter input, not encoder tensors or a verdict."""
    if not isinstance(part_name, str) or not part_name.replace("_", "").isalnum():
        raise ValueError("IP-Adapter 진단 부위 이름 오류")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f'debug_{part_name}_ip_adapter.png'
    record_path = directory / f'debug_{part_name}_ip_adapter.json'
    labels = {
        "garment": "의상", "identity": "얼굴", "hair": "헤어",
        "human_ears": "사람 귀", "animal_ears": "동물 귀",
        "ears": "동물 귀", "tail": "꼬리",
    }
    stage = f"IP-Adapter {labels.get(part_name, part_name)} 입력"

    def emit(message, level=logging.INFO):
        logger.log(level, '[%s] %s', stage, message)
        if run_log is not None:
            run_log.write_stage(stage, message)

    emit(f'임베딩 계산 전 입력 보존 위치={image_path}')
    # Save before checking, so even a rejected black input can be inspected.
    garment_image.save(image_path, format='PNG')
    with garment_image.convert('RGB') as rgb:
        pixels = np.asarray(rgb)
        channel_std = pixels.std(axis=(0, 1), dtype=np.float64)
        record = dict(
            stage='before_ip_adapter_image_embeds', part_name=part_name,
            mode=garment_image.mode,
            size=list(garment_image.size), image_path=str(image_path),
            min=int(pixels.min()), max=int(pixels.max()),
            mean=float(pixels.mean()), std=float(pixels.std()),
            channel_spatial_std=channel_std.tolist(),
            spatial_std=float(np.sqrt(np.mean(channel_std ** 2))),
            black_pixel_ratio=float(np.all(pixels == 0, axis=2).mean()),
            white_pixel_ratio=float(np.all(pixels == 255, axis=2).mean()),
            low_variation_threshold=5.0, warnings=[], errors=[],
            note='RGB 전역/공간 통계만 측정. 배경 대비도 포함하므로 의상 텍스처, 의미, 성별, 비율 일치의 증거가 아님.')
    if 'A' in garment_image.getbands() or 'transparency' in garment_image.info:
        with garment_image.convert('RGBA') as rgba:
            alpha = np.asarray(rgba)[:, :, 3]
            record['visible_alpha_pixels'] = int(np.count_nonzero(alpha))
            if not record['visible_alpha_pixels']:
                record['errors'].append('참조 이미지가 완전히 투명합니다. 마스크/크롭 연결을 확인하세요.')
    if record['max'] == 0:
        record['errors'].append('참조 이미지가 완전히 검은색입니다. 마스크/크롭 연결을 확인하세요.')
    if record['spatial_std'] < 5.0:
        record['warnings'].append('공간 색상 변화가 작습니다. 단색 입력일 수 있으며 저장 이미지를 확인해야 합니다.')
    record['status'] = 'rejected' if record['errors'] else ('warning' if record['warnings'] else 'measured')
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    emit(f"Size={record['size']} Min={record['min']} Max={record['max']} "
         f"Mean={record['mean']:.2f} Std={record['std']:.2f} "
         f"SpatialStd={record['spatial_std']:.2f} 상태={record['status']} JSON={record_path}")
    for message in record['warnings']:
        emit(message, logging.WARNING)
    if record['errors']:
        message = ' '.join(record['errors']) + f' 저장 위치={image_path}'
        emit(message, logging.ERROR)
        raise ValueError(message)
    emit('입력 통계 기록 완료 — 텍스처/디자인 정상 판정은 하지 않았습니다.')
    return record




@contextmanager
def timed_stage(log, stage, candidate_number, timings):
    started = perf_counter()
    status = 'failed'
    log.write_stage(stage, f'후보={candidate_number}, 시작')
    try:
        yield
        status = 'completed'
    finally:
        elapsed = perf_counter() - started
        timings[stage] = dict(status=status, elapsed_seconds=elapsed)
        log.write_stage(stage, f'후보={candidate_number}, 상태={status}, 소요={elapsed:.2f}초')
