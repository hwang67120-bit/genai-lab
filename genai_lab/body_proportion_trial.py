"""Opt-in proportion trial: confirmed geometry -> prompt -> generation -> comparison.

No GUI defaults, model installation, extra inference, or generation retries.
The existing backend owns GPU lifecycle; output measurement is injected so CPU
analysis can run after GPU teardown if the caller chooses a separate review.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math

from genai_lab.body_proportions import (
    BodyProportionReference, compare_body_proportions, image_digest)
from genai_lab.onepass_prompt import plan_prompt_chunks
from genai_lab.onepass_generation import (
    OnePassCancelled, generate_onepass_image, write_record)
from genai_lab.onepass_generation_settings import OnePassGenerationSettings


def prepare_body_proportion_inputs(inputs, reference=None, *, tokenizers=None):
    """None returns the identical request. Confirmed ratios append numeric text.

    Text guidance is experimental, not a spatial constraint or a guarantee of
    correct anatomy. Gender, clothing, pose, face image and negative text stay.
    """
    if reference is None:
        return inputs
    if not isinstance(reference, BodyProportionReference):
        raise TypeError('확인된 비율 참조가 필요합니다.')
    reference.require_confirmed()
    if reference.face_sha256 != inputs.face_sha256:
        raise ValueError('비율 참조와 얼굴 참조가 서로 다른 입력입니다.')
    if 'body_proportion_condition' in inputs.prompt.rules:
        raise ValueError('이미 비율 조건이 적용된 요청입니다.')
    phrases = reference.prompt_phrases()
    positive = ', '.join((inputs.prompt.positive, *phrases))
    encoders = plan_prompt_chunks(positive, inputs.prompt.negative, tokenizers)
    rules = dict(inputs.prompt.rules)
    rules['body_proportion_condition'] = dict(
        reference=asdict(reference), phrases=phrases, mode='numeric_text_trial',
        baseline_positive=inputs.prompt.positive,
        positive_tokens_before=tuple(p.positive_tokens for p in inputs.prompt.encoders),
        positive_tokens_after=tuple(p.positive_tokens for p in encoders),
        spatial_constraint=False,
        efficacy_verified=False, output_review_required=True)
    rules['chunks'] = len(encoders[0].positive)
    prompt = replace(inputs.prompt, positive=positive, encoders=encoders, rules=rules)
    return replace(inputs, prompt=prompt)


@dataclass(frozen=True)
class BodyProportionTrialResult:
    candidate: object
    comparison: dict
    comparison_path: Path


def review_body_proportion_candidate(candidate, reference, measure_output, *, relative_tolerance=0.05):
    """Keep raw/run.json untouched even when measurement fails or is cancelled."""
    path = candidate.path.parent / 'body-proportion-review.json'
    if path.exists():
        raise FileExistsError('기존 비율 검토 기록을 덮어쓰지 않습니다.')
    record = dict(status='measurement_pending', raw_sha256=candidate.record['raw_sha256'],
                  reference=asdict(reference), product_approved=False, user_review_required=True)
    write_record(path, record)
    try:
        reference.require_confirmed()
        if candidate.record.get('inputs', {}).get('face_sha256') != reference.face_sha256:
            raise ValueError('생성 결과와 비율 참조의 얼굴 입력이 다릅니다.')
        recorded = candidate.record.get('prompt', {}).get('rules', {}).get('body_proportion_condition')
        if recorded is not None and json.dumps(recorded['reference'], sort_keys=True) != json.dumps(asdict(reference), sort_keys=True):
            raise ValueError('생성에 사용한 비율과 대조할 비율이 다릅니다.')
        if not candidate.record.get('valid'):
            raise ValueError('실행 검증에 실패한 결과입니다.')
        if image_digest(candidate.path) != candidate.record['raw_sha256']:
            raise ValueError('생성 이미지 SHA가 달라졌습니다.')
        observation = measure_output(candidate.path)
        if observation.image_sha256 != candidate.record['raw_sha256']:
            raise ValueError('측정 대상이 생성 결과와 다릅니다.')
        if image_digest(candidate.path) != candidate.record['raw_sha256']:
            raise ValueError('측정 중 생성 이미지가 바뀌었습니다.')
        record.update(compare_body_proportions(reference, observation,
                                               relative_tolerance=relative_tolerance))
    except OnePassCancelled as error:
        record.update(status='measurement_cancelled', error=str(error))
        raise
    except Exception as error:
        record.update(status='measurement_failed', error_type=type(error).__name__, error=str(error))
    finally:
        write_record(path, record)
    return BodyProportionTrialResult(candidate, record, path)


def generate_body_proportion_trial(backend, inputs, reference, *, source_image,
                                   tokenizers, seed, directory, measure_output,
                                   settings=OnePassGenerationSettings(),
                                   relative_tolerance=0.05, cancelled=lambda: False):
    """One real generation call with confirmed ratio text and explicit review.

    A comparison failure returns measurement_failed while preserving raw output.
    Generation failures/cancellation propagate with the existing run.json record.
    """
    if not isinstance(reference, BodyProportionReference):
        raise TypeError('비율 시험에는 명시적인 참조가 필요합니다.')
    if type(relative_tolerance) not in (int, float) or not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError('비교 허용 오차는 0 이상 1 미만입니다.')
    prepared = prepare_body_proportion_inputs(inputs, reference, tokenizers=tokenizers)
    if image_digest(source_image) != reference.measurement.image_sha256:
        raise ValueError('비율을 측정한 원본 이미지가 달라졌습니다.')
    if not callable(measure_output):
        raise TypeError('생성 후 비율을 측정할 함수가 필요합니다.')
    candidate = generate_onepass_image(backend, prepared, seed, directory,
                                       settings=settings, cancelled=cancelled)
    return review_body_proportion_candidate(candidate, reference, measure_output,
                                            relative_tolerance=relative_tolerance)
