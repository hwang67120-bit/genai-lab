"""실험 제안을 승인한 입력과 실제 인코딩 입력을 연결한다."""
import hashlib
import json


def needs_experiment_review(inputs):
    record = getattr(inputs, 'analysis_record', None) or {}
    detection = record.get("part_detection", {})
    return bool(getattr(inputs, "reference_conditions", None) is not None
                or record.get("condition_separation") or record.get("part_region_experiment")
                or detection.get("filter_policy", {}).get("requires_input_review"))


def experiment_fingerprint(inputs):
    def image_hash(image):
        digest = hashlib.sha256(str((image.mode, image.size)).encode("utf-8"))
        digest.update(image.tobytes())
        return digest.hexdigest()
    record = getattr(inputs, 'analysis_record', None) or {}
    from genai_lab.reference_order import adapter_reference_names
    payload = {
        "adapter_reference_order": adapter_reference_names(inputs),
        "source": image_hash(inputs.source),
        "identity": image_hash(inputs.identity),
        "garment": image_hash(inputs.garment),
        "identity_mask": image_hash(inputs.identity_mask),
        "garment_mask": image_hash(inputs.garment_mask),
        "hair": (None if getattr(inputs, "hair_reference", None) is None else {
            "rgb": image_hash(inputs.hair_reference),
            "region": image_hash(inputs.hair_mask),
            "scale": inputs.hair_reference_scale,
        }),
        "parts": [{
            "name": part.name, "rgb": image_hash(part.rgb),
            "region": image_hash(part.region), "scale": part.scale,
        } for part in inputs.extra_references],
        "filter_policy": record.get("part_detection", {}).get("filter_policy"),
        "region_proposal": record.get("part_region_experiment"),
        "independent_conditions": (inputs.reference_conditions.fingerprint()
                                  if getattr(inputs, "reference_conditions", None) is not None else None),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")).hexdigest()


def approve_reference_experiment(inputs):
    """실제 사용자 입력 승인 이벤트에서만 호출한다. 모델 판정 정답 인증은 아니다."""
    if not needs_experiment_review(inputs):
        return
    inputs.analysis_record["experimental_input_approval"] = {
        "status": "approved_by_user",
        "fingerprint": experiment_fingerprint(inputs),
        "semantic_accuracy_verified": False,
    }


def require_reference_experiment_approval(inputs):
    if not needs_experiment_review(inputs):
        return
    approval = (getattr(inputs, "analysis_record", None) or {}).get("experimental_input_approval", {})
    if approval.get("status") != "approved_by_user":
        raise ValueError("추가 부위·영향 영역의 실험 제안을 먼저 승인하세요.")
    if approval.get("fingerprint") != experiment_fingerprint(inputs):
        raise ValueError("승인 이후 참조 이미지·영역·정책이 바뀌었습니다. 다시 확인하세요.")
