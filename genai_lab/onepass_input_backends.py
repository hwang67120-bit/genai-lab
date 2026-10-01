"""Injected detector/session adapters. This module installs/loads no model.

Clients must be provided explicitly by a configured worker; availability is
checked at deployment time, not inferred from historical environment notes. No hardcoded cache directories or process-wide chdir/env mutation.
"""
from genai_lab.onepass_pose import PoseObservation, joints_from_records, summarize_hands
from genai_lab.onepass_character import HeadDetection
from genai_lab.onepass_input_settings import OnePassInputSettings
from scripts.pose_reference_runner import create_pose_preview_and_coordinates


class InputBackendUnavailable(RuntimeError):
    pass


class DwPoseInputBackend:
    """Reuse the existing preview/coordinate function with actual count capture.

    The caller owns the detector's lifetime and supplies a locally configured
    detector. Missing or zero-person observations must not be converted to count 1.
    """
    def __init__(self, detector, settings=OnePassInputSettings(), *,
                 preview_builder=create_pose_preview_and_coordinates):
        if detector is None:
            raise InputBackendUnavailable('DWPose 백엔드가 필요합니다. 패키지를 자동 설치하지 않습니다.')
        self.detector = detector
        self.settings = settings
        self.preview_builder = preview_builder

    def __call__(self, image):
        owner = self
        class CountedDetector:
            count = None
            face_points = ()
            face_scores = ()
            hands = ()
            def pose_estimation(self, array):
                points, scores = owner.detector.pose_estimation(array)
                import numpy as np
                self.count = len(points)
                if self.count:
                    points, scores = np.asarray(points), np.asarray(scores)
                    if (points.ndim != 3 or scores.ndim != 2 or points.shape[:2] != scores.shape
                            or points.shape[2] != 2 or scores.shape[1] < 18):
                        raise ValueError('DWPose 좌표/신뢰도 형식 오류')
                    # Same selection as create_pose_preview_and_coordinates, before rendering normalization.
                    selected = int(np.argmax(np.mean(scores[:, :18], axis=1)))
                    if points.shape[1] >= 92:
                        self.face_points = tuple(tuple(map(float, xy)) for xy in points[selected, 24:92])
                        self.face_scores = tuple(map(float, scores[selected, 24:92]))
                    if points.shape[1] >= 134 and scores.shape[1] >= 134:
                        self.hands = summarize_hands(tuple(map(float, scores[selected, 92:113])),
                                                     tuple(map(float, scores[selected, 113:134])))
                return points, scores
        counted = CountedDetector()
        try:
            overlay, records, control = self.preview_builder(image, counted, self.settings.checks.confidence)
        except RuntimeError:
            if counted.count != 0:
                raise
            # The legacy builder rejects zero persons. Retain that fact for K1/K2.
            from PIL import Image
            return PoseObservation((), 0, image.convert('RGB'), Image.new('RGB', image.size))
        try:
            if counted.count is None:
                raise ValueError('검출 인원 수가 반환되지 않았습니다.')
            return PoseObservation(joints_from_records(records), counted.count, overlay, control,
                                   counted.face_points, counted.face_scores, counted.hands)
        except Exception:
            overlay.close()
            control.close()
            raise


class WdInputTagger:
    """Reuse an existing WdTagSession; raw general scores are not top-N truncated."""
    def __init__(self, session):
        if session is None:
            raise InputBackendUnavailable('WD 태그 세션이 필요합니다.')
        self.session = session

    def __call__(self, image):
        result = self.session.analyze(image)
        return dict(result.raw_general_scores)


class HeadInputBackend:
    """Interface for imgutils.detect.detect_heads; caller supplies a configured client."""
    def __init__(self, detect_heads, settings=OnePassInputSettings()):
        if detect_heads is None:
            raise InputBackendUnavailable('머리 검출 클라이언트가 필요합니다. 패키지를 자동 설치하지 않습니다.')
        self.detect_heads = detect_heads
        self.settings = settings

    def __call__(self, image):
        cfg = self.settings.character
        return tuple(HeadDetection(tuple(box), float(score)) for box, _, score in self.detect_heads(
            image, model_name=cfg.head_model, conf_threshold=cfg.head_confidence, iou_threshold=cfg.head_iou))


class ForegroundInputBackend:
    """Reuse extract_anime_character_foreground_mask via explicit injection.

    Pass scripts.body_comparison_runner.extract_anime_character_foreground_mask
    with a provisioned cache. The caller owns model provisioning/offline policy,
    just as for HeadInputBackend; constructing this adapter performs no inference.
    """
    def __init__(self, extractor, model_cache_dir):
        if extractor is None or model_cache_dir is None:
            raise InputBackendUnavailable('기존 isnet-anime 추출기와 모델 캐시 설정이 필요합니다.')
        self.extractor = extractor
        self.model_cache_dir = model_cache_dir

    def __call__(self, image):
        return self.extractor(image, model_id='isnet-anime', model_cache_dir=self.model_cache_dir)
