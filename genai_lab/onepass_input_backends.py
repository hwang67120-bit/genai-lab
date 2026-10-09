"""외부에서 주입한 검출기·세션 연결부다. 이 모듈에서 모델 설치나 로드는 없다. 설정된 작업자가 클라이언트를 제공하며 배포 시 실제 가용성을 확인한다. 과거
기록으로 추정하지 않는다. 캐시 경로 고정이나 전역 작업 폴더·환경 변경은 없다.
"""
from genai_lab.onepass_pose import PoseObservation, joints_from_records, summarize_hands
from genai_lab.onepass_character import HeadDetection
from genai_lab.onepass_input_settings import OnePassInputSettings
from scripts.pose_reference_runner import create_pose_preview_and_coordinates


class InputBackendUnavailable(RuntimeError):
    pass


class DwPoseInputBackend:
    """실제 인원수를 함께 수집하며 기존 미리보기·좌표 함수를 재사용한다. 호출부가 검출기 수명과 로컬 설정을 관리한다. 결과 없음이나 0명을 1명으로 바꾸지
    않는다.
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
                    # 그림 정규화 전에 기존 미리보기·좌표 함수와 같은 사람을 선택한다.
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
            # 구형 조립기는 0명을 거부한다. K1·K2를 위해 이 사실을 보존한다.
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
    """기존 WdTagSession을 재사용하며 일반 원점수를 상위 항목만으로 자르지 않는다."""
    def __init__(self, session):
        if session is None:
            raise InputBackendUnavailable('WD 태그 세션이 필요합니다.')
        self.session = session

    def __call__(self, image):
        result = self.session.analyze(image)
        return dict(result.raw_general_scores)


class HeadInputBackend:
    """설정한 클라이언트를 받아 imgutils.detect.detect_heads를 호출하는 연결부다."""
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
    """명시적으로 주입한 기존 캐릭터 외곽 마스크 함수를 재사용한다. 준비된 캐시와
    scripts.body_comparison_runner.extract_anime_character_foreground_mask를 전달한다. 모델 준비·오프라인
    정책은 호출부가 관리하며 연결부 생성만으로 추론하지 않는다.
    """
    def __init__(self, extractor, model_cache_dir):
        if extractor is None or model_cache_dir is None:
            raise InputBackendUnavailable('기존 isnet-anime 추출기와 모델 캐시 설정이 필요합니다.')
        self.extractor = extractor
        self.model_cache_dir = model_cache_dir

    def __call__(self, image):
        return self.extractor(image, model_id='isnet-anime', model_cache_dir=self.model_cache_dir)
