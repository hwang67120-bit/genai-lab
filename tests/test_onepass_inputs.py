from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from PIL import Image, ImageDraw

from genai_lab.onepass_input_settings import (
    OnePassInputSettings, InputPolicies, PoseCheckSettings, PoseNormalizationSettings,
)
from genai_lab.onepass_pose import (
    Finding, PoseObservation, PoseInputError, assess_pose, apply_input_policy, joints_from_records,
    normalize_pose_image, prepare_onepass_control, prepare_onepass_pose,
)
from genai_lab.onepass_character import (
    HeadDetection, prepare_c6_crop, prepare_manual_face_crop, measure_slimness,
    select_character_tags, select_pose_tags,
)
from genai_lab.onepass_input_backends import (
    InputBackendUnavailable, DwPoseInputBackend, HeadInputBackend, WdInputTagger, ForegroundInputBackend,
)
from genai_lab.pose_estimation import PoseJointCoordinateCandidate as Joint

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/onepass_input_e2e.json').read_text(encoding='utf-8'))


def standard_joints():
    points = {'nose': (50, 10), 'neck': (50, 30),
              'left_shoulder': (30, 30), 'right_shoulder': (70, 30),
              'left_elbow': (25, 65), 'right_elbow': (75, 65),
              'left_wrist': (20, 90), 'right_wrist': (80, 90),
              'left_hip': (35, 100), 'right_hip': (65, 100),
              'left_knee': (35, 150), 'right_knee': (65, 150),
              'left_ankle': (35, 200), 'right_ankle': (65, 200),
              'left_eye': (45, 8), 'right_eye': (55, 8),
              'left_ear': (40, 10), 'right_ear': (60, 10)}
    return tuple(Joint(n, x, y, .9, True) for n, (x, y) in points.items())


def change(joints, name, **values):
    return tuple(replace(j, **values) if j.joint_name == name else j for j in joints)


@pytest.mark.parametrize('key', tuple(FIXTURE['scenarios']))
def test_recorded_e2e_geometry_and_policy(key):
    row = FIXTURE['scenarios'][key]
    assessment = assess_pose(joints_from_records(row['joints']), row['person_count'], row['face_direction'])
    decision = apply_input_policy(assessment)
    assert decision.status == row['decision']
    # HIGH_ANGLE is an explicit additional product warning, not a new rejection.
    assert set(decision.rejects) == {f[0] for f in row['rejects']}
    assert set(decision.warnings) - {'HIGH_ANGLE'} == {f[0] for f in row['warnings']}
    char = row['character']
    assert measure_slimness(joints_from_records(char['joints']))['slim'] == char['slim']


@pytest.mark.parametrize('key', tuple(FIXTURE['demo']))
def test_demo_stored_findings_policy_only(key):
    row = FIXTURE['demo'][key]
    assessment = assess_pose(standard_joints(), 1, {'looking_at_viewer': 1})
    # Explicit policy replay: these demo files do not contain normalized joints.
    facts = tuple(Finding(f[0], {'recorded': f[1:]}) for f in row['rejects'] + row['warnings'])
    decision = apply_input_policy(replace(assessment, findings=facts))
    assert decision.status == row['decision']
    assert decision.rejects == tuple(dict.fromkeys(f[0] for f in row['rejects']))


def test_policy_change_does_not_change_measurement():
    joints = change(standard_joints(), 'left_ankle', confidence_score=.1, detected=False)
    facts = assess_pose(joints, 2, {'looking_at_viewer': 1})
    assert apply_input_policy(facts).status == 'reject'
    all_warn = InputPolicies(rules=tuple((k, 'warn') for k, _ in InputPolicies().rules))
    assert apply_input_policy(facts, all_warn).status == 'warn_user_choice'
    assert [f.code for f in facts.findings] == ['K1', 'K2']


@pytest.mark.parametrize('count', [None, -1, 1.0, True])
def test_person_count_is_not_inferred(count):
    with pytest.raises(ValueError):
        assess_pose(standard_joints(), count, {})


def test_no_person_has_k1_k2_and_no_geometry_division():
    facts = assess_pose((), 0, {'looking_at_viewer': 1})
    assert {f.code for f in facts.findings} == {'K1', 'K2'}
    assert facts.torso_length is None


def test_guessed_range_is_lower_inclusive_upper_exclusive():
    js = change(standard_joints(), 'left_wrist', confidence_score=.3)
    js = change(js, 'right_wrist', confidence_score=.5)
    assert len(assess_pose(js, 1, {}).guessed_joints) == 1
    js = change(js, 'right_wrist', confidence_score=.499)
    assert 'GUESS' in {f.code for f in assess_pose(js, 1, {}).findings}


def test_k7_requires_both_knees_and_ankles_close():
    js = change(standard_joints(), 'right_knee', x=35)
    assert 'K7' not in {f.code for f in assess_pose(js, 1, {}).findings}
    js = change(js, 'right_ankle', x=35)
    facts = assess_pose(js, 1, {})
    assert 'K7' in {f.code for f in facts.findings}
    assert apply_input_policy(facts).status == 'warn_user_choice'


def test_degenerate_torso_and_missing_joint_do_not_produce_nan():
    js = change(standard_joints(), 'left_hip', y=30)
    js = change(js, 'right_hip', y=30)
    assert 'K6' in {f.code for f in assess_pose(js, 1, {}).findings}
    assert measure_slimness(change(js, 'neck', y=30))['slim'] is None
    assert measure_slimness(())['reason'] == 'missing_joints'


def test_nonfinite_input_is_rejected():
    with pytest.raises(ValueError):
        assess_pose(change(standard_joints(), 'nose', x=float('nan')), 1, {})
    with pytest.raises(ValueError):
        assess_pose(standard_joints(), 1, {'profile': float('inf')})


def test_direction_boundary_and_high_angle_policy():
    facts = assess_pose(standard_joints(), 1, {'looking_at_viewer': .35, 'from_above': .35})
    assert facts.nonfrontal and facts.high_angle
    assert apply_input_policy(facts).status == 'warn_user_choice'
    assert apply_input_policy(facts, InputPolicies(high_angle='out_of_scope')).status == 'reject'
    assert not assess_pose(standard_joints(), 1, {'looking_at_viewer': .35}).nonfrontal


def test_normalization_padding_ratio_source_unchanged():
    source = Image.new('RGB', (100, 210), 'white')
    original = source.tobytes()
    normalized, box = normalize_pose_image(source, standard_joints())
    assert normalized.size == (736, 1232)
    assert box[1] < 0 and box[3] > source.height
    assert normalized.getpixel((0, 0)) == (128, 128, 128)
    assert source.tobytes() == original
    normalized.close(); source.close()


def test_normalization_rejects_zero_height_and_resource_exhaustion():
    with Image.new('RGB', (100, 210)) as source:
        with pytest.raises(ValueError):
            normalize_pose_image(source, tuple(replace(j, y=1) for j in standard_joints()))
        settings = OnePassInputSettings(normalization=PoseNormalizationSettings(maximum_canvas_pixels=10))
        with pytest.raises(ValueError):
            normalize_pose_image(source, standard_joints(), settings)


def test_control_channel_swap_and_padding_uses_existing_preparer():
    with Image.new('RGB', (100, 200), (255, 8, 12)) as image:
        out = prepare_onepass_control(image, standard_joints())
        try:
            assert out.size == (736, 1232)
            assert out.getpixel((368, 616)) == (12, 8, 255)
            assert out.getpixel((0, 0)) == (0, 0, 0)
            assert image.getpixel((0, 0)) == (255, 8, 12)
        finally:
            out.close()


def test_pipeline_two_passes_tags_original_and_preserves_first_count():
    images = []; tagged = []
    def estimator(image):
        images.append(image.size)
        return PoseObservation(standard_joints(), 2 if len(images) == 1 else 1,
                               image.copy(), Image.new('RGB', image.size, 'red'))
    def tagger(image):
        tagged.append(image.size)
        return {'looking_at_viewer': 1}
    with Image.new('RGB', (100, 210)) as source:
        result = prepare_onepass_pose(source, estimate=estimator, tag=tagger)
    try:
        assert images == [(100, 210), (736, 1232)]
        assert tagged == [(100, 210)]
        assert result.decision.rejects == ('K1',)
        assert result.decision.choices == ('proceed_without_pose', 'choose_other_image')
    finally:
        result.close()


def test_empty_detection_carries_rule_evidence():
    def empty(image):
        return PoseObservation((), 0, image.copy(), image.copy())
    with Image.new('RGB', (100, 210)) as source:
        with pytest.raises(PoseInputError) as caught:
            prepare_onepass_pose(source, estimate=empty, tag=lambda im: {})
    assert set(caught.value.decision.rejects) == {'K1', 'K2'}


def test_character_gender_is_candidate_only_and_pose_is_not_appearance():
    selected = select_character_tags({'1boy': .99, 'black_hair': .9, 'arms_up': .9,
                                      'looking_at_viewer': .7, 'small_breasts': .5})
    assert selected.gender_candidates == (('1boy', .99),)
    assert selected.appearance == (('black_hair', .9), ('small_breasts', .5))


def test_pocket_rule_uses_approved_garment_not_detector_gender():
    scores = {'hands_in_pockets': .9, 'from_above': .8, 'standing': .9}
    assert select_pose_tags(scores, ['white crop top']) == (('from_above', .8),)
    assert select_pose_tags(scores, ['black_shorts'])[0] == ('hands_in_pockets', .9)


def face_landmarks():
    points = [(50., 20.)]*68
    points[0], points[16], points[8] = (35., 20.), (65., 20.), (50., 35.)
    return points, [.9]*68


def crop_with_data(source, heads, joints=(), **kwargs):
    points, scores = face_landmarks()
    with Image.new('L', source.size, 255) as fg:
        return prepare_c6_crop(source, heads, joints, foreground_mask=fg,
                               face_points=points, face_scores=scores, **kwargs)


def test_c6_nose_selection_beats_largest_and_higher_score_outside_nose():
    heads = (HeadDetection((0, 0, 190, 190), .5), HeadDetection((40, 0, 70, 45), .8),
             HeadDetection((100, 100, 150, 150), .99))
    with Image.new('RGB', (200, 300), 'blue') as source:
        result = crop_with_data(source, heads, (standard_joints()[0],))
    try:
        assert result.head_box == (40, 0, 70, 45)
        assert result.details['expanded_box'] == (37, 0, 73, 45)
        assert result.image.width == result.image.height
        assert result.mask.getpixel((50, 38)) == 0  # Below chin + .03H.
        assert result.status == 'review' and result.requires_user_review
        assert result.image.getpixel((0, 0)) == (255, 255, 255)
    finally:
        result.close()


def test_c6_no_nose_chooses_confidence_not_area():
    with Image.new('RGB', (200, 300)) as source:
        result = crop_with_data(source, (HeadDetection((0,0,190,190), .5), HeadDetection((30,10,70,45), .9)))
    assert result.head_box == (30,10,70,45)
    assert result.details['choice'] == '전체 최고 신뢰도'
    result.close()


def test_c6_arm_guard_rejects_without_crop_or_manual_fallback():
    js = change(standard_joints(), 'left_elbow', x=50, y=40)
    with Image.new('RGB', (200, 300)) as source:
        result = crop_with_data(source, (HeadDetection((50,20,100,80), .9),), js)
    assert result.status == 'reject' and 'left_elbow' in result.arm_points
    assert result.image is None and result.mask is None
    assert '팔이 머리를 가리지 않는' in result.reason
    result.close()


def test_c6_foreground_and_chin_masks_are_not_box_crop():
    points, scores = face_landmarks()
    with Image.new('RGB', (100,100), 'blue') as source, Image.new('L',(100,100)) as fg:
        ImageDraw.Draw(fg).rectangle((40,5,60,60), fill=255)
        result = prepare_c6_crop(source, (HeadDetection((20,0,80,60), .9),), (),
                                 foreground_mask=fg, face_points=points, face_scores=scores)
    assert result.mask.getpixel((30,20)) == 0
    assert result.mask.getpixel((50,20)) == 255
    assert result.mask.getpixel((50,37)) == 0
    assert result.image.size == (36,36)  # Retained y=5..36 plus 2px border; no upscaling.
    result.close()


def test_c6_arm_band_preserves_face_hull():
    # Endpoints outside guard, arm segment crosses head. Guard must not mask band behavior.
    js = (Joint('right_shoulder',0,20,.9,True), Joint('left_shoulder',0,70,.9,True),
          Joint('right_elbow',99,20,.9,True))
    with Image.new('RGB', (100,100)) as source:
        result = crop_with_data(source, (HeadDetection((30,0,70,45), .9),), js)
    assert result.status == 'review'
    assert result.details['arm_thickness'] == 18
    assert result.mask.getpixel((50,22)) == 255  # Inside face hull.
    assert result.mask.getpixel((30,22)) == 0   # Arm outside face hull.
    result.close()


def test_c6_low_face_confidence_and_empty_foreground_fail():
    points, scores = face_landmarks()
    with Image.new('RGB',(100,100)) as source, Image.new('L',(100,100)) as fg:
        low = prepare_c6_crop(source,(HeadDetection((30,0,70,45),.9),),(),
                              foreground_mask=fg,face_points=points,face_scores=[.1]*68)
        empty = prepare_c6_crop(source,(HeadDetection((30,0,70,45),.9),),(),
                                foreground_mask=fg,face_points=points,face_scores=scores)
    assert low.status == empty.status == 'reject'
    assert '신뢰도' in low.reason and empty.reason == '보존 픽셀 0'


def test_c6_missing_head_and_manual_crop_are_explicit():
    with Image.new('RGB', (64,64)) as source:
        assert prepare_c6_crop(source,(),()).reason == '머리 검출 없음'
        assert prepare_c6_crop(source,(HeadDetection((10,10,30,30),.9),),()).reason == '얼굴 68점 없음'
        manual = prepare_manual_face_crop(source)
        assert manual.method == 'manual' and manual.requires_user_review
        manual.close()


def test_face_backend_returns_same_selected_person_without_extra_inference():
    import numpy as np
    points=np.zeros((2,134,2)); points[1,24:92]=[23,47]
    scores=np.full((2,134),.4); scores[1,:18]=.9; scores[1,24:92]=.8
    calls=[]
    def detect(a):
        calls.append(a)
        return points,scores
    def preview(image,wrapped,confidence):
        wrapped.pose_estimation(None)
        return image.copy(), [], image.copy()
    with Image.new('RGB',(100,100)) as image:
        result=DwPoseInputBackend(SimpleNamespace(pose_estimation=detect),preview_builder=preview)(image)
    assert len(calls)==1 and result.person_count==2
    assert result.face_points==((23.,47.),)*68
    assert result.face_scores==(.8,)*68
    points[1,24:92]=0
    assert result.face_points[0]==(23.,47.)
    result.close()


@pytest.mark.parametrize('factory', [lambda: DwPoseInputBackend(None), lambda: HeadInputBackend(None), lambda: WdInputTagger(None)])
def test_missing_backend_never_downloads_or_falls_back(factory):
    with pytest.raises(InputBackendUnavailable):
        factory()


def test_wd_backend_preserves_untruncated_scores():
    session = SimpleNamespace(analyze=lambda image: SimpleNamespace(raw_general_scores=(('profile', .2), ('black hair', .8))))
    assert WdInputTagger(session)(None) == {'profile': .2, 'black hair': .8}


def test_count_capture_does_not_infer_one_from_selected_person():
    import numpy as np
    detector = SimpleNamespace(pose_estimation=lambda a: (np.zeros((2, 18, 2)), np.ones((2, 18))))
    def preview(image, wrapped, confidence):
        wrapped.pose_estimation(None)
        from dataclasses import asdict
        return image.copy(), [asdict(j) for j in standard_joints()], image.copy()
    backend = DwPoseInputBackend(detector, preview_builder=preview)
    with Image.new('RGB', (100, 210)) as image:
        result = backend(image)
    assert result.person_count == 2
    result.close()


@pytest.mark.parametrize('constructor,kwargs', [
    (PoseCheckSettings, {'confidence': float('nan')}),
    (PoseCheckSettings, {'guessed_confidence_upper': .2}),
    (PoseNormalizationSettings, {'width': 735}),
    (InputPolicies, {'rules': (('K1', 'warn'),)}),
    (InputPolicies, {'high_angle': 'silent'}),
])
def test_invalid_settings_fail_explicitly(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)


def test_foreground_backend_reuses_injected_extractor_and_explicit_cache(tmp_path):
    calls=[]
    def extract(image, *, model_id, model_cache_dir):
        calls.append((image.size,model_id,model_cache_dir))
        return Image.new('L',image.size,128)
    backend=ForegroundInputBackend(extract,tmp_path)
    assert calls==[]
    with Image.new('RGB',(50,70)) as source, backend(source) as mask:
        assert mask.size==source.size and mask.getpixel((0,0))==128
    assert calls==[((50,70),'isnet-anime',tmp_path)]


def test_c6_missing_chin_keeps_lower_hair_and_records_reason():
    points,scores=face_landmarks();scores[8]=.1
    with Image.new('RGB',(100,100)) as source, Image.new('L',(100,100),255) as fg:
        result=prepare_c6_crop(source,(HeadDetection((30,0,70,60),.9),),(),
                               foreground_mask=fg,face_points=points,face_scores=scores)
    assert result.status=='review' and result.mask.getpixel((50,59))==255
    assert len(result.details['notes'])==2  # Both shoulder data and chin unavailable.
    result.close()


@pytest.mark.parametrize('face_points,face_scores', [([(0,0)]*67,[.9]*68), ([(float('nan'),0)]*68,[.9]*68)])
def test_c6_malformed_face_coordinates_do_not_silently_fallback(face_points,face_scores):
    with Image.new('RGB',(100,100)) as source:
        with pytest.raises(ValueError,match='얼굴 68점'):
            prepare_c6_crop(source,(HeadDetection((30,0,70,60),.9),),(),
                            face_points=face_points,face_scores=face_scores)


@pytest.mark.parametrize('status,count,scores', [('pass',1,{'looking_at_viewer':1}),
                                                ('warn_user_choice',1,{}), ('reject',2,{})])
def test_explicit_pose_choices_have_no_default(status, count, scores):
    decision = apply_input_policy(assess_pose(standard_joints(), count, scores))
    assert decision.status == status
    assert decision.choices == (('proceed_without_pose','choose_other_image') if status=='reject' else
                               ('proceed_with_pose','proceed_without_pose','choose_other_image'))
    assert not hasattr(decision, 'default_choice')


def test_normalized_zero_joints_is_reviewable_k2():
    calls=[]
    def estimate(image):
        calls.append(image.size)
        joints = standard_joints() if len(calls)==1 else tuple(
            replace(j, confidence_score=.29, detected=False) for j in standard_joints())
        return PoseObservation(joints,1,image.copy(),image.copy())
    with Image.new('RGB',(100,210),'red') as source:
        with pytest.raises(PoseInputError) as caught:
            prepare_onepass_pose(source,estimate=estimate,tag=lambda _: {'looking_at_viewer':1})
    error=caught.value
    try:
        assert error.decision.rejects==('K2',)
        assert error.decision.choices==('proceed_without_pose','choose_other_image')
        assert error.original_overlay.size==(100,210)
        assert error.overlay.size==(736,1232)
        assert len(error.k2_hints)==8
    finally:
        error.close()


def test_other_control_estimation_error_propagates_unchanged(monkeypatch):
    import genai_lab.onepass_pose as module
    from genai_lab.pose_estimation import PoseReferenceEstimationError
    failure=PoseReferenceEstimationError('실행 실패: 출력 없음')
    def failed(*args):
        raise failure
    monkeypatch.setattr(module,'prepare_onepass_control',failed)
    def estimate(image):
        return PoseObservation(standard_joints(),1,image.copy(),image.copy())
    with Image.new('RGB',(100,210)) as source:
        with pytest.raises(PoseReferenceEstimationError) as caught:
            prepare_onepass_pose(source,estimate=estimate,tag=lambda _: {})
    assert caught.value is failure


def test_normalization_failure_keeps_original_evidence_only_even_with_warn_policy():
    from genai_lab.onepass_pose import summarize_hands
    hands=summarize_hands((.3,)*21,(.2,)*21)
    def estimate(image):
        return PoseObservation((),0,image.copy(),image.copy(),hands=hands)
    cfg=OnePassInputSettings(policies=InputPolicies(rules=tuple((k,'warn') for k,_ in InputPolicies().rules)))
    with Image.new('RGB',(100,210),'blue') as source:
        with pytest.raises(PoseInputError) as caught:
            prepare_onepass_pose(source,estimate=estimate,tag=lambda _: {},settings=cfg)
    error=caught.value
    try:
        assert error.original_overlay.getpixel((0,0))==(0,0,255)
        assert error.overlay is None and error.hands==()
        assert error.original_hands==hands
        assert error.decision.status=='reject'
        assert 'proceed_with_pose' not in error.decision.choices
        assert set(error.decision.warnings)>={'K1','K2'}
    finally:
        error.close()


def test_display_evidence_does_not_change_control_or_decision():
    from genai_lab.onepass_pose import summarize_hands
    first_hands=summarize_hands((.29,)*20+(.3,),(.9,)*21)
    second_hands=summarize_hands((.8,)*21,(.1,)*21)
    def run(with_evidence):
        calls=[]
        def estimate(image):
            calls.append(1)
            hands=(first_hands if len(calls)==1 else second_hands) if with_evidence else ()
            return PoseObservation(standard_joints(),1,image.copy(),Image.new('RGB',image.size,(20,30,40)),hands=hands)
        with Image.new('RGB',(100,210),'red') as source:
            return prepare_onepass_pose(source,estimate=estimate,tag=lambda _: {'looking_at_viewer':1})
    before,after=run(False),run(True)
    try:
        assert before.decision==after.decision
        assert before.assessment==after.assessment
        assert before.control_image.tobytes()==after.control_image.tobytes()
        assert after.original_overlay.size==(100,210)
        assert after.original_overlay.getpixel((1,1))==(255,0,0)
        assert after.original_hands==first_hands and after.hands==second_hands
        assert first_hands[0].confident_count==1 and first_hands[0].total==21
    finally:
        before.close();after.close()


@pytest.mark.parametrize('x,y,hint',[(1,100,'out_of_frame_likely'),(-4,100,'out_of_frame_likely'),
                                   (50,209,'out_of_frame_likely'),(50,100,'low_confidence')])
def test_k2_display_hint_uses_original_coordinates_only(x,y,hint):
    from genai_lab.onepass_pose import missing_joint_hints
    original=change(standard_joints(),'left_ankle',x=x,y=y,confidence_score=.1,detected=False)
    normalized=change(standard_joints(),'left_ankle',x=50,y=100,confidence_score=.1,detected=False)
    facts=assess_pose(normalized,1,{'looking_at_viewer':1})
    before=apply_input_policy(facts)
    values=missing_joint_hints(facts,original,(100,210))
    assert len(values)==1 and values[0].hint==hint
    assert values[0].margin_pixels==2 and not values[0].verified
    assert apply_input_policy(facts)==before


def test_dwpose_hand_scores_from_selected_person_are_display_only():
    import numpy as np
    points=np.zeros((2,134,2)); scores=np.full((2,134),.1)
    scores[1,:18]=.9; scores[1,92:113]=[.3]+[.2]*20; scores[1,113:134]=.8
    calls=[]
    def detect(_):
        calls.append(1)
        return points,scores
    def preview(image,wrapped,confidence):
        wrapped.pose_estimation(None)
        return image.copy(),[],image.copy()
    with Image.new('RGB',(100,100)) as image:
        result=DwPoseInputBackend(SimpleNamespace(pose_estimation=detect),preview_builder=preview)(image)
    try:
        assert len(calls)==1 and len(result.hands)==2
        left,right=result.hands
        assert (left.side,left.confident_count,left.total)==('left',1,21)
        assert left.mean_score==pytest.approx((.3+.2*20)/21)
        assert right.confident_count==21 and right.mean_score==pytest.approx(.8)
        scores[:]=0
        assert right.mean_score==pytest.approx(.8)
    finally:
        result.close()
