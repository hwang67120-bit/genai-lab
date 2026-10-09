"""기본은 CPU 머리·측정 준비이며 --run을 명시해야 W0/W1을 생성한다."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from genai_lab.proportion_inputs import sha, require, json_value, validate_proportion_request, FOREGROUND_SHA, neck_gap_only
from genai_lab.qwen_record_io import write_json
from genai_lab.studio_generation import StudioRuntime, analyze_inputs, build_request
from genai_lab.studio_proportion import prepare_studio_pose, prepare_studio_head, confirmed_options
from genai_lab.identity_measure_service import measure_image
from genai_lab.proportion_generation import generate_stage, build_maps, finish_candidates, save_state
from genai_lab.proportion_backend import ProportionBackend, GuardedBase
from genai_lab.onepass_generation import DiffusersOnePassBackend, OnePassCancelled, validate_seed
from genai_lab.proportion_foreground import AnimeForeground
from scripts.verify_proportion_product import replay_inputs, replay_expectations
from scripts.body_outline_source_trial import restore_options, verify_manifest
from scripts.head_contour_trial_inputs import restore_request

SEEDS=(209212001,209212002,209212003,209212004)
CODE_NAMES=('body_widths.py','identity_measurement.py','identity_measure_service.py',
    'identity_measure_worker.py','identity_report.py','proportion_inputs.py','proportion_generation.py',
    'proportion_backend.py','proportion_foreground.py','onepass_generation.py','studio_proportion.py',
    'studio_proportion_worker.py','studio_generation.py','studio_analysis.py')


def code_manifest():
    return {str(p):sha(p) for p in [Path(__file__),*[ROOT/'genai_lab'/n for n in CODE_NAMES]]}


def lock_preparation(directory, details):
    details['files']={str(p):sha(p) for p in directory.rglob('*') if p.is_file()}
    details['code_sha256']=code_manifest()
    write_json(directory/'preflight.json',json_value(details))
    digest=sha(directory/'preflight.json');(directory/'preflight.sha256').write_text(digest+'\n')
    write_json(directory/'status.json',{'status':'needs_user_review','review_sha256':digest,'gpu_generation':False})
    return digest


def prepare(args, runtime):
    directory=args.output;directory.mkdir(parents=True,exist_ok=False)
    try:
        if args.r6:
            from genai_lab.studio_background import MODEL_RELATIVE
            foreground=args.foreground_model or runtime.model_cache/MODEL_RELATIVE
            inputs,settings,options=replay_inputs(foreground)
            validate_proportion_request(inputs,settings,options)
            # 사용자가 확인한 기존 R6 머리를 재사용한다. 검출기로 새로 승인하지 않는다.
            measurement=measure_image(options.head.normalized_file,directory/'original',runtime)
            bases,maps,raws=replay_expectations()
            details={'case':'R6','inputs':asdict(inputs),'settings':asdict(settings),'options':asdict(options),
                'expected_base':{str(s):bases[s] for s in args.seeds},
                'expected_W0_map':{str(s):maps[s] for s in args.seeds},
                'expected_W0_raw':{str(s):raws[s] for s in args.seeds},'measurement':measurement}
            details['source_files']={str(p):sha(p) for p in (inputs.control_file,inputs.face_file,
                options.head.normalized_file,options.head.mask_file,options.head.contour_file)}
        else:
            require(args.character is not None and args.garment is not None,'캐릭터와 의상 파일이 필요합니다.')
            analysis=analyze_inputs(args.character,args.garment,directory/'inputs',runtime)
            pose=prepare_studio_pose(analysis,runtime,directory/'pose')
            measurement=measure_image(pose['normalized_file'],directory/'original',runtime)
            # 검출 상자는 제안일 뿐이다. 선택 영역을 제공했다고 윤곽이 자동 승인되지 않는다.
            if args.head_review:
                selection=json.loads(args.head_review.read_text(encoding='utf-8'))
                require(selection.get('normalized_sha256')==pose['normalized_sha256'],'머리 선택의 원본 SHA가 다릅니다.')
            else:
                require('head_box' in measurement,'머리 초안을 만들지 못했습니다. 다른 입력이 필요합니다.')
                selection={'box':[int(round(v)) for v in measurement['head_box']],'face_point':pose['nose']}
            draft=prepare_studio_head(pose,selection,runtime,directory/'head')
            details={'case':'references','analysis':analysis,'pose':pose,'draft':draft,'measurement':measurement,
                'source_files':{str(p):sha(p) for p in (args.character,args.garment)}}
            write_json(directory/'input-review-template.json',{'confirmed':False,'gender':None,
                'garment_tags':analysis['garment_tags'],'note':'성별·의상 설명과 머리 윤곽 미리보기를 확인한 뒤 별도 사본에서 confirmed=true로 설정'})
        details.update(seeds=list(args.seeds),body_widths=['off','match_original'],strengths=[1.2,.5],
            shared_adapter_steps=list(range(11)),runtime=json_value(asdict(runtime)))
        return lock_preparation(directory,details)
    except BaseException as error:
        write_json(directory/'status.json',{'status':'failed','gpu_generation':False,'error':str(error)})
        raise


def generate_comparison(inputs, settings, options, seeds, output, *, expected_base=None,
        expected_map=None, expected_raw=None, base_factory=DiffusersOnePassBackend,
        contour_factory=ProportionBackend, foreground_factory=AnimeForeground, cancelled=lambda:False, preparation=None):
    """공통 1단계 하나와 서로 다른 2단계 조건 두 개를 실행한다. 자동 재시도는 없다."""
    mask,contour,models=validate_proportion_request(inputs,settings,options)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    state={'status':'started','BASE':[],'CONTOUR':[],'products':[],'seeds':list(seeds),'conditions':{}}
    lock=json_value({'inputs':asdict(inputs),'settings':asdict(settings),'options':asdict(options),
        'models':models,'seeds':list(seeds),'body_widths':['off','match_original'],'adapter_strengths':[1.2,.5],
        'shared_adapter_steps':list(range(11)),'code_sha256':code_manifest(),'preparation':preparation})
    write_json(output/'preflight.json',lock)
    (output/'preflight.sha256').write_text(sha(output/'preflight.json')+'\n')
    try:
        bases=generate_stage(inputs,seeds,output,settings,lambda:GuardedBase(base_factory(settings),settings),
            cancelled,state,'BASE',expected_base or {})
        for label,mode in (('W0','off'),('W1','match_original')):
            active=replace(options,body_widths=mode)
            directory=output/label;directory.mkdir()
            local={'status':'started','BASE':[],'CONTOUR':[],'products':[],'body_widths':mode,'base_stage':'shared'}
            state['conditions'][label]=local
            try:
                maps=build_maps(bases,mask,contour,directory,foreground_factory,active,cancelled,
                    expected_map or {} if label=='W0' else {})
                local['maps']=maps
                raws=generate_stage(inputs,seeds,directory,settings,lambda:contour_factory(settings,active,maps),
                    cancelled,local,'CONTOUR',expected_raw or {} if label=='W0' else {})
                finish_candidates(raws,directory,foreground_factory,active,cancelled,local,lambda _:None)
                local['status']='review_pending'
            except BaseException as error:
                local.update(status='cancelled' if isinstance(error,OnePassCancelled) else 'failed',error=str(error));raise
            finally:
                save_state(directory,local);write_json(directory/'run.json',json_value({**lock,**local}))
        state['status']='review_pending'
    except BaseException as error:
        state.update(status='cancelled' if isinstance(error,OnePassCancelled) else 'failed',error=str(error));raise
    finally:
        save_state(output,state);write_json(output/'run.json',json_value({**lock,**state}))
    return state


def run_prepared(args,runtime):
    prepared=args.prepared
    require(prepared is not None and args.review_sha is not None,'CPU 준비 폴더와 확인한 SHA가 필요합니다.')
    require(sha(prepared/'preflight.json')==args.review_sha==(prepared/'preflight.sha256').read_text().strip(),'미리보기 잠금 SHA 불일치')
    lock=json.loads((prepared/'preflight.json').read_text(encoding='utf-8'))
    require(len(lock['seeds'])==4 and len(set(lock['seeds']))==4,'잠금 seed는 중복 없는 4개여야 합니다.')
    for seed in lock['seeds']:validate_seed(seed)
    verify_manifest({**lock['files'],**lock['source_files'],**lock['code_sha256']})
    require(lock['runtime']==json_value(asdict(runtime)),'준비 이후 실행 환경 설정이 변경됐습니다.')
    if lock['case']=='R6':
        inputs,settings=restore_request({'inputs':{'BASE':lock['inputs']},'settings':lock['settings']})
        options=restore_options(lock['options'])
    else:
        require(args.input_review is not None,'성별·의상·머리 검토 기록이 필요합니다.')
        approval=json.loads(args.input_review.read_text(encoding='utf-8'))
        require(approval.get('confirmed') is True and approval.get('review_sha256')==args.review_sha,'현재 미리보기의 사용자 확인이 필요합니다.')
        error=lock['draft'].get('geometry_error')
        # 2026-10-09 결정: 목 위치 검사만 명시적 확인 후 통과할 수 있다.
        require(not error or (neck_gap_only(error) and approval.get('geometry_override')=='user_confirmed'),
                '머리 윤곽 좌표 오류를 먼저 고쳐 주세요.' if not (error and neck_gap_only(error)) else
                '목 위치 경고를 확인했다는 기록(geometry_override: user_confirmed)이 필요합니다.')
        options=confirmed_options(lock['pose'],lock['draft'],runtime)
        from PySide6.QtCore import QSettings
        preferences=QSettings(str(prepared/'trial-preferences.ini'),QSettings.Format.IniFormat)
        request=build_request(lock['analysis'],approval['garment_tags'],approval['gender'],confirmed=True,
            runtime=runtime,preferences=preferences,seeds=tuple(lock['seeds']),proportion_pose=lock['pose'])
        inputs,settings=request.inputs,request.settings
    expectations=lambda name:{int(k):v for k,v in lock.get(name,{}).items()}
    return generate_comparison(inputs,settings,options,tuple(lock['seeds']),args.output,
        expected_base=expectations('expected_base'),expected_map=expectations('expected_W0_map'),
        expected_raw=expectations('expected_W0_raw'),preparation={'path':str(prepared),'sha256':args.review_sha,
            'user_review_sha256':sha(args.input_review) if args.input_review else None})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--r6',action='store_true');source.add_argument('--character',type=Path);source.add_argument('--prepared',type=Path)
    parser.add_argument('--garment',type=Path);parser.add_argument('--head-review',type=Path)
    parser.add_argument('--input-review',type=Path);parser.add_argument('--foreground-model',type=Path)
    parser.add_argument('--seeds',nargs=4,type=int,default=SEEDS);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-sha');parser.add_argument('--run',action='store_true')
    args=parser.parse_args(argv);args.output=args.output.resolve()
    require(not args.output.exists(),'새 출력 폴더만 사용합니다.')
    require(len(set(args.seeds))==4,'중복 없는 seed 4개가 필요합니다.')
    for seed in args.seeds:validate_seed(seed)
    if args.r6:require(tuple(args.seeds)==SEEDS,'R6는 잠금 seed 001~004만 사용합니다.')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    runtime=StudioRuntime.from_environment()
    if args.run:
        state=run_prepared(args,runtime);print(state['status'])
    else:
        require(args.prepared is None,'CPU 준비에는 원본 입력이나 --r6를 사용합니다.')
        digest=prepare(args,runtime);print(json.dumps({'status':'needs_user_review','review_sha256':digest,'gpu_generation':0}))


if __name__=='__main__':main()
