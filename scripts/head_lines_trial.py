"""CPU 미리보기를 먼저 만든다. 검토 잠금과 별도 --run으로 CONTOUR만 생성한다."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from PIL import Image
import numpy as np
from genai_lab.proportion_inputs import sha,require,json_value,validate_proportion_request,checked_image
from genai_lab.proportion_generation import build_maps,generate_stage,finish_candidates,save_state
from genai_lab.proportion_backend import ProportionBackend
from genai_lab.proportion_foreground import AnimeForeground
from genai_lab.onepass_generation import OnePassCandidate
from genai_lab.head_lines import prepare_head_lines,compose_head_lines,checked_lines,save_overlay
from genai_lab.qwen_record_io import write_json
from scripts.body_outline_source_trial import load_studio,restore_options,verify_manifest,checked_record
from scripts.head_contour_trial_inputs import DESIGN,POSE_DIR,BASE_DIR,read,restore_request,canonical
from scripts.verify_proportion_product import replay_inputs,replay_expectations

R6_SEEDS=(209212001,209212002)
CODE_FILES=(Path(__file__),*[ROOT/'genai_lab'/name for name in (
    'head_lines.py','proportion_inputs.py','proportion_generation.py','proportion_backend.py',
    'proportion_foreground.py','onepass_generation.py','onepass_generation_settings.py','tail_complexity_worker.py')],
    ROOT/'scripts/head_contour_trial_inputs.py',ROOT/'scripts/verify_proportion_product.py',ROOT/'scripts/body_outline_source_trial.py')


def nose_from(joints, observation=None):
    nose = next((dict(j) for j in joints if j['joint_name']=='nose'), None)
    if nose is not None and observation:
        nose.update(face_points=observation.get('face_points', []), face_scores=observation.get('face_scores', []))
    return nose


def ensure_cheek_observation(nose, head, directory):
    if nose and nose.get('face_points'):
        return nose
    # 옛 잠금에는 정규화 얼굴 관절이 없다. 같은 화면을 CPU로 다시 검출한다.
    from genai_lab.studio_generation import StudioRuntime, cpu_process
    import shutil
    runtime = StudioRuntime.from_environment()
    folder = directory / 'face-pose'
    folder.mkdir()
    shutil.copyfile(head.normalized_file, folder / 'character.png')
    worker = ROOT / 'genai_lab/studio_analysis.py'
    cpu_process([str(runtime.pose_python), str(worker), 'pose', str(folder), str(runtime.pose_models),
        str(runtime.model_cache), str(runtime.head_cache)], folder / 'cpu.log', lambda: False, runtime.analysis_timeout)
    observed = read(folder / 'pose.json')
    require(observed['person_count'] == 1, '볼 표본에 사용할 얼굴 한 명이 필요합니다.')
    result = dict(nose or {})
    result.update(face_points=observed['face_points'], face_scores=observed['face_scores'])
    return result


def load_case(studio,foreground_model):
    if studio is not None:
        inputs,settings,options,all_seeds,manifest=load_studio(studio,outline_source='base')
        require(options.head_lines=='none','C0는 머리 안쪽 선이 없는 실행이어야 합니다.')
        seeds=all_seeds[:2]
        confirmation=read(options.head.contour_file.parent/'user-confirmation.json')
        nose=nose_from(confirmation['pose'].get('joints',[]), confirmation['pose'])
        generation=Path(studio)/'generation'
        bases={seed:generation/f'BASE_{seed}' for seed in seeds}
        old_maps=read(generation/'maps.json')
        expected={seed:old_maps[str(seed)]['sha256'] for seed in seeds}
        comparisons={seed:generation/f'CONTOUR_{seed}/raw.png' for seed in seeds}
        name=Path(studio).name
    else:
        require(foreground_model is not None,'R6에는 로컬 foreground-model이 필요합니다.')
        inputs,settings,options=replay_inputs(foreground_model)
        _,expected_all,_=replay_expectations()
        seeds=R6_SEEDS;manifest={}
        points=checked_record(POSE_DIR/'observations.json',manifest)['observations'][1]['joints']
        nose=nose_from(points)
        bases={seed:DESIGN/('gpu-02' if seed==209212001 else 'gpu-04')/f'BASE_{seed}' for seed in seeds}
        expected={seed:expected_all[seed] for seed in seeds}
        comparisons={seed:DESIGN/f'gpu-05/CONTOUR_{seed}/raw.png' for seed in seeds}
        for path in (BASE_DIR/'plan.json',BASE_DIR/'plan.sha256',DESIGN/'head_draft/head_confirmed.json',
                     DESIGN/'twopass/maps.json',DESIGN/'download.json'):
            manifest[str(path)]=sha(path)
        name='R6'
    candidates=[]
    for seed,folder in bases.items():
        record=checked_record(folder/'run.json',manifest)
        require(record['valid'] and record['completed'] and record['seed']==seed,'C0 BASE가 미완료입니다.')
        require(canonical(record['prompt'])==canonical(asdict(inputs.prompt)),'C0와 입력 프롬프트가 다릅니다.')
        require(sha(folder/'raw.png')==record['raw_sha256'],'C0 BASE SHA 불일치')
        manifest[str(folder/'raw.png')]=record['raw_sha256']
        candidates.append(OnePassCandidate(seed,folder/'raw.png',folder/'run.json',record))
        c0=checked_record(comparisons[seed].parent/'run.json',manifest)
        require(c0['valid'] and c0['completed'] and sha(comparisons[seed])==c0['raw_sha256'],'C0 CONTOUR SHA 불일치')
        manifest[str(comparisons[seed])]=c0['raw_sha256']
    for path,digest in ((inputs.control_file,inputs.control_sha256),(inputs.face_file,inputs.face_sha256),
        (options.head.normalized_file,options.head.normalized_sha256),(options.head.mask_file,options.head.mask_sha256),
        (options.head.contour_file,options.head.contour_sha256)):
        require(sha(path)==digest,'원본 입력 변경');manifest[str(path)]=digest
    return inputs,settings,options,seeds,candidates,expected,manifest,nose,name,comparisons


def prepare_case(case,directory,segmenter_factory=None,foreground_factory=AnimeForeground,*,face_method="skin_color_v2",
                 hair_confirmation=None):
    inputs,settings,options,seeds,bases,expected,sources,nose,name,comparisons=case
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    state={'status':'started','gpu_generation':False,'case':name}
    try:
        mask,contour,models=validate_proportion_request(inputs,settings,options)
        if face_method == "skin_color_v2":
            nose = ensure_cheek_observation(nose, options.head, directory)
        prepare_head_lines(options.head,nose,directory/'head-lines',segmenter_factory,face_method=face_method,
                           hair_confirmation=hair_confirmation)
        record_file=directory/'head-lines/head-lines.json'
        baseline=directory/'baseline-maps';baseline.mkdir()
        base_maps=build_maps(bases,mask,contour,baseline,foreground_factory,options,lambda:False,expected)
        modes={}
        for mode in ('hair','all'):
            active=replace(options,head_lines=mode,head_lines_file=record_file,head_lines_sha256=sha(record_file))
            lines,details=checked_lines(active,(settings.width,settings.height),require_review=False)
            folder=directory/mode;folder.mkdir();maps={}
            line_record=read(record_file)
            rgb=checked_image(options.head.normalized_file,options.head.normalized_sha256,(736,1232),'RGB')
            hair=checked_image(line_record['files']['hair-mask']['path'],line_record['files']['hair-mask']['sha256'],(736,1232),'L')
            face=checked_image(line_record['files']['face']['path'],line_record['files']['face']['sha256'],(736,1232),'L')
            for seed in seeds:
                base=base_maps[seed]
                pixels=checked_image(base['path'],base['sha256'],(736,1232),'RGB')
                sketch,details=compose_head_lines(pixels,active,require_review=False)
                path=folder/f'sketch_{seed}.png';Image.fromarray(sketch).save(path)
                overlay=folder/f'preview_{seed}.png';save_overlay(rgb,hair,face,sketch[...,0],overlay)
                maps[seed]={**base,'path':str(path),'sha256':sha(path),'head_lines':details,
                            'preview_file':str(overlay),'preview_sha256':sha(overlay)}
            modes[mode]={'options':asdict(active),'maps':maps}
        files={str(p):sha(p) for p in directory.rglob('*') if p.is_file()}
        lock=json_value({'case':name,'inputs':asdict(inputs),'settings':asdict(settings),'seeds':seeds,'models':models,
            'modes':modes,'source_files':sources,'prepared_files':files,'C0':comparisons,
            'base_stage':'reused','new_generation_count':4,'code_sha256':{str(p):sha(p) for p in CODE_FILES}})
        verify_manifest(sources)
        write_json(directory/'preflight.json',lock)
        digest=sha(directory/'preflight.json');(directory/'preflight.sha256').write_text(digest+'\n')
        state.update(status='needs_user_review',review_sha256=digest)
        return lock,digest
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:write_json(directory/'status.json',state)


def restore_mode(saved):
    options=restore_options(saved)
    return replace(options,head_lines_file=Path(options.head_lines_file),head_lines_confirmed=True)


def run_prepared(prepared,review_sha,output,*,contour_factory=ProportionBackend,foreground_factory=AnimeForeground):
    prepared,output=Path(prepared),Path(output)
    require(review_sha is not None and sha(prepared/'preflight.json')==review_sha
            and (prepared/'preflight.sha256').read_text().strip()==review_sha,'사용자가 확인한 미리보기 잠금 SHA가 필요합니다.')
    lock=read(prepared/'preflight.json')
    verify_manifest({**lock['source_files'],**lock['prepared_files'],**lock['code_sha256']})
    inputs,settings=restore_request({'inputs':{'BASE':lock['inputs']},'settings':lock['settings']})
    require(set(lock['modes'])=={'hair','all'} and len(lock['seeds'])==2 and len(set(lock['seeds']))==2,'시험 조건·seed 수가 다릅니다.')
    options={mode:restore_mode(saved['options']) for mode,saved in lock['modes'].items()}
    for mode,opt in options.items():
        require(opt.head_lines==mode,'머리 선 모드 불일치')
        require(validate_proportion_request(inputs,settings,opt)[2]==lock['models'],'준비 이후 모델 변경')
    output.mkdir(parents=True,exist_ok=False)
    state={'status':'started','BASE':[],'CONTOUR':[],'products':[],'case':lock['case'],
           'seeds':lock['seeds'],'review_sha256':review_sha,'base_stage':'reused','conditions':{}}
    write_json(output/'preflight.json',lock)
    write_json(output/'preview-confirmation.json',{'review_sha256':review_sha,'explicit_cli_confirmation':True})
    try:
        for mode in ('hair','all'):
            folder=output/mode;folder.mkdir()
            maps={int(k):v for k,v in lock['modes'][mode]['maps'].items()}
            local={'status':'started','phase':'CONTOUR','BASE':[],'CONTOUR':[],'products':[],
                   'head_lines':mode,'seeds':lock['seeds'],'base_stage':'reused'}
            state['conditions'][mode]=local
            try:
                verify_manifest({**lock['source_files'],**lock['prepared_files'],**lock['code_sha256']})
                from genai_lab.shoulder_control import resolve_shoulder_inputs
                stage_inputs,_=resolve_shoulder_inputs(inputs,options[mode])  # 제품과 같은 어깨 보정 골격
                raw=generate_stage(stage_inputs,lock['seeds'],folder,settings,
                    lambda:contour_factory(settings,options[mode],maps),lambda:False,local,'CONTOUR',{})
                local['phase']='white_background'
                finish_candidates(raw,folder,foreground_factory,options[mode],lambda:False,local,lambda _:None)
                local.update(status='review_pending',phase='done')
            except BaseException as error:
                local.update(status='failed',error_type=type(error).__name__,error=str(error));raise
            finally:
                save_state(folder,local)
                write_json(folder/'run.json',json_value({**local,'maps':maps,'options':asdict(options[mode])}))
        state['status']='review_pending'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:write_json(output/'run.json',json_value(state));save_state(output,state)
    return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--r6',action='store_true');source.add_argument('--studio-run',type=Path)
    source.add_argument('--prepared',type=Path)
    parser.add_argument('--foreground-model',type=Path)
    parser.add_argument('--model-cache',type=Path,default=Path('D:/genai-cache/huggingface'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--face-method',choices=('skin_color_v1','skin_color_v2','sam2','confirmed_hair'),default='skin_color_v2')
    parser.add_argument('--hair-confirmation',type=Path,help='confirmed_hair용: 머리 붙이기에서 사용자가 확인한 머리카락 영역 폴더')
    parser.add_argument('--run',action='store_true');parser.add_argument('--review-sha')
    args=parser.parse_args(argv)
    require(not args.output.exists(),'새 출력 폴더만 사용할 수 있습니다.')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if args.run:
        require(args.prepared is not None and args.review_sha is not None,'먼저 CPU 미리보기를 확인하고 --prepared와 --review-sha를 지정하세요.')
        run_prepared(args.prepared,args.review_sha,args.output)
        return
    require(args.prepared is None and args.review_sha is None,'CPU 준비는 --r6 또는 --studio-run을 사용합니다.')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    factory=None
    if args.face_method=='sam2':
        from genai_lab.tail_complexity_worker import CpuTailSegmenter
        factory=lambda:CpuTailSegmenter(args.model_cache)
    case=load_case(args.studio_run,args.foreground_model)
    hair=None
    if args.face_method=='confirmed_hair':
        require(args.hair_confirmation is not None,'confirmed_hair에는 --hair-confirmation이 필요합니다.')
        folder=args.hair_confirmation.resolve();info=read(folder/'result.json')
        hair={'directory':str(folder)}
        # 실행 전 재검사 대상에 확인 기록과 H 파일을 넣는다.
        case[6].update({str(folder/'confirmation.json'):sha(folder/'confirmation.json'),
                        str(folder/'result.json'):sha(folder/'result.json'),str(info['H_file']):info['H_sha256']})
    else:
        require(args.hair_confirmation is None,'--hair-confirmation은 confirmed_hair에서만 씁니다.')
    _,digest=prepare_case(case,args.output,factory,face_method=args.face_method,hair_confirmation=hair)
    print(json.dumps({'status':'needs_user_review','gpu_generation':0,'review_sha256':digest,'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
