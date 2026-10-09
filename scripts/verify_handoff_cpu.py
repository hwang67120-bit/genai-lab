"""시험 근거를 변경하지 않고 2026-10-09 인수인계 항목을 CPU로 재현한다."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='', ONNX_MODE='cpu', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import sha, json_value
from genai_lab.qwen_record_io import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def module_at(path):
    spec=importlib.util.spec_from_file_location('frozen_shoulder_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_shoulders(directory):
    from genai_lab.shoulder_control import prepare_shoulder_control
    reference=ROOT/'outputs/shoulder-skeleton-c2-wolf-20261009/common.py'
    loader=module_at(reference)
    expected={'R6':'3b9372b2a1a8a1eec6b9f43ec52feed8b00b82c1bf14227813ae960a48725289',
              'wolf':'ac215e048e592af63284f90c7c2024979122fd199bb2a0c50f7a61529f613f81'}
    rows=[]
    for name,digest in expected.items():
        inputs,settings,options,joints=loader.load_case(name)
        record=prepare_shoulder_control(inputs.control_file,inputs.control_sha256,joints,options.head,directory/name)
        rows.append({'case':name,'matched':record['status']=='applied' and record['control_sha256']==digest,
                     'expected':digest,'actual':record['control_sha256'],'status':record['status'], 'warning':record.get('warning')})
    return {'reference_code_sha256':sha(reference),'rows':rows,'passed':all(row['matched'] for row in rows)}


def verify_arms(directory):
    from genai_lab.identity_measure_worker import measure_folder
    from genai_lab.studio_generation import StudioRuntime
    source=ROOT/'outputs/identity-measure-design-20261009'
    expected_path=ROOT/'outputs/torso-arm-fix-20261009/results.json'
    expected=read(expected_path);old=read(source/'measurements_v2.json')
    runtime=StudioRuntime.from_environment();measurements={};retained=[]
    for iid in dict.fromkeys(row['id'] for row in expected['rows']):
        dest=directory/iid;dest.mkdir(parents=True)
        for filename in ('character.png','pose.json'):shutil.copyfile(source/'pose'/iid/filename,dest/filename)
        measure_folder(dest,runtime.model_cache,runtime.head_cache)
        actual=read(dest/'measurement.json');measurements[iid]=actual
        # 새 유효성 항목은 바뀔 수 있지만 기존 관찰 수치는 바뀌면 안 된다.
        errors=[]
        for key in ('B1','B2','B3','B4','B5','B6','B7','B8','H','skin_lab','hair_skin_dE'):
            a,b=actual.get(key),old[iid].get(key)
            if key in ('B3','B4','B5') and isinstance(a,dict):
                a={k:v for k,v in a.items() if k not in ('touch','invalid')}
                b={k:v for k,v in b.items() if k!='invalid'} if isinstance(b,dict) else b
            if a!=b:errors.append(key)
        retained.append({'id':iid,'numeric_errors':errors})
        print('arm',iid,errors,flush=True)
    rows=[]
    for row in expected['rows']:
        value=measurements[row['id']].get(row['metric'])
        status='no_axis' if value is None else value.get('invalid','valid')
        rows.append({'id':row['id'],'metric':row['metric'],'expected':row['status'],'actual':status,'matched':status==row['status']})
    return {'reference_sha256':sha(expected_path),'rows':rows,'numeric_checks':retained,
            'passed':all(row['matched'] for row in rows) and all(not row['numeric_errors'] for row in retained)}


def verify_head(directory):
    from genai_lab.head_lines import prepare_head_lines
    from scripts.head_lines_trial import ensure_cheek_observation, nose_from
    loader=module_at(ROOT/'outputs/shoulder-skeleton-c2-wolf-20261009/common.py')
    inputs,settings,options,joints=loader.load_case('R6')
    directory.mkdir()
    nose=ensure_cheek_observation(nose_from(joints),options.head,directory)
    results={mode:prepare_head_lines(options.head,nose,directory/mode,face_method=mode)
             for mode in ('skin_color_v1','skin_color_v2')}
    before,after=results.values()
    return {'status':after['status'],'old_face_ratio':before['face_area_ratio'],'new_face_ratio':after['face_area_ratio'],
        'all_lines_equal':before['files']['all']['sha256']==after['files']['all']['sha256'],
        'hair_lines_equal':before['files']['hair']['sha256']==after['files']['hair']['sha256'],
        'old_face_sha256':before['files']['face']['sha256'],'new_face_sha256':after['files']['face']['sha256'],
        'passed':after['status']=='needs_user_review' and before['files']['all']['sha256']==after['files']['all']['sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checks',nargs='+',choices=('shoulders','arms','head'),default=['shoulders','arms','head'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'gpu_generation':0,'checks':{}}
    for name in args.checks:
        try:report['checks'][name]={'shoulders':verify_shoulders,'arms':verify_arms,'head':verify_head}[name](args.output/name)
        except Exception as error:
            report['checks'][name]={'passed':False,'error':str(error)}
            import traceback;traceback.print_exc()
        write_json(args.output/'verification.json',json_value(report))
        print(name,report['checks'][name]['passed'],flush=True)
    if not all(check['passed'] for check in report['checks'].values()):raise SystemExit(1)


if __name__=='__main__':main()
