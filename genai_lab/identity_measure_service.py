"""독립적인 CPU 작업을 실행한다. 생성 작업을 점유하거나 CUDA를 로드하지 않는다."""
import json
from pathlib import Path
from PIL import Image
from genai_lab.proportion_inputs import sha, require
from genai_lab.studio_generation import cpu_process, save_analysis_reference


def measure_image(source, directory, runtime, *, cancelled=lambda:False):
    source=Path(source); directory=Path(directory)
    digest=sha(source);directory.mkdir(parents=True,exist_ok=False)
    with Image.open(source) as image:
        save_analysis_reference(image,directory/'character.png')
    worker=Path(__file__).with_name('studio_analysis.py')
    cpu_process([str(runtime.pose_python),str(worker),'pose',str(directory),str(runtime.pose_models),
        str(runtime.model_cache),str(runtime.head_cache)], directory/'pose.log',cancelled,runtime.analysis_timeout)
    worker=Path(__file__).with_name('identity_measure_worker.py')
    cpu_process([str(runtime.analysis_python),str(worker),str(directory),str(runtime.model_cache),str(runtime.head_cache)],
        directory/'measure.log',cancelled,runtime.analysis_timeout)
    require(sha(source)==digest,'측정 도중 입력 이미지가 변경됐습니다.')
    return json.loads((directory/'measurement.json').read_text(encoding='utf-8'))


def measure_pair(original, result, directory, runtime, *, cancelled=lambda:False):
    from genai_lab.identity_report import write_report
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    original_sha,result_sha=sha(original),sha(result)
    measure_image(original,directory/'original',runtime,cancelled=cancelled)
    measure_image(result,directory/'result',runtime,cancelled=cancelled)
    require(sha(original)==original_sha and sha(result)==result_sha,'측정 기준 이미지가 변경됐습니다.')
    report=write_report(directory/'original',directory/'result',directory/'report')
    report['source_original_sha256']=original_sha;report['source_result_sha256']=result_sha
    from genai_lab.qwen_record_io import write_json
    path=directory/'report/identity-report.json';write_json(path,report)
    return {'path':str(path),'sha256':sha(path),'result_sha256':result_sha,'report':report}
