"""머리 붙이기 제품 재현: 기본 CPU 검사, --run을 붙여야 GPU 6건을 실행한다."""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from types import SimpleNamespace
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import require, sha, json_value
from genai_lab.qwen_record_io import write_json
from genai_lab.onepass_generation import OnePassCandidate
from genai_lab.proportion_generation import ProportionCandidate, ProportionBatch
from genai_lab.studio_generation import StudioRuntime
from genai_lab.studio_head_paste import read, selected_request, prepare_hair, confirm_hair, apply_head
from genai_lab.head_paste_rules import blend_result

TRIAL=ROOT/"outputs/head-hair-mask-confirm-20261010"
CASES=(("be2bacf58f2a41a884dd3ace5fc8b224",(2987514498599016057,2987514498599016058)),
       ("b70aacb6cfc740a685c904501b5b1dde",(2343252627519382913,2343252627519382914)),
       ("21bc423cd2a249c08f17efb4c52c3bce",(2945843103555359005,2945843103555359006)))


def trial_selection(run,seed):
    """기존 실행은 읽기만 한다. 제품의 후보·기록 검사 계약을 그대로 사용한다."""
    from genai_lab.studio_generation import StudioResults
    generation=run/"generation"
    record_path=generation/f"seed-{seed}"/"run.json"
    product=read(record_path)
    raw_path=generation/f"CONTOUR_{seed}"/"raw.png"
    raw_record_path=raw_path.parent/"run.json"
    raw_record=read(raw_record_path)
    raw=OnePassCandidate(seed,raw_path,raw_record_path,raw_record)
    candidate=ProportionCandidate(seed,raw,Path(product["product_file"]),record_path,product)
    # 초기화가 user-review.json을 쓰므로 읽기 전용 복원에서는 실행하지 않는다.
    result=StudioResults.__new__(StudioResults)
    result.batch=ProportionBatch(generation,(candidate,))
    result.two_pass=True; result.product_records=(product,); result.selected=0
    result.tail_edit=None; result.optional_finishing=None
    result.backgrounds=(None,); result.finishings=(None,)
    return result


def reference_entry(run,seed):
    return read(TRIAL/"paste"/run[:8]/"inputs.json")["per_seed"][str(seed)]


def cpu_check(runtime):
    confirmations=read(TRIAL/"out/user-confirmation.json")
    require(confirmations["confirmed"],"시험 머리 영역 승인 없음")
    completed=[]
    for run,seeds in CASES:
        folder=TRIAL/"paste"/run[:8]
        lock=read(folder/"inputs.json")
        H=TRIAL/"out"/run[:8]/"H.png"
        require(sha(H)==confirmations["H_sha256"][run[:8]],"승인된 시험 머리 영역 변경")
        for seed in seeds:
            request=selected_request(trial_selection(ROOT/"outputs/studio-runs"/run,seed),runtime)
            e=lock["per_seed"][str(seed)]
            for key in ("raw","init","mask","features","paste"):
                require(sha(e[key])==e[key+"_sha256"],"고정 자료 변경: "+key)
            raw=np.asarray(Image.open(e["raw"]).convert("RGB"))
            mask=np.asarray(Image.open(e["mask"]).convert("L")); features=np.asarray(Image.open(e["features"]).convert("L"))
            with Image.open(e["init"]) as image:
                result,checks=blend_result(raw,image,mask,features,lock["box"])
            expected=np.asarray(Image.open(e["paste"]).convert("RGB"))
            require(np.array_equal(result,expected),"CPU 붙이기 합성 불일치")
            completed.append(dict(run=run,seed=seed,checks=checks,cache_key=request["cache_key"]))
    return completed


def pixel_difference(actual,expected):
    with Image.open(actual) as a,Image.open(expected) as b:
        left=np.asarray(a.convert("RGB"),dtype=np.int16); right=np.asarray(b.convert("RGB"),dtype=np.int16)
    require(left.shape==right.shape,"GPU 재현 크기가 다릅니다")
    distance=np.abs(left-right)
    return dict(mean_absolute=float(distance.mean()),max_absolute=int(distance.max()),
                changed_pixel_ratio=float(np.any(distance>0,axis=2).mean()))


def replay(output,runtime):
    output.mkdir(parents=True,exist_ok=False)
    approved=read(TRIAL/"out/user-confirmation.json")
    state=dict(status="started",completed=[],reference_sha256=sha(TRIAL/"verify_reference.md"))
    write_json(output/"status.json",state)
    try:
        for run,seeds in CASES:
            for seed in seeds:
                request=selected_request(trial_selection(ROOT/"outputs/studio-runs"/run,seed),runtime)
                preview=prepare_hair(request,output/"hair-cache",runtime,progress=print)
                require(preview["H_sha256"]==approved["H_sha256"][run[:8]],"보정 머리 영역이 승인된 시험 영역과 다릅니다. 중단합니다.")
                if not preview.get("remembered"):
                    # 기존 사용자 승인 자료와 픽셀이 같을 때만 재현 시험의 확인을 가져온다.
                    confirm_hair(preview,True)
                    write_json(Path(preview["directory"])/"imported-approval.json",dict(source=str(TRIAL/"out/user-confirmation.json"),sha256=sha(TRIAL/"out/user-confirmation.json")))
                info=apply_head(request,preview,output/run[:8]/f"seed-{seed}",runtime,progress=print)
                folder=Path(info["directory"])
                e=reference_entry(run,seed)
                rec=read(folder/"paste-inputs.json")["per_seed"][str(seed)]
                for key in ("init","mask","features","paste"):
                    require(rec[key+"_sha256"]==e[key+"_sha256"],"CPU 단계 전체 재현 불일치: "+key)
                expected_gpu=TRIAL/("gpu-02" if run.startswith("21bc") else "gpu-01")/run[:8]/f"seed-{seed}"
                generated=Path(info["redraw_record"]).parent
                differences={name:pixel_difference(generated/name,expected_gpu/name) for name in ("head_1024.png","raw_redraw.png")}
                exact=all(sha(generated/name)==sha(expected_gpu/name) for name in differences)
                state["completed"].append(dict(run=run,seed=seed,sha_match=exact,difference=differences,directory=str(folder)))
                write_json(output/"status.json",state)
        state["status"]="sha_matched" if all(e["sha_match"] for e in state["completed"]) else "needs_blind_review"
    except BaseException as error:
        state.update(status="failed",error=str(error));raise
    finally:
        write_json(output/"status.json",json_value(state))
    print(state["status"],len(state["completed"]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/head-paste-product-verify-20261010/gpu-01")
    args=parser.parse_args()
    runtime=StudioRuntime.from_environment()
    checks=cpu_check(runtime)
    print(f"CPU 보호 합성·원본 입력 계약 {len(checks)}/6 통과",flush=True)
    if args.run: replay(args.output,runtime)
    else: print("GPU 실행 없음. H2·1024 붙이기 전체 재계산은 분할 고정 자료가 없어 GPU 재현에서 확인합니다.",flush=True)


if __name__=="__main__":
    main()
