"""고정된 시제품 픽셀·자세를 제품 측정과 CPU로 비교한다."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import shutil
import numpy as np
from genai_lab.identity_measure_worker import measure_folder
from genai_lab.studio_generation import StudioRuntime
from genai_lab.qwen_record_io import write_json
from genai_lab.proportion_inputs import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ids',nargs='+',default=['orig_R6','R6_twopass_1','orig_wolf'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/identity-measure-design-20261009'
    expected=json.loads((source/'measurements_v2.json').read_text(encoding='utf-8'))
    runtime=StudioRuntime.from_environment();records=[]
    for iid in args.ids:
        old=source/'pose'/iid;dest=args.output/iid;dest.mkdir()
        for name in ('character.png','pose.json'):shutil.copyfile(old/name,dest/name)
        measure_folder(dest,runtime.model_cache,runtime.head_cache)
        actual=json.loads((dest/'measurement.json').read_text(encoding='utf-8'))
        keys=('B1','B2','B3','B4','B5','B6','B7','B8','head_box','H','skin_lab','hair_skin_dE')
        errors={k:{'old':expected[iid].get(k),'new':actual.get(k)} for k in keys if actual.get(k)!=expected[iid].get(k)}
        oldq,newq=old/'hair_quant_v2.npy',dest/'hair-quantiles.npy'
        quantile_match=oldq.exists()==newq.exists() and (not oldq.exists() or np.array_equal(np.load(oldq),np.load(newq)))
        records.append({'id':iid,'input_sha256':sha(old/'character.png'),'errors':errors,'quantiles_equal':quantile_match})
        write_json(args.output/'verification.json',{'passed':all(not r['errors'] and r['quantiles_equal'] for r in records),'cases':records})
        print(iid,errors,quantile_match,flush=True)
    if any(r['errors'] or not r['quantiles_equal'] for r in records):raise SystemExit(1)

if __name__=='__main__':main()
