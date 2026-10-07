"""Run once on Windows CPU against existing saved masks; no model imports."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
import sys
sys.dont_write_bytecode=True
import hashlib,json,time,platform,threading,unittest
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import psutil
from component_probe import collect_components,connection_candidates,inside_box

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from genai_lab.tail_complexity import keep_parts,centerline_v2

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def source_for(case):
    paths=list((ROOT/"outputs/suin-tail-test-20261006/inputs").glob(case+"_*"))
    if len(paths)!=1:
        raise ValueError(f"ambiguous source: {case} {len(paths)}")
    path=paths[0]
    if "143960292_19" in path.name or "0e0f64558c7c6f84b97b43b5b60e693e" in path.name:
        raise ValueError("blocked input")
    return path

def draw_dashed(draw,a,b):
    a,b=np.array(a,float),np.array(b,float)
    length=np.linalg.norm(b-a)
    for start in np.arange(0,length,12):
        end=min(start+6,length)
        draw.line([tuple(a+(b-a)*start/length),tuple(a+(b-a)*end/length)],fill=(245,130,0),width=3)

def make_overlay(rgb,mask,box,baseline,components,links,truth,case,out):
    rgb=rgb.copy()
    rgb[mask]=(rgb[mask]*.72+np.array([20,230,120])*.28).astype(np.uint8)
    panels=[Image.fromarray(rgb),Image.fromarray(rgb)]
    old=ImageDraw.Draw(panels[0])
    if baseline is not None:
        old.line([tuple(p) for p in baseline["fine"]],fill=(200,0,190),width=3)
        for x,y in [baseline["fine"][0],baseline["fine"][-1]]:
            old.ellipse([x-5,y-5,x+5,y+5],fill=(200,0,190))
    draw=ImageDraw.Draw(panels[1])
    colors=[(230,30,30),(20,70,240),(180,20,170),(100,80,0)]
    for i,part in enumerate(components):
        color=colors[i%len(colors)]
        for xy in part["skeleton_xy"]:
            draw.point(tuple(xy),fill=color)
        for endpoint in part["endpoints"]:
            x,y=endpoint["xy"]; draw.ellipse([x-5,y-5,x+5,y+5],outline=color,width=3)
    for link in links:
        draw_dashed(draw,link["a"],link["b"])
    for panel in panels:
        if "box" in truth:
            ImageDraw.Draw(panel).rectangle(truth["box"],outline=(0,180,190),width=3)
    panels=[panel.crop(box) for panel in panels]
    factor=min(1.4,520/panels[0].width,500/panels[0].height)
    panels=[panel.resize((round(panel.width*factor),round(panel.height*factor))) for panel in panels]
    sheet=Image.new("RGB",(2*panels[0].width+20,panels[0].height+65),"white")
    sheet.paste(panels[0],(0,60)); sheet.paste(panels[1],(panels[0].width+20,60))
    d=ImageDraw.Draw(sheet)
    d.text((5,5),case+" | largest component only        | all visible component endpoints",fill="black")
    d.text((5,24),"cyan box: existing tip truth; orange dashed: geometry ONLY, not a confirmed tail",fill="black")
    sheet.save(out/(case+".png"))
    return sheet

def evaluate_case(case,reg,old_t1b,out):
    cached=ROOT/"outputs/tail-complexity-claude-verify-20261006/replay/original"/case
    record_path=cached/"comparison.json"
    recorded=json.loads(record_path.read_text(encoding="utf-8"))
    source=source_for(case); mask_path=cached/"mask.png"
    source_sha=sha(source)
    if source_sha!=recorded["source_sha256"]:
        raise ValueError("source SHA mismatch: "+case)
    image=Image.open(source).convert("RGB"); rgb=np.array(image)
    mask=np.array(Image.open(mask_path).convert("L"))>0
    if mask.shape!=rgb.shape[:2]:
        raise ValueError("mask coordinate mismatch: "+case)
    box=reg["user_tail_box"][case]
    if list(recorded["actual"]["box"])!=box:
        raise ValueError("box mismatch: "+case)
    if int(mask.sum())!=recorded["actual"]["mask_px"]:
        raise ValueError("mask count mismatch: "+case)
    before=mask.copy()
    _,main=keep_parts(mask)
    baseline=centerline_v2(main)
    components=collect_components(mask)
    links=connection_candidates(components)
    np.testing.assert_array_equal(mask,before)
    endpoints=[e["xy"] for c in components for e in c["endpoints"]]
    oldpoints=[] if baseline is None else [baseline["fine"][0].tolist(),baseline["fine"][-1].tolist()]
    truth=reg["tip_location_truth"][case]
    hit=lambda points: any(inside_box(p,truth["box"]) for p in points) if "box" in truth else None
    prior=old_t1b.get("auto",{}).get(case,{})
    row=dict(case=case,source=str(source),source_sha256=source_sha,
        mask_sha256=sha(mask_path),mask_source=str(mask_path),mask_px=int(mask.sum()),
        prior_t1b_mask_px=prior.get("mask_px"),same_t1b_pixel_count=prior.get("mask_px")==int(mask.sum()),
        comparison_sha256=sha(record_path),box=box,truth=truth,
        baseline_endpoints=oldpoints,baseline_candidate_hit=hit(oldpoints),
        all_candidate_hit=hit(endpoints),components=components,links=links,
        endpoint_count=len(endpoints),component_count=len(components),
        final_tip=None,final_tip_status="not_selected",mask_unchanged=True)
    sheet=make_overlay(rgb,mask,box,baseline,components,links,truth,case,out)
    return row,sheet

def main():
    started=time.perf_counter()
    baseline=json.loads((HERE/"before.json").read_text(encoding="utf-8"))
    for rel,digest in baseline.items():
        if sha(ROOT/rel)!=digest: raise ValueError("pre-run source change: "+rel)
    protocol_sha=sha(HERE/"criteria.md")
    lock=json.loads((HERE/"protocol_lock.json").read_text(encoding="utf-8"))
    for name,digest in lock.items():
        if sha(HERE/name)!=digest: raise ValueError("protocol mismatch: "+name)
    out=HERE/"run"; out.mkdir(exist_ok=False)
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern="test_component_probe.py")
    with (out/"tests.txt").open("w",encoding="utf-8") as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not tests.wasSuccessful(): raise RuntimeError("CPU unit tests failed")
    proc=psutil.Process(); peak=[proc.memory_info().rss]; stopped=threading.Event()
    def sample():
        while not stopped.wait(.05): peak[0]=max(peak[0],proc.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    reg=json.loads((ROOT/"outputs/recognition-detail-design/t1b/regions_t1b.json").read_text(encoding="utf-8"))
    old=json.loads((ROOT/"outputs/recognition-detail-design/t1b/run/t1b_results.json").read_text(encoding="utf-8"))
    rows=[];panels=[]
    try:
        for case in ("S03","S04","S05","S06","S07","S17"):
            tick=time.perf_counter()
            row,panel=evaluate_case(case,reg,old,out)
            row["seconds"]=time.perf_counter()-tick
            rows.append(row);panels.append(panel)
            write_json(out/(case+".json"),row)
            print(case,"components",row["component_count"],"candidates",row["endpoint_count"],
                  "old/new",row["baseline_candidate_hit"],row["all_candidate_hit"],
                  "geometry links",len(row["links"]),flush=True)
    finally:
        stopped.set();monitor.join()
    merged=Image.new("RGB",(max(p.width for p in panels),sum(p.height for p in panels)+10*(len(panels)-1)),"white")
    y=0
    for panel in panels: merged.paste(panel,(0,y));y+=panel.height+10
    merged.save(out/"sheet.png")
    changed=[rel for rel,digest in baseline.items() if sha(ROOT/rel)!=digest]
    info=proc.memory_info()
    report=dict(rows=rows,tests_run=tests.testsRun,tests_passed=tests.wasSuccessful(),
        seconds=time.perf_counter()-started,peak_ram_bytes=max(peak[0],info.rss,getattr(info,"peak_wset",0)),
        protected_files_checked=len(baseline),protected_files_changed=changed,
        protocol_sha256=protocol_sha,os=platform.platform(),python=sys.version,
        model_inference_count=0,image_generation_count=0)
    write_json(out/"results.json",report)
    lines=["# 끊긴 꼬리 후보 보존 진단 결과","",
        "기존 마스크 재사용 / 모델 추론 0회 / 생성 0회 / 운영 미반영.",
        "기존 중심선과 모든 조각 끝점의 정답 상자 포함을 비교한다. 자동 말단 선택 성공률이 아니다.","",
        "| 입력 | 조각 | 끝점 후보 | 기존 후보 포함 | 모든 조각 후보 포함 | 미확인 연결 후보 |",
        "|---|---:|---:|---|---|---:|"]
    for r in rows:
        yn=lambda v:"해당 없음" if v is None else "예" if v else "아니오"
        lines.append(f"| {r['case']} | {r['component_count']} | {r['endpoint_count']} | {yn(r['baseline_candidate_hit'])} | {yn(r['all_candidate_hit'])} | {len(r['links'])} |")
    lines += ["",f"CPU 단위 테스트 {tests.testsRun}개 통과. 처리 {report['seconds']:.2f}초, 프로세스 최대 RAM {report['peak_ram_bytes']/2**30:.3f} GiB.",
        f"보호 파일 {len(baseline)}개 SHA 비교: 변경 {len(changed)}개.",
        "", "## 해석 한계",
        "- 이전 재현의 저장 마스크다. T1b 원시 마스크와 바이트 동일은 확인 불가. mask_px 비교는 개별 JSON에 기록.",
        "- 이미 본 6장에 대한 진단. 조각 연결은 거리·방향만의 후보이며 가림 물체의 의미나 동일 꼬리 여부를 검증하지 못했다.",
        "- 최종 말단을 자동 선택하지 않는다. 특히 여러 갈래 털끝과 솜꼬리에서 사용자 판단을 대체하지 않는다.",
        "- 3D 복원, Qwen 입력 변경, 색 견본 24곳 판정, 생성 개선은 실행하지 않았다.",
        "- 임계값이나 정답은 결과 확인 후 수정하지 않았다."]
    (HERE/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("DONE",report["seconds"],"seconds;",len(changed),"protected files changed",flush=True)

if __name__=="__main__":
    main()
