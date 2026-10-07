"""S06 SAM2 GPU forward once; compare with recorded CPU kept mask, no generation."""
import os,sys
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
sys.dont_write_bytecode=True
import argparse,json,hashlib,time,platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from component_probe import collect_components,connection_candidates,inside_box

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache-dir",required=True)
    args=parser.parse_args()
    lock=json.loads((HERE/"gpu_protocol_lock.json").read_text(encoding="utf-8"))
    for name,digest in lock.items():
        if sha(HERE/name)!=digest: raise ValueError("GPU protocol changed: "+name)
    out=HERE/"gpu_run";out.mkdir(exist_ok=False)
    start=time.perf_counter()
    result=dict(status="failed",model_forward_calls=0,image_generation_count=0,
                gpu_protocol_sha256=sha(HERE/"gpu_criteria.md"))
    torch=None
    try:
        import torch,transformers
        from huggingface_hub import snapshot_download
        from transformers import Sam2Processor
        from genai_lab.clothing_reference import load_sam2_image_model
        from genai_lab.tail_complexity import keep_parts
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        device="cuda:0"
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        result.update(torch=torch.__version__,transformers=transformers.__version__,
            device=torch.cuda.get_device_name(0),python=platform.python_version(),
            cuda_runtime=torch.version.cuda,dtype="float32",
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32)
        record=json.loads((ROOT/"outputs/tail-complexity-claude-verify-20261006/replay/original/S06/comparison.json").read_text(encoding="utf-8"))
        source=ROOT/"outputs/suin-tail-test-20261006/inputs"/Path(record["source"].replace("\\","/")).name
        if sha(source)!=record["source_sha256"]: raise ValueError("S06 source changed")
        cpu_path=ROOT/"outputs/tail-complexity-claude-verify-20261006/replay/original/S06/mask.png"
        cpu=np.array(Image.open(cpu_path).convert("L"))>0
        expected=json.loads((HERE/"run/S06.json").read_text(encoding="utf-8"))
        if sha(cpu_path)!=expected["mask_sha256"]: raise ValueError("CPU mask changed")
        reg=json.loads((ROOT/"outputs/recognition-detail-design/t1b/regions_t1b.json").read_text(encoding="utf-8"))
        box=reg["user_tail_box"]["S06"]; truth=reg["tip_location_truth"]["S06"]["box"]
        if box!=record["actual"]["box"]: raise ValueError("box changed")
        image=Image.open(source).convert("RGB")
        result.update(source_sha256=sha(source),cpu_mask_sha256=sha(cpu_path),box=box,size=list(image.size))
        tick=time.perf_counter()
        snapshot=snapshot_download("facebook/sam2.1-hiera-tiny",cache_dir=args.cache_dir,local_files_only=True)
        processor=Sam2Processor.from_pretrained(snapshot,local_files_only=True)
        model,_,_=load_sam2_image_model(SimpleNamespace(model_id=snapshot,cache_dir=args.cache_dir),device)
        model.eval()
        torch.cuda.synchronize()
        result.update(load_seconds=time.perf_counter()-tick,model_id="facebook/sam2.1-hiera-tiny",
            model_revision=Path(snapshot).name)
        inputs=processor(images=image,input_boxes=[[box]],return_tensors="pt")
        original_sizes=inputs["original_sizes"].cpu()
        inputs={k:v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
        tick=time.perf_counter()
        with torch.inference_mode():
            result["model_forward_calls"]+=1
            output=model(**inputs,multimask_output=False)
            torch.cuda.synchronize()
            result["forward_seconds"]=time.perf_counter()-tick
            masks=processor.post_process_masks(output.pred_masks.cpu(),original_sizes,binarize=True)[0]
        raw=np.asarray(masks).reshape(-1,image.height,image.width)[0].astype(bool)
        Image.fromarray(raw.astype(np.uint8)*255).save(out/"raw_mask.png")
        x0,y0,x1,y1=box
        clipped=raw.copy();clipped[:y0]=False;clipped[y1:]=False;clipped[:,:x0]=False;clipped[:,x1:]=False
        gpu,_=keep_parts(clipped)
        Image.fromarray(gpu.astype(np.uint8)*255).save(out/"kept_mask.png")
        before=gpu.copy()
        components=collect_components(gpu);links=connection_candidates(components)
        np.testing.assert_array_equal(gpu,before)
        points=[e["xy"] for c in components for e in c["endpoints"]]
        cpu_points=[e["xy"] for c in expected["components"] for e in c["endpoints"]]
        changed=int(np.count_nonzero(cpu!=gpu))
        union=int(np.count_nonzero(cpu|gpu))
        result.update(status="completed",mask_pixels=int(gpu.sum()),cpu_mask_pixels=int(cpu.sum()),
            differing_pixels=changed,mask_iou=float(np.count_nonzero(cpu&gpu)/union) if union else 1.,
            mask_pixels_equal=changed==0,kept_mask_sha256=sha(out/"kept_mask.png"),
            component_count=len(components),endpoint_count=len(points),
            endpoint_coordinates_equal=sorted(points)==sorted(cpu_points),
            tip_candidate_in_truth=any(inside_box(p,truth) for p in points),
            final_tip_selected=False,components=components,geometry_links=links,
            sam_iou_score=float(output.iou_scores.cpu().numpy().ravel()[0]))
        rgb=np.array(image); panels=[]
        for mask in (cpu,gpu,cpu!=gpu):
            px=rgb.copy();px[mask]=(px[mask]*.45+np.array([255,70,30])*.55).astype(np.uint8)
            panels.append(Image.fromarray(px).crop(box))
        sheet=Image.new("RGB",(panels[0].width*3,panels[0].height+35),"white")
        draw=ImageDraw.Draw(sheet)
        for i,p in enumerate(panels):
            sheet.paste(p,(i*p.width,35))
            draw.text((i*p.width+5,5),["CPU saved","GPU once","pixel difference"][i],fill="black")
        sheet.save(out/"comparison.png")
    except Exception as exc:
        result.update(status="failed",error=repr(exc))
        raise
    finally:
        if torch is not None and torch.cuda.is_available():
            result.update(allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                          reserved_peak_bytes=torch.cuda.max_memory_reserved())
        result["total_seconds"]=time.perf_counter()-start
        (out/"run.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k not in ("components","geometry_links")},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
