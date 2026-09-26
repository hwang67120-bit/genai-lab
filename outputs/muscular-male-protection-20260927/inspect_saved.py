from pathlib import Path
import json,os,datetime,hashlib
R=Path(__file__).resolve().parents[2]
O=Path(__file__).parent
rows=[]
for p in Path(os.environ["TEMP"]).glob("genai-parts-*/parts.json"):
 try:d=json.loads(p.read_text(encoding="utf-8"))
 except Exception:continue
 parts=d.get("parts",{})
 if "output_hair" not in parts:continue
 if p.stat().st_mtime<1790400000:continue
 rows.append({"path":str(p),"time":datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),"timestamp":p.stat().st_mtime,"size":d.get("image_size"),"parts":{k:{x:v.get(x) for x in ["boxes","detection_scores","mask_quality_scores","accepted_boxes","mask_pixels","mask_path","status"]} for k,v in parts.items()}})
rows.sort(key=lambda x:x["timestamp"])
(O/"saved-detection-index.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
for d in rows:print(d["time"],d["path"],d["size"],[(k,v.get("mask_pixels")) for k,v in d["parts"].items()])
for c in ["muscular-male","ordinary-female"]:
 for g in ["ferrari","partial","minimal"]:
  p=R/f"outputs/generalization-text-20260926/cases/{c}/{g}/off/actual-calls.json"
  if not p.exists():continue
  d=json.loads(p.read_text(encoding="utf-8"));print("CALL",c,g,str(d)[:500])
