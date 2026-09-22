from __future__ import annotations
import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path
ARCHIVES = {
 "makehuman-community-1.3.0-windows.zip": ("https://files2.makehumancommunity.org/releases/makehuman-community-1.3.0-windows.zip","authoring_tool","mixed_terms"),
 "makehuman_system_assets_cc0.zip": ("https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip","body_hair_sources","CC0"),
 "hair01_cc0.zip": ("https://files2.makehumancommunity.org/asset_packs/hair01/hair01_cc0.zip","hair_sources","CC0"),
 "haireditor_cc0.zip": ("https://files2.makehumancommunity.org/functional/haireditor.zip","hair_authoring","CC0")}
COMPOUND={"bangs","braid","bun","headband","messy","ponytail","shaggy"}
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()
def archive_record(path):
 url,role,license_scope=ARCHIVES[path.name]
 with zipfile.ZipFile(path) as z:
  broken=z.testzip()
  if broken: raise ValueError(f"corrupt zip entry: {path.name}:{broken}")
  return {"name":path.name,"url":url,"role":role,"license_scope":license_scope,"bytes":path.stat().st_size,"sha256":sha256(path),"zip_entry_count":len(z.infolist()),"zip_valid":True}
def decision(asset_id,asset_type,description):
 text=f"{asset_id} {description}".lower()
 if asset_type=="proxymeshes": return "reference_only","Use one parametric base mesh; proxy topology must not define a body class."
 found=sorted(term for term in COMPOUND if term in text)
 if found: return "deferred_compound_style","Deferred from v1: "+", ".join(found)
 return "authoring_source_only","Create a controlled matched render series before training."
def collect_assets(cache,metadata_dir):
 packs=[("makehuman_system_assets_cc0.zip","packs/makehuman_system_assets.json","makehuman_system_assets"),("hair01_cc0.zip","packs/hair01.json","hair01")]
 records=[]; metadata_dir.mkdir(parents=True,exist_ok=True)
 for archive_name,entry,pack in packs:
  with zipfile.ZipFile(cache/archive_name) as z: raw=json.loads(z.read(entry).decode("utf-8"))
  (metadata_dir/f"{pack}.json").write_text(json.dumps(raw,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
  for asset_id,value in sorted(raw.items()):
   asset_type=str(value.get("type",""))
   system_ok=pack=="makehuman_system_assets" and asset_type in {"hair","proxymeshes"}
   hair_ok=pack=="hair01" and str(value.get("category","")).lower()=="hair"
   if not(system_ok or hair_ok): continue
   state,reason=decision(asset_id,asset_type,str(value.get("description","")))
   records.append({"asset_id":asset_id,"pack":pack,"type":asset_type,"description":value.get("description",""),"author":value.get("author",""),"license":value.get("license",""),"source":value.get("source",""),"decision":state,"decision_reason":reason,"training_ready":False})
 return records
def contact_sheet(cache,output):
 try: from PIL import Image,ImageDraw,ImageOps
 except ImportError: return {"created":False,"reason":"Pillow is not installed"}
 chosen={}
 for archive_name,pack in (("makehuman_system_assets_cc0.zip","system"),("hair01_cc0.zip","hair01")):
  with zipfile.ZipFile(cache/archive_name) as z:
   for entry in z.infolist():
    parts=entry.filename.split("/")
    if len(parts)<3 or parts[0] not in {"hair","proxymeshes"} or entry.file_size<=0 or not entry.filename.lower().endswith((".thumb",".png")): continue
    key=(pack,parts[1]); score=2 if entry.filename.lower().endswith(".thumb") else 1
    if key not in chosen or score>chosen[key][0]: chosen[key]=(score,z.read(entry))
 tiles=[]
 for (pack,asset_id),(_,payload) in sorted(chosen.items()):
  try:
   with Image.open(io.BytesIO(payload)) as im: tiles.append((f"{pack}/{asset_id}",im.convert("RGB").copy()))
  except Exception: pass
 if not tiles: return {"created":False,"reason":"No readable thumbnails"}
 tw,th,lh,cols=240,240,46,5; rows=(len(tiles)+cols-1)//cols
 sheet=Image.new("RGB",(cols*tw,rows*(th+lh)),"white"); draw=ImageDraw.Draw(sheet)
 for i,(label,im) in enumerate(tiles):
  x,y=(i%cols)*tw,(i//cols)*(th+lh); fit=ImageOps.contain(im,(tw-12,th-12))
  sheet.paste(fit,(x+(tw-fit.width)//2,y+(th-fit.height)//2)); draw.text((x+6,y+th+4),label[:38],fill="black")
 output.parent.mkdir(parents=True,exist_ok=True); sheet.save(output)
 return {"created":True,"path":str(output),"asset_count":len(tiles),"sha256":sha256(output)}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--cache-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
 cache,output=a.cache_dir.resolve(),a.output_dir.resolve(); missing=[n for n in ARCHIVES if not(cache/n).is_file()]
 if missing: raise FileNotFoundError(f"missing archives: {missing}")
 output.mkdir(parents=True,exist_ok=True); archives=[archive_record(cache/n) for n in ARCHIVES]; assets=collect_assets(cache,output/"upstream_metadata")
 if not assets or any(x["license"]!="CC0" for x in assets): raise ValueError("Body/hair inventory contains missing or non-CC0 license")
 preview=contact_sheet(cache,output/"previews"/"source_contact_sheet.png")
 manifest={"version":"body_hair_cc0_source_inventory_v1","policy":{"training_input_requires_self_render":True,"one_target_axis_per_run":True,"upstream_assets_are_not_training_images":True,"unknown_source_sets_included":False},"cache_dir":str(cache),"archives":archives,"asset_count":len(assets),"assets":assets,"contact_sheet":preview}
 (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 with (output/"asset_inventory.csv").open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(assets[0])); w.writeheader(); w.writerows(assets)
 print(json.dumps({"archives":len(archives),"assets":len(assets),"contact_sheet":preview},ensure_ascii=False))
if __name__=="__main__": main()

