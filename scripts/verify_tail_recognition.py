"""Validate local recognition assets or inspect an existing tail selection; never edit an image."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from genai_lab.qwen_tail_edit import TailEditSpec
from genai_lab.tail_recognition import RecognitionSettings, settings_path, validate_model, check_versions, recognize_tail


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default=str(settings_path()))
    parser.add_argument("--selection",help="selection.json or replay preflight.json with a spec")
    parser.add_argument("--output")
    parser.add_argument("--recognize",action="store_true",help="Run visual recognition only; no Qwen image editing")
    parser.add_argument("--cpu",action="store_true")
    args=parser.parse_args()
    settings=RecognitionSettings.load(args.settings)
    if args.cpu: settings=replace(settings,device="cpu",timeout_seconds=900)
    if not args.recognize:
        manifest=validate_model(settings)
        print(json.dumps({"versions":check_versions(),"files_verified":len(manifest["files"]),
                          "revision":settings.model_revision,"inference":False},indent=2))
        return
    if not args.selection or not args.output: parser.error("--recognize requires --selection and --output")
    payload=json.loads(Path(args.selection).read_text(encoding="utf-8"))
    spec=TailEditSpec(**payload["spec"])
    report=recognize_tail(spec,settings,Path(args.output),progress=lambda text:print(text,flush=True))
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=="__main__": main()
