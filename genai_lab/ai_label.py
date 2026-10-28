"""저장 결과의 AI 생성 표시(D-152, 2026-10-22 사용자 "AI 생성 표시는 보류하지 말고 파일 정보에 표시").

픽셀은 한 점도 바꾸지 않고 PNG 파일 정보만 더한다(그림을 해치지 않는 표시).
- PNG 텍스트: Software = GenAI Lab, AI-Generated = true, Description(한국어 안내)
- XMP(IPTC 디지털 출처 유형): 전부 생성 = trainedAlgorithmicMedia, 원본 픽셀이 섞인 합성(옷 바꾸기의 원본 머리 붙이기) =
  compositeWithTrainedAlgorithmicMedia. 사진·편집 도구들이 읽는 표준 항목이다.
서명이 들어가는 C2PA 증명서는 아니다(키 관리·배포 단계에서 검토). 파일 정보는 다른 프로그램이 다시 저장하면 지워질 수 있다.
"""
import io
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
SOURCE_TYPES = {"generated": IPTC + "trainedAlgorithmicMedia", "composite": IPTC + "compositeWithTrainedAlgorithmicMedia"}
SOFTWARE = "GenAI Lab"
DESCRIPTION = {"generated": "AI로 생성한 이미지입니다 (GenAI Lab)",
               "composite": "원본 그림의 일부에 AI 생성 부분을 합성한 이미지입니다 (GenAI Lab)"}


def label_record(kind, model="Qwen-Image-Edit-2511"):
    """승인 기록(.review.json)에 함께 남길 표시 내용."""
    if kind not in SOURCE_TYPES:
        raise ValueError("AI 생성 표시 종류 오류")
    return dict(ai_generated=True, kind=kind, digital_source_type=SOURCE_TYPES[kind], software=SOFTWARE, model=model,
                description=DESCRIPTION[kind], method="PNG text + XMP (IPTC Digital Source Type)", signed=False)


def xmp_packet(record):
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
            ' xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"'
            f' Iptc4xmpExt:DigitalSourceType="{escape(record["digital_source_type"])}" xmp:CreatorTool="{escape(record["software"])}">'
            f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(record["description"])}</rdf:li></rdf:Alt></dc:description>'
            '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def labeled_png(data, record):
    """PNG 바이트 → 같은 픽셀 + AI 생성 표시가 들어간 PNG 바이트. 픽셀이 하나라도 달라지면 오류."""
    with Image.open(io.BytesIO(data)) as source:
        if source.format != "PNG":
            raise ValueError("AI 생성 표시는 PNG에만 넣습니다.")
        source.load()
        info = PngInfo()
        info.add_text("Software", record["software"])
        info.add_text("AI-Generated", "true")
        info.add_itxt("Description", record["description"])
        info.add_itxt("XML:com.adobe.xmp", xmp_packet(record))
        out = io.BytesIO()
        extra = {"icc_profile": source.info["icc_profile"]} if source.info.get("icc_profile") else {}
        source.save(out, format="PNG", pnginfo=info, **extra)
        labeled = out.getvalue()
        with Image.open(io.BytesIO(labeled)) as check:
            if check.mode != source.mode or check.size != source.size or not np.array_equal(np.asarray(check), np.asarray(source)):
                raise ValueError("AI 생성 표시를 넣는 중 픽셀이 바뀌었습니다.")
    return labeled


def read_label(path):
    """저장 파일의 AI 생성 표시(PNG 텍스트·XMP). 없으면 None."""
    with Image.open(path) as image:
        text = dict(getattr(image, "text", {}) or {})
    if text.get("AI-Generated") != "true":
        return None
    xmp = text.get("XML:com.adobe.xmp", "")
    kind = next((k for k, v in SOURCE_TYPES.items() if v + '"' in xmp), None)
    return dict(ai_generated=True, kind=kind, software=text.get("Software"), description=text.get("Description"))
