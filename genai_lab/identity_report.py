"""참조 비교만 한다. 점수 검사나 생성 의존성은 없다."""
from pathlib import Path
import json
import numpy as np
from genai_lab.proportion_inputs import sha
from genai_lab.qwen_record_io import write_json

LABELS = dict(zip(('B1','B2','B3','B4','B5','B6','B7','B8','H5'),
    ('등신','어깨폭','가슴 폭','허리 폭','골반 폭','허벅지 폭','종아리 폭','위팔 폭','머리색')))


def body_value(measurement, key):
    value = measurement.get(key)
    if value is None:
        return None, 'measurement_unavailable'
    if isinstance(value, dict):
        if 'invalid' in value:
            return None, value['invalid']
        value = (value['w'], value['bare'])
    else:
        value = (value, None)
    if not np.isfinite(value[0]) or value[0] <= 0:
        return None, 'invalid_value'
    return value, None


def compare_measurements(original, result, original_quantiles=None, result_quantiles=None):
    rows = {}
    for key in list(LABELS)[:-1]:
        a, ar = body_value(original, key); b, br = body_value(result, key)
        reason = ar or br
        if not reason and a[1] is not None and a[1] != b[1]:
            reason = 'exposure_mismatch'
        rows[key] = {'label': LABELS[key], 'original': None if a is None else a[0],
            'result': None if b is None else b[0], 'original_bare': None if a is None else a[1],
            'result_bare': None if b is None else b[1], 'measurable': reason is None,
            'reason': reason, 'difference': None if reason else round(abs(b[0]-a[0])/a[0],3)}
    invalid = ('hair_skin_inseparable','no_hair_band','no_eyes','no_head','no_nose')
    reason = next((n for m in (original,result) for n in m.get('notes',[]) if n in invalid), None)
    if original_quantiles is None or result_quantiles is None:
        reason = reason or 'hair_measurement_unavailable'
    distance = None
    if not reason:
        a, b = np.asarray(original_quantiles), np.asarray(result_quantiles)
        if a.shape != (101,3) or b.shape != a.shape or not np.isfinite([a,b]).all():
            reason = 'invalid_quantiles'
        else:
            distance = round(float(np.sqrt(np.sum(np.mean(np.abs(a-b),axis=0)**2))),2)
    rows['H5'] = {'label':LABELS['H5'], 'original':original_quantiles,
        'result':result_quantiles,'difference':distance,'measurable':reason is None,'reason':reason}
    values = [rows[k]['difference'] for k in list(LABELS)[1:-1] if rows[k]['measurable']]
    return {'schema':'identity_measure_v2_arm_touch','informational_only':True,'items':rows,
        'body_difference':round(float(np.median(values)),3) if values else None,
        'hair_distance':distance,'unavailable_count':sum(not x['measurable'] for x in rows.values())}


def summary(report):
    body = report.get('body_difference'); hair = report.get('hair_distance')
    return '체형 차이 ' + (f'{body*100:.1f}%' if body is not None else '측정 불가') + \
        ' · 머리색 거리 ' + (f'{hair:g}' if hair is not None else '측정 불가') + \
        f" · 측정 불가 {report['unavailable_count']}항목 (참고용)"


def write_report(original_folder, result_folder, destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=False)
    def read(folder):
        folder=Path(folder)
        data=json.loads((folder/'measurement.json').read_text(encoding='utf-8'))
        quantile=folder/'hair-quantiles.npy'
        return data,np.load(quantile).tolist() if quantile.exists() else None
    a,aq=read(original_folder); b,bq=read(result_folder)
    report=compare_measurements(a,b,aq,bq)
    report['sources']={name:{'file':str(Path(folder)/'character.png'),
        'sha256':sha(Path(folder)/'character.png')} for name,folder in
        (('original',original_folder),('result',result_folder))}
    from PIL import Image
    panels=[]
    for folder in (original_folder,result_folder):
        path=Path(folder)/'overlay.png'
        with Image.open(path if path.exists() else Path(folder)/'character.png') as im:
            panels.append(im.convert('RGB'))
    canvas=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white')
    x=0
    for panel in panels:
        canvas.paste(panel,(x,0));x+=panel.width;panel.close()
    canvas.save(destination/'identity-overlay.png');canvas.close()
    report['overlay']={'path':str(destination/'identity-overlay.png'),'sha256':sha(destination/'identity-overlay.png')}
    write_json(destination/'identity-report.json',report)
    return report
