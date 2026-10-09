"""외곽선 폭 맞추기는 선택 기능이다. 옷의 외곽을 숨은 인체로 해석하지 않는다."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from genai_lab.identity_report import body_value
from genai_lab.identity_measurement import largest
from genai_lab.proportion_inputs import sha, require
from genai_lab.qwen_record_io import write_json


def joints(pose):
    return {j['joint_name']:np.array([j['x'],j['y']],dtype=float) for j in pose['joints']
            if j['confidence_score']>=.3}


def body_segments(pose):
    j=joints(pose); neck=j.get('neck'); hips=[j.get(s+'_hip') for s in ('right','left')]
    out=[]
    if neck is not None and all(h is not None for h in hips):
        hip=(hips[0]+hips[1])/2; axis=hip-neck
        # 가슴·허리·골반 폭이 서로 영향을 주지 않도록 몸통을 나눈다.
        for key,lo,hi,t in [('B2',0,.3,.15),('B3',.3,.45,.3),('B4',.45,.8,.6),('B5',.8,1.,1.)]:
            out.append((key,'torso',neck+lo*axis,neck+hi*axis,neck+t*axis))
    for key,start,end,t in [('B6','hip','knee',.3),('B7','knee','ankle',.4),('B8','shoulder','elbow',.5)]:
        for side in ('right','left'):
            a,b=j.get(side+'_'+start),j.get(side+'_'+end)
            if a is not None and b is not None:
                out.append((key,side,a,b,a+t*(b-a)))
    return out


def width_scale(original, result, key):
    a,ar=body_value(original,key);b,br=body_value(result,key)
    record={'original_width':None if a is None else a[0],'base_width':None if b is None else b[0],
            'scale':None,'applied':False,'reason':ar or br}
    if record['reason']: return record
    if a[1] is not None and a[1]!=b[1]:
        record['reason']='exposure_mismatch';return record
    scale=a[0]/b[0];record['scale']=scale
    if not .6<=scale<=1.4:
        record['reason']='scale_out_of_range';return record
    record['applied']=True
    return record


def capsule(shape, start, end, radius):
    yy,xx=np.indices(shape,dtype=np.float32)
    axis=end-start;length=float(np.linalg.norm(axis))
    if length<1:return None
    tangent=axis/length;normal=np.array([-tangent[1],tangent[0]])
    along=(xx-start[0])*tangent[0]+(yy-start[1])*tangent[1]
    across=(xx-start[0])*normal[0]+(yy-start[1])*normal[1]
    excess=np.maximum(-along,np.maximum(along-length,0))
    mask=(across**2+excess**2<=radius**2)
    return xx,yy,across,normal,mask


def match_widths(alpha, original, result, pose, *, protected=None):
    """지정한 캡슐 영역만 바꾼다. 모든 변환은 변경 전 1단계 결과에서 표본을 얻는다."""
    require(alpha.dtype==np.uint8 and alpha.ndim==2,'폭 조정 알파 형식 오류')
    baseline=alpha.copy()
    total=np.zeros(alpha.shape,np.float32);weight=np.zeros_like(total)
    support=np.zeros(alpha.shape,bool);records=[]
    scales={key:width_scale(original,result,key) for key in ('B2','B3','B4','B5','B6','B7','B8')}
    seen=set()
    for key,side,start,end,center in body_segments(pose):
        seen.add(key);entry=dict(scales[key],metric=key,side=side,start=start.tolist(),end=end.tolist())
        records.append(entry)
        if not entry['applied']:continue
        width=entry['base_width']*result['H']
        region=capsule(alpha.shape,start,end,width*.75)
        if region is None:
            entry.update(applied=False,reason='short_axis');continue
        xx,yy,across,normal,allowed=region
        if protected is not None:allowed &= ~protected
        # 5px 흐림을 적용한 뒤 캡슐 영역으로 제한한다. 머리 보호 영역은 정확히 유지한다.
        blend=cv2.GaussianBlur(allowed.astype(np.float32),(5,5),0)*allowed
        delta=across*(1/entry['scale']-1)
        sampled=cv2.remap(baseline,(xx+normal[0]*delta).astype(np.float32),
            (yy+normal[1]*delta).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        total+=sampled*blend;weight+=blend;support|=allowed
        entry.update(base_width_px=width,offset_px=(1-entry['scale'])*width/2,
                     capsule_radius_px=width*.75,center=center.tolist())
    for key in scales.keys()-seen:
        records.append(dict(scales[key],metric=key,applied=False,reason='missing_joints'))
    changed=weight>0
    averaged=total/np.maximum(weight,1e-6)
    blend=np.minimum(weight,1)
    output=baseline.copy()
    output[changed]=np.clip(np.round(averaged[changed]*blend[changed]+baseline[changed]*(1-blend[changed])),0,255)
    # 처리 영역 밖에서는 가장자리 반투명 값까지 원래 알파 값을 그대로 유지한다.
    output[~support]=alpha[~support]
    return output,{'method':'perpendicular_axis_scale','boundary_blur_px':5,
        'torso_intervals':[[0,.3],[.3,.45],[.45,.8],[.8,1]],'items':records},support


def adjust_base_widths(base_path, alpha, options, directory, *, runtime=None, cancelled=lambda:False):
    from genai_lab.identity_measure_service import measure_image
    from genai_lab.studio_generation import StudioRuntime
    runtime=runtime or StudioRuntime.from_environment();directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    original=measure_image(options.head.normalized_file,directory/'original',runtime,cancelled=cancelled)
    result=measure_image(base_path,directory/'base',runtime,cancelled=cancelled)
    pose=json.loads((directory/'base/pose.json').read_text(encoding='utf-8'))
    with Image.open(options.head.mask_file) as im: head=np.asarray(im)>0
    protected=cv2.dilate(head.astype(np.uint8),np.ones((13,13),np.uint8))>0
    output,record,support=match_widths(alpha,original,result,pose,protected=protected)
    Image.fromarray(alpha).save(directory/'before.png');Image.fromarray(output).save(directory/'after.png')
    Image.fromarray(support.astype(np.uint8)*255).save(directory/'capsules.png')
    with Image.open(base_path) as im: canvas=im.convert('RGB')
    pixels=np.asarray(canvas).copy();canvas.close()
    contours,_=cv2.findContours((output>127).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pixels,contours,-1,(0,210,100),2)
    Image.fromarray(pixels).save(directory/'adjusted-overlay.png')
    # 보정 외곽선과 함께 원본·기준 이미지의 측정선을 표시한다.
    panels=[]
    for folder in ('original','base'):
        path=directory/folder/'overlay.png'
        with Image.open(path if path.exists() else directory/folder/'character.png') as im:panels.append(im.convert('RGB'))
    panels.append(Image.fromarray(pixels));preview=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white')
    x=0
    for panel in panels:preview.paste(panel,(x,0));x+=panel.width;panel.close()
    preview.save(directory/'preview.png');preview.close()
    record.update(original=original,base=result,before_sha256=sha(directory/'before.png'),
        after_sha256=sha(directory/'after.png'),preview=str(directory/'preview.png'),
        limitation='옷 포함 폭은 실제 인체 폭이 아니며 가려진 체형을 복원하지 않습니다.')
    write_json(directory/'widths.json',record)
    return output,{'path':str(directory/'widths.json'),'sha256':sha(directory/'widths.json'),'mode':'match_original'}
