"""분리된 CPU 작업이다. 모델은 오프라인 캐시에 이미 있어야 한다."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',ONNX_MODE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def measure_folder(folder, model_cache, head_cache):
    os.environ['HF_HUB_CACHE']=str(head_cache)
    import onnxruntime
    import numpy as np
    from PIL import Image
    from imgutils.detect import detect_heads
    from scripts.body_comparison_runner import extract_anime_character_foreground_mask
    from genai_lab.identity_measurement import measure_pixels
    from genai_lab.qwen_record_io import write_json
    pose=json.loads((folder/'pose.json').read_text(encoding='utf-8'))
    with Image.open(folder/'character.png') as src:
        image=src.convert('RGB')
    try:
        if pose.get('person_count',1)!=1:
            write_json(folder/'measurement.json',{'notes':['person_count_not_one']})
            return
        alpha_image=extract_anime_character_foreground_mask(image,model_id='isnet-anime',model_cache_dir=model_cache)
        with alpha_image:
            alpha=np.asarray(alpha_image.convert('L').resize(image.size))
        heads=detect_heads(image,model_name='head_detect_v2.0_s')
        head=max(heads,key=lambda t:t[2])[0] if heads else None
        measurement,artifacts=measure_pixels(image,pose,alpha,head)
        # 폭 맞추기가 같은 픽셀을 쓰도록 입력 분할 결과를 유지한다.
        Image.fromarray(alpha).save(folder/'alpha.png')
        write_json(folder/'measurement.json',measurement)
        if 'overlay' in artifacts:
            artifacts['overlay'].save(folder/'overlay.png');artifacts['overlay'].close()
        if 'hair_quantiles' in artifacts:
            np.save(folder/'hair-quantiles.npy',artifacts['hair_quantiles'])
    finally:image.close()


if __name__=='__main__':
    folder,cache,heads=map(Path,sys.argv[1:])
    measure_folder(folder,cache,heads)
