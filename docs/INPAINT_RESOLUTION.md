# 복원·의상 합성 연산 해상도

실행 전 해상도 창에서 가로 256~2048px(8배수)를 입력한다. 세로는 원본 비율을 기준으로 8배수 반올림하며 최대 4096px다. 768×1344 원본에서 가로 512는 512×896이다. 기본은 원본 크기이며 앱 실행 중 마지막 선택을 기억한다. 기준 캐릭터 생성 해상도는 변경하지 않는다.

원본/승인 마스크/자세 자료는 그대로 보관한다. 별도 생성 프로세스 안에서 초기 RGB는 LANCZOS, 하드 마스크는 NEAREST, 자세 RGB는 BILINEAR로 같은 크기로 맞춘다. 변경 해상도에서는 padding_mask_crop을 끄고 전체 캔버스로 연산한다. 출력은 원본 크기로 되돌린 뒤 기존 원본 해상도 보호 합성을 적용한다. 확대가 세부 품질을 복원한다는 보장은 없다.

metadata.json에 inference_size와 effective_padding_mask_crop, prompt_execution.json에 inference_size와 output_size를 기록한다. 작은 해상도가 OOM 해결이나 특정 속도를 보장하지 않는다. 실제 GPU 품질·성능 비교는 별도 검증이 필요하다. 이미 실행 중인 작업은 변경하지 않으며 다음 앱 실행부터 UI가 적용된다.

근거: [Diffusers 메모리 관리](https://huggingface.co/docs/diffusers/optimization/memory), [Pillow resize](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize), [ControlNet Inpaint API](https://huggingface.co/docs/diffusers/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline).
