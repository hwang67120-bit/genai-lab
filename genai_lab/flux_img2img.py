"""FLUX.2-klein으로 '원래 그림 위에 살짝만 다시 그리기'(img2img, S41 얼굴 다듬기).

편집 모델은 참조 그림을 보고 구도를 새로 잡는다(얼굴을 잘라 넣으니 더 확대되고 표정·위치가 바뀌어 붙일 수 없었다). 그래서 원래 그림의
잠재값에서 출발해 4단계 일정의 뒤쪽만 다시 그린다: 일정 [1, .75, .5, .25] 중 strength 비율만큼의 뒤쪽 시그마를 쓰고, 첫 시그마
(파이프라인과 같은 동적 이동 적용)만큼 잡음을 섞은 원래 잠재값을 시작점으로 넘긴다. 참조 그림(image)은 그대로 조건으로 쓴다.
"""
import numpy as np


def partial_sigmas(steps, strength):
    """전체 일정 linspace(1, 1/steps, steps)의 뒤쪽 strength 비율(최소 1단계)."""
    full = np.linspace(1.0, 1.0 / steps, steps)
    keep = max(1, int(round(steps * strength)))
    return full[-keep:]


def redraw(pipe, image, refs, prompt, strength, seed, steps=4, guidance=1.0, max_sequence_length=256):
    """image(PIL, 16의 배수 크기)를 출발점으로 뒤쪽 단계만 다시 그린다. refs: 조건 그림 목록. 반환 PIL."""
    import torch
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    width, height = image.size
    device = pipe._execution_device
    tensor = pipe.image_processor.preprocess(image, height=height, width=width).to(device=device, dtype=pipe.vae.dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        clean = pipe._encode_vae_image(tensor, generator=None)                     # (1, 128, h/16, w/16)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device=clean.device, dtype=clean.dtype)
    sigmas = partial_sigmas(steps, strength)
    seq_len = clean.shape[2] * clean.shape[3]
    mu = compute_empirical_mu(image_seq_len=seq_len, num_steps=steps)
    start = float(pipe.scheduler.time_shift(mu, 1.0, torch.tensor([sigmas[0]], dtype=torch.float64))[0])
    latents = (1.0 - start) * clean + start * noise
    return pipe(image=refs, prompt=prompt, width=width, height=height, num_inference_steps=steps, sigmas=list(sigmas),
                latents=latents, guidance_scale=guidance, max_sequence_length=max_sequence_length,
                generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
