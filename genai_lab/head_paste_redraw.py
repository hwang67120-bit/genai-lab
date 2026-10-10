"""선택 머리 한 장의 마지막 4단계를 재생성한다. H2/G3 붙이기 뒤에만 호출한다."""
import gc
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from dataclasses import replace
from genai_lab.proportion_inputs import require, sha, checked_image, json_value
from genai_lab.qwen_record_io import write_json
from genai_lab.head_paste_rules import blend_result
from genai_lab.onepass_prompt import plan_prompt_chunks
LIMIT = 6.5
START = 24
HEAD_TAGS = ('portrait', 'close-up', 'white background')

def head_prompt(inputs, analysis, tokenizers):
    """라쿤 규칙: 성별 앞부분 + 외형 태그 + 귀 태그 + 머리 확대 태그. 잠긴 문장에 있는 태그만 쓴다."""
    p0 = inputs.prompt
    require(plan_prompt_chunks(p0.positive, p0.negative, tokenizers) == p0.encoders, '원래 문장 토큰 재현 실패')
    terms = [t.strip() for t in p0.positive.split(',')]
    keep = list(p0.rules['prefix']) + [t.replace('_', ' ') for t in analysis['groups']['appearance']] + [t.replace('_', ' ') for t in analysis['groups']['fixed'] if t.endswith('ears')]
    require(all((t in terms for t in keep)), f'잠긴 문장에 없는 머리 태그: {[t for t in keep if t not in terms]}')
    positive = ', '.join(keep + list(HEAD_TAGS))
    return replace(p0, positive=positive, encoders=plan_prompt_chunks(positive, p0.negative, tokenizers))

class RestoreFrom:
    """start 이전 단계는 출발 그림 + 그 단계 잡음으로 덮어쓰고, 이후에는 마스크 밖만 되돌린다."""

    def __init__(self, initial, noise, mask, sigmas, start, guard=lambda: None):
        self.initial, self.noise, self.mask, self.sigmas, self.start = (initial, noise, mask, sigmas, start)
        self.guard, self.calls = (guard, [])

    def __call__(self, index, timestep, latents):
        import torch
        require(index == len(self.calls), '단계 콜백 순서 불일치')
        reference = self.initial + self.noise * self.sigmas[index + 1].to(latents.device)
        latents.copy_(reference if index < self.start else torch.where(self.mask, latents, reference))
        require(torch.equal(latents[~self.mask], reference[~self.mask]), '잠재 공간 보호 실패')
        self.calls.append({'index': index, 'sigma_next': float(self.sigmas[index + 1]), 'mode': 'overwrite' if index < self.start else 'blend'})
        self.guard()

def cleanup(steps, errors):
    """정리 단계를 모두 시도하고 오류는 따로 모은다(원래 실행 오류와 분리)."""
    for name, step in steps:
        try:
            step()
        except BaseException as error:
            errors.append({'step': name, 'error': repr(error)})

def memory_guard(torch):
    value = torch.cuda.max_memory_reserved() / 2 ** 30
    require(value <= LIMIT, f'reserved 메모리 한도 초과: {value:.3f} GiB')

def memory_snapshot(torch, pipe, output, stage):
    """실제 사용량·예약량과 모델의 GPU 상주량을 단계마다 기록한다."""
    row = {'stage': stage, 'seconds': time.perf_counter(), 'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    row['gpu_parameter_bytes'] = {name: sum((p.numel() * p.element_size() for p in getattr(pipe, name).parameters() if p.device.type == 'cuda')) for name in ('text_encoder', 'text_encoder_2', 'image_encoder', 'adapter', 'unet', 'vae')}
    with (output / 'memory.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row

def stage_guard(torch, cancelled):
    from genai_lab.onepass_generation import OnePassCancelled
    if cancelled():
        raise OnePassCancelled('머리 붙이기를 취소했습니다. 원본은 그대로 유지합니다.')
    memory_guard(torch)

def load_redraw_pipeline(settings, options):
    """고정 스케치·얼굴 참조 모델을 CPU에 읽는다. 다운로드는 하지 않는다."""
    import torch
    from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
    from transformers import CLIPVisionModelWithProjection
    adapter = T2IAdapter.from_pretrained(str(options.sketch_root), variant='fp16', torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(settings.ip_root / settings.image_encoder_subfolder), torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(str(settings.model_root), adapter=adapter, image_encoder=encoder, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe.load_ip_adapter(str(settings.ip_root), subfolder=settings.ip_subfolder, weight_name=settings.ip_weight_name, image_encoder_folder=None, local_files_only=True)
    pipe.set_ip_adapter_scale(0.9)
    return pipe

def encode_initial_heads(pipe, prepared, state, cancelled):
    """1024 붙이기 그림을 fp32 VAE로 인코딩한 뒤 VAE를 GPU에서 내린다."""
    import torch
    initial = {}
    pipe.vae.to(device='cuda', dtype=torch.float32)
    for p in prepared:
        for seed in p['rec']['seeds']:
            stage_guard(torch, cancelled)
            with Image.open(p['rec']['per_seed'][str(seed)]['init']) as image:
                tensor = pipe.image_processor.preprocess(image.convert('RGB')).to('cuda', torch.float32)
            with torch.inference_mode():
                latent = pipe.vae.encode(tensor).latent_dist.mean * pipe.vae.config.scaling_factor
            initial[p['run'].name, seed] = latent.cpu().half()
            del tensor, latent
            memory_guard(torch)
    state['vae_encode_peak_gib'] = torch.cuda.max_memory_reserved() / 2 ** 30
    pipe.vae.to(device='cpu', dtype=torch.float16)
    torch.cuda.empty_cache()
    return initial

def execute_redraw(pipe, prepared, seed, embeds, generator, callback):
    """시험과 같은 스케치·얼굴 참조·스케줄러 입력을 한 장에 전달한다."""
    from genai_lab.finishing_memory import unet_attention
    with unet_attention(pipe.unet), Image.open(prepared['paste_dir'] / f'sketch_{seed}.png') as sketch, Image.open(prepared['inputs'].face_file) as face:
        return pipe(**embeds, image=sketch.convert('RGB'), ip_adapter_image=face.convert('RGB'), width=1024, height=1024, num_inference_steps=28, guidance_scale=prepared['settings'].guidance_scale, adapter_conditioning_scale=0.8, adapter_conditioning_factor=1.0, generator=generator, callback=callback, callback_steps=1).images[0]

def redraw_candidate(pipe, p, seed, initial, embeds, output, state, offload, adapter_hook, cancelled, progress):
    """고정 단계 재생성 → 호출 수 검사 → 보호 합성 → 실행·정리 결과 저장."""
    import torch
    from diffusers import EulerAncestralDiscreteScheduler
    from genai_lab.onepass_generation_settings import scheduler_config
    from genai_lab.finishing_memory import unet_attention
    stage_guard(torch, cancelled)
    e = p['rec']['per_seed'][str(seed)]
    tag = f"{p['run'].name[:8]}-{seed}"
    offload.begin_stage()
    destination = output / p['run'].name[:8] / f'seed-{seed}'
    destination.mkdir(parents=True)
    for path, digest in p['manifest'].items():
        require(sha(path) == digest, '실행 중 입력 변경: ' + path)
    torch.cuda.reset_peak_memory_stats()
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config())
    pipe.scheduler.set_timesteps(28, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(seed)
    init = initial[p['run'].name, seed].to('cuda')
    noise = torch.randn(init.shape, generator=generator, device='cuda', dtype=init.dtype)
    mask = np.asarray(Image.open(e['mask']).convert('L'))
    down = cv2.resize(mask.astype(np.float32) / 255, (128, 128), interpolation=cv2.INTER_AREA) >= 0.5
    latent_mask = torch.from_numpy(down).to('cuda')[None, None].expand_as(init)
    callback = RestoreFrom(init, noise, latent_mask, pipe.scheduler.sigmas, START, lambda: stage_guard(torch, cancelled))
    observations, counts = ([], [])

    def observe(module, args, kwargs, result):
        observations.append({'batch': int(args[0].shape[0]), 'adapter': kwargs.get('down_intrablock_additional_residuals') is not None})
        memory_snapshot(torch, pipe, output, f'{tag}:unet-{len(observations)}')
        memory_guard(torch)
    handles = [pipe.unet.register_forward_hook(observe, with_kwargs=True), pipe.adapter.register_forward_hook(lambda *a: counts.append(1))]
    started = time.perf_counter()
    record = {'run': str(p['run']), 'seed': seed, 'status': 'generating', 'prompt': p['prompt'].positive, 'negative': p['prompt'].negative, 'sketch_strength': 0.8, 'ip_scale': 0.9, 'steps': 28, 'actual_start_index': START, 'edited_unet_steps': 28 - START, 'nominal_fraction': round((28 - START) / 28, 3), 'init': e['init'], 'memory_limit_gib': LIMIT, 'memory_mode': 'block'}
    write_json(destination / 'run.json', record)
    failure = None
    try:
        memory_snapshot(torch, pipe, output, f'{tag}:before_pipeline')
        result = execute_redraw(pipe, p, seed, embeds, generator, callback)
        offload.verify_stage(28)
        memory_snapshot(torch, pipe, output, f'{tag}:decoded')
        result.save(destination / 'head_1024.png')
        require(len(observations) == 28 and len(callback.calls) == 28 and (len(counts) == 1), '모델·복원 호출 수 불일치')
        require(all((x['batch'] == 2 and x['adapter'] for x in observations)), 'CFG 또는 스케치 적용 불일치')
        raw = np.asarray(Image.open(e['raw']).convert('RGB'))
        features = np.asarray(Image.open(e['features']).convert('L'))
        blended, checks = blend_result(raw, result, mask, features, p['rec']['box'])
        Image.fromarray(blended).save(destination / 'raw_redraw.png')
        memory_guard(torch)
        record.update(status='review_pending', seconds=time.perf_counter() - started, checks=checks, max_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, head_sha256=sha(destination / 'head_1024.png'), result_sha256=sha(destination / 'raw_redraw.png'))
        state['completed'].append([p['run'].name[:8], seed])
        print(tag, record['status'], round(record['seconds'], 1), round(record['max_reserved_gib'], 3), flush=True)
    except BaseException as error:
        failure = error
        record.update(status='failed', error=repr(error))
        cleanup([('failed_snapshot', lambda: record.__setitem__('failure_memory', memory_snapshot(torch, pipe, output, f'{tag}:failed')))], record.setdefault('cleanup_errors', []))
    for path, digest in p['manifest'].items():
        if sha(path) != digest:
            failure = ValueError('실행 중 입력 변경: ' + path)
            record.update(status='failed', error=str(failure))
    record.update(unet_calls=observations, callback=callback.calls, adapter_calls=len(counts))
    errors = record.setdefault('cleanup_errors', [])
    cleanup([('ip_observations', lambda: record.__setitem__('ip_observations', dict(offload.observations))), ('offload.release', offload.release), ('adapter_hook.offload', adapter_hook.offload), *((f'remove_hook_{i}', h.remove) for i, h in enumerate(handles)), ('free_tensors', torch.cuda.empty_cache), ('release_snapshot', lambda: memory_snapshot(torch, pipe, output, f'{tag}:release'))], errors)
    write_json(destination / 'run.json', json_value(record))
    if errors:
        state.setdefault('cleanup_errors', []).append({'run': tag, 'errors': errors})
    write_json(output / 'status.json', json_value(state))
    del init, noise, latent_mask
    if failure is not None:
        raise failure
    require(not errors, f'{tag} 정리 오류: {errors}')

def redraw_prepared(prepared, output, state, cancelled=lambda: False, progress=lambda _: None):
    import torch
    from accelerate import cpu_offload_with_hook
    from genai_lab.finishing_offload import configure_offload
    from genai_lab.finishing_memory import attention_policy
    from genai_lab.onepass_generation import encode_prompt_plan
    settings, options = (prepared[0]['settings'], prepared[0]['options'])
    for p in prepared:
        require(p['settings'].model_root == settings.model_root and p['settings'].ip_root == settings.ip_root and (p['options'].sketch_root == options.sketch_root), '실행마다 모델이 다릅니다')
    pipe = load_redraw_pipeline(settings, options)
    initial, offload, adapter_hook = ({}, None, None)
    try:
        memory_snapshot(torch, pipe, output, 'models_loaded_cpu')
        torch.cuda.reset_peak_memory_stats()
        initial = encode_initial_heads(pipe, prepared, state, cancelled)
        state['offload'] = {}
        offload = configure_offload(pipe, torch, state['offload'])
        _, adapter_hook = cpu_offload_with_hook(pipe.adapter, torch.device('cuda:0'))
        offload.observer_hooks.append(pipe.unet.register_forward_pre_hook(lambda *_: adapter_hook.offload()))
        state['attention_policy'] = attention_policy(pipe)
        pipe.enable_vae_tiling()
        for p in prepared:
            encoders = (pipe.text_encoder.to('cuda'), pipe.text_encoder_2.to('cuda'))
            embeds = encode_prompt_plan(p['prompt'], encoders, 'cuda', torch.float16)
            pipe.text_encoder.to('cpu')
            pipe.text_encoder_2.to('cpu')
            torch.cuda.empty_cache()
            for seed in p['rec']['seeds']:
                redraw_candidate(pipe, p, seed, initial, embeds, output, state, offload, adapter_hook, cancelled, progress)
    finally:
        errors = state.setdefault('final_cleanup_errors', [])
        cleanup([('offload.close', lambda: offload is not None and offload.close()), ('adapter_hook.offload', lambda: adapter_hook is not None and adapter_hook.offload()), ('adapter_hook.remove', lambda: adapter_hook is not None and adapter_hook.remove()), ('models_released_snapshot', lambda: memory_snapshot(torch, pipe, output, 'models_released'))], errors)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
