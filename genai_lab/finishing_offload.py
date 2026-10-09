"""ON-only block offload, with explicit ownership of non-UNet model hooks."""
import time
from genai_lab.proportion_inputs import require


def group_inventory(unet):
    """Read installed Diffusers hooks and prove every IP weight belongs to a group."""
    groups = {}
    for name, module in unet.named_modules():
        registry = getattr(module, "_diffusers_hook", None)
        for hook in getattr(registry, "hooks", {}).values():
            if type(hook).__name__ != "GroupOffloadingHook":
                continue
            group = hook.group
            if id(group) in groups:
                continue
            require(group.stream is None and not group.record_stream, "스트림 오프로드는 허용하지 않습니다.")
            weights = {id(p):p for child in group.modules for p in child.parameters()}
            weights.update({id(p):p for p in group.parameters})
            groups[id(group)] = (group, weights, name or "unet")
    require(groups, "UNet 블록 오프로드 훅이 없습니다.")
    ip = {name:p for name,p in unet.named_parameters()
          if ".to_k_ip." in name or ".to_v_ip." in name or name.startswith("encoder_hid_proj.")}
    require(ip and any(".to_k_ip." in name for name in ip) and
            any(".to_v_ip." in name for name in ip) and
            any(name.startswith("encoder_hid_proj.") for name in ip), "IP 가중치 구성을 확인할 수 없습니다.")
    membership = []
    for name, parameter in ip.items():
        owners = [location for _,weights,location in groups.values() if id(parameter) in weights]
        require(len(owners)==1, "IP 가중치의 오프로드 소속이 없거나 중복입니다: "+name)
        membership.append({"parameter":name,"group":owners[0],
                           "bytes":parameter.numel()*parameter.element_size(),"dtype":str(parameter.dtype)})
    return list(groups.values()), {"groups":[{"module":location,
        "parameter_bytes":sum(p.numel()*p.element_size() for p in weights.values())}
        for _,weights,location in groups.values()], "ip_membership":membership,
        "ip_parameter_bytes":sum(row["bytes"] for row in membership), "ip_resident_bytes":0,
        "ip_policy":"with_owning_group", "projection_policy":"unet_root_group"}


class FinishingOffload:
    def __init__(self, pipe, record):
        self.pipe, self.record = pipe, record
        self.model_hooks, self.observer_hooks, self.groups = [], [], []
        self.observations = {}
        self.ip_module_names = set()

    def release_models(self):
        for hook in self.model_hooks:
            hook.offload()

    def release(self):
        """Also release partially executed groups after failure; never reconfigure them."""
        self.release_models()
        for group, _, _ in self.groups:
            group.offload_()

    def begin_stage(self):
        self.observations = {}

    def verify_stage(self, calls):
        require(set(self.observations)==self.ip_module_names and
                all(row["calls"]==calls for row in self.observations.values()),
                "IP 층의 실제 실행 횟수가 UNet 호출 수와 다릅니다.")
        require(all(p.device.type=="cpu" for p in self.pipe.unet.parameters()),
                "UNet 블록이 실행 뒤 CPU로 내려가지 않았습니다.")

    def observe_ip(self, name, expected_device, expected_dtype):
        def observe(module, *_):
            parameters = list(module.parameters())
            require(parameters and all(p.device == expected_device and p.dtype == expected_dtype
                    for p in parameters), "IP 층의 실행 장치 또는 dtype이 다릅니다: "+name)
            row = self.observations.setdefault(name, {"calls":0,"device":str(expected_device),
                                                       "dtype":str(expected_dtype)})
            row["calls"] += 1
        return observe

    def close(self):
        try:
            self.release()
        finally:
            for hook in self.observer_hooks:
                hook.remove()
            for hook in self.model_hooks:
                hook.remove()
            self.observer_hooks.clear()
            self.model_hooks.clear()
            self.groups.clear()
            self.pipe = None


def configure_offload(pipe, torch, record, *, device="cuda:0"):
    """Keep Accelerate on encoders/VAE and use Diffusers groups exclusively on UNet."""
    from accelerate import cpu_offload_with_hook
    started = time.perf_counter()
    owner = FinishingOffload(pipe, record)
    record.update(status="started", unet="block_level", num_blocks_per_group=1,
                  use_stream=False, record_stream=False, non_blocking=False,
                  other_models="accelerate_model_cpu_offload", retry=False,
                  onload_device=str(device),offload_device="cpu")
    try:
        require(not getattr(pipe,"_all_hooks",[]) and
                not any(hasattr(m,"_hf_hook") for m in pipe.unet.modules()),
                "UNet에 기존 모델 오프로드 훅을 섞을 수 없습니다.")
        pipe.unet.enable_group_offload(onload_device=torch.device(device),
            offload_device=torch.device("cpu"), offload_type="block_level", num_blocks_per_group=1,
            use_stream=False, record_stream=False, non_blocking=False)
        owner.groups, inventory = group_inventory(pipe.unet)
        record.update(inventory)
        previous = None
        for name in ("text_encoder","text_encoder_2","image_encoder","vae"):
            _, previous = cpu_offload_with_hook(getattr(pipe,name),torch.device(device),prev_module_hook=previous)
            owner.model_hooks.append(previous)
        # The pipeline otherwise reinstalls whole-model hooks after every call, conflicting with groups.
        # Own these handles here instead; refine() releases models after success AND failure.
        pipe._all_hooks = []
        owner.observer_hooks.append(pipe.unet.register_forward_pre_hook(lambda *_:owner.release_models()))
        for name,module in pipe.unet.named_modules():
            if (".to_k_ip." in name or ".to_v_ip." in name or name=="encoder_hid_proj") and list(module.parameters(recurse=False) if name!="encoder_hid_proj" else module.parameters()):
                owner.ip_module_names.add(name)
                dtype = next(module.parameters()).dtype
                owner.observer_hooks.append(module.register_forward_pre_hook(owner.observe_ip(name,torch.device(device),dtype)))
        require(owner.observer_hooks and len(owner.observer_hooks)>1,"IP 이동 관찰 지점이 없습니다.")
        require(all(p.device.type=="cpu" for p in pipe.unet.parameters()),"초기 UNet이 CPU에 없습니다.")
        record["status"] = "configured"
        return owner
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error))
        try:
            owner.close()
        except BaseException as cleanup:
            record["cleanup_error"] = str(cleanup)
        raise
    finally:
        record["setup_seconds"] = time.perf_counter()-started
