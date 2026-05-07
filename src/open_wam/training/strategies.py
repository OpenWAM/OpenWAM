from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from open_wam.configs import StrategyName, TrainerAccelerator, TrainerConfig, TrainerPrecision


def _resolve_device(accelerator: TrainerAccelerator | str, local_rank: int = 0) -> torch.device:
    if accelerator == TrainerAccelerator.GPU:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested `accelerator=gpu` but CUDA is not available.")
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _apply_block_activation_checkpointing(module: nn.Module) -> None:
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    except ImportError:
        return

    for child in module.modules():
        blocks = getattr(child, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            continue
        for block_index, block in enumerate(blocks):
            if getattr(block, "_open_wam_activation_checkpoint_wrapped", False):
                continue
            wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
            setattr(wrapped, "_open_wam_activation_checkpoint_wrapped", True)
            blocks[block_index] = wrapped


def _apply_composable_fsdp_sharding(
    model: nn.Module,
    *,
    mesh,
    mp_policy,
) -> nn.Module:
    from torch.distributed.fsdp import fully_shard

    # Optional CPU offload of params + grads + optimizer state. Enabled via
    # `OPEN_WAM_FSDP_CPU_OFFLOAD=1`. Useful when the M5 packed-coupling path
    # makes both video DiT and action expert trainable on a 4×L40S box.
    cpu_offload = os.environ.get("OPEN_WAM_FSDP_CPU_OFFLOAD", "0") == "1"
    offload_policy = None
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy

        offload_policy = CPUOffloadPolicy(pin_memory=True)

    def _shard_kwargs() -> dict:
        kwargs = {
            "mesh": mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": True,
        }
        if offload_policy is not None:
            kwargs["offload_policy"] = offload_policy
        return kwargs

    def _shard_block_stack(owner: nn.Module | None) -> None:
        if owner is None:
            return
        blocks = getattr(owner, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            return
        for block in blocks:
            if hasattr(block, "attn1"):
                fully_shard(block.attn1, **_shard_kwargs())
            if hasattr(block, "attn2"):
                fully_shard(block.attn2, **_shard_kwargs())
            if hasattr(block, "ffn"):
                fully_shard(block.ffn, **_shard_kwargs())
            fully_shard(block, **_shard_kwargs())

    visual_tower = getattr(model, "visual_tower", None)
    core = getattr(visual_tower, "core", None) if visual_tower is not None else None
    policy_variant = getattr(model, "policy_variant", None)
    action_expert = getattr(policy_variant, "action_expert", None) if policy_variant is not None else None

    # MoT packed-coupling path: blocks have been transferred from
    # core.blocks / action_expert.blocks into a MoTPackedBlockStack at
    # pipeline-build time. FSDP wraps each MoTPackedBlock as one unit so the
    # joint video+action attention runs through standard FSDP pre/post-forward
    # hooks (no manual `_summon_full_params` / `linear_with_materialized_params`
    # bypass during forward, which was causing
    # `setStorage out of bounds for storage of size 0` during backward).
    packed_block_stack = getattr(policy_variant, "packed_block_stack", None)
    if packed_block_stack is not None:
        for packed_block in packed_block_stack.packed_blocks:
            fully_shard(packed_block, **_shard_kwargs())
        # core.blocks and action_expert.blocks are intentionally empty in this
        # path; calling `_shard_block_stack` on them is a no-op.
        _shard_block_stack(core)
        _shard_block_stack(action_expert)
    else:
        _shard_block_stack(core)
        _shard_block_stack(action_expert)

    return model


def _set_module_gradient_sync(module: nn.Module, enabled: bool) -> bool:
    toggled = False
    setter = getattr(module, "set_requires_gradient_sync", None)
    if callable(setter):
        setter(enabled)
        return True
    if hasattr(module, "require_backward_grad_sync"):
        setattr(module, "require_backward_grad_sync", enabled)
        toggled = True
    if hasattr(module, "require_forward_param_sync"):
        setattr(module, "require_forward_param_sync", enabled)
        toggled = True
    return toggled


def _set_gradient_sync_recursive(module: nn.Module, enabled: bool) -> None:
    visited: set[int] = set()
    for submodule in module.modules():
        module_id = id(submodule)
        if module_id in visited:
            continue
        visited.add(module_id)
        _set_module_gradient_sync(submodule, enabled)


def _local_grad_tensor(grad: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None
    if DTensor is not None and isinstance(grad, DTensor):
        return grad.to_local()
    return grad


def _is_dtensor_grad(grad: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(grad, DTensor)


def _clip_grad_norm_mixed(
    parameters,
    max_grad_norm: float,
    *,
    distributed: bool,
) -> torch.Tensor:
    grads: list[torch.Tensor] = [param.grad for param in parameters if getattr(param, "grad", None) is not None]
    if not grads:
        return torch.tensor(0.0)

    device = _local_grad_tensor(grads[0]).device
    local_tensor_sq = torch.zeros((), device=device, dtype=torch.float32)
    local_dtensor_sq = torch.zeros((), device=device, dtype=torch.float32)

    for grad in grads:
        local_grad = _local_grad_tensor(grad).detach()
        grad_norm_sq = local_grad.float().pow(2).sum()
        if _is_dtensor_grad(grad):
            local_dtensor_sq = local_dtensor_sq + grad_norm_sq
        else:
            local_tensor_sq = local_tensor_sq + grad_norm_sq

    total_dtensor_sq = local_dtensor_sq
    if distributed and dist.is_initialized():
        dist.all_reduce(total_dtensor_sq, op=dist.ReduceOp.SUM)

    total_norm = torch.sqrt(local_tensor_sq + total_dtensor_sq)
    max_norm_tensor = torch.tensor(float(max_grad_norm), device=device, dtype=torch.float32)
    clip_coef = torch.clamp(max_norm_tensor / (total_norm + 1e-6), max=1.0)

    if clip_coef.item() < 1.0:
        for grad in grads:
            local_grad = _local_grad_tensor(grad)
            local_grad.mul_(clip_coef.to(device=local_grad.device, dtype=local_grad.dtype))

    return total_norm


@dataclass
class SingleDeviceStrategy:
    """Single-process training strategy with optional autocast support."""

    accelerator: TrainerAccelerator
    precision: TrainerPrecision

    def __post_init__(self) -> None:
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.distributed = False
        self.is_main_process = True
        self.device = _resolve_device(self.accelerator)
        self._use_fp16_scaler = self.precision == TrainerPrecision.FP16 and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_fp16_scaler)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        model.to(device=self.device)
        return model

    def autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.precision == TrainerPrecision.BF16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.precision == TrainerPrecision.FP16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_scaler.is_enabled():
            self.grad_scaler.unscale_(optimizer)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_scaler.is_enabled():
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()

    def clip_grad_norm_(self, parameters, max_grad_norm: float) -> torch.Tensor:
        return _clip_grad_norm_mixed(parameters, max_grad_norm, distributed=False)

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def set_gradient_sync(self, model: nn.Module, enabled: bool) -> None:
        del model, enabled
        return None

    def state_dict(self) -> dict[str, object]:
        return {"grad_scaler": self.grad_scaler.state_dict() if self.grad_scaler.is_enabled() else None}

    def load_state_dict(self, raw: dict[str, object] | None) -> None:
        if not self.grad_scaler.is_enabled():
            return
        payload = raw or {}
        scaler_state = payload.get("grad_scaler")
        if isinstance(scaler_state, dict):
            self.grad_scaler.load_state_dict(scaler_state)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def barrier(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class DistributedStrategy(SingleDeviceStrategy):
    """Distributed strategy that can wrap a model in DDP or FSDP.

    When launched without distributed environment variables, this degrades
    cleanly to the single-device behavior so config changes remain low-risk.
    """

    kind: StrategyName = StrategyName.DDP

    def __post_init__(self) -> None:
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        if self.accelerator == TrainerAccelerator.GPU and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        self.device = _resolve_device(self.accelerator, local_rank=self.local_rank)
        self._use_fp16_scaler = self.precision == TrainerPrecision.FP16 and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_fp16_scaler)
        self._owns_process_group = False
        if self.distributed and not dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)
            self._owns_process_group = True
        self._device_mesh = (
            init_device_mesh(self.device.type, (self.world_size,))
            if self.distributed and self.device.type != "cpu"
            else None
        )

    def prepare_model(self, model: nn.Module) -> nn.Module:
        model.to(device=self.device)
        if not self.distributed:
            return model
        if self.kind == StrategyName.DDP:
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
            )
        if self.kind == StrategyName.FSDP:
            from torch.distributed.fsdp import MixedPrecisionPolicy

            _apply_block_activation_checkpointing(model)
            mp_policy = MixedPrecisionPolicy(cast_forward_inputs=False)
            if self.device.type == "cuda":
                if self.precision == TrainerPrecision.BF16:
                    mp_policy = MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        output_dtype=torch.bfloat16,
                        cast_forward_inputs=False,
                    )
                elif self.precision == TrainerPrecision.FP16:
                    mp_policy = MixedPrecisionPolicy(
                        param_dtype=torch.float16,
                        reduce_dtype=torch.float32,
                        output_dtype=torch.float16,
                        cast_forward_inputs=False,
                    )
            return _apply_composable_fsdp_sharding(
                model,
                mesh=self._device_mesh,
                mp_policy=mp_policy,
            )
        raise ValueError(f"Unsupported distributed strategy kind {self.kind!r}.")

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return getattr(model, "module", model)

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def set_gradient_sync(self, model: nn.Module, enabled: bool) -> None:
        if not self.distributed:
            return
        _set_gradient_sync_recursive(model, enabled)

    def clip_grad_norm_(self, parameters, max_grad_norm: float) -> torch.Tensor:
        return _clip_grad_norm_mixed(parameters, max_grad_norm, distributed=self.distributed)

    def close(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def build_training_strategy(config: TrainerConfig) -> SingleDeviceStrategy:
    strategy_name = config.strategy
    if strategy_name in {StrategyName.LIGHTNING, StrategyName.SINGLE_DEVICE}:
        return SingleDeviceStrategy(accelerator=config.accelerator, precision=config.precision)
    if strategy_name in {StrategyName.DDP, StrategyName.FSDP}:
        return DistributedStrategy(accelerator=config.accelerator, precision=config.precision, kind=strategy_name)
    raise NotImplementedError(f"Unsupported training strategy {strategy_name!r}.")
