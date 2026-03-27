from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from open_wam.configs import StrategyName, TrainerAccelerator, TrainerConfig, TrainerPrecision


def _resolve_device(accelerator: TrainerAccelerator | str, local_rank: int = 0) -> torch.device:
    if accelerator == TrainerAccelerator.GPU:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested `accelerator=gpu` but CUDA is not available.")
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


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
        return torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

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
        self.device = _resolve_device(self.accelerator, local_rank=self.local_rank)
        self._use_fp16_scaler = self.precision == TrainerPrecision.FP16 and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_fp16_scaler)
        self._owns_process_group = False
        if self.distributed and not dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)
            self._owns_process_group = True

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
            from torch.distributed.fsdp import FullyShardedDataParallel

            return FullyShardedDataParallel(
                model,
                device_id=self.device if self.device.type == "cuda" else None,
                use_orig_params=True,
            )
        raise ValueError(f"Unsupported distributed strategy kind {self.kind!r}.")

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return getattr(model, "module", model)

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

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
