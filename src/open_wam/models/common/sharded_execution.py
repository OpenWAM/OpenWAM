"""Runtime parameter contexts for custom execution over sharded modules."""

from __future__ import annotations

from contextlib import ExitStack

try:  # pragma: no cover - import surface depends on torch build
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - CPU-only or non-FSDP env
    FSDP = None


def summon_full_parameters(*modules):
    """Materialize parameters managed by classic FSDP wrappers."""

    stack = ExitStack()
    if FSDP is None:
        return stack
    seen_ids: set[int] = set()
    for module in modules:
        fsdp_modules = tuple(FSDP.fsdp_modules(module, root_only=False))
        if not fsdp_modules:
            continue
        for fsdp_module in fsdp_modules:
            module_id = id(fsdp_module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            stack.enter_context(
                FSDP.summon_full_params(
                    fsdp_module,
                    recurse=False,
                    writeback=False,
                )
            )
    return stack


class _FSDP2UnshardContext:
    def __init__(self, *modules) -> None:
        self._modules = modules
        self._unsharded: list[object] = []

    def __enter__(self) -> "_FSDP2UnshardContext":
        seen_ids: set[int] = set()
        for module in self._modules:
            for submodule in module.modules():
                module_id = id(submodule)
                if module_id in seen_ids:
                    continue
                seen_ids.add(module_id)
                unshard = getattr(submodule, "unshard", None)
                reshard = getattr(submodule, "reshard", None)
                if not callable(unshard) or not callable(reshard):
                    continue
                unshard()
                self._unsharded.append(submodule)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for submodule in reversed(self._unsharded):
            reshard = getattr(submodule, "reshard", None)
            if callable(reshard):
                reshard()
        self._unsharded.clear()
        return False


def unshard_runtime_parameters(*modules):
    """Materialize FSDP1/FSDP2 parameters for custom module execution.

    Construction eagerly enters both sharding contexts. The returned
    ``ExitStack`` owns those active contexts and reshards them when it exits.
    """

    stack = ExitStack()
    stack.enter_context(summon_full_parameters(*modules))
    stack.enter_context(_FSDP2UnshardContext(*modules))
    return stack


def checkpoint_unshard_context(*modules):
    """Prepare independently owned forward/recompute checkpoint contexts.

    Both contexts eagerly materialize parameters while this tuple is built.
    """

    return (
        unshard_runtime_parameters(*modules),
        unshard_runtime_parameters(*modules),
    )
