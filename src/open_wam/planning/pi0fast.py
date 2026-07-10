from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import ActionChunk, PlanningContext


@dataclass(frozen=True)
class Pi0FastBatchMapping:
    """Map Open-WAM planning observations into LeRobot PI0-family batch keys."""

    image_keys: Mapping[str, str]
    image_transform: str = "libero_180"
    state_key: str = "observation.state"
    task_key: str = "task"
    device: str = "cuda"
    dtype: str = "float32"
    # Raw dataset-state dimension expected by the saved LeRobot preprocessor.
    # `lerobot/pi0fast-libero` declares an internal max_state_dim of 32, but its
    # normalizer statistics are 8D and the pi0-fast processor tokenizes that
    # normalized raw state. Do not pre-pad to the model's max_state_dim here.
    state_dim: int = 8


class Pi0FastPolicySampler:
    """Optional LeRobot PI0/PI0-fast adapter for planner action sampling."""

    def __init__(
        self,
        *,
        policy: Any,
        batch_mapping: Pi0FastBatchMapping,
        policy_family: str = "pi0fast",
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        action_clip: float | None = 1.0,
        short_horizon_strategy: str = "repeat_last",
        action_selection_mode: str = "select_action",
    ) -> None:
        self.policy = policy
        self.batch_mapping = batch_mapping
        self.policy_family = str(policy_family)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_clip = action_clip
        self.short_horizon_strategy = str(short_horizon_strategy)
        self.action_selection_mode = str(action_selection_mode)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        batch_mapping: Pi0FastBatchMapping,
        policy_family: str = "pi0fast",
        action_clip: float | None = 1.0,
        use_saved_processors: bool = True,
        short_horizon_strategy: str = "repeat_last",
        action_selection_mode: str = "select_action",
        compile_model: bool | None = None,
        **kwargs: Any,
    ) -> "Pi0FastPolicySampler":
        policy_cls = _load_lerobot_policy_class(policy_family)
        policy_config = None
        if compile_model is not None:
            policy_config = _load_lerobot_policy_config(model_path, **kwargs)
            if hasattr(policy_config, "compile_model"):
                _set_config_attr(policy_config, "compile_model", bool(compile_model))
        try:
            policy = policy_cls.from_pretrained(model_path, config=policy_config, **kwargs)
        except Exception as exc:
            message = _exception_chain_text(exc)
            if "google/paligemma-3b-pt-224" in message or "gated repo" in message.lower():
                raise RuntimeError(
                    f"Loading {policy_family} requires authenticated Hugging Face access to the gated "
                    "`google/paligemma-3b-pt-224` tokenizer/model assets. Log in with a token that has "
                    "accepted that model's license, then retry this runner."
                ) from exc
            raise
        preprocessor = None
        postprocessor = None
        if use_saved_processors:
            preprocessor, postprocessor = _load_lerobot_policy_processors(
                model_path,
                policy_family=policy_family,
                **kwargs,
            )
        return cls(
            policy=policy,
            batch_mapping=batch_mapping,
            policy_family=policy_family,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            action_clip=action_clip,
            short_horizon_strategy=short_horizon_strategy,
            action_selection_mode=action_selection_mode,
        )

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.preprocessor is not None and hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if self.postprocessor is not None and hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def snapshot_state(self) -> dict[str, Any]:
        """Capture mutable rollout state for speculative planner sampling.

        LeRobot PI0-family policies keep action queues inside the policy object.
        Candidate planning may query several hypothetical chunks from the same
        real observation; those queries must not mutate the live policy queue
        used by later replans.
        """

        return {
            "policy_action_queue": _snapshot_deque(getattr(self.policy, "_action_queue", None)),
            "policy_queues": _snapshot_queue_mapping(getattr(self.policy, "_queues", None)),
            "policy_temperature": _get_policy_temperature(self.policy),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if "policy_action_queue" in state and hasattr(self.policy, "_action_queue"):
            setattr(self.policy, "_action_queue", _restore_deque(state["policy_action_queue"]))
        if "policy_queues" in state and hasattr(self.policy, "_queues"):
            setattr(self.policy, "_queues", _restore_queue_mapping(state["policy_queues"]))
        if "policy_temperature" in state:
            _restore_policy_temperature(self.policy, state["policy_temperature"])

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ) -> list[ActionChunk]:
        if seed is not None:
            _seed_torch(seed)
        _set_policy_temperature(self.policy, temperature)
        chunks: list[ActionChunk] = []
        for sample_index in range(int(num_samples)):
            batch = build_pi0fast_raw_batch(context, self.batch_mapping)
            if self.preprocessor is not None:
                batch = self.preprocessor(batch)
            if self.action_selection_mode == "select_action":
                actions = self._select_native_action_chunk(batch, target_steps=int(chunk_action_steps))
            elif self.action_selection_mode == "predict_chunk":
                actions = _predict_action_chunk(self.policy, batch)
                if self.postprocessor is not None:
                    actions = self.postprocessor(actions)
            else:
                raise ValueError(
                    f"Unsupported {self.policy_family} action_selection_mode={self.action_selection_mode!r}."
                )
            actions = _to_numpy(actions).astype(np.float32, copy=False)
            if actions.ndim == 3:
                actions = actions[0]
            if actions.ndim != 2:
                raise ValueError(
                    f"{self.policy_family} action output must have shape [T,D] or [B,T,D], got {actions.shape}."
                )
            native_action_steps = int(actions.shape[0])
            actions = _adapt_action_horizon(
                actions,
                target_steps=int(chunk_action_steps),
                strategy=self.short_horizon_strategy,
            )
            if self.action_clip is not None:
                actions = np.clip(actions, -float(self.action_clip), float(self.action_clip))
            chunks.append(
                ActionChunk(
                    actions=actions,
                    metadata={
                        "policy": self.policy_family,
                        "sample_index": sample_index,
                        "temperature": float(temperature),
                        "native_action_steps": native_action_steps,
                        "requested_action_steps": int(chunk_action_steps),
                        "short_horizon_strategy": self.short_horizon_strategy,
                        "action_selection_mode": self.action_selection_mode,
                    },
                )
            )
        return chunks

    def _select_native_action_chunk(self, batch: dict[str, Any], *, target_steps: int) -> Any:
        native_steps = int(getattr(getattr(self.policy, "config", None), "n_action_steps", target_steps))
        if int(target_steps) != native_steps:
            raise ValueError(
                f"{self.policy_family} action_selection_mode='select_action' must use the policy's native action horizon "
                f"to avoid stale internal action-queue leftovers: native_steps={native_steps}, requested={target_steps}. "
                "Use action_selection_mode='predict_chunk' for planner-side horizon adaptation."
            )
        if not hasattr(self.policy, "select_action"):
            raise AttributeError(f"{self.policy_family} select_action mode requires policy.select_action().")
        selected_actions: list[Any] = []
        for _ in range(native_steps):
            action = self.policy.select_action(batch)
            if self.postprocessor is not None:
                action = self.postprocessor(action)
            selected_actions.append(_to_numpy(action))
        return np.stack(selected_actions, axis=1)


def build_pi0fast_raw_batch(context: PlanningContext, mapping: Pi0FastBatchMapping) -> dict[str, Any]:
    """Build the raw LeRobot batch consumed by the saved pi0-fast preprocessor."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - torch is a project extra
        raise ModuleNotFoundError("pi0-fast batching requires torch.") from exc
    batch: dict[str, Any] = {}
    for model_key, context_key in mapping.image_keys.items():
        frame = _resolve_context_frame(context, context_key)
        tensor = torch.from_numpy(_image_to_chw_float(_transform_image(frame, mapping.image_transform)))
        batch[model_key] = tensor
    if context.state is not None:
        state = torch.from_numpy(_pad_vector(np.asarray(context.state, dtype=np.float32), mapping.state_dim))
        batch[mapping.state_key] = state
    if context.task_text is not None:
        batch[mapping.task_key] = context.task_text
    return batch


def build_pi0fast_batch(context: PlanningContext, mapping: Pi0FastBatchMapping) -> dict[str, Any]:
    """Backward-compatible minimal batch builder for tests without saved processors."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - torch is a project extra
        raise ModuleNotFoundError("pi0-fast batching requires torch.") from exc
    batch = build_pi0fast_raw_batch(context, mapping)
    for key, value in list(batch.items()):
        if hasattr(value, "to"):
            batch[key] = value.unsqueeze(0).to(device=mapping.device)
    return batch


def _resolve_context_frame(context: PlanningContext, key: str) -> np.ndarray:
    if key == "__predicted_last__":
        if context.predicted_video is None:
            raise ValueError("Context has no predicted_video for image key '__predicted_last__'.")
        video = np.asarray(context.predicted_video)
        if video.ndim != 4:
            raise ValueError(f"predicted_video must have shape [T,H,W,C], got {video.shape}.")
        return np.asarray(video[-1])
    if key not in context.views:
        raise KeyError(f"Planning context has no view {key!r}; available={sorted(context.views)}.")
    return np.asarray(context.views[key])


def _image_to_chw_float(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {image.shape}.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.moveaxis(image.astype(np.float32) / 255.0, -1, 0)


def _transform_image(frame: np.ndarray, transform: str) -> np.ndarray:
    image = np.asarray(frame)
    if transform == "none":
        return image
    if transform == "libero_180":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(f"Unsupported pi0-fast image_transform={transform!r}.")


def _pad_vector(vector: np.ndarray, size: int) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    size = int(size)
    if flat.shape[0] == size:
        return flat
    if flat.shape[0] > size:
        return flat[:size]
    out = np.zeros(size, dtype=np.float32)
    out[: flat.shape[0]] = flat
    return out


def _adapt_action_horizon(actions: np.ndarray, *, target_steps: int, strategy: str) -> np.ndarray:
    target_steps = int(target_steps)
    if target_steps <= 0:
        raise ValueError(f"chunk_action_steps must be positive, got {target_steps}.")
    if actions.shape[0] >= target_steps:
        return actions[:target_steps]
    if actions.shape[0] <= 0:
        raise ValueError("pi0-fast produced an empty action chunk.")
    if strategy == "error":
        raise ValueError(
            "pi0-fast produced fewer actions than requested: "
            f"native_steps={actions.shape[0]}, requested={target_steps}. "
            "Use short_horizon_strategy='repeat_last' only when this policy-side adapter is acceptable."
        )
    if strategy != "repeat_last":
        raise ValueError(f"Unsupported pi0-fast short_horizon_strategy={strategy!r}.")
    padding = np.repeat(actions[-1:], target_steps - int(actions.shape[0]), axis=0)
    return np.concatenate([actions, padding], axis=0)


def _predict_action_chunk(policy: Any, batch: dict[str, Any]) -> Any:
    if hasattr(policy, "predict_action_chunk"):
        output = policy.predict_action_chunk(batch)
    elif hasattr(policy, "select_action"):
        output = policy.select_action(batch)
    else:
        raise AttributeError("pi0-fast policy object has neither predict_action_chunk() nor select_action().")
    return output


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _set_policy_temperature(policy: Any, temperature: float) -> None:
    value = float(temperature)
    if hasattr(policy, "config") and hasattr(policy.config, "temperature"):
        policy.config.temperature = value
    elif hasattr(policy, "temperature"):
        policy.temperature = value


def _get_policy_temperature(policy: Any) -> float | None:
    if hasattr(policy, "config") and hasattr(policy.config, "temperature"):
        return float(policy.config.temperature)
    if hasattr(policy, "temperature"):
        return float(policy.temperature)
    return None


def _restore_policy_temperature(policy: Any, value: float | None) -> None:
    if value is None:
        return
    _set_policy_temperature(policy, float(value))


def _snapshot_deque(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, deque):
        return None
    return {"items": list(value), "maxlen": value.maxlen}


def _restore_deque(snapshot: Any) -> Any:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        return snapshot
    return deque(snapshot.get("items", ()), maxlen=snapshot.get("maxlen"))


def _snapshot_queue_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return {str(key): _snapshot_deque(queue) for key, queue in value.items()}


def _restore_queue_mapping(snapshot: Any) -> Any:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        return snapshot
    return {key: _restore_deque(queue_snapshot) for key, queue_snapshot in snapshot.items()}


def _seed_torch(seed: int) -> None:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_lerobot_policy_class(policy_family: str) -> Any:
    family = str(policy_family)
    try:
        if family == "pi0fast":
            from lerobot.policies.pi0_fast import PI0FastPolicy

            return PI0FastPolicy
        if family == "pi0":
            from lerobot.policies.pi0 import PI0Policy

            return PI0Policy
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "LeRobot PI0-family policies are not installed. Install LeRobot with PI0 support before "
            "using Pi0FastPolicySampler, e.g. `uv pip install 'lerobot[pi]'`."
        ) from exc
    raise ValueError(f"Unsupported LeRobot policy_family={policy_family!r}; expected 'pi0fast' or 'pi0'.")


def _import_lerobot_policy_processor(policy_family: str) -> None:
    family = str(policy_family)
    try:
        if family == "pi0fast":
            import lerobot.policies.pi0_fast.processor_pi0_fast  # noqa: F401

            return
        if family == "pi0":
            import lerobot.policies.pi0.processor_pi0  # noqa: F401

            return
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(f"LeRobot processor support is required for {family} saved processors.") from exc
    raise ValueError(f"Unsupported LeRobot policy_family={policy_family!r}; expected 'pi0fast' or 'pi0'.")


def _load_lerobot_policy_config(model_path: str, **kwargs: Any) -> Any:
    try:
        from lerobot.configs.policies import PreTrainedConfig
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("LeRobot policy config support is required for PI0-family policies.") from exc

    passthrough_keys = {
        "force_download",
        "resume_download",
        "proxies",
        "token",
        "cache_dir",
        "local_files_only",
        "revision",
    }
    load_kwargs = {key: value for key, value in kwargs.items() if key in passthrough_keys}
    return PreTrainedConfig.from_pretrained(pretrained_name_or_path=model_path, **load_kwargs)


def _set_config_attr(config: Any, key: str, value: Any) -> None:
    try:
        setattr(config, key, value)
    except Exception:
        object.__setattr__(config, key, value)


def _load_lerobot_policy_processors(
    model_path: str,
    *,
    policy_family: str = "pi0fast",
    **kwargs: Any,
) -> tuple[Any, Any]:
    try:
        from lerobot.processor import DataProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("LeRobot processor support is required for PI0-family saved processors.") from exc
    _import_lerobot_policy_processor(policy_family)

    passthrough_keys = {
        "force_download",
        "resume_download",
        "proxies",
        "token",
        "cache_dir",
        "local_files_only",
        "revision",
    }
    load_kwargs = {key: value for key, value in kwargs.items() if key in passthrough_keys}
    try:
        preprocessor = DataProcessorPipeline.from_pretrained(
            model_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            **load_kwargs,
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            model_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            **load_kwargs,
        )
    except Exception as exc:
        message = _exception_chain_text(exc)
        if "google/paligemma-3b-pt-224" in message or "gated repo" in message.lower():
            raise RuntimeError(
                f"Loading {policy_family} processors requires authenticated Hugging Face access to "
                "`google/paligemma-3b-pt-224`."
            ) from exc
        raise
    return preprocessor, postprocessor


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(parts)
