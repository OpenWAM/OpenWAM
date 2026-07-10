from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ActionChunk, DynamicsPrediction, PlanningContext


@dataclass(frozen=True)
class OpenWamFdmContextKeys:
    """Metadata keys used to carry Open-WAM FDM runtime state across branches."""

    session: str = "openwam_fdm_session"
    text_context: str = "openwam_text_context"
    negative_text_context: str = "openwam_negative_text_context"
    context_start_frame: str = "openwam_context_start_frame"
    current_frame: str = "openwam_current_frame"
    proprio_state: str = "openwam_proprio_state"
    hidden_proprio_history: str = "openwam_hidden_proprio_history"
    predicted_latents: str = "openwam_predicted_latents"


class OpenWamActionConditionedFdm:
    """Planner adapter for Open-WAM action-conditioned video prediction.

    The adapter intentionally targets the existing FDM rollout interface instead
    of a method-specific class. Any rollout object with `reset_and_warmup`,
    `infer_chunk`, `action_per_frame`, and `frame_chunk_size` can be used.
    """

    def __init__(
        self,
        *,
        rollout: Any,
        mode: Any,
        decode_video_fn: Any | None = None,
        keys: OpenWamFdmContextKeys | None = None,
        predicted_view_key: str | None = None,
        fork_session: bool = True,
        drop_text_conditioning: bool = True,
        short_action_horizon_strategy: str = "repeat_last",
        crop_decoded_video_to_action_steps: bool = True,
        project_prediction_to_context_view: bool = False,
        return_canonical_prediction_video: bool = False,
        return_context_canvas_prediction_video: bool = False,
        context_canvas_view_keys: Sequence[str] | None = None,
        predicted_view_aliases: Mapping[str, str] | None = None,
        input_view_transform: str = "none",
    ) -> None:
        if bool(return_canonical_prediction_video) and bool(return_context_canvas_prediction_video):
            raise ValueError(
                "OpenWAM FDM prediction output cannot be both canonical and context-canvas. "
                "Use return_canonical_prediction_video for raw Open-WAM debug videos, or "
                "return_context_canvas_prediction_video for policy/evaluator-facing videos."
            )
        if bool(return_context_canvas_prediction_video) and not bool(project_prediction_to_context_view):
            raise ValueError(
                "OpenWAM FDM context-canvas prediction output requires project_prediction_to_context_view=True."
            )
        self.rollout = rollout
        self.mode = mode
        self.decode_video_fn = decode_video_fn
        self.keys = keys or OpenWamFdmContextKeys()
        self.predicted_view_key = predicted_view_key
        self.fork_session = bool(fork_session)
        self.drop_text_conditioning = bool(drop_text_conditioning)
        self.short_action_horizon_strategy = str(short_action_horizon_strategy)
        self.crop_decoded_video_to_action_steps = bool(crop_decoded_video_to_action_steps)
        self.project_prediction_to_context_view = bool(project_prediction_to_context_view)
        self.return_canonical_prediction_video = bool(return_canonical_prediction_video)
        self.return_context_canvas_prediction_video = bool(return_context_canvas_prediction_video)
        self.context_canvas_view_keys = tuple(str(key) for key in (context_canvas_view_keys or ()))
        self.predicted_view_aliases = dict(predicted_view_aliases or {})
        self.input_view_transform = _normalize_view_transform(input_view_transform)

    def predict(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> DynamicsPrediction:
        session = self._resolve_session(context)
        session = _fork_session(session) if self.fork_session else session
        requested_action_steps = int(action_chunk.actions.shape[0])
        fdm_actions = self._adapt_action_horizon(action_chunk.actions)
        raw_actions = _action_chunk_to_tensor(fdm_actions, device=self._runtime_device())
        with _torch_inference_mode():
            chunk = self.rollout.infer_chunk(
                session=session,
                mode=self.mode,
                raw_action_chunk=raw_actions,
                video_condition_latents=None,
                seed=seed,
                drop_text_conditioning=self.drop_text_conditioning,
                proprio_state=_optional_tensor(
                    context.metadata.get(self.keys.proprio_state),
                    device=self._runtime_device(),
                ),
            )
        predicted_video = None
        next_views = None
        predicted_latents = chunk.predicted_latents.detach().cpu()
        if self.decode_video_fn is not None:
            predicted_video, next_views = self.decode_latent_sequence_for_video(
                context,
                predicted_latents,
                max_frames=requested_action_steps if self.crop_decoded_video_to_action_steps else None,
                return_next_views=True,
            )
        current_frame = int(context.metadata.get(self.keys.current_frame, 0)) + int(self.rollout.frame_chunk_size)
        next_context = context.with_prediction(
            predicted_video=predicted_video,
            views=next_views,
            metadata_updates={
                self.keys.session: chunk.session,
                self.keys.current_frame: current_frame,
                "openwam_fdm_debug": {
                    **dict(getattr(chunk, "debug", {}) or {}),
                    "policy_action_steps": requested_action_steps,
                    "internal_action_steps": int(fdm_actions.shape[0]),
                    "padded_action_steps": int(fdm_actions.shape[0]) - requested_action_steps,
                    "cropped_decoded_video_to_action_steps": bool(self.crop_decoded_video_to_action_steps),
                    "projected_prediction_to_context_view": bool(self.project_prediction_to_context_view),
                    "returned_canonical_prediction_video": bool(self.return_canonical_prediction_video),
                    "returned_context_canvas_prediction_video": bool(self.return_context_canvas_prediction_video),
                },
            },
        )
        return DynamicsPrediction(
            predicted_video=predicted_video,
            next_context=next_context,
            metadata={
                "debug": dict(getattr(chunk, "debug", {}) or {}),
                "policy_action_steps": requested_action_steps,
                "internal_action_steps": int(fdm_actions.shape[0]),
                "padded_action_steps": int(fdm_actions.shape[0]) - requested_action_steps,
                self.keys.predicted_latents: predicted_latents,
            },
        )

    def decode_latent_sequence_for_video(
        self,
        context: PlanningContext,
        latents: Any,
        *,
        max_frames: int | None = None,
        return_next_views: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, Mapping[str, np.ndarray] | None]:
        """Decode Open-WAM FDM latents into the same video space as `predict`.

        Online planning still decodes one chunk at a time so future policy
        context is available immediately. Debug artifacts can call this with a
        stitched latent sequence to avoid per-chunk VAE decode discontinuities
        while preserving the same projection/view-transform contract.
        """

        if self.decode_video_fn is None:
            raise ValueError("Cannot decode Open-WAM FDM latents without decode_video_fn.")
        with _torch_inference_mode():
            predicted_video = self.decode_video_fn(latents)
        if max_frames is not None:
            predicted_video = _truncate_decoded_prediction(
                predicted_video,
                max_frames=int(max_frames),
            )
        predicted_video = _decoded_prediction_to_uint8(predicted_video)
        context_prediction = predicted_video
        if self.project_prediction_to_context_view:
            canonical_prediction = predicted_video
            context_prediction = _project_prediction_to_context_view(
                predicted_video,
                current_views=context.views,
                predicted_view_key=self.predicted_view_key,
                predicted_view_aliases=self.predicted_view_aliases,
                placements=_canonical_placements(self.rollout),
            )
            context_prediction = _transform_decoded_prediction(
                context_prediction,
                transform=_inverse_view_transform(self.input_view_transform),
            )
            context_prediction = _decoded_prediction_to_uint8(context_prediction)
            if self.return_context_canvas_prediction_video:
                predicted_video = _prediction_canvas_from_context_views(
                    context_prediction,
                    current_views=context.views,
                    view_keys=self.context_canvas_view_keys,
                    predicted_view_aliases=self.predicted_view_aliases,
                )
            elif not self.return_canonical_prediction_video:
                predicted_video = _prediction_video_for_key(
                    context_prediction,
                    predicted_view_key=self.predicted_view_key,
                )
            else:
                predicted_video = canonical_prediction
        next_views = None
        if return_next_views:
            next_views, _ = _views_from_decoded_prediction(
                context.views,
                context_prediction,
                predicted_view_key=self.predicted_view_key,
            )
            return predicted_video, next_views
        return predicted_video

    def warm_context(
        self,
        context: PlanningContext,
        *,
        video_context_latents: Any,
        action_context: Any,
        action_space: str = "raw",
        action_conditioning_mode: str = "forced_action_joint_fdm",
    ) -> PlanningContext:
        """Create an Open-WAM FDM cache/session and store it in context metadata."""

        proprio_state = _optional_tensor(context.metadata.get(self.keys.proprio_state), device=self._runtime_device())
        hidden_proprio_history = context.metadata.get(self.keys.hidden_proprio_history)
        if hidden_proprio_history is None and proprio_state is not None:
            hidden_proprio_history = _repeat_proprio_for_video_context(
                proprio_state,
                video_context_latents=video_context_latents,
            )
        else:
            hidden_proprio_history = _optional_tensor(hidden_proprio_history, device=self._runtime_device())
        with _torch_inference_mode():
            session = self.rollout.reset_and_warmup(
                task_text=(context.task_text,),
                video_context=video_context_latents,
                action_context=action_context,
                text_context=context.metadata.get(self.keys.text_context),
                negative_text_context=context.metadata.get(self.keys.negative_text_context),
                context_start_frame=int(context.metadata.get(self.keys.context_start_frame, 0)),
                action_space=action_space,
                action_conditioning_mode=action_conditioning_mode,
                drop_text_conditioning=self.drop_text_conditioning,
                proprio_state=proprio_state,
                hidden_proprio_history=hidden_proprio_history,
            )
        return context.with_prediction(
            predicted_video=context.predicted_video,
            metadata_updates={
                self.keys.session: session,
                self.keys.current_frame: int(context.metadata.get(self.keys.current_frame, 0)),
            },
        )

    def warm_context_from_views(
        self,
        context: PlanningContext,
        *,
        view_aliases: Mapping[str, str] | None = None,
        action_dim: int | None = None,
        action_space: str = "raw",
        action_conditioning_mode: str = "forced_action_joint_fdm",
    ) -> PlanningContext:
        """Encode policy views and warm a persistent Open-WAM FDM session."""

        pipeline = self.rollout.runner.pipeline
        device = self._runtime_device()
        views = _views_to_tensors(
            context.views,
            view_aliases=view_aliases,
            device=device,
            transform=self.input_view_transform,
        )
        with _torch_inference_mode():
            canonical = pipeline.canonicalize(views)
            latents = pipeline.visual_tower.frontend.encode_video(
                canonical.video,
                placements=canonical.placements,
                reset_reference_cache=True,
            )
        resolved_action_dim = int(action_dim or _resolve_rollout_action_dim(self.rollout))
        action_context = _zeros_like_action_context(
            video_context_latents=latents,
            action_per_frame=int(self.rollout.action_per_frame),
            action_dim=resolved_action_dim,
            device=device,
        )
        return self.warm_context(
            context,
            video_context_latents=latents,
            action_context=action_context,
            action_space=action_space,
            action_conditioning_mode=action_conditioning_mode,
        )

    def _resolve_session(self, context: PlanningContext) -> Any:
        try:
            return context.metadata[self.keys.session]
        except KeyError as exc:
            raise ValueError(
                "OpenWamActionConditionedFdm requires a warmed planning context. "
                "Call warm_context(...) with encoded video/action context before planning."
            ) from exc

    def _runtime_device(self) -> Any:
        try:
            parameter = next(self.rollout.runner.pipeline.visual_tower.parameters())
            return parameter.device
        except Exception:
            return None

    def expected_action_steps_per_chunk(self) -> int:
        return int(self.rollout.frame_chunk_size) * int(self.rollout.action_per_frame)

    def _adapt_action_horizon(self, actions: np.ndarray) -> np.ndarray:
        array = np.asarray(actions)
        if array.ndim != 2:
            raise ValueError(f"Open-WAM FDM action chunks must have shape [T,D], got {array.shape}.")
        expected = self.expected_action_steps_per_chunk()
        action_steps = int(array.shape[0])
        if action_steps == expected:
            return array.astype(np.float32, copy=False)
        if action_steps > expected:
            raise ValueError(
                "Open-WAM GJD FDM expects one full low-level action horizon per video chunk: "
                f"frame_chunk_size={int(self.rollout.frame_chunk_size)}, "
                f"action_per_frame={int(self.rollout.action_per_frame)}, expected {expected} actions, "
                f"got {action_steps}. Shorter policy/control horizons can be padded for FDM scoring, "
                "but longer horizons would silently truncate control intent."
            )
        if action_steps <= 0:
            raise ValueError("Open-WAM FDM action chunks must contain at least one action.")
        if self.short_action_horizon_strategy != "repeat_last":
            raise ValueError(
                "Open-WAM FDM received a shorter policy horizon than its internal chunk horizon: "
                f"expected {expected}, got {action_steps}. "
                "Set short_action_horizon_strategy='repeat_last' to pad only the internal FDM input."
            )
        padding = np.repeat(array[-1:], expected - action_steps, axis=0)
        return np.concatenate([array, padding], axis=0).astype(np.float32, copy=False)


def make_zero_action_context(action_per_frame: int, action_dim: int, *, frame_count: int = 1) -> np.ndarray:
    return np.zeros((1, int(frame_count) * int(action_per_frame), int(action_dim)), dtype=np.float32)


def merge_context_metadata(context: PlanningContext, metadata: Mapping[str, Any]) -> PlanningContext:
    return context.with_prediction(predicted_video=context.predicted_video, metadata_updates=metadata)


def _action_chunk_to_tensor(actions: np.ndarray, *, device: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("OpenWAM FDM planning requires torch.") from exc
    tensor = torch.from_numpy(np.asarray(actions, dtype=np.float32)).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _optional_tensor(value: Any, *, device: Any) -> Any:
    if value is None:
        return None
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("OpenWAM FDM planning requires torch.") from exc
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device=device, dtype=torch.float32)
    return tensor


def _repeat_proprio_for_video_context(proprio_state: Any, *, video_context_latents: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("OpenWAM FDM planning requires torch.") from exc
    if not isinstance(proprio_state, torch.Tensor):
        proprio_state = torch.as_tensor(proprio_state, dtype=torch.float32)
    if proprio_state.ndim == 1:
        proprio_state = proprio_state.unsqueeze(0)
    context_frames = int(getattr(video_context_latents, "shape")[2])
    return proprio_state.unsqueeze(1).expand(-1, context_frames, -1).contiguous()


def _fork_session(session: Any) -> Any:
    try:
        return copy.deepcopy(session)
    except Exception:
        forked = copy.copy(session)
        policy_state = copy.copy(getattr(session, "policy_state", None))
        if policy_state is not None:
            cache = getattr(policy_state, "cache", None)
            if isinstance(cache, dict):
                policy_state.cache = dict(cache)
            forked.policy_state = policy_state
        return forked


def _views_to_tensors(
    views: Mapping[str, np.ndarray],
    *,
    view_aliases: Mapping[str, str] | None,
    device: Any,
    transform: str = "none",
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("OpenWAM FDM planning requires torch.") from exc
    tensors: dict[str, Any] = {}
    for key, value in views.items():
        array = np.asarray(value)
        array = _transform_view_array(array, transform=transform)
        if array.ndim == 3:
            array = array[None, ...]
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if device is not None:
            tensor = tensor.to(device=device)
        tensors[str(key)] = tensor
    for target_key, source_key in dict(view_aliases or {}).items():
        if source_key in tensors and target_key not in tensors:
            tensors[str(target_key)] = tensors[source_key]
    return tensors


def _normalize_view_transform(transform: str | None) -> str:
    value = "none" if transform is None else str(transform)
    aliases = {
        "none": "none",
        "identity": "none",
        "vertical_flip": "vertical_flip",
        "flip_ud": "vertical_flip",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Open-WAM FDM view transform "
            f"{transform!r}; expected one of {sorted(aliases)}."
        ) from exc


def _inverse_view_transform(transform: str) -> str:
    normalized = _normalize_view_transform(transform)
    if normalized in {"none", "vertical_flip"}:
        return normalized
    raise ValueError(f"Unsupported Open-WAM FDM inverse view transform {transform!r}.")


def _transform_view_array(array: np.ndarray, *, transform: str) -> np.ndarray:
    normalized = _normalize_view_transform(transform)
    value = np.asarray(array)
    if normalized == "none":
        return value
    if normalized == "vertical_flip":
        if value.ndim == 3:
            return np.ascontiguousarray(value[::-1])
        if value.ndim == 4:
            return np.ascontiguousarray(value[:, ::-1])
        raise ValueError(
            "Open-WAM FDM vertical_flip expects RGB frame [H,W,C] or video [T,H,W,C], "
            f"got {value.shape}."
        )
    raise ValueError(f"Unsupported Open-WAM FDM view transform {transform!r}.")


def _transform_decoded_prediction(decoded_prediction: Any, *, transform: str) -> Any:
    normalized = _normalize_view_transform(transform)
    if decoded_prediction is None or normalized == "none":
        return decoded_prediction
    if isinstance(decoded_prediction, Mapping):
        return {
            key: _transform_view_array(np.asarray(value), transform=normalized)
            for key, value in decoded_prediction.items()
        }
    return _transform_view_array(np.asarray(decoded_prediction), transform=normalized)


def _resolve_rollout_action_dim(rollout: Any) -> int:
    pipeline = rollout.runner.pipeline
    for owner in (getattr(pipeline, "policy_variant", None), getattr(pipeline, "action_decoder", None)):
        value = getattr(owner, "action_dim", None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer Open-WAM rollout action_dim; pass action_dim explicitly.")


def _zeros_like_action_context(
    *,
    video_context_latents: Any,
    action_per_frame: int,
    action_dim: int,
    device: Any,
) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("OpenWAM FDM planning requires torch.") from exc
    batch_size = int(video_context_latents.shape[0])
    context_frames = int(video_context_latents.shape[2])
    return torch.zeros(
        batch_size,
        context_frames * int(action_per_frame),
        int(action_dim),
        device=device,
        dtype=torch.float32,
    )


def _torch_inference_mode():
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return nullcontext()
    return torch.inference_mode()


def _views_from_decoded_prediction(
    current_views: Mapping[str, np.ndarray],
    decoded_prediction: Any,
    *,
    predicted_view_key: str | None,
) -> tuple[Mapping[str, np.ndarray] | None, np.ndarray | None]:
    if decoded_prediction is None:
        return None, None
    if isinstance(decoded_prediction, Mapping):
        next_views = dict(current_views)
        representative: np.ndarray | None = None
        for key, value in decoded_prediction.items():
            video = np.asarray(value)
            if video.ndim != 4:
                raise ValueError(
                    "Decoded Open-WAM FDM view predictions must have shape [T,H,W,C], "
                    f"got key={key!r}, shape={video.shape}."
                )
            next_views[str(key)] = video[-1]
            if representative is None:
                representative = video
        return next_views, representative
    video = np.asarray(decoded_prediction)
    if video.ndim != 4:
        raise ValueError(f"Decoded Open-WAM FDM prediction must have shape [T,H,W,C], got {video.shape}.")
    if predicted_view_key is None:
        return None, video
    next_views = dict(current_views)
    next_views[predicted_view_key] = video[-1]
    return next_views, video


def _prediction_video_for_key(decoded_prediction: Any, *, predicted_view_key: str | None) -> np.ndarray:
    if isinstance(decoded_prediction, Mapping):
        if predicted_view_key is None:
            raise ValueError("A predicted_view_key is required to select a projected prediction from a mapping.")
        if predicted_view_key not in decoded_prediction:
            raise KeyError(
                f"Projected prediction is missing predicted_view_key={predicted_view_key!r}; "
                f"available={sorted(decoded_prediction)}."
            )
        video = np.asarray(decoded_prediction[predicted_view_key])
        if video.ndim != 4:
            raise ValueError(
                "Projected Open-WAM FDM prediction must have shape [T,H,W,C], "
                f"got key={predicted_view_key!r}, shape={video.shape}."
            )
        return video
    video = np.asarray(decoded_prediction)
    if video.ndim != 4:
        raise ValueError(f"Decoded Open-WAM FDM prediction must have shape [T,H,W,C], got {video.shape}.")
    return video


def _prediction_canvas_from_context_views(
    decoded_prediction: Any,
    *,
    current_views: Mapping[str, np.ndarray],
    view_keys: Sequence[str],
    predicted_view_aliases: Mapping[str, str] | None,
) -> np.ndarray:
    if not isinstance(decoded_prediction, Mapping):
        return _prediction_video_for_key(decoded_prediction, predicted_view_key=None)
    resolved_view_keys = [str(key) for key in view_keys if str(key) in decoded_prediction]
    if not resolved_view_keys:
        aliases = dict(predicted_view_aliases or {})
        for source_key in aliases:
            context_key = aliases[source_key]
            if context_key in decoded_prediction and context_key not in resolved_view_keys:
                resolved_view_keys.append(context_key)
    if not resolved_view_keys:
        resolved_view_keys = [str(key) for key in decoded_prediction]
    videos: list[np.ndarray] = []
    for key in resolved_view_keys:
        if key not in decoded_prediction:
            raise KeyError(
                f"Projected prediction is missing context-canvas view {key!r}; "
                f"available={sorted(decoded_prediction)}."
            )
        video = np.asarray(decoded_prediction[key])
        if video.ndim != 4 or int(video.shape[-1]) != 3:
            raise ValueError(
                "Projected Open-WAM FDM context-canvas prediction videos must have shape [T,H,W,3], "
                f"got key={key!r}, shape={video.shape}."
            )
        if key in current_views:
            video = _project_video_to_frame_shape(video, target_shape=np.asarray(current_views[key]).shape)
        videos.append(video)
    frame_counts = {int(video.shape[0]) for video in videos}
    if len(frame_counts) != 1:
        raise ValueError(
            "Projected Open-WAM FDM context-canvas views must have matching frame counts, "
            f"got {[video.shape for video in videos]}."
        )
    heights = {int(video.shape[1]) for video in videos}
    if len(heights) != 1:
        target_height = min(heights)
        videos = [
            _resize_video_nearest(video, height=target_height, width=int(video.shape[2]))
            for video in videos
        ]
    return np.ascontiguousarray(np.concatenate(videos, axis=2))


def _decoded_prediction_to_uint8(decoded_prediction: Any) -> Any:
    if decoded_prediction is None:
        return None
    if isinstance(decoded_prediction, Mapping):
        return {key: _video_to_uint8(value) for key, value in decoded_prediction.items()}
    return _video_to_uint8(decoded_prediction)


def _video_to_uint8(video: Any) -> np.ndarray:
    array = np.asarray(video)
    if array.dtype == np.uint8:
        return array
    if array.size and float(np.nanmax(array)) <= 1.0001:
        return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _truncate_decoded_prediction(decoded_prediction: Any, *, max_frames: int) -> Any:
    frame_count = int(max_frames)
    if frame_count <= 0:
        raise ValueError(f"max_frames must be positive, got {frame_count}.")
    if isinstance(decoded_prediction, Mapping):
        return {
            key: _truncate_video_frames(value, max_frames=frame_count)
            for key, value in decoded_prediction.items()
        }
    return _truncate_video_frames(decoded_prediction, max_frames=frame_count)


def _project_prediction_to_context_view(
    decoded_prediction: Any,
    *,
    current_views: Mapping[str, np.ndarray],
    predicted_view_key: str | None,
    predicted_view_aliases: Mapping[str, str] | None = None,
    placements: tuple[Any, ...] = (),
) -> Any:
    if predicted_view_key is None:
        raise ValueError("Open-WAM context-view projection requires predicted_view_key.")
    if predicted_view_key not in current_views:
        raise KeyError(
            f"Open-WAM context-view projection requires current view {predicted_view_key!r}; "
            f"available={sorted(current_views)}."
        )
    split_prediction = _split_canonical_prediction_to_context_views(
        decoded_prediction,
        current_views=current_views,
        predicted_view_aliases=predicted_view_aliases,
        placements=placements,
    )
    if split_prediction:
        return split_prediction
    target_frame = np.asarray(current_views[predicted_view_key])
    if target_frame.ndim != 3 or int(target_frame.shape[-1]) != 3:
        raise ValueError(f"Expected context RGB frame [H,W,3], got {target_frame.shape}.")
    if isinstance(decoded_prediction, Mapping):
        if predicted_view_key in decoded_prediction:
            return {
                predicted_view_key: _project_video_to_frame_shape(
                    decoded_prediction[predicted_view_key],
                    target_shape=target_frame.shape,
                )
            }
        if len(decoded_prediction) == 1:
            key, value = next(iter(decoded_prediction.items()))
            del key
            return {
                predicted_view_key: _project_video_to_frame_shape(value, target_shape=target_frame.shape)
            }
        raise ValueError(
            "Cannot project multi-view Open-WAM FDM prediction without a matching predicted_view_key; "
            f"keys={sorted(decoded_prediction)}."
        )
    return _project_video_to_frame_shape(decoded_prediction, target_shape=target_frame.shape)


def _split_canonical_prediction_to_context_views(
    decoded_prediction: Any,
    *,
    current_views: Mapping[str, np.ndarray],
    predicted_view_aliases: Mapping[str, str] | None,
    placements: tuple[Any, ...],
) -> dict[str, np.ndarray]:
    if isinstance(decoded_prediction, Mapping):
        return _split_mapping_prediction_to_context_views(
            decoded_prediction,
            current_views=current_views,
            predicted_view_aliases=predicted_view_aliases,
        )
    if not placements:
        return {}
    video = np.asarray(decoded_prediction)
    if video.ndim != 4 or int(video.shape[-1]) != 3:
        return {}
    aliases = dict(predicted_view_aliases or {})
    resolved: dict[str, np.ndarray] = {}
    for placement in placements:
        source_name = str(getattr(placement, "source_name", ""))
        context_key = aliases.get(source_name, source_name)
        if context_key not in current_views:
            continue
        top = int(getattr(placement, "top"))
        left = int(getattr(placement, "left"))
        height = int(getattr(placement, "height"))
        width = int(getattr(placement, "width"))
        if top < 0 or left < 0 or height <= 0 or width <= 0:
            continue
        if top + height > int(video.shape[1]) or left + width > int(video.shape[2]):
            continue
        view_video = np.ascontiguousarray(video[:, top : top + height, left : left + width])
        target_frame = np.asarray(current_views[context_key])
        resolved[context_key] = _project_video_to_frame_shape(view_video, target_shape=target_frame.shape)
    return resolved


def _split_mapping_prediction_to_context_views(
    decoded_prediction: Mapping[str, Any],
    *,
    current_views: Mapping[str, np.ndarray],
    predicted_view_aliases: Mapping[str, str] | None,
) -> dict[str, np.ndarray]:
    aliases = dict(predicted_view_aliases or {})
    resolved: dict[str, np.ndarray] = {}
    for key, value in decoded_prediction.items():
        context_key = aliases.get(str(key), str(key))
        if context_key not in current_views:
            continue
        target_frame = np.asarray(current_views[context_key])
        resolved[context_key] = _project_video_to_frame_shape(value, target_shape=target_frame.shape)
    return resolved


def _project_video_to_frame_shape(video: Any, *, target_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(video)
    if array.ndim != 4 or int(array.shape[-1]) != 3:
        raise ValueError(f"Expected decoded video [T,H,W,3], got {array.shape}.")
    target_height, target_width, target_channels = map(int, target_shape)
    if target_channels != 3:
        raise ValueError(f"Expected target RGB frame shape [H,W,3], got {target_shape}.")
    if int(array.shape[1]) == target_height and int(array.shape[2]) == target_width:
        return array
    if int(array.shape[1]) == target_height and int(array.shape[2]) > target_width:
        return np.ascontiguousarray(array[:, :, :target_width])
    return _resize_video_nearest(array, height=target_height, width=target_width)


def _canonical_placements(rollout: Any) -> tuple[Any, ...]:
    pipeline = getattr(getattr(rollout, "runner", None), "pipeline", None)
    preprocessor = getattr(pipeline, "preprocessor", None)
    placements = getattr(preprocessor, "placements", ())
    return tuple(placements or ())


def _resize_video_nearest(video: np.ndarray, *, height: int, width: int) -> np.ndarray:
    array = np.asarray(video)
    y_indices = np.linspace(0, int(array.shape[1]) - 1, int(height)).round().astype(np.int64)
    x_indices = np.linspace(0, int(array.shape[2]) - 1, int(width)).round().astype(np.int64)
    return np.ascontiguousarray(array[:, y_indices][:, :, x_indices])


def _truncate_video_frames(video: Any, *, max_frames: int) -> np.ndarray:
    array = np.asarray(video)
    if array.ndim != 4:
        raise ValueError(f"Decoded Open-WAM FDM prediction must have shape [T,H,W,C], got {array.shape}.")
    return array[: min(int(max_frames), int(array.shape[0]))]
