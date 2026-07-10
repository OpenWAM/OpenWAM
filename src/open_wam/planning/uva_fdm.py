from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .contracts import ActionChunk, DynamicsPrediction, PlanningContext


@dataclass(frozen=True)
class UvaFdmContextKeys:
    """Metadata keys used by the UVA planning adapter."""

    debug: str = "uva_fdm_debug"


class UvaLiberoActionConditionedFdm:
    """Forward-dynamics wrapper for UVA's released LIBERO checkpoint.

    This is an adapter for planning diagnostics, not a native Open-WAM model.
    The planner still passes Open-WAM/LIBERO raw 7D OSC candidate actions. This
    wrapper converts those actions into UVA's native 32-step absolute-EEF
    window without using any candidate-executed future state before scoring.
    """

    def __init__(
        self,
        *,
        uva_root: str | Path,
        checkpoint: str | Path = "checkpoints/libero10.ckpt",
        dataset_dir: str | Path = "data/libero_10",
        output_dir: str | Path,
        runtime_device: str = "cuda:0",
        view_key: str = "agentview_image",
        action_space: str = "openwam_raw7_delta_osc_libero_uva_frame",
        internal_horizon: int = 32,
        action_target_start: int = 15,
        short_action_horizon_strategy: str = "repeat_last",
        predicted_view_key: str | None = "agentview_image",
        resize_prediction_to_context_view: bool = True,
        keys: UvaFdmContextKeys | None = None,
    ) -> None:
        self.uva_root = Path(uva_root).expanduser()
        self.checkpoint = checkpoint
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir).expanduser()
        self.runtime_device = str(runtime_device)
        self.view_key = str(view_key)
        self.action_space = str(action_space)
        self.internal_horizon = int(internal_horizon)
        self.action_target_start = int(action_target_start)
        self.short_action_horizon_strategy = str(short_action_horizon_strategy)
        self.predicted_view_key = predicted_view_key
        self.resize_prediction_to_context_view = bool(resize_prediction_to_context_view)
        self.keys = keys or UvaFdmContextKeys()
        if self.internal_horizon <= 0:
            raise ValueError("UVA internal_horizon must be positive.")
        if self.action_target_start < 0 or self.action_target_start >= self.internal_horizon:
            raise ValueError(
                "UVA action_target_start must be inside the internal horizon, "
                f"got start={self.action_target_start}, horizon={self.internal_horizon}."
            )
        self._runtime = _load_uva_runtime(
            uva_root=self.uva_root,
            checkpoint=self.checkpoint,
            dataset_dir=self.dataset_dir,
            output_dir=self.output_dir,
            runtime_device=self.runtime_device,
        )

    def predict(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> DynamicsPrediction:
        del seed
        action_window = self._build_action_window(action_chunk.actions)
        frame_window = self._build_frame_window(context)
        state_window = self._build_state_window(context)
        debug_module = self._runtime["debug_module"]
        uva = self._runtime["uva"]
        with _torch_inference_mode():
            result = debug_module._run_uva_single_dense_window(
                cfg=uva["cfg"],
                policy=uva["policy"],
                tokenizer=uva["tokenizer"],
                rotation_transformer=uva["rotation_transformer"],
                task_text=context.task_text or "",
                frames=frame_window,
                actions=action_window,
                states=state_window,
                action_space=self.action_space,
                runtime_device=self._runtime["device"],
            )
        predicted_video = _ensure_uint8_video(result["fdm_video"])
        context_frame = np.asarray(context.views[self.view_key], dtype=np.uint8)
        if self.resize_prediction_to_context_view:
            predicted_video = _resize_video_nearest(
                predicted_video,
                height=int(context_frame.shape[0]),
                width=int(context_frame.shape[1]),
            )
        next_views = None
        if self.predicted_view_key is not None and predicted_video.shape[0] > 0:
            next_views = dict(context.views)
            next_views[str(self.predicted_view_key)] = predicted_video[-1]
        debug = {
            **dict(result.get("fdm_debug", {}) or {}),
            "adapter": "uva_libero_fdm",
            "internal_horizon": self.internal_horizon,
            "action_target_start": self.action_target_start,
            "policy_action_steps": int(action_chunk.actions.shape[0]),
            "action_window_shape": list(action_window.shape),
            "frame_window_shape": list(frame_window.shape),
            "state_window_shape": list(state_window.shape),
            "resized_prediction_to_context_view": bool(self.resize_prediction_to_context_view),
            "view_key": self.view_key,
            "action_space": self.action_space,
        }
        return DynamicsPrediction(
            predicted_video=predicted_video,
            next_context=context.with_prediction(
                predicted_video=predicted_video,
                views=next_views,
                metadata_updates={self.keys.debug: debug},
            ),
            metadata={"debug": debug},
        )

    def _build_frame_window(self, context: PlanningContext) -> np.ndarray:
        if self.view_key not in context.views:
            raise KeyError(f"UVA FDM requires context view {self.view_key!r}; available={sorted(context.views)}.")
        frame = np.asarray(context.views[self.view_key], dtype=np.uint8)
        if frame.ndim != 3 or int(frame.shape[-1]) != 3:
            raise ValueError(f"UVA FDM expected RGB context frame [H,W,3], got {frame.shape}.")
        return np.repeat(frame[None], self.internal_horizon, axis=0)

    def _build_state_window(self, context: PlanningContext) -> np.ndarray:
        if context.state is None:
            raise ValueError("UVA FDM requires PlanningContext.state for raw7-to-absolute action conversion.")
        state = np.asarray(context.state, dtype=np.float32)
        if state.ndim != 1 or int(state.shape[0]) < 6:
            raise ValueError(f"UVA FDM expected current state [D>=6], got {state.shape}.")
        return np.repeat(state[None], self.internal_horizon, axis=0)

    def _build_action_window(self, actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(actions, dtype=np.float32)
        if raw_actions.ndim != 2 or int(raw_actions.shape[1]) != 7:
            raise ValueError(f"UVA FDM expected raw LIBERO actions [T,7], got {raw_actions.shape}.")
        target_slots = self.internal_horizon - self.action_target_start
        if int(raw_actions.shape[0]) > target_slots:
            raise ValueError(
                "UVA FDM candidate chunk is longer than the target action slots: "
                f"actions={raw_actions.shape[0]}, target_slots={target_slots}."
            )
        if int(raw_actions.shape[0]) <= 0:
            raise ValueError("UVA FDM candidate chunk must contain at least one action.")
        if int(raw_actions.shape[0]) < target_slots:
            if self.short_action_horizon_strategy != "repeat_last":
                raise ValueError(
                    "UVA FDM received a short candidate chunk. Set "
                    "short_action_horizon_strategy='repeat_last' to pad the UVA-only suffix."
                )
            pad = np.repeat(raw_actions[-1:], target_slots - int(raw_actions.shape[0]), axis=0)
            target_actions = np.concatenate((raw_actions, pad), axis=0)
        else:
            target_actions = raw_actions
        history_actions = np.zeros((self.action_target_start, int(raw_actions.shape[1])), dtype=np.float32)
        return np.concatenate((history_actions, target_actions), axis=0).astype(np.float32, copy=False)


def _load_uva_runtime(
    *,
    uva_root: Path,
    checkpoint: str | Path,
    dataset_dir: str | Path,
    output_dir: Path,
    runtime_device: str,
) -> dict[str, Any]:
    import torch

    repo_root = Path(__file__).resolve().parents[3]
    debug_path = repo_root / "scripts" / "debug_gjd_uva_mode_videos.py"
    if not debug_path.exists():
        raise FileNotFoundError(f"Could not locate UVA debug adapter script: {debug_path}")
    spec = importlib.util.spec_from_file_location("_openwam_uva_debug_adapter", debug_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load UVA debug adapter script: {debug_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._load_uva_dependencies(uva_root)
    device = torch.device(runtime_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        uva_root=str(uva_root),
        uva_checkpoint=str(checkpoint),
        uva_dataset_dir=str(dataset_dir),
    )
    uva = module._build_uva(args, runtime_device=device, output_root=output_dir)
    return {"debug_module": module, "uva": uva, "device": device}


def _ensure_uint8_video(video: np.ndarray) -> np.ndarray:
    array = np.asarray(video)
    if array.ndim != 4 or int(array.shape[-1]) != 3:
        raise ValueError(f"Expected video [T,H,W,3], got {array.shape}.")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = array.astype(np.float32, copy=False)
    if float(np.nanmax(array)) <= 1.5:
        array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _resize_video_nearest(video: np.ndarray, *, height: int, width: int) -> np.ndarray:
    array = _ensure_uint8_video(video)
    if int(array.shape[1]) == int(height) and int(array.shape[2]) == int(width):
        return array
    y_indices = np.linspace(0, int(array.shape[1]) - 1, int(height)).round().astype(np.int64)
    x_indices = np.linspace(0, int(array.shape[2]) - 1, int(width)).round().astype(np.int64)
    return np.ascontiguousarray(array[:, y_indices][:, :, x_indices])


def _torch_inference_mode():
    import torch

    return torch.inference_mode()
