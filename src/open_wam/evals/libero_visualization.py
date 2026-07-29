"""LIBERO observation preparation and rollout-artifact helpers.

This optional evaluation module owns simulator-facing visualization glue. It
does not define policy, cache, or sequence semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor

from open_wam.configs import ProprioContextMode
from open_wam.data.action_transforms import quaternion_to_axis_angle
from open_wam.integrations import (
    LIBERO_ROLLOUT_VIEW_KEYS,
    LiberoTaskSpec,
    resolve_libero_task_by_id,
)


LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS


def resolve_task_spec(benchmark_name: str, task_id: int) -> tuple[LiberoTaskSpec, str]:
    """Resolve one benchmark task and return its model prompt."""

    task_spec = resolve_libero_task_by_id(benchmark_name, task_id)
    return task_spec, task_spec.task_language


def initialize_raw_observation(env: Any, init_state: Any) -> Mapping[str, Any]:
    """Reset a LIBERO environment and return its five-step startup observation."""

    env.reset()
    env.set_init_state(init_state)
    observation = None
    for _ in range(5):
        observation, _, _, _ = env.step([0.0] * 7)
    if observation is None:
        raise RuntimeError(
            "LIBERO env did not return an observation during initialization."
        )
    return observation


def extract_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract vertically corrected RGB views under canonical rollout keys."""

    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(observation["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1]
        ),
    }


def proprio_context_enabled(config: Any) -> bool:
    """Return whether the selected policy consumes exact-rollout proprio."""

    policy_config = getattr(config, "policy_variant", None)
    mode = getattr(policy_config, "proprio_context_mode", ProprioContextMode.NONE)
    return ProprioContextMode(mode) in {
        ProprioContextMode.TEXT_CONTEXT_TOKEN,
        ProprioContextMode.PER_CHUNK_ADDITIVE,
    }


def extract_proprio_context_tensor(
    observation: Mapping[str, Any],
    *,
    config: Any,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the exact M1 proprio tensor when the policy enables it."""

    if not proprio_context_enabled(config):
        return None
    state_encoding = getattr(
        getattr(config.data, "action_target", None), "state_encoding", None
    )
    if state_encoding != "eef_pos_axisangle_gripper_2d":
        raise ValueError(
            "LIBERO exact proprio context currently supports only "
            f"state_encoding='eef_pos_axisangle_gripper_2d', got {state_encoding!r}."
        )
    state = extract_eef_axisangle_gripper_state(observation)
    expected_dim = int(
        getattr(getattr(config.data, "action_schema", None), "state_dim", 0) or 0
    )
    if expected_dim > 0 and state.shape[0] != expected_dim:
        raise ValueError(
            "LIBERO proprio context state dim does not match data.action_schema.state_dim, "
            f"got {state.shape[0]} and expected {expected_dim}."
        )
    return torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)


def extract_eef_axisangle_gripper_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Convert raw LIBERO EEF state to the retained exact-rollout 8D layout."""

    eef_pos = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(
        observation["robot0_gripper_qpos"], dtype=np.float32
    ).reshape(-1)
    if eef_pos.shape[0] != 3:
        raise ValueError(
            f"Expected LIBERO robot0_eef_pos to have dim 3, got {eef_pos.shape[0]}."
        )
    if eef_quat.shape[0] != 4:
        raise ValueError(
            f"Expected LIBERO robot0_eef_quat to have dim 4, got {eef_quat.shape[0]}."
        )
    if gripper_qpos.shape[0] != 2:
        raise ValueError(
            f"Expected LIBERO robot0_gripper_qpos to have dim 2, got {gripper_qpos.shape[0]}."
        )
    axis_angle = (
        quaternion_to_axis_angle(
            torch.from_numpy(eef_quat).to(dtype=torch.float32).unsqueeze(0)
        )[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if axis_angle.shape[0] != 3:
        raise ValueError(
            f"Expected axis-angle proprio dim 3, got {axis_angle.shape[0]}."
        )
    return np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=0).astype(
        np.float32, copy=False
    )


def observations_to_views(
    observations: Sequence[Mapping[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack canonical observation views into time-major tensors."""

    if not observations:
        raise ValueError(
            "Cannot build LIBERO views from an empty observation sequence."
        )
    return {
        key: torch.from_numpy(
            np.stack([observation[key] for observation in observations], axis=0)
        ).to(device=device)
        for key in LIBERO_OBS_KEYS
    }


def prepare_exact_runtime_inputs(
    runner: Any,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
    preserve_stream_cache: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Run canonicalization and the shared visual frontend for exact rollout."""

    canonical_batch = runner.pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend_output = runner.pipeline.visual_tower.run_frontend(
        canonical_video,
        placements=canonical_batch.placements,
        task_text=task_text,
        text_context=None
        if text_context is None
        else text_context.to(device=frontend_device),
        negative_text_context=(
            None
            if negative_text_context is None
            else negative_text_context.to(device=frontend_device)
        ),
        preserve_stream_cache=preserve_stream_cache,
    )
    return {
        "video_latents": frontend_output.video_latents.to(device=runtime_device),
        "text_context": (
            None
            if frontend_output.conditioning.text_context is None
            else frontend_output.conditioning.text_context.to(device=runtime_device)
        ),
        "negative_text_context": (
            None
            if frontend_output.conditioning.negative_text_context is None
            else frontend_output.conditioning.negative_text_context.to(
                device=runtime_device
            )
        ),
    }


def decode_imagined_video(
    runner: Any,
    predicted_latent_chunks: Sequence[torch.Tensor],
    *,
    decode_device: torch.device,
) -> np.ndarray | None:
    """Decode accumulated predicted latent chunks when a VAE is available."""

    if not predicted_latent_chunks:
        return None
    assets = runner.pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None

    latents = torch.cat(tuple(predicted_latent_chunks), dim=2)
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype

    target_dtype = torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    if original_device != decode_device or original_dtype != target_dtype:
        vae = vae.to(device=decode_device, dtype=target_dtype)
    latents = latents.to(device=decode_device, dtype=target_dtype)

    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    imagined_video = video_processor.postprocess_video(decoded, output_type="np")[0]

    if (
        next(assets.vae.parameters()).device != original_device
        or next(assets.vae.parameters()).dtype != original_dtype
    ):
        assets.vae = assets.vae.to(device=original_device, dtype=original_dtype)
    return imagined_video


def build_comparison_video_frames(
    *,
    real_observations: Sequence[Mapping[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> list[np.ndarray]:
    """Build side-by-side real and imagined rollout artifact frames."""

    final_frames: list[np.ndarray] = []
    imagined_frames = [] if imagined_video is None else list(imagined_video)
    panel_height = 300

    for index, observation in enumerate(real_observations):
        agentview = np.ascontiguousarray(observation[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(observation[LIBERO_OBS_KEYS[1]])
        real_row = np.ascontiguousarray(np.hstack([agentview, wrist]))
        real_row = np.array(
            with_title(Image.fromarray(real_row), "Real (AgentView / Wrist)"),
            copy=True,
        )
        target_width = real_row.shape[1]

        if index < len(imagined_frames):
            imagined = Image.fromarray(_to_uint8(imagined_frames[index]))
            scale = min(target_width / imagined.width, panel_height / imagined.height)
            resized = imagined.resize(
                (
                    max(1, int(imagined.width * scale)),
                    max(1, int(imagined.height * scale)),
                )
            )
            imagined_row = Image.new(
                "RGB", (target_width, panel_height), color=(0, 0, 0)
            )
            imagined_row.paste(
                resized,
                (
                    (target_width - resized.width) // 2,
                    (panel_height - resized.height) // 2,
                ),
            )
        else:
            imagined_row = Image.new(
                "RGB", (target_width, panel_height), color=(0, 0, 0)
            )
            draw = ImageDraw.Draw(imagined_row)
            draw.text(
                (max(10, target_width // 2 - 140), 150),
                "No imagined video",
                fill=(120, 120, 120),
            )
        imagined_row = with_title(imagined_row, "Imagined (Open-WAM Exact)")
        final_frames.append(
            np.ascontiguousarray(
                np.vstack([real_row, np.array(imagined_row, copy=True)])
            )
        )
    return final_frames


def with_title(image: Image.Image, title: str) -> Image.Image:
    """Add a fixed-height title bar without resizing the source image."""

    title_height = 36
    canvas = Image.new(
        "RGB", (image.width, image.height + title_height), color=(0, 0, 0)
    )
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return canvas


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def resolve_device(
    device_arg: str | None,
    *,
    fallback: torch.device | None = None,
) -> torch.device:
    """Resolve an explicit device, fallback, or the available default."""

    if device_arg is not None:
        return torch.device(device_arg)
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
