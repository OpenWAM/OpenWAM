"""Render and persist LIBERO rollout artifacts independently from execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from einops import rearrange
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor

from open_wam.configs.enums import RolloutArtifactProfile
from open_wam.integrations import LIBERO_ROLLOUT_VIEW_KEYS
from open_wam.pipelines import VariantPipeline


LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS

__all__ = [
    "LiberoExactStartupDebugOptions",
    "LiberoExactStartupDebugPayload",
    "LiberoRolloutArtifactIdentity",
    "LiberoRolloutArtifactOptions",
    "LiberoRolloutArtifactOutput",
    "LiberoRolloutArtifactPayload",
    "LiberoRealtimeArtifactIdentity",
    "LiberoRealtimeArtifactOptions",
    "LiberoRealtimeArtifactOutput",
    "LiberoRealtimeArtifactPayload",
    "RolloutArtifactPolicy",
    "append_predicted_latent_chunk",
    "build_libero_exact_startup_debug_report",
    "build_libero_fallback_timeline_video_frames",
    "build_libero_realtime_output_stem",
    "build_libero_realtime_video_frames",
    "build_libero_rollout_output_path",
    "capture_torch_rng_debug_state",
    "decode_latent_video_chunks",
    "extract_predicted_latents",
    "iter_comparison_video_frames",
    "iter_rollout_video_frames",
    "persist_libero_realtime_artifacts",
    "persist_libero_rollout_artifacts",
    "to_uint8",
    "with_title",
    "write_video_frames",
]


@dataclass(frozen=True)
class LiberoRolloutArtifactIdentity:
    """Stable coordinates used to derive one episode's artifact paths."""

    benchmark: str
    task_id: int
    prompt: str
    episode_idx: int
    success: bool
    suffix: str


@dataclass(frozen=True)
class LiberoRolloutArtifactOptions:
    """Output and video choices independent from simulator execution."""

    output_root: Path
    video_fps: float
    save_rollout_video: bool = False
    skip_comparison_video: bool = False


@dataclass(frozen=True)
class LiberoRolloutArtifactPayload:
    """Episode traces consumed only by artifact rendering and persistence."""

    real_observations: Sequence[Mapping[str, np.ndarray]]
    predicted_latent_chunks: Sequence[torch.Tensor]
    action_trace: Sequence[np.ndarray]
    chunk_events: Sequence[Mapping[str, object]]
    component_report: Mapping[str, object]


@dataclass(frozen=True)
class LiberoRolloutArtifactOutput:
    """Persisted artifact paths plus the path-enriched legacy summary."""

    summary: dict[str, object]
    summary_path: Path
    action_trace_path: Path
    chunk_events_path: Path
    component_report_path: Path
    comparison_video_path: Path | None
    rollout_video_path: Path | None


@dataclass(frozen=True)
class RolloutArtifactPolicy:
    """Artifact profile decisions shared by collection and persistence."""

    profile: RolloutArtifactProfile
    write_fallback_timeline_video: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", RolloutArtifactProfile(self.profile))

    @classmethod
    def from_value(
        cls,
        profile: RolloutArtifactProfile | str,
        *,
        write_fallback_timeline_video: bool = False,
    ) -> RolloutArtifactPolicy:
        return cls(
            profile=RolloutArtifactProfile(profile),
            write_fallback_timeline_video=bool(write_fallback_timeline_video),
        )

    @property
    def writes_rollout_video(self) -> bool:
        return self.profile in {
            RolloutArtifactProfile.STANDARD,
            RolloutArtifactProfile.DEBUG,
        }

    @property
    def writes_debug_artifacts(self) -> bool:
        return self.profile in {
            RolloutArtifactProfile.STANDARD,
            RolloutArtifactProfile.DEBUG,
        }

    @property
    def writes_fallback_timeline_video(self) -> bool:
        return (
            self.profile is RolloutArtifactProfile.DEBUG
            or self.write_fallback_timeline_video
        )

    @property
    def collects_video_records(self) -> bool:
        return self.writes_rollout_video or self.writes_fallback_timeline_video


@dataclass(frozen=True)
class LiberoRealtimeArtifactIdentity:
    """Stable coordinates used for one realtime episode's artifact stem."""

    benchmark: str
    task_id: int
    prompt: str
    episode_idx: int
    suffix: str


@dataclass(frozen=True)
class LiberoRealtimeArtifactOptions:
    """Realtime output choices independent from simulator execution."""

    output_root: Path
    video_fps: float
    action_per_frame: int
    policy: RolloutArtifactPolicy


@dataclass(frozen=True)
class LiberoRealtimeArtifactPayload:
    """Realtime traces consumed only by artifact rendering and persistence."""

    action_records: Sequence[Mapping[str, Any]]
    action_video_records: Sequence[Mapping[str, Any]]
    replan_records: Sequence[Mapping[str, Any]]
    extension_records: Sequence[Mapping[str, Any]]
    component_report: Mapping[str, Any]
    startup_debug_report: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LiberoRealtimeArtifactOutput:
    """Persisted realtime paths plus the path-enriched summary."""

    summary: dict[str, Any]
    summary_path: Path
    video_path: Path | None
    fallback_timeline_video_path: Path | None
    action_trace_path: Path | None
    replan_trace_path: Path | None
    extension_trace_path: Path | None
    load_report_path: Path | None
    startup_debug_path: Path | None


@dataclass(frozen=True)
class LiberoExactStartupDebugOptions:
    """Stable runtime metadata for one exact-rollout startup report."""

    prompt: str
    seed: int
    runtime_device: torch.device
    frontend_device: torch.device
    decode_device: torch.device
    reference_assets_device_policy: str
    runtime_mode: str
    video_num_inference_steps: int
    action_num_inference_steps: int
    guidance_scale: float
    action_guidance_scale: float
    frame_chunk_size: int
    action_per_frame: int
    exact_startup_bootstrap_padding: bool = False
    startup_warmup_debug: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LiberoExactStartupDebugPayload:
    """Observed and model-produced values summarized by a startup report."""

    first_observation: Mapping[str, np.ndarray]
    video_latents: torch.Tensor | None
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None
    session_text_context: torch.Tensor | None
    session_negative_text_context: torch.Tensor | None
    rng_before_startup_infer: Mapping[str, Any]
    rng_after_startup_infer: Mapping[str, Any]
    first_chunk_debug: Mapping[str, Any]
    chunk_action_pred: torch.Tensor | None
    raw_chunk_action_pred: torch.Tensor | None
    predicted_latents: torch.Tensor | None


def capture_torch_rng_debug_state() -> dict[str, Any]:
    """Capture hash-oriented CPU and CUDA RNG summaries without changing RNG."""

    return {
        "torch_cpu": _debug_tensor_summary(torch.get_rng_state()),
        "torch_cuda": (
            [_debug_tensor_summary(state) for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def build_libero_exact_startup_debug_report(
    *,
    options: LiberoExactStartupDebugOptions,
    payload: LiberoExactStartupDebugPayload,
) -> dict[str, Any]:
    """Build the canonical first-chunk diagnostic report for exact rollouts."""

    cuda_device_name = None
    if options.runtime_device.type == "cuda" and torch.cuda.is_available():
        device_index = (
            torch.cuda.current_device()
            if options.runtime_device.index is None
            else int(options.runtime_device.index)
        )
        cuda_device_name = torch.cuda.get_device_name(device_index)
    return {
        "schema_version": 1,
        "purpose": "startup_first_chunk_cross_gpu_debug",
        "prompt": str(options.prompt),
        "seed": int(options.seed),
        "torch_version": str(torch.__version__),
        "cuda_device_name": cuda_device_name,
        "runtime_device": str(options.runtime_device),
        "frontend_device": str(options.frontend_device),
        "decode_device": str(options.decode_device),
        "reference_assets_device_policy": str(options.reference_assets_device_policy),
        "runtime_mode": str(options.runtime_mode),
        "video_num_inference_steps": int(options.video_num_inference_steps),
        "action_num_inference_steps": int(options.action_num_inference_steps),
        "guidance_scale": float(options.guidance_scale),
        "action_guidance_scale": float(options.action_guidance_scale),
        "exact_startup_bootstrap_padding": bool(options.exact_startup_bootstrap_padding),
        "startup_warmup_debug": (
            None
            if options.startup_warmup_debug is None
            else dict(options.startup_warmup_debug)
        ),
        "first_obs": {
            key: _debug_array_summary(value)
            for key, value in sorted(payload.first_observation.items())
        },
        "initial_inputs": {
            "video_latents": _debug_tensor_summary(payload.video_latents),
            "text_context": _debug_tensor_summary(payload.text_context),
            "negative_text_context": _debug_tensor_summary(payload.negative_text_context),
        },
        "session_text_context": _debug_tensor_summary(payload.session_text_context),
        "session_negative_text_context": _debug_tensor_summary(
            payload.session_negative_text_context
        ),
        "rng_before_startup_infer": payload.rng_before_startup_infer,
        "rng_after_startup_infer": payload.rng_after_startup_infer,
        "first_chunk": {
            "debug": dict(payload.first_chunk_debug),
            "chunk_action_pred": _debug_tensor_summary(payload.chunk_action_pred),
            "raw_chunk_action_pred": _debug_tensor_summary(payload.raw_chunk_action_pred),
            "predicted_latents": _debug_tensor_summary(payload.predicted_latents),
            "raw_action_grid": _debug_raw_action_grid(
                raw_chunk_action_pred=payload.raw_chunk_action_pred,
                generation_frame_start=int(
                    payload.first_chunk_debug.get("generation_frame_start", 0)
                ),
                frame_chunk_size=options.frame_chunk_size,
                action_per_frame=options.action_per_frame,
            ),
        },
    }


def persist_libero_realtime_artifacts(
    *,
    identity: LiberoRealtimeArtifactIdentity,
    options: LiberoRealtimeArtifactOptions,
    payload: LiberoRealtimeArtifactPayload,
    summary: dict[str, Any],
) -> LiberoRealtimeArtifactOutput:
    """Persist one realtime rollout and enrich its established summary in place."""

    output_stem = build_libero_realtime_output_stem(
        root=options.output_root,
        identity=identity,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    policy = options.policy
    summary["artifact_profile"] = policy.profile.value

    video_path: Path | None = None
    if policy.writes_rollout_video:
        video_frames = build_libero_realtime_video_frames(
            action_video_records=payload.action_video_records,
            target_action_hz=float(summary["target_action_hz"]),
            action_per_frame=options.action_per_frame,
        )
        video_path = output_stem.with_suffix(".mp4")
        imageio.mimsave(video_path, video_frames, fps=float(options.video_fps))
        summary["video_path"] = str(video_path.resolve())

    fallback_timeline_video_path: Path | None = None
    if policy.writes_fallback_timeline_video:
        fallback_timeline_frames = build_libero_fallback_timeline_video_frames(
            action_video_records=payload.action_video_records,
            target_action_hz=float(summary["target_action_hz"]),
            action_per_frame=options.action_per_frame,
        )
        fallback_timeline_video_path = output_stem.with_name(
            f"{output_stem.stem}_fallback_timeline.mp4"
        )
        imageio.mimsave(
            fallback_timeline_video_path,
            fallback_timeline_frames,
            fps=float(options.video_fps),
        )
        summary["fallback_timeline_video_path"] = str(
            fallback_timeline_video_path.resolve()
        )

    summary_path = output_stem.with_suffix(".json")
    action_trace_path = output_stem.with_name(f"{output_stem.stem}_actions.jsonl")
    replan_trace_path = output_stem.with_name(f"{output_stem.stem}_replans.jsonl")
    extension_trace_path = output_stem.with_name(f"{output_stem.stem}_extensions.jsonl")
    load_report_path = output_stem.with_name(f"{output_stem.stem}_load_report.json")
    startup_debug_path = output_stem.with_name(f"{output_stem.stem}_startup_debug.json")
    summary["summary_path"] = str(summary_path.resolve())
    if policy.writes_debug_artifacts:
        summary["action_trace_path"] = str(action_trace_path.resolve())
        summary["replan_trace_path"] = str(replan_trace_path.resolve())
        summary["extension_trace_path"] = str(extension_trace_path.resolve())
        summary["load_report_path"] = str(load_report_path.resolve())
    if payload.startup_debug_report is not None:
        summary["startup_debug_path"] = str(startup_debug_path.resolve())

    if policy.writes_debug_artifacts:
        _write_jsonl_records(action_trace_path, payload.action_records)
        _write_jsonl_records(replan_trace_path, payload.replan_records)
        _write_jsonl_records(extension_trace_path, payload.extension_records)
        load_report_path.write_text(
            json.dumps(dict(payload.component_report), indent=2),
            encoding="utf-8",
        )
    if payload.startup_debug_report is not None:
        startup_debug_path.write_text(
            json.dumps(dict(payload.startup_debug_report), indent=2),
            encoding="utf-8",
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return LiberoRealtimeArtifactOutput(
        summary=summary,
        summary_path=summary_path,
        video_path=video_path,
        fallback_timeline_video_path=fallback_timeline_video_path,
        action_trace_path=action_trace_path if policy.writes_debug_artifacts else None,
        replan_trace_path=replan_trace_path if policy.writes_debug_artifacts else None,
        extension_trace_path=extension_trace_path if policy.writes_debug_artifacts else None,
        load_report_path=load_report_path if policy.writes_debug_artifacts else None,
        startup_debug_path=(
            startup_debug_path
            if payload.startup_debug_report is not None
            else None
        ),
    )


def persist_libero_rollout_artifacts(
    *,
    pipeline: VariantPipeline,
    identity: LiberoRolloutArtifactIdentity,
    options: LiberoRolloutArtifactOptions,
    payload: LiberoRolloutArtifactPayload,
    summary: Mapping[str, object],
    decode_device: torch.device,
) -> LiberoRolloutArtifactOutput:
    """Render and persist one rollout while preserving the legacy file schema."""

    output_path = build_libero_rollout_output_path(
        root=options.output_root,
        identity=identity,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_video_path: Path | None = None
    if not options.skip_comparison_video:
        imagined_video = decode_latent_video_chunks(
            pipeline,
            payload.predicted_latent_chunks,
            decode_device=decode_device,
            restore_vae=False,
        )
        write_video_frames(
            output_path,
            iter_comparison_video_frames(
                real_observations=payload.real_observations,
                imagined_video=imagined_video,
            ),
            fps=options.video_fps,
        )
        comparison_video_path = output_path.resolve()

    rollout_video_path: Path | None = None
    if options.save_rollout_video:
        rollout_video_path = output_path.with_name(
            f"{output_path.stem}_rollout.mp4"
        )
        write_video_frames(
            rollout_video_path,
            iter_rollout_video_frames(
                real_observations=payload.real_observations,
            ),
            fps=options.video_fps,
        )
        rollout_video_path = rollout_video_path.resolve()

    resolved_summary = dict(summary)
    resolved_comparison_path = (
        None
        if comparison_video_path is None
        else str(comparison_video_path)
    )
    resolved_summary["video_path"] = resolved_comparison_path
    resolved_summary["comparison_video_path"] = resolved_comparison_path
    resolved_summary["rollout_video_path"] = (
        None if rollout_video_path is None else str(rollout_video_path)
    )

    summary_path = output_path.with_suffix(".json")
    action_trace_path = output_path.with_name(
        f"{output_path.stem}_actions.jsonl"
    )
    chunk_events_path = output_path.with_name(
        f"{output_path.stem}_chunks.json"
    )
    component_report_path = output_path.with_name(
        f"{output_path.stem}_load_report.json"
    )
    resolved_summary["action_trace_path"] = str(action_trace_path.resolve())

    summary_path.write_text(
        json.dumps(resolved_summary, indent=2),
        encoding="utf-8",
    )
    _write_action_trace(action_trace_path, payload.action_trace)
    chunk_events_path.write_text(
        json.dumps(list(payload.chunk_events), indent=2, default=str),
        encoding="utf-8",
    )
    component_report_path.write_text(
        json.dumps(dict(payload.component_report), indent=2),
        encoding="utf-8",
    )
    return LiberoRolloutArtifactOutput(
        summary=resolved_summary,
        summary_path=summary_path,
        action_trace_path=action_trace_path,
        chunk_events_path=chunk_events_path,
        component_report_path=component_report_path,
        comparison_video_path=comparison_video_path,
        rollout_video_path=rollout_video_path,
    )


def build_libero_realtime_output_stem(
    *,
    root: Path,
    identity: LiberoRealtimeArtifactIdentity,
) -> Path:
    """Resolve one sanitized realtime artifact stem without a suffix."""

    safe_prompt = _safe_path_token(identity.prompt)
    safe_suffix = _safe_path_token(identity.suffix)
    return (
        root
        / identity.benchmark
        / f"{identity.task_id}_{safe_prompt}"
        / f"{identity.episode_idx}_{safe_suffix}"
    )


def build_libero_rollout_output_path(
    *,
    root: Path,
    identity: LiberoRolloutArtifactIdentity,
) -> Path:
    """Resolve the maintained per-task, per-episode artifact path."""

    safe_prompt = identity.prompt.replace(" ", "_")
    return (
        root
        / identity.benchmark
        / f"{identity.task_id}_{safe_prompt}"
        / (
            f"{identity.episode_idx}_{identity.success}_"
            f"{identity.suffix}.mp4"
        )
    )


def append_predicted_latent_chunk(
    predicted_latent_chunks: list[torch.Tensor],
    predicted_latents: torch.Tensor,
    *,
    max_imagined_latent_frames: int | None,
) -> None:
    """Append a detached CPU latent chunk without exceeding an artifact cap."""

    if predicted_latents.ndim != 5:
        raise ValueError(
            "Predicted latent chunks must have shape [B, C, T, H, W], "
            f"got {tuple(predicted_latents.shape)}."
        )
    if max_imagined_latent_frames is not None:
        cap = int(max_imagined_latent_frames)
        if cap <= 0:
            return
        retained_frames = sum(
            int(chunk.shape[2]) for chunk in predicted_latent_chunks
        )
        if retained_frames >= cap:
            return
        predicted_latents = predicted_latents[
            :, :, : cap - retained_frames
        ]
    if int(predicted_latents.shape[2]) <= 0:
        return
    predicted_latent_chunks.append(predicted_latents.detach().cpu())


def extract_predicted_latents(infer_output: Any) -> torch.Tensor | None:
    """Read optional imagined latents from decoder or policy diagnostics."""

    predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
    if not isinstance(predicted_latents, torch.Tensor):
        predicted_latents = infer_output.policy_output.aux.get(
            "predicted_latents"
        )
    return (
        predicted_latents
        if isinstance(predicted_latents, torch.Tensor)
        else None
    )


def write_video_frames(
    output_path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
) -> None:
    """Stream contiguous frames to one imageio video writer."""

    wrote_frame = False
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
            wrote_frame = True
    if not wrote_frame:
        raise ValueError(
            f"No frames were produced for video output {output_path}."
        )


def iter_rollout_video_frames(
    *,
    real_observations: Sequence[Mapping[str, np.ndarray]],
) -> Iterable[np.ndarray]:
    """Yield titled agent-view/wrist rows for the real rollout."""

    for observation in real_observations:
        agentview = np.ascontiguousarray(observation[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(observation[LIBERO_OBS_KEYS[1]])
        real_row = np.ascontiguousarray(np.hstack([agentview, wrist]))
        titled = with_title(
            Image.fromarray(real_row),
            "MoT Rollout (AgentView / Wrist)",
        )
        yield np.ascontiguousarray(np.array(titled, copy=True))


def build_libero_realtime_video_frames(
    *,
    action_video_records: Sequence[Mapping[str, Any]],
    target_action_hz: float,
    action_per_frame: int,
) -> list[np.ndarray]:
    """Render the established live-observation realtime video layout."""

    frames: list[np.ndarray] = []
    for record in action_video_records:
        obs = record["obs"]
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        titled = with_title(
            Image.fromarray(np.ascontiguousarray(row_real)),
            "Live LIBERO (AgentView / Wrist)",
        )
        info_panel = Image.new("RGB", (titled.width, 108), color=(0, 0, 0))
        draw = ImageDraw.Draw(info_panel)
        header = (
            f"Action {int(record['action_index']) + 1} | "
            f"Frame {int(record['absolute_frame_index'])} "
            f"[{int(record['action_offset']) + 1}/{action_per_frame}]"
        )
        source = str(record["source"])
        lag_text = (
            "fallback"
            if record["generation_lag_frames"] is None
            else str(int(record["generation_lag_frames"]))
        )
        lines = [
            header,
            (
                f"Source: {source} | Target: {target_action_hz:.1f} Hz | "
                f"Lateness: {1000.0 * float(record['lateness_s']):.1f} ms"
            ),
            (
                f"Env step: {1000.0 * float(record['env_step_s']):.1f} ms | "
                f"Generation lag: {lag_text} frame(s)"
            ),
        ]
        text_color = (255, 255, 255) if source == "policy" else (255, 180, 120)
        for index, line in enumerate(lines):
            draw.text((10, 10 + index * 28), line, fill=text_color)
        full_frame = np.vstack(
            [np.array(titled, copy=True), np.array(info_panel, copy=True)]
        )
        frames.append(np.ascontiguousarray(full_frame))
    return frames


def build_libero_fallback_timeline_video_frames(
    *,
    action_video_records: Sequence[Mapping[str, Any]],
    target_action_hz: float,
    action_per_frame: int,
) -> list[np.ndarray]:
    """Render the established fallback-aware realtime timeline layout."""

    frames: list[np.ndarray] = []
    for record_index, record in enumerate(action_video_records):
        obs = record["obs"]
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        titled = with_title(
            Image.fromarray(np.ascontiguousarray(row_real)),
            "Live LIBERO fallback timeline (AgentView / Wrist)",
        )
        source_color = _fallback_timeline_record_color(record)
        bordered = Image.new(
            "RGB",
            (titled.width + 12, titled.height + 12),
            color=source_color,
        )
        bordered.paste(titled, (6, 6))

        info_panel = Image.new("RGB", (bordered.width, 164), color=(0, 0, 0))
        draw = ImageDraw.Draw(info_panel)
        source = str(record.get("source", "unknown"))
        history_decision = str(record.get("frame_history_decision", "not_recorded"))
        generation_lag = (
            "fallback"
            if record.get("generation_lag_frames") is None
            else f"{int(record['generation_lag_frames'])} frame(s)"
        )
        generation_frame = (
            "NA"
            if record.get("generation_frame_start") is None
            else str(record["generation_frame_start"])
        )
        ready_delay = (
            "NA"
            if record.get("plan_ready_delay_s") is None
            else f"{float(record['plan_ready_delay_s']):.2f}s"
        )
        lines = [
            (
                f"Action {int(record['action_index']) + 1} | "
                f"Frame {int(record['absolute_frame_index'])} "
                f"[{int(record['action_offset']) + 1}/{action_per_frame}] | "
                f"Target {target_action_hz:.1f} Hz"
            ),
            (
                f"Source: {source} | History decision: {history_decision} | "
                "Frame has fallback: "
                f"{bool(record.get('frame_contains_fallback_action', source.startswith('fallback_')))}"
            ),
            (
                f"Gen frame: {generation_frame} | Gen lag: {generation_lag} | "
                f"Ready delay: {ready_delay} | "
                f"Late: {1000.0 * float(record['lateness_s']):.1f} ms"
            ),
            _format_fallback_timeline_action(record.get("action")),
            (
                "Timeline colors: red fallback, orange hidden/washout, "
                "blue history, green extension, violet startup."
            ),
        ]
        for index, line in enumerate(lines):
            fill = source_color if index == 1 else (255, 255, 255)
            draw.text((10, 10 + index * 28), line, fill=fill)

        timeline = _build_fallback_timeline_strip(
            action_video_records=action_video_records,
            width=bordered.width,
            height=42,
            current_index=record_index,
        )
        full_frame = np.vstack(
            [
                np.array(bordered, copy=True),
                np.array(info_panel, copy=True),
                np.array(timeline, copy=True),
            ]
        )
        frames.append(np.ascontiguousarray(full_frame))
    return frames


def iter_comparison_video_frames(
    *,
    real_observations: Sequence[Mapping[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> Iterable[np.ndarray]:
    """Yield the maintained real/imagined comparison layout lazily."""

    panel_height = 300
    target_length = len(real_observations)
    for frame_index, real_observation in enumerate(real_observations):
        imagined_frame = _imagined_frame_for_rollout_index(
            imagined_video=imagined_video,
            frame_index=frame_index,
            target_length=target_length,
        )
        agentview = np.ascontiguousarray(
            real_observation[LIBERO_OBS_KEYS[0]]
        )
        wrist = np.ascontiguousarray(real_observation[LIBERO_OBS_KEYS[1]])
        real_row = np.ascontiguousarray(np.hstack([agentview, wrist]))
        real_row = np.array(
            with_title(
                Image.fromarray(real_row),
                f"Real Rollout Frame {frame_index}",
            ),
            copy=True,
        )
        target_width = real_row.shape[1]
        if imagined_frame is None:
            imagined_row = Image.new(
                "RGB",
                (target_width, panel_height),
                color=(0, 0, 0),
            )
            draw = ImageDraw.Draw(imagined_row)
            draw.text(
                (10, panel_height // 2),
                "No imagined frame",
                fill=(120, 120, 120),
            )
        else:
            image = Image.fromarray(to_uint8(imagined_frame))
            scale = min(
                target_width / image.width,
                panel_height / image.height,
            )
            resized = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                )
            )
            imagined_row = Image.new(
                "RGB",
                (target_width, panel_height),
                color=(0, 0, 0),
            )
            imagined_row.paste(
                resized,
                (
                    (target_width - resized.width) // 2,
                    (panel_height - resized.height) // 2,
                ),
            )
        imagined_row = with_title(
            imagined_row,
            f"Imagined Frame {frame_index}",
        )
        yield np.ascontiguousarray(
            np.vstack([real_row, np.array(imagined_row, copy=True)])
        )


def decode_latent_video_chunks(
    pipeline: VariantPipeline,
    latent_chunks: Sequence[torch.Tensor],
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    """Decode accumulated imagined latents when the frontend has a VAE."""

    if not latent_chunks:
        return None
    return _decode_latent_video(
        pipeline,
        torch.cat(tuple(latent_chunks), dim=2),
        decode_device=decode_device,
        restore_vae=restore_vae,
    )


def with_title(image: Image.Image, title: str) -> Image.Image:
    """Add a fixed-height title bar without resizing the source image."""

    title_height = 36
    canvas = Image.new(
        "RGB",
        (image.width, image.height + title_height),
        color=(0, 0, 0),
    )
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return canvas


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize an RGB array to contiguous display-ready uint8 values."""

    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _debug_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _debug_array_summary(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    flat = array.reshape(-1)
    numeric = flat.astype(np.float64, copy=False) if flat.size else flat
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "sha256": _debug_sha256_bytes(array.tobytes()),
        "preview": flat[:12].tolist(),
        "mean": None if flat.size == 0 else float(numeric.mean()),
        "std": None if flat.size == 0 else float(numeric.std()),
    }


def _debug_tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach().contiguous().cpu()
    byte_tensor = tensor.view(torch.uint8)
    flat = tensor.reshape(-1)
    numeric = flat.to(dtype=torch.float32) if flat.numel() else flat
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(value.device),
        "sha256": _debug_sha256_bytes(byte_tensor.numpy().tobytes()),
        "preview": flat[:12].to(dtype=torch.float32).tolist(),
        "mean": None if flat.numel() == 0 else float(numeric.mean().item()),
        "std": None if flat.numel() == 0 else float(numeric.std(unbiased=False).item()),
    }


def _debug_raw_action_grid(
    *,
    raw_chunk_action_pred: torch.Tensor | None,
    generation_frame_start: int,
    frame_chunk_size: int,
    action_per_frame: int,
) -> dict[str, Any] | None:
    if raw_chunk_action_pred is None:
        return None
    raw_actions = rearrange(
        raw_chunk_action_pred[0].detach().to(dtype=torch.float32).cpu(),
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    executable: list[list[float]] = []
    for frame_offset in range(raw_actions.shape[0]):
        if int(generation_frame_start) + frame_offset < 1:
            continue
        for action_offset in range(raw_actions.shape[1]):
            executable.append(
                [float(value) for value in raw_actions[frame_offset, action_offset].tolist()]
            )
    return {
        "generation_frame_start": int(generation_frame_start),
        "all_gripper_by_frame": [
            [
                float(raw_actions[frame_offset, action_offset, 6].item())
                for action_offset in range(raw_actions.shape[1])
            ]
            for frame_offset in range(raw_actions.shape[0])
        ],
        "first_executable_actions": executable[:16],
    }


def _imagined_frame_for_rollout_index(
    *,
    imagined_video: np.ndarray | None,
    frame_index: int,
    target_length: int,
) -> np.ndarray | None:
    if imagined_video is None or target_length <= 0:
        return None
    imagined_frame_count = len(imagined_video)
    if imagined_frame_count <= 0:
        return None
    if imagined_frame_count == 1 or target_length == 1:
        imagined_index = 0
    elif imagined_frame_count == target_length:
        imagined_index = frame_index
    else:
        imagined_index = int(
            round(
                frame_index
                * (imagined_frame_count - 1)
                / (target_length - 1)
            )
        )
    imagined_index = max(
        0,
        min(imagined_frame_count - 1, imagined_index),
    )
    return np.array(imagined_video[imagined_index], copy=True)


def _decode_latent_video(
    pipeline: VariantPipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    target_dtype = (
        torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    )
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
    imagined_video = video_processor.postprocess_video(
        decoded,
        output_type="np",
    )[0]
    if (
        restore_vae
        and (
            next(assets.vae.parameters()).device != original_device
            or next(assets.vae.parameters()).dtype != original_dtype
        )
    ):
        assets.vae = assets.vae.to(
            device=original_device,
            dtype=original_dtype,
        )
    return imagined_video


def _safe_path_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "run"


def _fallback_timeline_record_color(
    record: Mapping[str, Any],
) -> tuple[int, int, int]:
    source = str(record.get("source", ""))
    history_decision = str(record.get("frame_history_decision", ""))
    if source.startswith("fallback_"):
        return (230, 50, 40)
    if history_decision in {"fallback", "washout"}:
        return (235, 165, 35)
    if source == "history_replan":
        return (70, 150, 255)
    if source == "open_loop_extension":
        return (70, 210, 120)
    if source == "startup_plan":
        return (175, 150, 255)
    return (180, 180, 180)


def _format_fallback_timeline_action(action: Any) -> str:
    if action is None:
        return "Action: NA"
    values = [float(value) for value in action]
    delta_values = " ".join(f"{value:+.2f}" for value in values[:6])
    tail_values = " ".join(f"{value:+.2f}" for value in values[6:])
    return (
        f"Action delta[0:6]: {delta_values} | "
        f"absolute[6:]: {tail_values or 'NA'}"
    )


def _build_fallback_timeline_strip(
    *,
    action_video_records: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    current_index: int,
) -> Image.Image:
    strip = Image.new("RGB", (int(width), int(height)), color=(18, 18, 18))
    draw = ImageDraw.Draw(strip)
    total = max(1, len(action_video_records))
    bar_top = 8
    bar_bottom = int(height) - 10
    for index, record in enumerate(action_video_records):
        x0 = int(index * int(width) / total)
        x1 = max(x0 + 1, int((index + 1) * int(width) / total))
        draw.rectangle(
            [x0, bar_top, min(int(width) - 1, x1), bar_bottom],
            fill=_fallback_timeline_record_color(record),
        )
    current_x = int(current_index * int(width) / total)
    draw.line(
        [(current_x, 0), (current_x, int(height) - 1)],
        fill=(255, 255, 255),
        width=3,
    )
    draw.text(
        (10, int(height) - 10),
        f"{current_index + 1}/{total}",
        fill=(255, 255, 255),
    )
    return strip


def _write_jsonl_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True))
            handle.write("\n")


def _write_action_trace(
    path: Path,
    actions: Sequence[np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for action_index, action in enumerate(actions):
            handle.write(
                json.dumps(
                    {
                        "action_index": int(action_index),
                        "action": np.asarray(
                            action,
                            dtype=np.float32,
                        ).tolist(),
                    }
                )
                + "\n"
            )
