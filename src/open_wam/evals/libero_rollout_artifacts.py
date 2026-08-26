"""Render and persist LIBERO rollout artifacts independently from execution.

This stable facade owns persistence orchestration. Typed contracts, diagnostics,
frame rendering, and storage helpers live in role-specific sibling modules.
"""

# Keep the established public export order and persistence ASTs stable.
# ruff: noqa: RUF022

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch

from open_wam.evals.libero_rollout_artifact_contracts import (
    LiberoExactStartupDebugOptions,
    LiberoExactStartupDebugPayload,
    LiberoRealtimeArtifactIdentity,
    LiberoRealtimeArtifactOptions,
    LiberoRealtimeArtifactOutput,
    LiberoRealtimeArtifactPayload,
    LiberoRolloutArtifactIdentity,
    LiberoRolloutArtifactOptions,
    LiberoRolloutArtifactOutput,
    LiberoRolloutArtifactPayload,
    RolloutArtifactPolicy,
)
from open_wam.evals.libero_rollout_artifact_diagnostics import (
    build_libero_exact_startup_debug_report,
    capture_torch_rng_debug_state,
)
from open_wam.evals.libero_rollout_artifact_rendering import (
    append_predicted_latent_chunk,
    build_libero_fallback_timeline_video_frames,
    build_libero_realtime_video_frames,
    extract_predicted_latents,
    iter_comparison_video_frames,
    iter_rollout_video_frames,
    with_title,
)
from open_wam.evals.libero_rollout_artifact_storage import (
    _write_action_trace,
    _write_jsonl_records,
    build_libero_realtime_output_stem,
    build_libero_rollout_output_path,
)
from open_wam.evals.video_artifacts import (
    decode_latent_video_chunks,
    to_uint8,
    write_video_frames,
)
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
        extension_trace_path=extension_trace_path
        if policy.writes_debug_artifacts
        else None,
        load_report_path=load_report_path if policy.writes_debug_artifacts else None,
        startup_debug_path=(
            startup_debug_path if payload.startup_debug_report is not None else None
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
        rollout_video_path = output_path.with_name(f"{output_path.stem}_rollout.mp4")
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
        None if comparison_video_path is None else str(comparison_video_path)
    )
    resolved_summary["video_path"] = resolved_comparison_path
    resolved_summary["comparison_video_path"] = resolved_comparison_path
    resolved_summary["rollout_video_path"] = (
        None if rollout_video_path is None else str(rollout_video_path)
    )

    summary_path = output_path.with_suffix(".json")
    action_trace_path = output_path.with_name(f"{output_path.stem}_actions.jsonl")
    chunk_events_path = output_path.with_name(f"{output_path.stem}_chunks.json")
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
