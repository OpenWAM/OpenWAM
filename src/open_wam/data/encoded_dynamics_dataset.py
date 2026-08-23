"""Canonical target-only encoded-dynamics artifacts and dataset views."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from open_wam.configs import (
    DataConfig,
    DynamicsSource,
)
from open_wam.contracts import (
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON,
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE,
    ConditionalDynamicsSequenceLayout,
)

from .artifacts import (
    DatasetArtifactKind,
    DatasetArtifactPreflightError,
    DatasetArtifactRequirement,
    DatasetArtifactStatus,
    require_dataset_artifacts,
)
from .encoded_dynamics_materialization import (
    load_empty_text_embedding,
    materialize_target_only_sample,
)
from .encoded_dynamics_ordering import build_task_branch_balanced_indices
from .latent_contracts import LatentWAMSample

ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1 = "open_wam.encoded_dynamics.v1"
_ENCODED_DYNAMICS_GENERATION_CONTRACT = "t0_observation_plus_future"
_RAW_PAYLOAD_ROOT_FIELD = "raw_payload_root"
_REQUIRED_TRANSITION_FIELDS = frozenset(
    {"branch", "context_id", "sample_id", "sample_path", "target_latent_path"}
)


@dataclass(frozen=True)
class EncodedDynamicsArtifact:
    """Validated index for the canonical ``t0 + future`` artifact layout."""

    root: Path
    manifest: dict[str, Any]
    transition_rows: tuple[dict[str, Any], ...]

    @property
    def reference_branch(self) -> str:
        return str(self.manifest["reference_branch"])

    @property
    def raw_root(self) -> Path:
        return _encoded_dynamics_raw_root_from_manifest(
            self.manifest,
            encoded_root=self.root,
        )

    def rows_for_source(
        self,
        source: DynamicsSource | str,
    ) -> tuple[dict[str, Any], ...]:
        resolved_source = DynamicsSource(source)
        if resolved_source not in {
            DynamicsSource.REAL_DEMO,
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
        }:
            raise ValueError(
                "Encoded dynamics artifacts support only real_demo and "
                f"counterfactual_dynamics sources, got {resolved_source.value!r}."
            )
        select_reference = resolved_source == DynamicsSource.REAL_DEMO
        return tuple(
            row
            for row in self.transition_rows
            if (str(row["branch"]) == self.reference_branch) == select_reference
        )


@dataclass(frozen=True)
class EncodedDynamicsResources:
    """Artifact index and conditioning tensor shared by dataset views."""

    artifact: EncodedDynamicsArtifact
    empty_text_embedding: torch.Tensor | None

    @classmethod
    def load(
        cls,
        data_config: DataConfig,
        encoded_root: str | Path,
    ) -> EncodedDynamicsResources:
        return cls(
            artifact=load_encoded_dynamics_artifact(encoded_root),
            empty_text_embedding=load_empty_text_embedding(
                data_config.empty_text_embedding_path
            ),
        )

    def with_artifact_root(
        self,
        encoded_root: str | Path,
    ) -> EncodedDynamicsResources:
        """Reuse shared conditioning resources with another artifact index."""

        root = Path(encoded_root).expanduser().resolve()
        if root == self.artifact.root:
            return self
        return EncodedDynamicsResources(
            artifact=load_encoded_dynamics_artifact(root),
            empty_text_embedding=self.empty_text_embedding,
        )


def load_encoded_dynamics_artifact(
    encoded_root: str | Path,
) -> EncodedDynamicsArtifact:
    """Load and validate one canonical encoded-dynamics artifact index."""

    root = Path(encoded_root).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    declared_schema = manifest.get("artifact_schema")
    if declared_schema is None:
        raise ValueError(
            "Encoded dynamics artifact manifest is unversioned. Migrate it "
            "once with migrate_encoded_dynamics_artifact(...) "
            "before loading it."
        )
    if declared_schema != ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1:
        raise ValueError(
            f"Unsupported encoded dynamics artifact schema {declared_schema!r}; "
            f"expected {ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1!r}."
        )
    return _build_encoded_dynamics_artifact(root=root, manifest=manifest)


def migrate_encoded_dynamics_artifact(
    encoded_root: str | Path,
    *,
    raw_root: str | Path | None = None,
    reference_branch: str = "gt",
) -> EncodedDynamicsArtifact:
    """Canonicalize and validate a manifest without rewriting its payloads."""

    root = Path(encoded_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    declared_schema = manifest.get("artifact_schema")
    if declared_schema not in {None, ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1}:
        raise ValueError(
            f"Unsupported encoded dynamics artifact schema {declared_schema!r}; "
            f"expected {ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1!r}."
        )

    resolved_reference_branch = str(
        manifest.get("reference_branch", reference_branch)
    ).strip()
    if not resolved_reference_branch:
        raise ValueError("Encoded dynamics reference_branch must be non-empty.")
    resolved_raw_root = _resolve_migration_raw_root(
        manifest,
        encoded_root=root,
        raw_root=raw_root,
    )
    migrated_manifest = {
        **manifest,
        "artifact_schema": ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1,
        "reference_branch": resolved_reference_branch,
        _RAW_PAYLOAD_ROOT_FIELD: Path(
            os.path.relpath(resolved_raw_root, start=root)
        ).as_posix(),
    }
    artifact = _build_encoded_dynamics_artifact(
        root=root,
        manifest=migrated_manifest,
    )
    _validate_complete_artifact_payloads(artifact)
    if migrated_manifest != manifest:
        _write_json_atomic(manifest_path, migrated_manifest)
    return load_encoded_dynamics_artifact(root)


def _build_encoded_dynamics_artifact(
    *,
    root: Path,
    manifest: dict[str, Any],
) -> EncodedDynamicsArtifact:
    reference_branch = manifest.get("reference_branch")
    if not isinstance(reference_branch, str) or not reference_branch.strip():
        raise ValueError(
            "Encoded dynamics artifact manifest must declare a non-empty "
            "reference_branch."
        )
    transition_rows = tuple(
        _read_jsonl(root / "metadata" / "encoded_transitions.jsonl")
    )
    if not transition_rows:
        raise ValueError(f"Encoded dynamics artifact contains no transitions: {root}")
    for index, row in enumerate(transition_rows):
        missing = sorted(_REQUIRED_TRANSITION_FIELDS - row.keys())
        if missing:
            raise ValueError(
                "Encoded dynamics transition row "
                f"{index} is missing required fields: {', '.join(missing)}."
            )
    if not any(str(row["branch"]) == reference_branch for row in transition_rows):
        raise ValueError(
            "Encoded dynamics reference_branch is absent from the transition "
            f"index: reference_branch={reference_branch!r}."
        )
    artifact = EncodedDynamicsArtifact(
        root=root,
        manifest=manifest,
        transition_rows=transition_rows,
    )
    _ = artifact.raw_root
    return artifact


def preflight_encoded_dynamics_artifact(
    encoded_root: str | Path | None,
    *,
    sources: tuple[DynamicsSource, ...],
    config_path: str,
) -> tuple[DatasetArtifactStatus, ...]:
    """Validate files, schema, and requested source views before model creation."""

    root = None if encoded_root is None else Path(encoded_root).expanduser()
    requirements = (
        DatasetArtifactRequirement(
            name="encoded dynamics root",
            path=root,
            kind=DatasetArtifactKind.DIRECTORY,
            required=True,
            config_path=config_path,
            purpose="stores canonical target-only dynamics artifacts",
        ),
        DatasetArtifactRequirement(
            name="encoded dynamics manifest",
            path=None if root is None else root / "manifest.json",
            kind=DatasetArtifactKind.FILE,
            required=True,
            config_path=config_path,
            purpose="declares artifact provenance and the reference branch",
        ),
        DatasetArtifactRequirement(
            name="encoded dynamics transition index",
            path=(
                None
                if root is None
                else root / "metadata" / "encoded_transitions.jsonl"
            ),
            kind=DatasetArtifactKind.FILE,
            required=True,
            config_path=config_path,
            purpose="indexes target latent and action/state payloads",
        ),
    )
    statuses = require_dataset_artifacts(
        requirements,
        dataset_type="encoded_dynamics",
    )
    try:
        artifact = load_encoded_dynamics_artifact(root)
        source_rows = {
            source: artifact.rows_for_source(source)
            for source in sources
        }
        missing_sources = [
            source.value for source, rows in source_rows.items() if not rows
        ]
        if missing_sources:
            raise ValueError(
                "artifact has no rows for requested source views: "
                + ", ".join(missing_sources)
            )
        statuses += require_dataset_artifacts(
            (
                DatasetArtifactRequirement(
                    name="encoded dynamics raw payload root",
                    path=artifact.raw_root,
                    kind=DatasetArtifactKind.DIRECTORY,
                    required=True,
                    config_path=config_path,
                    purpose="stores aligned future actions and proprio state",
                ),
            ),
            dataset_type="encoded_dynamics",
        )
        _validate_indexed_payload_files(artifact, source_rows)
    except DatasetArtifactPreflightError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise DatasetArtifactPreflightError(
            f"Encoded dynamics artifact preflight failed for {root}: {exc}"
        ) from exc
    return statuses


class EncodedDynamicsLatentDataset(Dataset[LatentWAMSample]):
    """Target-only latent windows from simulator-rendered dynamics branches.

    ``real_demo`` selects the recorded ``gt`` branch; ``counterfactual_dynamics``
    selects every perturbed branch. Both views use the same rollout-local
    encoding: target latent 0 is the observed t0 frame and loss starts at target
    latent 1. Pre-t0 context artifacts remain provenance-only.
    """

    def __init__(
        self,
        data_config: DataConfig,
        resources: EncodedDynamicsResources,
        *,
        split: str,
        source: DynamicsSource | str,
    ) -> None:
        if not isinstance(resources, EncodedDynamicsResources):
            raise TypeError(
                "EncodedDynamicsLatentDataset requires loaded "
                "EncodedDynamicsResources; use from_root(...) for a path."
            )
        self.data_config = data_config
        self.artifact = resources.artifact
        self.encoded_root = self.artifact.root
        self.split = str(split)
        self.source = DynamicsSource(source)
        if self.source not in {
            DynamicsSource.REAL_DEMO,
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
        }:
            raise ValueError(
                "Encoded dynamics datasets support only real_demo and "
                f"counterfactual_dynamics sources, got {self.source.value!r}."
            )
        self.manifest = self.artifact.manifest
        self.raw_root = self.artifact.raw_root
        self.transition_rows = self.artifact.rows_for_source(self.source)
        self.empty_text_embedding = resources.empty_text_embedding
        if not self.transition_rows:
            raise ValueError(
                f"No {self.source.value!r} encoded dynamics transitions found under "
                f"{self.encoded_root}."
            )

    @classmethod
    def from_root(
        cls,
        data_config: DataConfig,
        encoded_root: str | Path,
        *,
        split: str,
        source: DynamicsSource | str,
    ) -> EncodedDynamicsLatentDataset:
        """Load resources and construct one standalone encoded-dynamics view."""

        return cls(
            data_config,
            EncodedDynamicsResources.load(data_config, encoded_root),
            split=split,
            source=source,
        )

    def __len__(self) -> int:
        return len(self.transition_rows)

    def build_balanced_source_indices(self) -> tuple[int, ...]:
        return build_task_branch_balanced_indices(self.transition_rows)

    def __getitem__(self, index: int) -> LatentWAMSample:
        row = self.transition_rows[int(index) % len(self.transition_rows)]
        latent_start = 0
        materialized = materialize_target_only_sample(
            data_config=self.data_config,
            target_latent_path=self._resolve_encoded_path(
                row,
                "target_latent_path",
            ),
            raw_sample_path=self._resolve_raw_path(row, "sample_path"),
            source_start_frame=int(row.get("t0_frame", 0)),
        )
        segment = materialized.segment
        video_latents = segment.video_latents
        actions = segment.actions
        action_mask = segment.action_mask
        total_frames = int(video_latents.shape[1])
        source_frames = materialized.source_frames
        segment_length = source_frames
        source_actions = materialized.source_actions
        action_per_frame = materialized.action_per_frame
        sampled_chunk_size = materialized.sampled_chunk_size
        sampled_window_size = materialized.sampled_window_size
        proprio_context_frames = materialized.proprio_context_frames
        proprio_context_frames_mask = materialized.proprio_context_frames_mask
        proprio_context_state = proprio_context_frames.clone()
        proprio_context_state_mask = proprio_context_frames_mask.clone()
        state = materialized.state
        state_mask = materialized.state_mask
        proprio_context_source = materialized.proprio_context_source
        text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        observed_frame_ids = list(materialized.observed_frame_ids)
        valid_frame_ids = observed_frame_ids[: max(0, segment.valid_source_frames)]
        sample_start_frame = int(valid_frame_ids[0]) if valid_frame_ids else 0
        sample_end_frame = (
            int(valid_frame_ids[-1]) + int(action_per_frame)
            if valid_frame_ids
            else sample_start_frame
        )
        loss_frame_start = segment.loss_frame_start
        loss_frame_end = segment.loss_frame_end
        first_loss_frame_id = (
            int(observed_frame_ids[loss_frame_start])
            if 0 <= loss_frame_start < len(observed_frame_ids)
            else sample_end_frame
        )
        last_loss_frame_end = (
            int(observed_frame_ids[loss_frame_end - 1]) + int(action_per_frame)
            if loss_frame_end > loss_frame_start and loss_frame_end - 1 < len(observed_frame_ids)
            else first_loss_frame_id
        )
        target_observation_frame = segment.target_observation_frame
        target_observation_frame_id = (
            int(observed_frame_ids[target_observation_frame])
            if 0 <= target_observation_frame < len(observed_frame_ids)
            else None
        )
        transition_action_steps_required = max(0, source_frames - 1) * action_per_frame
        extra_source_action_steps = max(0, int(source_actions.shape[0]) - int(transition_action_steps_required))
        layout_metadata = ConditionalDynamicsSequenceLayout().to_metadata(
            observed_num_frames=total_frames
        )

        metadata = {
            "dataset_id": str(self.encoded_root),
            "dataset_kind": "encoded_dynamics",
            "encoded_dynamics_source": self.source.value,
            **layout_metadata,
            # Stable aliases retained in encoded artifacts for provenance tools.
            "generalist_conditional_training_sequence": "target_only",
            "generalist_conditional_context_used_for_training": False,
            "counterfactual_contract": (
                DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE
            ),
            "counterfactual_generation_contract": (
                _ENCODED_DYNAMICS_GENERATION_CONTRACT
            ),
            "counterfactual_training_sequence": "target_only",
            "counterfactual_context_used_for_training": False,
            "split": self.split,
            "counterfactual_sample_id": int(row["sample_id"]),
            "counterfactual_context_id": int(row["context_id"]),
            "counterfactual_branch": row.get("branch"),
            "counterfactual_branch_family": row.get("branch_family"),
            "counterfactual_branch_strength": row.get("branch_strength"),
            "counterfactual_branch_is_ood": bool(row.get("branch_is_ood", False)),
            "episode_index": int(row.get("dataset_episode_index", -1)),
            "task_index": int(row.get("task_id", -1)),
            "init_state_index": row.get("init_state_index"),
            "t0_frame": int(row.get("t0_frame", 0)),
            "t0_action_frame": int(row.get("t0_frame", 0)) * action_per_frame,
            "context_start_frame": int(row.get("context_start_frame", 0)),
            "context_start_action_frame": int(row.get("context_start_frame", 0)) * action_per_frame,
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "observation_start": sample_start_frame,
            "observation_frame_indices": observed_frame_ids,
            "window_sampling_mode": self.data_config.sample_construction.mode,
            "window_start_frame": sample_start_frame,
            "window_end_frame": sample_end_frame,
            "anchor_frame_index": int(valid_frame_ids[-1]) if valid_frame_ids else sample_start_frame,
            "segment_length_frames": total_frames,
            "segment_valid_latent_frames": segment.valid_latent_frames,
            "segment_padded_latent_frames": segment.padded_latent_frames,
            "tail_padding_mode": "none" if segment.padded_latent_frames == 0 else "zero_order_hold",
            "target_observation_frame_index": target_observation_frame_id,
            "first_supervised_future_frame_in_sample": loss_frame_start,
            "first_supervised_future_frame_index": first_loss_frame_id,
            "supervised_future_latent_frames": max(0, loss_frame_end - loss_frame_start),
            "target_frame_start": first_loss_frame_id,
            "target_frame_end": last_loss_frame_end,
            "sampled_chunk_size": int(sampled_chunk_size),
            "counterfactual_gjd_chunk_contract": DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON,
            "conditional_history_policy": DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "counterfactual_conditional_history_policy": DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "sampled_window_size": int(sampled_window_size),
            "has_condition_latents": False,
            "condition_source_frame_offset": None,
            "condition_latents_source": "in_sequence_t0",
            "proprio_context_source": proprio_context_source,
            "proprio_context_chunk_count": int(proprio_context_state.shape[0]),
            "proprio_context_frame_count": int(proprio_context_frames.shape[0]),
            "state_source_key": materialized.state_source_key,
            "state_anchor_frame": segment.prefix_state_frame,
            "state_anchor_source_frame": segment.prefix_state_source_frame,
            "state_anchor_frame_in_sample": segment.prefix_state_frame_in_sample,
            "latent_frame_start": int(latent_start),
            "frame_shift": int(latent_start),
            "start_padding_frames": max(0, int(self.data_config.sample_construction.start_padding_frames)),
            "segment_pre_start_frames": segment.pre_start_frames,
            "start_padding_mode": "repeat_first_latent" if segment.pre_start_frames > 0 else "none",
            "subwindow_latent_start": int(latent_start),
            "subwindow_latent_end": int(latent_start) + int(segment_length),
            "subwindow_action_start": max(0, int(latent_start)) * action_per_frame,
            "subwindow_action_end": max(0, int(latent_start)) * action_per_frame + int(actions.shape[0]),
            "source_action_steps": int(source_actions.shape[0]),
            "transition_action_steps_required": int(transition_action_steps_required),
            "extra_source_action_steps": int(extra_source_action_steps),
            "lingbot_window_action_alignment": {
                "latent_num_frames": total_frames,
                "prefix_actions": action_per_frame,
                "required_action_num": int(actions.shape[0]),
                "leading_zero_action_frames": segment.leading_zero_action_frames,
                "leading_zero_action_steps": segment.leading_zero_action_frames * action_per_frame,
                "leading_zero_action_mask": segment.leading_zero_action_mask,
                "source_action_steps": int(source_actions.shape[0]),
                "transition_action_steps_required": int(transition_action_steps_required),
                "extra_source_action_steps": int(extra_source_action_steps),
            },
            "valid_action_steps": int(action_mask.float().sum(dim=-1).gt(0).sum().item()),
            "valid_action_values": int(action_mask.float().sum().item()),
            "counterfactual_source_row": {
                key: row.get(key)
                for key in (
                    "sample_id",
                    "context_id",
                    "branch",
                    "branch_family",
                    "branch_strength",
                    "action_delta_l2_mean",
                    "target_vs_gt_rgb_mse",
                )
            },
        }
        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            proprio_context_state=proprio_context_state,
            proprio_context_state_mask=proprio_context_state_mask,
            proprio_context_frames=proprio_context_frames,
            proprio_context_frames_mask=proprio_context_frames_mask,
            task_text=None,
            text_context=text_context,
            negative_text_context=text_context.clone() if text_context is not None else None,
            condition_latents=None,
            metadata=metadata,
        )

    def _resolve_encoded_path(self, row: dict[str, Any], key: str) -> Path:
        path = _artifact_path_candidate(self.encoded_root, row, key)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Missing encoded dynamics artifact for {key}: {row[key]}"
        )

    def _resolve_raw_path(self, row: dict[str, Any], key: str) -> Path:
        path = _artifact_path_candidate(self.raw_root, row, key)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Missing raw dynamics artifact for {key}: {row[key]}"
        )


def _artifact_path_candidate(
    root: Path,
    row: dict[str, Any],
    key: str,
) -> Path:
    relative = Path(str(row[key]))
    direct = root / relative
    if direct.exists():
        return direct
    shard = row.get("shard")
    if shard is not None:
        sharded = root / str(shard) / relative
        if sharded.exists():
            return sharded
    return direct


def _validate_indexed_payload_files(
    artifact: EncodedDynamicsArtifact,
    source_rows: Mapping[DynamicsSource, tuple[dict[str, Any], ...]],
) -> None:
    missing_count = 0
    missing_details: list[str] = []
    seen_paths: set[Path] = set()
    payload_specs = (
        ("target_latent_path", artifact.root),
        ("sample_path", artifact.raw_root),
    )
    for source, rows in source_rows.items():
        for row in rows:
            for key, root in payload_specs:
                path = _artifact_path_candidate(root, row, key)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                if path.is_file():
                    continue
                missing_count += 1
                if len(missing_details) < 8:
                    missing_details.append(
                        f"source={source.value}, sample_id={row.get('sample_id')}, "
                        f"field={key}, path={path}"
                    )
    if missing_count:
        details = "; ".join(missing_details)
        remainder = missing_count - len(missing_details)
        suffix = "" if remainder <= 0 else f"; and {remainder} more"
        raise FileNotFoundError(
            f"artifact index references {missing_count} missing payload file(s): "
            f"{details}{suffix}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            temporary_path = Path(handle.name)
        temporary_path.chmod(path.stat().st_mode & 0o777)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _encoded_dynamics_raw_root_from_manifest(
    manifest: dict[str, Any],
    *,
    encoded_root: Path,
) -> Path:
    value = manifest.get(_RAW_PAYLOAD_ROOT_FIELD)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(
            "Encoded dynamics manifest must declare non-empty, artifact-relative "
            f"`{_RAW_PAYLOAD_ROOT_FIELD}`; migrate legacy manifests with "
            "scripts/migrate_encoded_dynamics_artifact.py."
        )
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(
            f"Encoded dynamics `{_RAW_PAYLOAD_ROOT_FIELD}` must be relative to "
            f"the encoded artifact root, got {value!r}."
        )
    return (encoded_root / relative).resolve()


def _resolve_migration_raw_root(
    manifest: dict[str, Any],
    *,
    encoded_root: Path,
    raw_root: str | Path | None,
) -> Path:
    if raw_root is not None:
        resolved = Path(raw_root).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Encoded dynamics raw payload root is not a directory: {resolved}"
            )
        return resolved

    if _RAW_PAYLOAD_ROOT_FIELD in manifest:
        resolved = _encoded_dynamics_raw_root_from_manifest(
            manifest,
            encoded_root=encoded_root,
        )
        if not resolved.is_dir():
            raise FileNotFoundError(
                "Encoded dynamics manifest resolves "
                f"`{_RAW_PAYLOAD_ROOT_FIELD}` to a missing directory: {resolved}. "
                "Pass --raw-root to repair a moved artifact."
            )
        return resolved

    candidates: list[Path] = []
    for key in ("dataset_root", "source_dataset_root"):
        value = manifest.get(key)
        if value is not None:
            candidates.append(Path(str(value)).expanduser().resolve())

    for summary_key in ("source_summary", "source_aggregate_summary"):
        summary = manifest.get(summary_key)
        if not isinstance(summary, dict):
            continue
        for key in ("root", "dataset_root", "output_root"):
            value = summary.get(key)
            if value is not None:
                candidates.append(Path(str(value)).expanduser().resolve())

    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir():
            return candidate

    legacy_paths = ", ".join(str(path) for path in dict.fromkeys(candidates))
    detail = legacy_paths or "none"
    raise FileNotFoundError(
        "Could not infer an existing raw payload root from legacy manifest "
        f"provenance (candidates: {detail}). Pass --raw-root explicitly."
    )


def _validate_complete_artifact_payloads(
    artifact: EncodedDynamicsArtifact,
) -> None:
    _validate_indexed_payload_files(
        artifact,
        {
            source: artifact.rows_for_source(source)
            for source in (
                DynamicsSource.REAL_DEMO,
                DynamicsSource.COUNTERFACTUAL_DYNAMICS,
            )
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON objects in {path}.")
                rows.append(row)
    return rows


__all__ = [
    "ENCODED_DYNAMICS_ARTIFACT_SCHEMA_V1",
    "EncodedDynamicsArtifact",
    "EncodedDynamicsLatentDataset",
    "EncodedDynamicsResources",
    "load_encoded_dynamics_artifact",
    "migrate_encoded_dynamics_artifact",
    "preflight_encoded_dynamics_artifact",
]
