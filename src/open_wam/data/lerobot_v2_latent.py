"""Stable facade for local LeRobot latent datasets."""

# Compatibility imports intentionally preserve historical module identities.
# ruff: noqa: I001, PLC0414

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import math
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from open_wam.configs import (
    DataConfig,
    DataSplit,
    LatentWindowProfile,
    PaddedTargetPolicy,
    SampleOrderMode,
    SampleWeightMode,
    SampleTargetAlignment,
    TailPaddingPolicy,
    WindowSamplingMode,
)

from .latent_causal_sampling import LatentCausalPrefixSuffixWindowPlanner
from .latent_contracts import LatentWAMSample
from .latent_hierarchical_sampling import LocalLatentHierarchicalSegmentPlan
from .latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET as CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    latent_anchor_positions,
    observed_frame_ids_for_latent_segment,
    raw_span_for_latent_range,
)

# Keep storage symbols importable from this historical module while ownership
# lives in the repository adapter.
from .lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    LocalLatentRepository,
    LocalRepoBundle as LocalRepoBundle,
    assemble_canonical_latents as assemble_canonical_latents,
    condition_latent_offset_mismatches as condition_latent_offset_mismatches,
    discover_local_lerobot_repo_bundles,
    latent_filename as latent_filename,
    load_empty_text_embedding,
    load_lerobot_v2_local_metadata as load_lerobot_v2_local_metadata,
    read_json_local as read_json_local,
    read_jsonl_local as read_jsonl_local,
    reshape_latent_payload as reshape_latent_payload,
    resolve_latent_root as resolve_latent_root,
    scan_local_latent_windows,
    split_local_episode_indices as split_local_episode_indices,
)
from .lerobot_v2_latent_hierarchical_policy import (
    HierarchicalFixedSegmentSamplingPlan,
    HierarchicalFixedSegmentTaskSpec as HierarchicalFixedSegmentTaskSpec,
    HierarchicalFixedSegmentWindowSpec as HierarchicalFixedSegmentWindowSpec,
    build_hierarchical_fixed_segment_task_specs,
)
from .lerobot_v2_latent_sampler_adapters import (
    HierarchicalFixedSegmentTrainSampler as HierarchicalFixedSegmentTrainSampler,
    LocalLatentEpochOrderSampler as LocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler as LocalLatentWeightedTrainSampler,
)
from .lerobot_v2_latent_segment import LocalLatentSegmentAssembler
from .lerobot_v2_latent_source import (
    LocalLatentSampleSource,
    LocalLatentSampleSourceLoader,
)
from .lerobot_v2_latent_split import LocalLatentTrainValWindowPlanner
from .lerobot_v2_latent_supervision import LocalLatentSupervisionAssembler
from .lerobot_v2_latent_uniform_policy import (
    LocalLatentUniformSegmentSamplingPlan as _LocalLatentUniformSegmentSamplingPlan,
)
from .lerobot_v2_latent_weighting import LocalLatentWindowWeightPlan

from .lerobot_v2_latent_base_dataset import (
    FullSegmentLocalLeRobotLatentDataset,
    LocalLeRobotLatentWindowDataset,
)
from .lerobot_v2_latent_causal_dataset import (
    CausalPrefixSuffixLocalLeRobotLatentDataset,
)
from .lerobot_v2_latent_factory import (
    build_local_lerobot_latent_train_val_datasets,
)
from .lerobot_v2_latent_hierarchical_dataset import (
    HierarchicalFixedSegmentLocalLeRobotLatentDataset,
)
from .lerobot_v2_latent_uniform_dataset import (
    UniformSegmentLocalLeRobotLatentDataset,
)

_COMPATIBILITY_EXPORTS = (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    HierarchicalFixedSegmentSamplingPlan,
    HierarchicalFixedSegmentTaskSpec,
    HierarchicalFixedSegmentTrainSampler,
    HierarchicalFixedSegmentWindowSpec,
    LocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler,
    _LocalLatentUniformSegmentSamplingPlan,
    assemble_canonical_latents,
    build_hierarchical_fixed_segment_task_specs,
    condition_latent_offset_mismatches,
    discover_local_lerobot_repo_bundles,
    latent_anchor_positions,
    LocalRepoBundle,
    latent_filename,
    load_empty_text_embedding,
    load_lerobot_v2_local_metadata,
    read_json_local,
    read_jsonl_local,
    reshape_latent_payload,
    resolve_latent_root,
    scan_local_latent_windows,
    split_local_episode_indices,
)


# Preserve the historical wildcard-import surface.
__all__ = [
    "CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET",
    "Any",
    "CausalPrefixSuffixLocalLeRobotLatentDataset",
    "DataConfig",
    "DataSplit",
    "Dataset",
    "FullSegmentLocalLeRobotLatentDataset",
    "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
    "HierarchicalFixedSegmentSamplingPlan",
    "HierarchicalFixedSegmentTaskSpec",
    "HierarchicalFixedSegmentTrainSampler",
    "HierarchicalFixedSegmentWindowSpec",
    "Iterator",
    "LatentCausalPrefixSuffixWindowPlanner",
    "LatentWAMSample",
    "LatentWindowProfile",
    "LocalEpisodeWindow",
    "LocalLatentEpochOrderSampler",
    "LocalLatentHierarchicalSegmentPlan",
    "LocalLatentRepository",
    "LocalLatentSampleSource",
    "LocalLatentSampleSourceLoader",
    "LocalLatentSegmentAssembler",
    "LocalLatentSupervisionAssembler",
    "LocalLatentTrainValWindowPlanner",
    "LocalLatentWeightedTrainSampler",
    "LocalLatentWindowWeightPlan",
    "LocalLeRobotLatentWindowDataset",
    "LocalRepoBundle",
    "PaddedTargetPolicy",
    "SampleOrderMode",
    "SampleTargetAlignment",
    "SampleWeightMode",
    "Sampler",
    "TailPaddingPolicy",
    "UniformSegmentLocalLeRobotLatentDataset",
    "WindowSamplingMode",
    "annotations",
    "assemble_canonical_latents",
    "build_hierarchical_fixed_segment_task_specs",
    "build_local_lerobot_latent_train_val_datasets",
    "condition_latent_offset_mismatches",
    "discover_local_lerobot_repo_bundles",
    "latent_anchor_positions",
    "latent_filename",
    "load_empty_text_embedding",
    "load_lerobot_v2_local_metadata",
    "math",
    "observed_frame_ids_for_latent_segment",
    "raw_span_for_latent_range",
    "read_json_local",
    "read_jsonl_local",
    "replace",
    "reshape_latent_payload",
    "resolve_latent_root",
    "scan_local_latent_windows",
    "split_local_episode_indices",
    "torch",
]
