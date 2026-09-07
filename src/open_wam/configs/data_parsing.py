from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import enums as config_enums
from .coercion import (
    coerce_enum as _coerce_enum,
    coerce_optional_enum as _coerce_optional_enum,
)
from .data_benchmarks import (
    CalvinDataConfig,
    GenericDataConfig,
    LiberoDataConfig,
    RobotWinDataConfig,
)
from .data_consortium import (
    ConsortiumChannelMappingConfig,
    ConsortiumCloudCacheConfig,
    ConsortiumEpisodeSelectionConfig,
    ConsortiumLocalCacheConfig,
    ConsortiumMemberConfig,
    LeRobotConsortiumDataConfig,
)
from .data_contracts import (
    ActionMappingConfig,
    ActionNormalizationConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    BatchingConfig,
    CausalPrefixSuffixBucketConfig,
    DataConfig,
    DynamicsRoutingConfig,
    DynamicsRouteConfig,
    SampleConstructionConfig,
    ViewLayoutConfig,
)
from .data_mixed_video import (
    MixedVideoDataConfig,
    MixedVideoResizeBinConfig,
    MixedVideoSourceConfig,
    MixedVideoViewCombinationConfig,
)

__all__ = ["parse_data_config"]


def _batching_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("`data.batching` must be a mapping.")
    return dict(raw)


def _load_consortium_channel_mappings(raw_value: Any) -> tuple[ConsortiumChannelMappingConfig, ...]:
    mappings_raw = raw_value or ()
    if not isinstance(mappings_raw, (list, tuple)):
        raise ValueError("Expected `channel_mappings` to be a list of mappings.")
    return tuple(
        ConsortiumChannelMappingConfig(
            source_name=str(item["source_name"]),
            target_slot=str(item["target_slot"]),
        )
        for item in mappings_raw
    )


def _load_consortium_episode_selection(raw_value: Any) -> tuple[ConsortiumEpisodeSelectionConfig, ...]:
    selections_raw = raw_value or ()
    if not isinstance(selections_raw, (list, tuple)):
        raise ValueError("Expected explicit consortium episode selections to be a list.")
    return tuple(
        ConsortiumEpisodeSelectionConfig(
            member_id=str(item["member_id"]),
            episode_indices=tuple(int(value) for value in item.get("episode_indices", ())),
        )
        for item in selections_raw
    )


def _load_consortium_members(raw_value: Any) -> tuple[ConsortiumMemberConfig, ...]:
    members_raw = raw_value or ()
    if not isinstance(members_raw, (list, tuple)):
        raise ValueError("Expected `consortium_members` to be a list of member mappings.")
    members: list[ConsortiumMemberConfig] = []
    for item in members_raw:
        members.append(
            ConsortiumMemberConfig(
                member_id=item.get("member_id"),
                repo_id=item.get("repo_id"),
                local_root=item.get("local_root"),
                enabled=item.get("enabled", True),
                source_group=item.get("source_group"),
                include_channels=tuple(item.get("include_channels", ())),
                channel_mappings=_load_consortium_channel_mappings(item.get("channel_mappings")),
                sampling_weight=item.get("sampling_weight"),
            )
        )
    return tuple(members)


def _load_mixed_video_sources(raw_value: Any) -> tuple[MixedVideoSourceConfig, ...]:
    sources_raw = raw_value or ()
    if not isinstance(sources_raw, (list, tuple)):
        raise ValueError("Expected `video_sources` to be a list of source mappings.")
    sources: list[MixedVideoSourceConfig] = []
    for item in sources_raw:
        if not isinstance(item, dict):
            raise ValueError("Expected each `video_sources` entry to be a mapping.")
        sources.append(
            MixedVideoSourceConfig(
                source_id=str(item["source_id"]),
                manifest_csv=str(item["manifest_csv"]),
                repo_id=item.get("repo_id"),
                local_root=item.get("local_root"),
                latent_root=item.get("latent_root"),
                source_format=_coerce_enum(
                    config_enums.MixedVideoSourceFormat,
                    item.get("source_format", "rgb"),
                ),
                latent_key=str(item.get("latent_key", "video_latents")),
                enabled=item.get("enabled", True),
                source_group=item.get("source_group"),
                include_streams=tuple(item.get("include_streams", ())),
                channel_mappings=_load_consortium_channel_mappings(item.get("channel_mappings")),
                sampling_weight=item.get("sampling_weight"),
            )
        )
    return tuple(sources)


def _load_mixed_video_resize_bins(raw_value: Any) -> tuple[MixedVideoResizeBinConfig, ...] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError("Expected `decode_resize_bins` to be a list of bin mappings.")
    bins: list[MixedVideoResizeBinConfig] = []
    for item in raw_value:
        if not isinstance(item, dict):
            raise ValueError("Expected each `decode_resize_bins` entry to be a mapping.")
        bins.append(
            MixedVideoResizeBinConfig(
                name=str(item["name"]),
                aspect_width=int(item["aspect_width"]),
                aspect_height=int(item["aspect_height"]),
                target_height=int(item["target_height"]),
                target_width=int(item["target_width"]),
                max_pixels=item.get("max_pixels"),
            )
        )
    return tuple(bins)


def _load_mixed_video_view_combinations(raw_value: Any) -> tuple[MixedVideoViewCombinationConfig, ...]:
    combinations_raw = raw_value or ()
    if not isinstance(combinations_raw, (list, tuple)):
        raise ValueError("Expected `latent_view_combinations` to be a list of mappings.")
    combinations: list[MixedVideoViewCombinationConfig] = []
    for item in combinations_raw:
        if not isinstance(item, dict):
            raise ValueError("Expected each `latent_view_combinations` entry to be a mapping.")
        if "name" not in item:
            raise ValueError("Expected each `latent_view_combinations` entry to define `name`.")
        combinations.append(
            MixedVideoViewCombinationConfig(
                name=str(item["name"]),
                slots=tuple(str(value) for value in item.get("slots", ())),
                sampling_weight=float(item.get("sampling_weight", 1.0)),
                source_ids=tuple(str(value) for value in item.get("source_ids", ())),
                enabled=item.get("enabled", True),
            )
        )
    return tuple(combinations)


def _load_mixed_video_fit_mode(
    data_raw: dict[str, Any],
    data_defaults: MixedVideoDataConfig,
) -> config_enums.MixedVideoFrameFitMode:
    if "decode_fit_mode" in data_raw:
        return _coerce_enum(config_enums.MixedVideoFrameFitMode, data_raw["decode_fit_mode"])
    if "decode_center_crop" in data_raw:
        return (
            config_enums.MixedVideoFrameFitMode.CENTER_CROP
            if bool(data_raw["decode_center_crop"])
            else config_enums.MixedVideoFrameFitMode.LETTERBOX_PAD
        )
    return data_defaults.decode_fit_mode


def _load_action_mapping_config(raw_value: Any, defaults: ActionMappingConfig) -> ActionMappingConfig:
    raw = raw_value or {}
    if not isinstance(raw, dict):
        raise ValueError("Expected `data.action_mapping` to be a mapping.")
    normalization = _load_action_normalization_config(
        raw.get("normalization", None),
        defaults.normalization,
        field_path="data.action_mapping.normalization",
    )
    return ActionMappingConfig(
        mode=_coerce_enum(
            config_enums.ActionMappingMode,
            raw.get("mode", defaults.mode),
        ),
        source_dim=raw.get("source_dim", defaults.source_dim),
        target_dim=raw.get("target_dim", defaults.target_dim),
        source_to_target_indices=tuple(
            int(value) for value in raw.get("source_to_target_indices", defaults.source_to_target_indices)
        ),
        active_target_indices=tuple(
            int(value) for value in raw.get("active_target_indices", defaults.active_target_indices)
        ),
        inactive_value=float(raw.get("inactive_value", defaults.inactive_value)),
        loss_mask_mode=_coerce_enum(
            config_enums.ActionMappingLossMaskMode,
            raw.get("loss_mask_mode", defaults.loss_mask_mode),
        ),
        sampler_mask_mode=_coerce_enum(
            config_enums.ActionMappingSamplerMaskMode,
            raw.get("sampler_mask_mode", defaults.sampler_mask_mode),
        ),
        normalization=normalization,
    )


def _load_action_normalization_config(
    raw_value: Any,
    defaults: ActionNormalizationConfig,
    *,
    field_path: str,
) -> ActionNormalizationConfig:
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError(f"Expected `{field_path}` to be a mapping.")
    return ActionNormalizationConfig(
        mode=_coerce_enum(
            config_enums.ActionNormalizationMode,
            raw_value.get("mode", defaults.mode),
        ),
        mean=tuple(float(value) for value in raw_value.get("mean", defaults.mean)),
        std=tuple(float(value) for value in raw_value.get("std", defaults.std)),
        q01=tuple(float(value) for value in raw_value.get("q01", defaults.q01)),
        q99=tuple(float(value) for value in raw_value.get("q99", defaults.q99)),
        lower=tuple(float(value) for value in raw_value.get("lower", defaults.lower)),
        upper=tuple(float(value) for value in raw_value.get("upper", defaults.upper)),
        clip_min=raw_value.get("clip_min", defaults.clip_min),
        clip_max=raw_value.get("clip_max", defaults.clip_max),
    )


def _load_dynamics_routing_config(
    raw_value: Any,
    defaults: DynamicsRoutingConfig,
) -> DynamicsRoutingConfig:
    raw = raw_value or {}
    if not isinstance(raw, dict):
        raise ValueError("Expected `data.dynamics_routing` to be a mapping.")
    retired_fields = {
        "real_joint_weight",
        "real_action_conditioned_video_weight",
        "real_video_conditioned_action_weight",
        "counterfactual_action_conditioned_video_weight",
        "counterfactual_video_conditioned_action_weight",
        "conditional_history_frames",
    }
    authored_retired_fields = sorted(retired_fields.intersection(raw))
    if authored_retired_fields:
        joined = ", ".join(authored_retired_fields)
        raise ValueError(
            "Retired `data.dynamics_routing` fields were provided: "
            f"{joined}. Express each source/mode/weight choice in `routes` instead."
        )
    known_fields = {
        "train_latent_root",
        "val_latent_root",
        "allow_train_latent_root_for_val",
        "routes",
        "seed",
        "length_multiplier",
    }
    unknown_fields = sorted(set(raw).difference(known_fields, retired_fields))
    if unknown_fields:
        joined = ", ".join(str(field) for field in unknown_fields)
        raise ValueError(
            f"`data.dynamics_routing` contains unknown fields: {joined}."
        )
    routes_raw = raw.get("routes", defaults.routes)
    if not isinstance(routes_raw, (list, tuple)):
        raise ValueError("Expected `data.dynamics_routing.routes` to be a list.")
    parsed_routes: list[DynamicsRouteConfig] = []
    for index, route in enumerate(routes_raw):
        if isinstance(route, DynamicsRouteConfig):
            parsed_routes.append(route)
            continue
        if not isinstance(route, Mapping):
            raise ValueError(
                f"Expected `data.dynamics_routing.routes[{index}]` to be a mapping."
            )
        route_fields = {"source", "mode", "weight"}
        missing = route_fields.difference(route)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(
                f"`data.dynamics_routing.routes[{index}]` is missing: {fields}."
            )
        unknown = set(route).difference(route_fields)
        if unknown:
            fields = ", ".join(sorted(str(field) for field in unknown))
            raise ValueError(
                f"`data.dynamics_routing.routes[{index}]` contains unknown fields: {fields}."
            )
        parsed_routes.append(
            DynamicsRouteConfig(
                source=route["source"],
                mode=route["mode"],
                weight=route["weight"],
            )
        )
    return DynamicsRoutingConfig(
        train_latent_root=raw.get("train_latent_root", defaults.train_latent_root),
        val_latent_root=raw.get("val_latent_root", defaults.val_latent_root),
        allow_train_latent_root_for_val=raw.get(
            "allow_train_latent_root_for_val",
            defaults.allow_train_latent_root_for_val,
        ),
        routes=tuple(parsed_routes),
        seed=raw.get("seed", defaults.seed),
        length_multiplier=raw.get("length_multiplier", defaults.length_multiplier),
    )


def parse_data_config(raw_value: Mapping[str, Any] | None) -> DataConfig:
    """Parse one dataset section into the uniform typed data contract."""

    data_raw = raw_value or {}
    if "generalist_dynamics_mixture" in data_raw:
        raise ValueError(
            "`data.generalist_dynamics_mixture` is retired in authored configs; "
            "use `data.dynamics_routing`. Historical keys are accepted only by "
            "checkpoint runtime compatibility loading."
        )
    action_schema_raw = data_raw.get("action_schema", {})
    action_target_raw = data_raw.get("action_target", {})
    sample_construction_raw = data_raw.get("sample_construction", {})
    dataset_name = data_raw.get("dataset_name", "robotwin")
    dataset_type = data_raw.get("dataset_type")

    # Resolve defaults in two stages:
    # 1. known benchmark presets such as RobotWin and LIBERO
    # 2. a fully generic multiview fallback for custom sources
    #
    # This keeps new-source onboarding mostly declarative. A collaborator can
    # often add a new dataset config by specifying `dataset_type`, camera names,
    # layouts, and action/state schema in YAML without editing the loader.
    if dataset_name == "libero":
        if dataset_type == "libero_hdf5":
            data_defaults = LiberoDataConfig(dataset_type="libero_hdf5", repo_id=None)
        else:
            data_defaults = LiberoDataConfig()
        data_config_cls = LiberoDataConfig
    elif dataset_type == "lerobot_consortium" or dataset_name == "lerobot_consortium":
        data_defaults = LeRobotConsortiumDataConfig()
        data_config_cls = LeRobotConsortiumDataConfig
    elif dataset_type == "mixed_video" or dataset_name == "mixed_video":
        data_defaults = MixedVideoDataConfig(
            video_sources=_load_mixed_video_sources(data_raw.get("video_sources")),
        )
        data_config_cls = MixedVideoDataConfig
    elif dataset_name == "calvin" or dataset_type == "calvin_npz":
        data_defaults = CalvinDataConfig()
        data_config_cls = CalvinDataConfig
    elif dataset_name == "robotwin":
        data_defaults = RobotWinDataConfig()
        data_config_cls = RobotWinDataConfig
    elif dataset_type == "lerobot_v2":
        data_defaults = GenericDataConfig(dataset_name=dataset_name, dataset_type="lerobot_v2")
        data_config_cls = GenericDataConfig
    else:
        resolved_type = dataset_type or "synthetic_multiview"
        data_defaults = GenericDataConfig(dataset_name=dataset_name, dataset_type=resolved_type)
        data_config_cls = GenericDataConfig

    sample_construction_mode = _coerce_enum(
        config_enums.WindowSamplingMode,
        sample_construction_raw.get("mode", data_defaults.sample_construction.mode),
    )
    sample_target_alignment = _coerce_enum(
        config_enums.SampleTargetAlignment,
        sample_construction_raw.get(
            "target_alignment",
            data_defaults.sample_construction.target_alignment,
        ),
    )
    if sample_target_alignment == config_enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT:
        legacy_context_keys = [key for key in ("context_prefix_policy", "context_prefix_frames") if key in sample_construction_raw]
        if legacy_context_keys:
            joined = ", ".join(f"`{key}`" for key in legacy_context_keys)
            raise ValueError(
                "`sample_construction.target_alignment=next_after_context` uses "
                "`rollout_context_policy` / `rollout_context_frames`; remove legacy context fields: "
                f"{joined}."
            )
    if sample_construction_mode == config_enums.WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT:
        legacy_hierarchical_keys = (
            "segment_min_frames",
            "segment_max_frames",
            "randomize_segment_length",
            "randomize_segment_start",
            "require_full_segment",
            "sample_weight_mode",
            "sample_weight_length_power",
        )
        present_legacy_keys = [key for key in legacy_hierarchical_keys if key in sample_construction_raw]
        if present_legacy_keys:
            joined = ", ".join(f"`{key}`" for key in present_legacy_keys)
            raise ValueError(
                "`sample_construction.mode=hierarchical_fixed_segment` uses `segment_frames`, padding policies, "
                f"and hierarchical powers; remove legacy fields: {joined}."
            )

    default_view_layout = [
        {
            "source_name": view.source_name,
            "canonical_name": view.canonical_name,
            "top": view.top,
            "left": view.left,
            "height": view.height,
            "width": view.width,
        }
        for view in data_defaults.view_layout
    ]
    view_layout_raw = data_raw.get("view_layout", default_view_layout)
    view_layout = tuple(
        ViewLayoutConfig(
            source_name=view["source_name"],
            canonical_name=view.get("canonical_name", view["source_name"]),
            top=view["top"],
            left=view["left"],
            height=view["height"],
            width=view["width"],
        )
        for view in view_layout_raw
    )

    common_data_kwargs = dict(
        dataset_name=dataset_name,
        dataset_type=data_raw.get("dataset_type", data_defaults.dataset_type),
        repo_id=data_raw.get("repo_id", data_defaults.repo_id),
        local_root=data_raw.get("local_root", data_defaults.local_root),
        val_local_root=data_raw.get("val_local_root", data_defaults.val_local_root),
        empty_text_embedding_path=data_raw.get(
            "empty_text_embedding_path",
            data_defaults.empty_text_embedding_path,
        ),
        latent_root=data_raw.get("latent_root", data_defaults.latent_root),
        latent_subdir=data_raw.get("latent_subdir", data_defaults.latent_subdir),
        latent_window_profile=_coerce_enum(
            config_enums.LatentWindowProfile,
            data_raw.get("latent_window_profile", data_defaults.latent_window_profile),
        ),
        latent_temporal_layout=_coerce_enum(
            config_enums.LatentTemporalLayout,
            data_raw.get("latent_temporal_layout", data_defaults.latent_temporal_layout),
        ),
        split=_coerce_enum(config_enums.DataSplit, data_raw.get("split", data_defaults.split)),
        cache_dir=data_raw.get("cache_dir", data_defaults.cache_dir),
        camera_names=tuple(data_raw.get("camera_names", data_defaults.camera_names)),
        latent_camera_names=tuple(data_raw.get("latent_camera_names", data_defaults.latent_camera_names)),
        canonical_height=data_raw.get("canonical_height", data_defaults.canonical_height),
        canonical_width=data_raw.get("canonical_width", data_defaults.canonical_width),
        view_layout=view_layout,
        num_frames=data_raw.get("num_frames", data_defaults.num_frames),
        frame_stride=data_raw.get("frame_stride", data_defaults.frame_stride),
        sample_stride=data_raw.get("sample_stride", data_defaults.sample_stride),
        episode_cache_size=data_raw.get("episode_cache_size", data_defaults.episode_cache_size),
        train_fraction=data_raw.get("train_fraction", data_defaults.train_fraction),
        split_seed=data_raw.get("split_seed", data_defaults.split_seed),
        max_train_episodes=data_raw.get("max_train_episodes", data_defaults.max_train_episodes),
        max_val_episodes=data_raw.get("max_val_episodes", data_defaults.max_val_episodes),
        replay_status_path=data_raw.get("replay_status_path", data_defaults.replay_status_path),
        val_replay_status_path=data_raw.get("val_replay_status_path", data_defaults.val_replay_status_path),
        replay_status_policy=_coerce_enum(
            config_enums.ReplayStatusPolicy,
            data_raw.get("replay_status_policy", data_defaults.replay_status_policy),
        ),
        require_replay_status=data_raw.get("require_replay_status", data_defaults.require_replay_status),
        val_replay_status_policy=_coerce_optional_enum(
            config_enums.ReplayStatusPolicy,
            data_raw.get("val_replay_status_policy", data_defaults.val_replay_status_policy),
        ),
        val_require_replay_status=data_raw.get(
            "val_require_replay_status",
            data_defaults.val_require_replay_status,
        ),
        train_batch_size=data_raw.get("train_batch_size", data_defaults.train_batch_size),
        val_batch_size=data_raw.get("val_batch_size", data_defaults.val_batch_size),
        num_workers=data_raw.get("num_workers", data_defaults.num_workers),
        batching=BatchingConfig(**_batching_mapping(data_raw.get("batching", {}))),
        action_schema=ActionSchemaConfig(
            action_dim=action_schema_raw.get("action_dim", data_defaults.action_schema.action_dim),
            action_horizon=action_schema_raw.get("action_horizon", data_defaults.action_schema.action_horizon),
            state_dim=action_schema_raw.get("state_dim", data_defaults.action_schema.state_dim),
            state_horizon=action_schema_raw.get("state_horizon", data_defaults.action_schema.state_horizon),
        ),
        action_target=ActionTargetConfig(
            representation=_coerce_enum(
                config_enums.ActionTargetRepresentation,
                action_target_raw.get("representation", data_defaults.action_target.representation),
            ),
            source_key=action_target_raw.get("source_key", data_defaults.action_target.source_key),
            pose_source_key=action_target_raw.get("pose_source_key", data_defaults.action_target.pose_source_key),
            state_encoding=_coerce_enum(
                config_enums.ActionTargetStateEncoding,
                action_target_raw.get("state_encoding", data_defaults.action_target.state_encoding),
            ),
            reference_source=_coerce_enum(
                config_enums.ActionTargetReferenceSource,
                action_target_raw.get("reference_source", data_defaults.action_target.reference_source),
            ),
            rotation_representation=_coerce_enum(
                config_enums.RotationRepresentation,
                action_target_raw.get("rotation_representation", data_defaults.action_target.rotation_representation),
            ),
            include_gripper=action_target_raw.get("include_gripper", data_defaults.action_target.include_gripper),
            gripper_representation=_coerce_enum(
                config_enums.GripperRepresentation,
                action_target_raw.get("gripper_representation", data_defaults.action_target.gripper_representation),
            ),
            gripper_action_index=action_target_raw.get(
                "gripper_action_index",
                data_defaults.action_target.gripper_action_index,
            ),
            gripper_position_source_key=action_target_raw.get(
                "gripper_position_source_key",
                data_defaults.action_target.gripper_position_source_key,
            ),
            joint_position_source_key=action_target_raw.get(
                "joint_position_source_key",
                data_defaults.action_target.joint_position_source_key,
            ),
            joint_position_normalization=_load_action_normalization_config(
                action_target_raw.get("joint_position_normalization"),
                data_defaults.action_target.joint_position_normalization,
                field_path="data.action_target.joint_position_normalization",
            ),
            normalization=_load_action_normalization_config(
                action_target_raw.get("normalization"),
                data_defaults.action_target.normalization,
                field_path="data.action_target.normalization",
            ),
        ),
        action_mapping=_load_action_mapping_config(
            data_raw.get("action_mapping"),
            data_defaults.action_mapping,
        ),
        sample_construction=SampleConstructionConfig(
            mode=sample_construction_mode,
            anchor_policy=_coerce_enum(
                config_enums.AnchorPolicy,
                sample_construction_raw.get(
                    "anchor_policy",
                    data_defaults.sample_construction.anchor_policy,
                ),
            ),
            num_frames=sample_construction_raw.get(
                "num_frames",
                data_defaults.sample_construction.num_frames,
            ),
            action_horizon=sample_construction_raw.get(
                "action_horizon",
                data_defaults.sample_construction.action_horizon,
            ),
            state_horizon=sample_construction_raw.get(
                "state_horizon",
                data_defaults.sample_construction.state_horizon,
            ),
            state_anchor_mode=_coerce_enum(
                config_enums.SampleStateAnchorMode,
                sample_construction_raw.get(
                    "state_anchor_mode",
                    data_defaults.sample_construction.state_anchor_mode,
                ),
            ),
            frame_stride=sample_construction_raw.get(
                "frame_stride",
                data_defaults.sample_construction.frame_stride,
            ),
            chunk_size=sample_construction_raw.get(
                "chunk_size",
                data_defaults.sample_construction.chunk_size,
            ),
            window_size=sample_construction_raw.get(
                "window_size",
                data_defaults.sample_construction.window_size,
            ),
            predict_blocks_per_sample=sample_construction_raw.get(
                "predict_blocks_per_sample",
                data_defaults.sample_construction.predict_blocks_per_sample,
            ),
            randomize_geometry=sample_construction_raw.get(
                "randomize_geometry",
                data_defaults.sample_construction.randomize_geometry,
            ),
            allow_next_after_context_random_geometry=sample_construction_raw.get(
                "allow_next_after_context_random_geometry",
                data_defaults.sample_construction.allow_next_after_context_random_geometry,
            ),
            target_alignment=sample_target_alignment,
            rollout_context_policy=_coerce_enum(
                config_enums.RolloutContextPolicy,
                sample_construction_raw.get(
                    "rollout_context_policy",
                    data_defaults.sample_construction.rollout_context_policy,
                ),
            ),
            rollout_context_frames=sample_construction_raw.get(
                "rollout_context_frames",
                data_defaults.sample_construction.rollout_context_frames,
            ),
            segment_frames=sample_construction_raw.get(
                "segment_frames",
                data_defaults.sample_construction.segment_frames,
            ),
            segment_min_frames=sample_construction_raw.get(
                "segment_min_frames",
                data_defaults.sample_construction.segment_min_frames,
            ),
            segment_max_frames=sample_construction_raw.get(
                "segment_max_frames",
                data_defaults.sample_construction.segment_max_frames,
            ),
            segment_length_stride=sample_construction_raw.get(
                "segment_length_stride",
                data_defaults.sample_construction.segment_length_stride,
            ),
            segment_locality_block_size=sample_construction_raw.get(
                "segment_locality_block_size",
                data_defaults.sample_construction.segment_locality_block_size,
            ),
            randomize_segment_length=sample_construction_raw.get(
                "randomize_segment_length",
                data_defaults.sample_construction.randomize_segment_length,
            ),
            randomize_segment_start=sample_construction_raw.get(
                "randomize_segment_start",
                data_defaults.sample_construction.randomize_segment_start,
            ),
            require_full_segment=sample_construction_raw.get(
                "require_full_segment",
                data_defaults.sample_construction.require_full_segment,
            ),
            start_padding_frames=sample_construction_raw.get(
                "start_padding_frames",
                data_defaults.sample_construction.start_padding_frames,
            ),
            condition_source_frame_offset=sample_construction_raw.get(
                "condition_source_frame_offset",
                data_defaults.sample_construction.condition_source_frame_offset,
            ),
            context_prefix_policy=_coerce_enum(
                config_enums.SegmentContextPolicy,
                sample_construction_raw.get(
                    "context_prefix_policy",
                    data_defaults.sample_construction.context_prefix_policy,
                ),
            ),
            context_prefix_frames=sample_construction_raw.get(
                "context_prefix_frames",
                data_defaults.sample_construction.context_prefix_frames,
            ),
            tail_padding_policy=_coerce_enum(
                config_enums.TailPaddingPolicy,
                sample_construction_raw.get(
                    "tail_padding_policy",
                    data_defaults.sample_construction.tail_padding_policy,
                ),
            ),
            padded_target_policy=_coerce_enum(
                config_enums.PaddedTargetPolicy,
                sample_construction_raw.get(
                    "padded_target_policy",
                    data_defaults.sample_construction.padded_target_policy,
                ),
            ),
            task_start_power=sample_construction_raw.get(
                "task_start_power",
                data_defaults.sample_construction.task_start_power,
            ),
            demo_count_power=sample_construction_raw.get(
                "demo_count_power",
                data_defaults.sample_construction.demo_count_power,
            ),
            trajectory_start_power=sample_construction_raw.get(
                "trajectory_start_power",
                data_defaults.sample_construction.trajectory_start_power,
            ),
            sample_weight_mode=_coerce_enum(
                config_enums.SampleWeightMode,
                sample_construction_raw.get(
                    "sample_weight_mode",
                    data_defaults.sample_construction.sample_weight_mode,
                ),
            ),
            sample_order_mode=_coerce_enum(
                config_enums.SampleOrderMode,
                sample_construction_raw.get(
                    "sample_order_mode",
                    data_defaults.sample_construction.sample_order_mode,
                ),
            ),
            sample_weight_length_power=sample_construction_raw.get(
                "sample_weight_length_power",
                data_defaults.sample_construction.sample_weight_length_power,
            ),
            sample_weight_min=sample_construction_raw.get(
                "sample_weight_min",
                data_defaults.sample_construction.sample_weight_min,
            ),
            sample_weight_max=sample_construction_raw.get(
                "sample_weight_max",
                data_defaults.sample_construction.sample_weight_max,
            ),
            causal_prefix_suffix_buckets=tuple(
                CausalPrefixSuffixBucketConfig(
                    observed_frames=int(bucket["observed_frames"]),
                    future_frames=int(bucket["future_frames"]),
                )
                for bucket in sample_construction_raw.get(
                    "causal_prefix_suffix_buckets",
                    tuple(
                        {
                            "observed_frames": bucket.observed_frames,
                            "future_frames": bucket.future_frames,
                        }
                        for bucket in data_defaults.sample_construction.causal_prefix_suffix_buckets
                    ),
                )
            ),
        ),
        dynamics_routing=_load_dynamics_routing_config(
            data_raw.get("dynamics_routing"),
            data_defaults.dynamics_routing,
        ),
        adapter_options=data_raw.get("adapter_options", data_defaults.adapter_options),
    )
    if data_config_cls is LeRobotConsortiumDataConfig:
        common_data_kwargs.update(
            consortium_members=_load_consortium_members(data_raw.get("consortium_members")),
            channel_selection_mode=_coerce_enum(
                config_enums.ConsortiumChannelSelectionMode,
                data_raw.get("channel_selection_mode", data_defaults.channel_selection_mode),
            ),
            required_channels=tuple(data_raw.get("required_channels", data_defaults.required_channels)),
            channel_mappings=_load_consortium_channel_mappings(data_raw.get("channel_mappings")),
            view_packing_mode=_coerce_enum(
                config_enums.ConsortiumViewPackingMode,
                data_raw.get("view_packing_mode", data_defaults.view_packing_mode),
            ),
            frame_packing_order=_coerce_enum(
                config_enums.ConsortiumFramePackingOrder,
                data_raw.get("frame_packing_order", data_defaults.frame_packing_order),
            ),
            missing_channel_policy=_coerce_enum(
                config_enums.ConsortiumMissingChannelPolicy,
                data_raw.get("missing_channel_policy", data_defaults.missing_channel_policy),
            ),
            random_mode=_coerce_enum(
                config_enums.ConsortiumRandomMode,
                data_raw.get("random_mode", data_defaults.random_mode),
            ),
            weight_mode=_coerce_enum(
                config_enums.ConsortiumWeightMode,
                data_raw.get("weight_mode", data_defaults.weight_mode),
            ),
            sampling_seed=data_raw.get("sampling_seed", data_defaults.sampling_seed),
            split_mode=_coerce_enum(
                config_enums.ConsortiumSplitMode,
                data_raw.get("split_mode", data_defaults.split_mode),
            ),
            explicit_train_episodes=_load_consortium_episode_selection(data_raw.get("explicit_train_episodes")),
            explicit_val_episodes=_load_consortium_episode_selection(data_raw.get("explicit_val_episodes")),
            local_cache=ConsortiumLocalCacheConfig(
                mode=_coerce_enum(
                    config_enums.ConsortiumCacheMode,
                    (data_raw.get("local_cache", {}) or {}).get("mode", data_defaults.local_cache.mode),
                ),
                root=(data_raw.get("local_cache", {}) or {}).get("root", data_defaults.local_cache.root),
            ),
            cloud_cache=ConsortiumCloudCacheConfig(
                mode=_coerce_enum(
                    config_enums.ConsortiumCacheMode,
                    (data_raw.get("cloud_cache", {}) or {}).get("mode", data_defaults.cloud_cache.mode),
                ),
                backend=_coerce_enum(
                    config_enums.ConsortiumCloudCacheBackend,
                    (data_raw.get("cloud_cache", {}) or {}).get("backend", data_defaults.cloud_cache.backend),
                ),
                root=(data_raw.get("cloud_cache", {}) or {}).get("root", data_defaults.cloud_cache.root),
            ),
        )
    if data_config_cls is MixedVideoDataConfig:
        resize_bins = _load_mixed_video_resize_bins(data_raw.get("decode_resize_bins"))
        common_data_kwargs.update(
            video_sources=_load_mixed_video_sources(data_raw.get("video_sources")),
            latent_encoding_mode=_coerce_enum(
                config_enums.MixedVideoLatentEncodingMode,
                data_raw.get("latent_encoding_mode", data_defaults.latent_encoding_mode),
            ),
            latent_view_combinations=_load_mixed_video_view_combinations(data_raw.get("latent_view_combinations")),
            decode_size_mode=_coerce_enum(
                config_enums.MixedVideoDecodeSizeMode,
                data_raw.get("decode_size_mode", data_defaults.decode_size_mode),
            ),
            decode_resize_bins=data_defaults.decode_resize_bins if resize_bins is None else resize_bins,
            decode_height=int(data_raw.get("decode_height", data_defaults.decode_height)),
            decode_width=int(data_raw.get("decode_width", data_defaults.decode_width)),
            decode_fit_mode=_load_mixed_video_fit_mode(data_raw, data_defaults),
            decode_center_crop=bool(data_raw.get("decode_center_crop", data_defaults.decode_center_crop)),
            decode_allow_upscale=bool(data_raw.get("decode_allow_upscale", data_defaults.decode_allow_upscale)),
            target_observation_fps=(
                None
                if data_raw.get("target_observation_fps", data_defaults.target_observation_fps) is None
                else float(data_raw.get("target_observation_fps", data_defaults.target_observation_fps))
            ),
            missing_observation_fps=float(
                data_raw.get("missing_observation_fps", data_defaults.missing_observation_fps)
            ),
            missing_stream_policy=_coerce_enum(
                config_enums.MixedVideoMissingStreamPolicy,
                data_raw.get("missing_stream_policy", data_defaults.missing_stream_policy),
            ),
            random_mode=_coerce_enum(
                config_enums.MixedVideoRandomMode,
                data_raw.get("random_mode", data_defaults.random_mode),
            ),
            weight_mode=_coerce_enum(
                config_enums.MixedVideoWeightMode,
                data_raw.get("weight_mode", data_defaults.weight_mode),
            ),
            sampling_seed=data_raw.get("sampling_seed", data_defaults.sampling_seed),
        )
    if data_config_cls is CalvinDataConfig:
        common_data_kwargs.update(
            language_annotation_pickle_policy=_coerce_enum(
                config_enums.LegacyPicklePolicy,
                data_raw.get(
                    "language_annotation_pickle_policy",
                    data_defaults.language_annotation_pickle_policy,
                ),
            ),
        )

    # `data_config_cls` may be a benchmark-specific preset or the generic
    # fallback. In both cases, the instantiated object carries the exact view
    # layout and action/state schema that the rest of the code should trust.
    data_config = data_config_cls(
        **common_data_kwargs,
    )
    return data_config
