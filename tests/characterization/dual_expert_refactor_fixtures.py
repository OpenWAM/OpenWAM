from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from open_wam.configs import (
    DynamicsObjective,
    DynamicsSource,
    ExperimentConfig,
    load_experiment_config,
)
from open_wam.data import (
    build_dynamics_routing_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
)
from open_wam.data.lerobot_video import (
    LeRobotV2VideoWindowDataset,
    load_lerobot_v2_video_metadata,
)
from open_wam.utils import apply_config_overrides

from .dual_expert_refactor_artifacts import (
    RolloutInputFixture,
    save_latent_batch_fixture,
    save_rollout_input_fixture,
)
from .dual_expert_refactor_contract import (
    METHOD_BY_ASSET_ID,
    CharacterizationAssets,
    DualExpertTrainingProfile,
    apply_ground_truth_training_profile,
    load_characterization_assets,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "experiments"
FIXTURE_SEED = 20260727


def configure_characterization_paths(
    config: ExperimentConfig,
    assets: CharacterizationAssets,
    *,
    include_counterfactual: bool,
) -> ExperimentConfig:
    overrides: dict[str, Any] = {
        "data.local_root": str(assets.dataset_root),
        "data.empty_text_embedding_path": str(assets.empty_text_embedding),
        "data.num_workers": 0,
        "backbone.pretrained_model_name_or_path": str(assets.base_model_root),
        "backbone.transformer_subdir": str(assets.video_transformer_root),
    }
    if include_counterfactual:
        if (
            assets.counterfactual_train_root is None
            or assets.counterfactual_val_root is None
        ):
            raise ValueError(
                "GJD fixtures require both counterfactual_train_root and "
                "counterfactual_val_root in the asset manifest."
            )
        overrides.update(
            {
                "data.dynamics_routing.train_latent_root": str(
                    assets.counterfactual_train_root
                ),
                "data.dynamics_routing.val_latent_root": str(
                    assets.counterfactual_val_root
                ),
            }
        )
    return apply_config_overrides(config, overrides)


def build_all_characterization_fixtures(
    *,
    assets: CharacterizationAssets,
    output_root: str | Path,
    seed: int = FIXTURE_SEED,
) -> dict[str, Path]:
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    method = METHOD_BY_ASSET_ID["mot_joint"]
    full_batch = None
    for profile, fixture_id in (
        (DualExpertTrainingProfile.RANDOM_SEGMENT, "mot_random_segment"),
        (DualExpertTrainingProfile.FULL_SEGMENT_W64, "mot_full_segment_w64"),
    ):
        config = _load_non_gjd_fixture_config(assets, profile=profile)
        _seed_everything(seed)
        train_dataset, _ = build_train_val_latent_datasets(config.data)
        sample_index = _representative_source_index(train_dataset)
        sample = train_dataset[sample_index]
        batch = collate_latent_wam_samples([sample])
        generated[fixture_id] = save_latent_batch_fixture(
            output_path,
            fixture_id=fixture_id,
            batch=batch,
            provenance={
                "seed": seed,
                "sample_index": sample_index,
                "config_name": method.artifact_config_name,
                "training_profile": profile.value,
                "resolved_contract": _resolved_training_contract(config),
                "source_metadata": sample.metadata,
            },
        )
        if profile == DualExpertTrainingProfile.FULL_SEGMENT_W64:
            full_batch = batch

    generated.update(
        _build_gjd_fixtures(
            assets=assets,
            output_root=output_path,
            seed=seed,
        )
    )
    if full_batch is None:  # pragma: no cover - guarded by the fixed profile matrix
        raise RuntimeError("Full-segment fixture was not built.")
    generated["mot_streaming_startup"] = _build_rollout_fixture(
        assets=assets,
        output_root=output_path,
        full_batch=full_batch,
        seed=seed,
    )
    _write_fixture_index(output_path, generated=generated, seed=seed)
    return generated


def _load_non_gjd_fixture_config(
    assets: CharacterizationAssets,
    *,
    profile: DualExpertTrainingProfile,
) -> ExperimentConfig:
    method = METHOD_BY_ASSET_ID["mot_joint"]
    config = load_experiment_config(CONFIG_ROOT / f"{method.config_name}.yaml")
    config = configure_characterization_paths(
        config,
        assets,
        include_counterfactual=False,
    )
    return apply_ground_truth_training_profile(config, profile)


def _load_gjd_fixture_config(assets: CharacterizationAssets) -> ExperimentConfig:
    method = METHOD_BY_ASSET_ID["gjd_vanilla"]
    config = load_experiment_config(CONFIG_ROOT / f"{method.config_name}.yaml")
    return configure_characterization_paths(
        config,
        assets,
        include_counterfactual=True,
    )


def _build_gjd_fixtures(
    *,
    assets: CharacterizationAssets,
    output_root: Path,
    seed: int,
) -> dict[str, Path]:
    method = METHOD_BY_ASSET_ID["gjd_vanilla"]
    config = _load_gjd_fixture_config(assets)
    _seed_everything(seed)
    real_train, real_val = build_train_val_latent_datasets(config.data)
    train_mixture, _ = build_dynamics_routing_datasets(
        data_config=config.data,
        train_dataset=real_train,
        val_dataset=real_val,
    )
    source_specs = (
        ("gjd_real_joint", DynamicsSource.REAL_DEMO, DynamicsObjective.JOINT),
        (
            "gjd_real_fdm",
            DynamicsSource.REAL_DEMO,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            "gjd_real_idm",
            DynamicsSource.REAL_DEMO,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
        (
            "gjd_counterfactual_fdm",
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            "gjd_counterfactual_idm",
            DynamicsSource.COUNTERFACTUAL_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    )
    generated: dict[str, Path] = {}
    real_source_index = _representative_source_index(real_train)
    for fixture_offset, (fixture_id, source, mode) in enumerate(
        source_specs
    ):
        source_view = train_mixture.build_source_view(
            source=source,
            mode=mode,
            bucket_name=fixture_id,
        )
        _seed_everything(seed + fixture_offset)
        sample_index = (
            real_source_index
            if source == DynamicsSource.REAL_DEMO
            else _representative_counterfactual_source_index(source_view)
        )
        sample = source_view[sample_index]
        batch = collate_latent_wam_samples([sample])
        generated[fixture_id] = save_latent_batch_fixture(
            output_root,
            fixture_id=fixture_id,
            batch=batch,
            provenance={
                "seed": seed + fixture_offset,
                "sample_index": sample_index,
                "config_name": method.artifact_config_name,
                "source": source.value,
                "mode": mode.value,
                "drop_text": mode.is_conditional,
                "resolved_contract": _resolved_training_contract(config),
                "source_metadata": sample.metadata,
            },
        )
    return generated


def _build_rollout_fixture(
    *,
    assets: CharacterizationAssets,
    output_root: Path,
    full_batch,
    seed: int,
) -> Path:
    metadata = dict(full_batch.metadata[0])
    episode_index = int(metadata["episode_index"])
    observation_frame_indices = metadata.get("observation_frame_indices")
    if (
        not isinstance(observation_frame_indices, (tuple, list))
        or not observation_frame_indices
    ):
        raise ValueError(
            "Full-segment fixture lacks observation_frame_indices metadata."
        )
    observation_start = int(observation_frame_indices[0])

    config = _load_non_gjd_fixture_config(
        assets,
        profile=DualExpertTrainingProfile.FULL_SEGMENT_W64,
    )
    raw_config = apply_config_overrides(
        config,
        {
            "data.dataset_type": "lerobot_v2_video",
            "data.num_frames": 1,
            "data.frame_stride": 1,
            "data.sample_stride": 1,
        },
    )
    raw_metadata = load_lerobot_v2_video_metadata(assets.dataset_root)
    raw_dataset = LeRobotV2VideoWindowDataset(
        data_config=raw_config.data,
        episodes=[record.episode_index for record in raw_metadata.episodes],
    )
    raw_index = next(
        (
            index
            for index, window in enumerate(raw_dataset.sample_index)
            if int(window.episode_index) == episode_index
            and int(window.observation_start) == observation_start
        ),
        None,
    )
    if raw_index is None:
        raise ValueError(
            "Could not align the rollout startup fixture with the frozen latent "
            f"sample: episode={episode_index}, frame={observation_start}."
        )
    raw_sample = raw_dataset[raw_index]
    if full_batch.text_context is None or full_batch.negative_text_context is None:
        raise ValueError(
            "Strict rollout fixture requires positive and negative text embeddings."
        )
    if int(full_batch.video_latents.shape[2]) < 5:
        raise ValueError(
            "Strict rollout fixture requires t0 plus four future latent frames."
        )

    fixture = RolloutInputFixture(
        views={name: value.unsqueeze(0) for name, value in raw_sample.views.items()},
        state=raw_sample.state.unsqueeze(0)
        if raw_sample.state is not None
        else full_batch.state,
        text_context=full_batch.text_context,
        negative_text_context=full_batch.negative_text_context,
        action_conditioning=raw_sample.actions.unsqueeze(0),
        video_conditioning=full_batch.video_latents[:, :, 1:5].contiguous(),
        task_text=(raw_sample.task_text,),
        metadata={
            "seed": seed,
            "episode_index": episode_index,
            "observation_start": observation_start,
            "raw_dataset_index": raw_index,
            "raw_metadata": raw_sample.metadata,
            "latent_metadata": metadata,
        },
    )
    return save_rollout_input_fixture(
        output_root,
        fixture_id="mot_streaming_startup",
        fixture=fixture,
        provenance={
            "seed": seed,
            "episode_index": episode_index,
            "observation_start": observation_start,
            "frontend_encode_mode": "lingbot_streaming_vae",
            "startup_model_obs_frames": 1,
        },
    )


def _resolved_training_contract(config: ExperimentConfig) -> dict[str, Any]:
    sample = config.data.sample_construction
    return {
        "parallel_sequence_contract": str(
            config.policy_variant.sequence_contract.value
        ),
        "current_block_coupling": str(
            config.policy_variant.current_block_coupling.value
        ),
        "condition_source_frame_offset": int(sample.condition_source_frame_offset),
        "target_alignment": str(sample.target_alignment.value),
        "sample_mode": str(sample.mode.value),
        "sample_order_mode": str(sample.sample_order_mode.value),
        "segment_min_frames": int(sample.segment_min_frames),
        "segment_max_frames": int(sample.segment_max_frames),
        "segment_length_stride": int(sample.segment_length_stride),
        "randomize_geometry": bool(sample.randomize_geometry),
        "randomize_segment_length": bool(sample.randomize_segment_length),
        "randomize_segment_start": bool(sample.randomize_segment_start),
        "window_size": int(sample.window_size),
        "training_window_size": int(config.training.window_size),
        "chunk_size": int(config.training.chunk_size),
    }


def _write_fixture_index(
    root: Path,
    *,
    generated: dict[str, Path],
    seed: int,
) -> None:
    payload = {
        "schema_version": 1,
        "seed": seed,
        "fixtures": {
            fixture_id: path.name for fixture_id, path in sorted(generated.items())
        },
    }
    path = root / "fixture_index.json"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _representative_source_index(dataset) -> int:
    """Choose the first virtual sample from the longest available trajectory."""

    windows = getattr(dataset, "windows", None)
    virtual_index = getattr(dataset, "_virtual_index", None)
    if not isinstance(windows, (tuple, list)) or not windows:
        return 0
    longest_window_index = max(
        range(len(windows)),
        key=lambda index: (
            int(windows[index].end_frame) - int(windows[index].start_frame),
            -int(windows[index].episode_index),
        ),
    )
    if isinstance(virtual_index, (tuple, list)):
        for index, entry in enumerate(virtual_index):
            if int(entry[0]) == longest_window_index:
                return index
        raise ValueError(
            f"Longest trajectory window {longest_window_index} has no virtual sample."
        )
    return longest_window_index


def _representative_counterfactual_source_index(dataset) -> int:
    """Prefer a genuinely perturbed CF row over the easy ground-truth branch."""

    fallback_index = 0
    for index in range(min(len(dataset), 64)):
        metadata = getattr(dataset[index], "metadata", None) or {}
        if metadata.get("counterfactual_branch") != "gt":
            fallback_index = index
        if (
            metadata.get("counterfactual_branch_family") == "axis_pulse"
            and metadata.get("counterfactual_branch_strength") == "strong"
        ):
            return index
    return fallback_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze strict DualExpert/GJD characterization inputs from real LIBERO data."
    )
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=FIXTURE_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generated = build_all_characterization_fixtures(
        assets=load_characterization_assets(args.assets),
        output_root=args.output_root,
        seed=args.seed,
    )
    for fixture_id, path in sorted(generated.items()):
        print(f"{fixture_id}: {path}")


if __name__ == "__main__":
    main()
