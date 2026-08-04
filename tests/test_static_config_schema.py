from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_wam.configs.static_schema import validate_config_file, validate_config_files

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_maintained_configs_use_explicit_policy_and_decoder_sections() -> None:
    maintained_configs = tuple(
        path
        for directory in ("experiments", "examples")
        for path in sorted((REPO_ROOT / "configs" / directory).glob("*.yaml"))
    )

    legacy_configs = tuple(
        path.relative_to(REPO_ROOT).as_posix()
        for path in maintained_configs
        if "action_head" in yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    assert legacy_configs == ()


@pytest.mark.unit
def test_static_validator_accepts_public_tiny_configs() -> None:
    reports = validate_config_files(
        (
            REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml",
            REPO_ROOT / "configs/evals/public_tiny_synthetic_contract.yaml",
        )
    )

    assert all(report.ok for report in reports)


@pytest.mark.unit
def test_static_validator_catches_enum_typos(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
name: bad
data:
  dataset_name: bad
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 4
    action_horizon: 2
    state_dim: 3
    state_horizon: 1
backbone:
  implementation: not_a_backbone
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 4
  action_horizon: 2
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid BackboneImplementation" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_non_mapping_adapter_options(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_adapter_options.yaml"
    config_path.write_text(
        """
name: bad_adapter_options
data:
  dataset_name: custom
  dataset_type: custom_dataset
  adapter_options:
    - not
    - a
    - mapping
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.adapter_options" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_deprecated_equal_bucket_latent_layout(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_latent_layout.yaml"
    config_path.write_text(
        """
name: bad_latent_layout
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  latent_temporal_layout: equal_bucket_legacy
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
action_decoder:
  name: lingbot_parallel
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("equal_bucket_legacy" in issue.message and "deprecated" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_catches_parallel_stream_proprio_context_typo(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_parallel_proprio.yaml"
    config_path.write_text(
        """
name: bad_parallel_proprio
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
  proprio_context_mode: text_context_typo
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid ProprioContextMode" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_warns_on_deprecated_text_token_proprio(tmp_path: Path) -> None:
    config_path = tmp_path / "deprecated_text_token_proprio.yaml"
    config_path.write_text(
        """
name: deprecated_text_token_proprio
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
  proprio_context_mode: text_context_token  # deprecated
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok
    assert any("Deprecated text-space proprio token path" in issue.message for issue in report.warnings)


@pytest.mark.unit
def test_static_validator_accepts_parallel_stream_context_and_history_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "parallel_context_flags.yaml"
    config_path.write_text(
        """
name: parallel_context_flags
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  sample_construction:
    condition_source_frame_offset: -1
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
  current_block_coupling: decoupled_same_step
  context_condition_latent_source: single_frame_condition_latent
  history_stream_visibility: video_only
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok


@pytest.mark.unit
def test_static_validator_rejects_single_frame_context_without_previous_frame_offset(tmp_path: Path) -> None:
    config_path = tmp_path / "parallel_context_flags_leaky_offset.yaml"
    config_path.write_text(
        """
name: parallel_context_flags_leaky_offset
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  sample_construction:
    condition_source_frame_offset: 0
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
  current_block_coupling: decoupled_same_step
  context_condition_latent_source: single_frame_condition_latent
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(
        "single_frame_condition_latent" in issue.message
        and "offset 0 can expose the first target raw frame" in issue.message
        for issue in report.errors
    )


@pytest.mark.unit
def test_static_validator_catches_parallel_stream_history_visibility_typo(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_parallel_history_visibility.yaml"
    config_path.write_text(
        """
name: bad_parallel_history_visibility
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
  history_stream_visibility: typo
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid HistoryStreamVisibility" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_accepts_dual_expert_context_and_history_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "dual_expert_context_flags.yaml"
    config_path.write_text(
        """
name: dual_expert_context_flags
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  sample_construction:
    condition_source_frame_offset: -1
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  runtime_mode: non_joint_two_stream
  current_block_coupling: decoupled_same_step
  proprio_context_mode: per_chunk_additive
  context_condition_latent_source: single_frame_condition_latent
  history_stream_visibility: video_only
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok


@pytest.mark.unit
def test_static_validator_rejects_dual_expert_single_frame_context_without_previous_frame_offset(tmp_path: Path) -> None:
    config_path = tmp_path / "dual_expert_context_flags_leaky_offset.yaml"
    config_path.write_text(
        """
name: dual_expert_context_flags_leaky_offset
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  sample_construction:
    condition_source_frame_offset: 0
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  runtime_mode: non_joint_two_stream
  current_block_coupling: decoupled_same_step
  proprio_context_mode: per_chunk_additive
  context_condition_latent_source: single_frame_condition_latent
  history_stream_visibility: video_only
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(
        "single_frame_condition_latent" in issue.message
        and "offset 0 can expose the first target raw frame" in issue.message
        for issue in report.errors
    )


@pytest.mark.unit
def test_static_validator_catches_dual_expert_proprio_context_typo(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_dual_expert_proprio.yaml"
    config_path.write_text(
        """
name: bad_dual_expert_proprio
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  runtime_mode: video_prefill_action_denoise
  proprio_context_mode: text_context_typo
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid ProprioContextMode" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_accepts_hierarchical_fixed_segment_sampler(tmp_path: Path) -> None:
    config_path = tmp_path / "hierarchical_fixed_segment.yaml"
    config_path.write_text(
        """
name: hierarchical_fixed_segment
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 128
    sample_order_mode: epoch_order
    start_padding_frames: 3
    context_prefix_policy: rollout_history
    tail_padding_policy: zero_order_hold
    padded_target_policy: mask_loss
    task_start_power: 0.5
    demo_count_power: 0.0
    trajectory_start_power: 1.0
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok


@pytest.mark.unit
def test_static_validator_rejects_replacement_order_on_hierarchical_sampler(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_hierarchical_replacement_order.yaml"
    config_path.write_text(
        """
name: bad_hierarchical_replacement_order
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 128
    sample_order_mode: replacement
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.sample_construction.sample_order_mode" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_accepts_replacement_order_with_mixed_dynamics(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics_replacement_order.yaml"
    config_path.write_text(
        """
name: mixed_dynamics_replacement_order
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    sample_order_mode: replacement
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok


@pytest.mark.unit
def test_static_validator_rejects_mixed_dynamics_views_batch_adapter(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics_views_adapter.yaml"
    config_path.write_text(
        """
name: mixed_dynamics_views_adapter
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: views
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "trainer.batch_adapter" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_non_uniform_sample_weight_with_mixed_dynamics(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics_weighted_source.yaml"
    config_path.write_text(
        """
name: mixed_dynamics_weighted_source
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: uniform_segment
    sample_weight_mode: valid_action_steps
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.sample_construction.sample_weight_mode" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_accepts_rollout_parity_fixed_segment_sampler(tmp_path: Path) -> None:
    config_path = tmp_path / "hierarchical_fixed_segment_rollout_parity.yaml"
    config_path.write_text(
        """
name: hierarchical_fixed_segment_rollout_parity
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 128
    chunk_size: 4
    window_size: 30
    randomize_geometry: false
    start_padding_frames: 0
    target_alignment: next_after_context
    rollout_context_policy: one_frame
    tail_padding_policy: zero_order_hold
    padded_target_policy: mask_loss
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok


@pytest.mark.unit
def test_static_validator_rejects_rollout_parity_legacy_head_padding(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_hierarchical_fixed_segment_rollout_parity.yaml"
    config_path.write_text(
        """
name: bad_hierarchical_fixed_segment_rollout_parity
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 128
    chunk_size: 4
    randomize_geometry: true
    start_padding_frames: 3
    target_alignment: next_after_context
    context_prefix_policy: none
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    error_paths = {issue.path for issue in report.errors}
    assert "data.sample_construction.randomize_geometry" in error_paths
    assert "data.sample_construction.start_padding_frames" in error_paths
    assert "data.sample_construction.context_prefix_policy" in error_paths


@pytest.mark.unit
def test_static_validator_rejects_rollout_parity_missing_randomize_geometry(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_hierarchical_fixed_segment_missing_randomize_geometry.yaml"
    config_path.write_text(
        """
name: bad_hierarchical_fixed_segment_missing_randomize_geometry
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 128
    chunk_size: 4
    target_alignment: next_after_context
    rollout_context_policy: one_frame
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.sample_construction.randomize_geometry" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_negative_context_prefix(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_hierarchical_context_prefix.yaml"
    config_path.write_text(
        """
name: bad_hierarchical_context_prefix
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 8
    context_prefix_policy: fixed
    context_prefix_frames: -1
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.sample_construction.context_prefix_frames" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_sample_state_anchor_typo(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_state_anchor.yaml"
    config_path.write_text(
        """
name: bad_state_anchor
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: uniform_segment
    state_anchor_mode: definitely_not_a_mode
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid SampleStateAnchorMode" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_checks_auxiliary_validation_tasks(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_aux_validation.yaml"
    config_path.write_text(
        """
name: bad_aux_validation
data:
  dataset_name: robotwin
  dataset_type: synthetic_multiview
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
validation:
  auxiliary_tasks:
    - name: fdm_val
      mode_override: not_a_mode
      dataset_split: val
      max_batches: -1
      report_prefix: val_probe
    - name: fdm_val
      mode_override: action_conditioned_video
      dataset_split: made_up
      source: bad_source
      report_prefix: val_probe
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid GeneralistDenoisingMode" in issue.message for issue in report.errors)
    assert any("Invalid DataSplit" in issue.message for issue in report.errors)
    assert any("Invalid AuxiliaryValidationSource" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("max_batches") for issue in report.errors)
    assert any("Duplicate auxiliary validation task name" in issue.message for issue in report.errors)
    assert any("Duplicate auxiliary validation report prefix" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_legacy_full_segment_flag_on_hierarchical_sampler(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_hierarchical_fixed_segment.yaml"
    config_path.write_text(
        """
name: bad_hierarchical_fixed_segment
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: hierarchical_fixed_segment
    segment_frames: 128
    require_full_segment: false
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: post_latent
  attach_site: post_visual_core
action_decoder:
  name: mlp_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.sample_construction.require_full_segment" for issue in report.errors)

@pytest.mark.unit
def test_static_validator_rejects_legacy_prefix_fastwam_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_legacy_prefix_fastwam.yaml"
    config_path.write_text(
        """
name: bad_legacy_prefix_fastwam
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: fastwam_first_frame
  sequence_contract: legacy_prefix_single_frame_perchunk_proprio
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("legacy_prefix_single_frame_perchunk_proprio" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_contract_owned_history_visibility(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_contract_history_visibility.yaml"
    config_path.write_text(
        """
name: bad_contract_history_visibility
data:
  dataset_name: libero
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 7
    action_horizon: 8
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  runtime_mode: lingbot_exact
  current_block_coupling: decoupled_same_step
  sequence_contract: rollout_parity_single_frame_perchunk_proprio
  history_stream_visibility: full
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 8
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("history_stream_visibility" in issue.path for issue in report.errors)


@pytest.mark.unit
def test_static_validator_warns_for_legacy_action_head(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_action_head.yaml"
    config_path.write_text(
        """
name: legacy_action_head
data:
  dataset_name: synthetic
  dataset_type: synthetic_multiview
  action_schema:
    action_dim: 4
    action_horizon: 2
    state_dim: 3
    state_horizon: 1
backbone:
  implementation: dummy
  hidden_size: 32
action_head:
  name: contract_only
  hidden_size: 32
  action_dim: 4
  action_horizon: 2
  state_dim: 3
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok
    assert any(issue.path == "action_head" for issue in report.warnings)


@pytest.mark.unit
def test_static_validator_checks_joint_denoise_mode_probabilities(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_joint_probs.yaml"
    config_path.write_text(
        """
name: bad_joint_probs
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  generalist_denoising_mode_probs:
    joint: 0.0
    typo_mode: 1.0
    action_conditioned_video: -0.2
    video_conditioned_action: .nan
action_decoder:
  name: parallel_stream_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("Invalid GeneralistDenoisingMode" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("action_conditioned_video") for issue in report.errors)
    assert any(issue.path.endswith("video_conditioned_action") and "finite" in issue.message for issue in report.errors)


@pytest.mark.unit
def test_static_validator_checks_parallel_stream_current_block_coupling(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_m1_coupling.yaml"
    config_path.write_text(
        """
name: bad_m1_coupling
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  current_block_coupling: typo_joint
action_decoder:
  name: parallel_stream_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "policy_variant.current_block_coupling" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_allows_dual_expert_shared_video_schedule(tmp_path: Path) -> None:
    config_path = tmp_path / "dual_expert_shared_video_schedule.yaml"
    config_path.write_text(
        """
name: dual_expert_shared_video_schedule
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  attach_site: post_visual_core
  runtime_mode: non_joint_two_stream
  current_block_coupling: joint
  joint_timestep_coupling: shared_video_schedule
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert report.ok


@pytest.mark.unit
def test_static_validator_checks_dual_expert_generalist_mode_probabilities(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_dual_expert_probs.yaml"
    config_path.write_text(
        """
name: bad_dual_expert_probs
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  attach_site: post_visual_core
  runtime_mode: non_joint_two_stream
  current_block_coupling: video_then_action
  generalist_denoising_mode_probs:
    joint: 0.0
    typo_mode: 1.0
    action_conditioned_video: -0.2
    video_conditioned_action: .nan
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any("current_block_coupling: joint" in issue.message for issue in report.errors)
    assert any("Invalid GeneralistDenoisingMode" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("action_conditioned_video") for issue in report.errors)
    assert any(issue.path.endswith("video_conditioned_action") and "finite" in issue.message for issue in report.errors)
    assert any(issue.path == "data.train_batch_size" for issue in report.errors)
    assert any(issue.path == "data.val_batch_size" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_checks_dual_expert_generalist_batch_size(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_dual_expert_gjd_batch_size.yaml"
    config_path.write_text(
        """
name: bad_dual_expert_gjd_batch_size
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 2
  val_batch_size: 1
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  attach_site: post_visual_core
  runtime_mode: non_joint_two_stream
  current_block_coupling: joint
  generalist_denoising_mode_probs:
    joint: 1.0
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "data.train_batch_size" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_checks_dual_expert_mode_token_requires_gjd(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_dual_expert_mode_token.yaml"
    config_path.write_text(
        """
name: bad_dual_expert_mode_token
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 1
  val_batch_size: 1
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  attach_site: post_visual_core
  runtime_mode: non_joint_two_stream
  current_block_coupling: joint
  generalist_mode_text_token: true
action_decoder:
  name: dual_expert_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "policy_variant.generalist_mode_text_token" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_rejects_boolean_joint_denoise_probability(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_joint_bool_prob.yaml"
    config_path.write_text(
        """
name: bad_joint_bool_prob
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  generalist_denoising_mode_probs:
    joint: true
    action_conditioned_video: 0.0
    video_conditioned_action: 0.0
action_decoder:
  name: parallel_stream_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path.endswith("joint") and "numeric" in issue.message for issue in report.errors)


@pytest.mark.parametrize("raw_probability", [".nan", ".inf"])
def test_static_validator_rejects_non_finite_joint_denoise_probability(
    tmp_path: Path,
    raw_probability: str,
) -> None:
    config_path = tmp_path / "bad_joint_non_finite_prob.yaml"
    config_path.write_text(
        f"""
name: bad_joint_non_finite_prob
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  generalist_denoising_mode_probs:
    joint: 0.6
    action_conditioned_video: {raw_probability}
    video_conditioned_action: 0.2
action_decoder:
  name: parallel_stream_decoder
  action_dim: 30
  action_horizon: 16
trainer:
  accelerator: gpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(
        issue.path.endswith("action_conditioned_video") and "finite" in issue.message
        for issue in report.errors
    )


@pytest.mark.unit
def test_static_validator_checks_extension_envelopes_without_importing_plugins(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_extensions.yaml"
    config_path.write_text(
        """
name: bad_extensions
data:
  dataset_name: custom
  dataset_type: custom
  action_schema:
    action_dim: 7
    action_horizon: 4
    state_dim: 8
    state_horizon: 1
backbone:
  implementation: shared_transformer
policy_variant:
  name: extension
  extension_type: " "
  attach_site: post_visual_core
  options: []
action_decoder:
  name: extension
  extension_type: " decoder.with.spaces "
  action_dim: 7
  action_horizon: 4
  options:
    1: invalid-key
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    report = validate_config_file(config_path, repo_root=tmp_path)

    assert not report.ok
    assert any(issue.path == "policy_variant.extension_type" for issue in report.errors)
    assert any(issue.path == "policy_variant.options" for issue in report.errors)
    assert any(issue.path == "action_decoder.extension_type" for issue in report.errors)
    assert any(issue.path == "action_decoder.options" for issue in report.errors)
