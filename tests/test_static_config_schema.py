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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
  program: video_then_action
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
  program: video_then_action
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
  program: video_then_action
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
  program: decoupled_same_step
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
def test_static_validator_rejects_retired_history_visibility_alias(
    tmp_path: Path,
) -> None:
    source = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["policy_variant"]["preserve_video_pretrain_history"] = True
    config_path = tmp_path / "retired_history_alias.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        "preserve_video_pretrain_history" in issue.message
        and "retired" in issue.message
        for issue in report.errors
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "decoder", "field_name"),
    [
        ("parallel_stream", "parallel_stream_decoder", "use_condition_latents"),
        ("parallel_stream", "parallel_stream_decoder", "require_condition_latents"),
        ("dual_expert", "dual_expert_decoder", "use_condition_latents"),
        ("dual_expert", "dual_expert_decoder", "require_condition_latents"),
    ],
)
def test_static_validator_rejects_disabled_single_frame_condition_flags(
    tmp_path: Path,
    architecture: str,
    decoder: str,
    field_name: str,
) -> None:
    config_path = tmp_path / f"{architecture}_{field_name}_false.yaml"
    use_condition_latents = "false" if field_name == "use_condition_latents" else "true"
    require_condition_latents = (
        "false" if field_name == "require_condition_latents" else "true"
    )
    config_path.write_text(
        f"""
name: disabled_single_frame_condition_flag
data:
  dataset_name: synthetic
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
  name: {architecture}
  program: decoupled_same_step
  context_condition_latent_source: single_frame_condition_latent
  use_condition_latents: {use_condition_latents}
  require_condition_latents: {require_condition_latents}
action_decoder:
  name: {decoder}
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
        issue.path == f"policy_variant.{field_name}"
        and "requires" in issue.message
        for issue in report.errors
    )


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
  program: decoupled_same_step
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
  program: video_then_action
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
  program: decoupled_same_step
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
  program: decoupled_same_step
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
  program: video_then_action
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
def test_static_validator_accepts_replacement_order_with_dynamics_routing(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed_replacement_order.yaml"
    config_path.write_text(
        """
name: dynamics_routed_replacement_order
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 1
  val_batch_size: 1
  sample_construction:
    sample_order_mode: replacement
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  dynamics_routing:
    routes:
      - {source: real_demo, mode: joint, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
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
def test_static_validator_rejects_epoch_order_with_dynamics_routing(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["sample_construction"]["sample_order_mode"] = "epoch_order"
    config_path = tmp_path / "dynamics_routed_epoch_order.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == "data.sample_construction.sample_order_mode"
        and "replacement" in issue.message
        for issue in report.errors
    )


@pytest.mark.unit
def test_static_validator_accepts_default_replacement_order_with_dynamics_routing(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["sample_construction"].pop("sample_order_mode")
    config_path = tmp_path / "dynamics_routed_default_replacement_order.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert report.ok


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("lenght_multiplier", 2.0),
        ("seed", True),
        ("length_multiplier", True),
    ],
)
def test_static_validator_rejects_invalid_dynamics_routing_scalars(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"][field_name] = bad_value
    config_path = tmp_path / f"bad_dynamics_routing_{field_name}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == f"data.dynamics_routing.{field_name}"
        for issue in report.errors
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["generalist_training_paradigm", "dynamics_routing_requirement"],
)
def test_static_validator_rejects_retired_routing_markers(
    tmp_path: Path,
    field_name: str,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"][field_name] = "mixed_dynamics"
    config_path = tmp_path / "legacy_mixed_dynamics.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

    assert not report.ok
    assert any(
        issue.path == f"policy_variant.{field_name}"
        and "retired" in issue.message
        for issue in report.errors
    )


@pytest.mark.unit
def test_static_validator_rejects_dynamics_routing_views_batch_adapter(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed_views_adapter.yaml"
    config_path.write_text(
        """
name: dynamics_routed_views_adapter
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  dynamics_routing:
    routes:
      - {source: real_demo, mode: joint, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
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
def test_static_validator_rejects_non_uniform_sample_weight_with_dynamics_routing(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed_weighted_source.yaml"
    config_path.write_text(
        """
name: dynamics_routed_weighted_source
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
  dynamics_routing:
    routes:
      - {source: real_demo, mode: joint, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
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
    sample_order_mode: epoch_order
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
    sample_order_mode: epoch_order
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
  name: dual_expert
  program: video_then_action
action_decoder:
  name: dual_expert_decoder
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
    sample_order_mode: epoch_order
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
  name: dual_expert
  program: video_then_action
action_decoder:
  name: dual_expert_decoder
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
    sample_order_mode: epoch_order
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
validation:
  auxiliary_tasks:
    - name: fdm_val
      mode_override: not_a_mode
      dataset_split: val
      max_batches: -1
      report_prefix: val_probe
    - name: fdm_val
      mode_override: action_conditioned_video
      drop_text_conditioning: false
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
    assert any("Invalid DynamicsObjective" in issue.message for issue in report.errors)
    assert any("Invalid DataSplit" in issue.message for issue in report.errors)
    assert any("Invalid AuxiliaryValidationSource" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("max_batches") for issue in report.errors)
    assert any("Duplicate auxiliary validation task name" in issue.message for issue in report.errors)
    assert any("Duplicate auxiliary validation report prefix" in issue.message for issue in report.errors)
    assert any(
        issue.path.endswith("drop_text_conditioning")
        and "always removes task text" in issue.message
        for issue in report.errors
    )


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
    sample_order_mode: epoch_order
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
  name: dual_expert
  program: video_then_action
  attach_site: post_visual_core
action_decoder:
  name: dual_expert_decoder
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
def test_static_validator_rejects_retired_parallel_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "retired_parallel_runtime.yaml"
    config_path.write_text(
        """
name: retired_parallel_runtime
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
  program: video_then_action
  runtime_mode: fastwam_first_frame
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
    assert any(issue.path == "policy_variant.runtime_mode" for issue in report.errors)


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
  program: decoupled_same_step
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
def test_static_validator_rejects_removed_action_head(tmp_path: Path) -> None:
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
  name: legacy
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

    assert not report.ok
    assert any(issue.path == "action_head" for issue in report.errors)


@pytest.mark.unit
def test_static_validator_checks_parallel_stream_generalist_routes(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_generalist_routes.yaml"
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
  dynamics_routing:
    routes:
      - {source: real_demo, mode: typo_mode, weight: 1.0}
      - {source: real_demo, mode: action_conditioned_video, weight: -0.2}
      - {source: counterfactual_dynamics, mode: video_conditioned_action, weight: .nan}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
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
    assert any(issue.path.endswith("routes.0.mode") for issue in report.errors)
    assert any(issue.path.endswith("routes.1.weight") for issue in report.errors)
    assert any(issue.path.endswith("routes.2.weight") and "finite" in issue.message for issue in report.errors)


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
  program: joint
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
  program: joint
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
def test_static_validator_checks_dual_expert_generalist_routes(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_dual_expert_routes.yaml"
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
  dynamics_routing:
    routes:
      - {source: real_demo, mode: typo_mode, weight: 1.0}
      - {source: real_demo, mode: action_conditioned_video, weight: -0.2}
      - {source: counterfactual_dynamics, mode: video_conditioned_action, weight: .nan}
      - {source: real_demo, mode: joint, weight: "1.0", ratio: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: dual_expert
  attach_site: post_visual_core
  program: generalist_joint_denoising
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
    assert any(issue.path.endswith("routes.0.mode") for issue in report.errors)
    assert any(issue.path.endswith("routes.1.weight") for issue in report.errors)
    assert any(issue.path.endswith("routes.2.weight") and "finite" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("routes.3.weight") and "numeric" in issue.message for issue in report.errors)
    assert any(issue.path.endswith("routes.3.ratio") and "Unknown" in issue.message for issue in report.errors)
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
  program: generalist_joint_denoising
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
def test_static_validator_checks_parallel_generalist_batch_size(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["train_batch_size"] = 2
    config_path = tmp_path / "bad_parallel_gjd_batch_size.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    report = validate_config_file(config_path, repo_root=REPO_ROOT)

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
  program: joint
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
def test_static_validator_rejects_boolean_generalist_route_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_route_bool_weight.yaml"
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
  dynamics_routing:
    routes:
      - {source: real_demo, mode: joint, weight: true}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
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
    assert any(issue.path.endswith("routes.0.weight") and "numeric" in issue.message for issue in report.errors)


@pytest.mark.parametrize("raw_probability", [".nan", ".inf"])
def test_static_validator_rejects_non_finite_generalist_route_weight(
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
  dynamics_routing:
    routes:
      - source: real_demo
        mode: action_conditioned_video
        weight: {raw_probability}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
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
        issue.path.endswith("routes.0.weight") and "finite" in issue.message
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
