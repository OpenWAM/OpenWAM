from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from open_wam.configs.enums import (
    PolicyVariantName,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    WindowSamplingMode,
)


ALLOW_DEPRECATED_LIBERO_CONFIG_ENV = "OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG"

_DEPRECATED_LIBERO_POLICY_CONFIG_REASONS = {
    "mot_libero_latent_local": "legacy M5 local config without strict one-frame fixed-128 rollout parity",
    "mot_libero_latent_local_idm": "legacy M5 IDM config without strict one-frame fixed-128 rollout parity",
    "mot_libero_latent_local_joint": "legacy M5 joint config without strict one-frame fixed-128 rollout parity",
    "mot_libero_latent_local_joint_full_segment": "legacy M5 full-segment config",
    "mot_libero_latent_local_full_segment": "legacy M5 full-segment config",
    "mot_libero_latent_local_full_segment_non_joint_aligned": "legacy M5 aligned full-segment config",
    "mot_libero_latent_local_full_segment_with_latent": "legacy M5 full-segment latent config",
    "parallel_stream_libero_lingbot_exact_local": "legacy local M1 exact config",
    "parallel_stream_libero_lingbot_joint_denoise_heng_compatible_contextual_fixed_geometry": (
        "legacy contextual-subwindow M1 joint config"
    ),
    "parallel_stream_libero_lingbot_joint_denoise_heng_compatible_contextual_subwindow": (
        "legacy contextual-subwindow M1 joint config"
    ),
    "parallel_stream_libero_lingbot_joint_denoise_heng_compatible_random_subwindow": (
        "legacy random-subwindow M1 joint config"
    ),
}

_DEPRECATED_LIBERO_SCRIPT_REPLACEMENTS = {
    "run_libero_exact_realtime_sandbox.py": "scripts/run_libero_realtime_sandbox.py",
    "run_libero_exact_visualization.py": "scripts/run_libero_realtime_sandbox.py or scripts/run_libero_sampled_eval.py",
    "run_libero_mot_visualization.py": "scripts/run_libero_realtime_sandbox.py or scripts/run_libero_sampled_eval.py",
    "run_libero_realtime_ablation.py": "scripts/run_libero_sampled_eval.py",
    "run_mot_non_joint_aligned_libero_A.sh": (
        "scripts/run_mot_nonjoint_posttrain_libero.sh with a current *_heng_compatible CONFIG_NAME"
    ),
    "run_mot_non_joint_action_only_libero_B.sh": (
        "scripts/run_mot_nonjoint_posttrain_libero.sh with a current *_heng_compatible CONFIG_NAME"
    ),
    "run_mot_full_segment_nonjoint_libero.sh": "scripts/run_mot_nonjoint_posttrain_libero.sh",
}


def normalize_config_stem(config_path: str | Path | None) -> str:
    if config_path is None:
        return ""
    name = Path(str(config_path)).name
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def deprecated_libero_policy_config_reason(config_path: str | Path | None) -> str | None:
    return _DEPRECATED_LIBERO_POLICY_CONFIG_REASONS.get(normalize_config_stem(config_path))


def normalize_libero_script_name(script_path: str | Path | None) -> str:
    if script_path is None:
        return ""
    return Path(str(script_path)).name


def deprecated_libero_script_replacement(script_path: str | Path | None) -> str | None:
    return _DEPRECATED_LIBERO_SCRIPT_REPLACEMENTS.get(normalize_libero_script_name(script_path))


def require_current_libero_script(
    script_path: str | Path | None,
    *,
    allow_deprecated: bool = False,
    source: str | None = None,
) -> None:
    replacement = deprecated_libero_script_replacement(script_path)
    if replacement is None or allow_deprecated or _env_allows_deprecated_libero_config():
        return

    script_label = str(source or script_path or "<unknown>")
    raise ValueError(
        f"{script_label} is deprecated for current LIBERO M1/M5 launch paths. "
        f"Use {replacement}. Set {ALLOW_DEPRECATED_LIBERO_CONFIG_ENV}=1 or pass "
        "--allow-deprecated-libero-config only for historical debugging."
    )


def collect_current_libero_policy_paradigm_issues(
    config: Any,
    *,
    config_path: str | Path | None = None,
    require_proprio: bool = True,
) -> list[str]:
    """Return issues that make a LIBERO M1/M5 config legacy for new launches."""

    policy_variant = getattr(config, "policy_variant", None)
    policy_name = _enum_value(getattr(policy_variant, "name", None))
    if policy_name not in {PolicyVariantName.PARALLEL_STREAM.value, PolicyVariantName.MOT.value}:
        return []

    config_name = _enum_value(getattr(config, "name", ""))
    data = getattr(config, "data", None)
    dataset_name = _enum_value(getattr(data, "dataset_name", ""))
    if "libero" not in f"{config_name} {dataset_name} {config_path or ''}".lower():
        return []

    deprecated_reason = deprecated_libero_policy_config_reason(config_path)

    issues: list[str] = []
    sample = getattr(data, "sample_construction", None)
    expectations = (
        ("data.sample_construction.mode", getattr(sample, "mode", None), WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT.value),
        ("data.sample_construction.segment_frames", getattr(sample, "segment_frames", None), 128),
        ("data.sample_construction.chunk_size", getattr(sample, "chunk_size", None), 4),
        ("data.sample_construction.window_size", getattr(sample, "window_size", None), 30),
        ("data.sample_construction.randomize_geometry", getattr(sample, "randomize_geometry", None), False),
        ("data.sample_construction.start_padding_frames", getattr(sample, "start_padding_frames", None), 0),
        (
            "data.sample_construction.target_alignment",
            getattr(sample, "target_alignment", None),
            SampleTargetAlignment.NEXT_AFTER_CONTEXT.value,
        ),
        (
            "data.sample_construction.rollout_context_policy",
            getattr(sample, "rollout_context_policy", None),
            RolloutContextPolicy.ONE_FRAME.value,
        ),
    )
    for field_name, actual, expected in expectations:
        if _normalized_value(actual) != _normalized_value(expected):
            issues.append(f"{field_name}={_display_value(actual)!r}, expected {_display_value(expected)!r}")

    if require_proprio:
        proprio_mode = getattr(policy_variant, "proprio_context_mode", ProprioContextMode.NONE)
        if _normalized_value(proprio_mode) != ProprioContextMode.TEXT_CONTEXT_TOKEN.value:
            issues.append(
                "policy_variant.proprio_context_mode="
                f"{_display_value(proprio_mode)!r}, expected {ProprioContextMode.TEXT_CONTEXT_TOKEN.value!r}"
            )

    if deprecated_reason is not None and issues:
        issues.insert(0, deprecated_reason)

    return issues


def require_current_libero_policy_paradigm(
    config: Any,
    *,
    config_path: str | Path | None = None,
    source: str,
    allow_deprecated: bool = False,
    require_proprio: bool = True,
) -> None:
    issues = collect_current_libero_policy_paradigm_issues(
        config,
        config_path=config_path,
        require_proprio=require_proprio,
    )
    if not issues or allow_deprecated or _env_allows_deprecated_libero_config():
        return

    issue_lines = "\n".join(f"  - {issue}" for issue in issues)
    config_label = str(config_path) if config_path is not None else str(getattr(config, "name", "<unknown>"))
    raise ValueError(
        f"{source} refuses deprecated LIBERO M1/M5 config {config_label!r}.\n"
        "The current training/eval paradigm requires strict fixed-128 samples, one-frame rollout "
        "conditioning, fixed 4-frame chunks, no head padding, and proprio text-context tokens.\n"
        f"Issues:\n{issue_lines}\n"
        f"Use a current *_heng_compatible config with proprio enabled, or set "
        f"{ALLOW_DEPRECATED_LIBERO_CONFIG_ENV}=1 / pass --allow-deprecated-libero-config "
        "only for historical debugging."
    )


def _env_allows_deprecated_libero_config() -> bool:
    return os.environ.get(ALLOW_DEPRECATED_LIBERO_CONFIG_ENV, "").strip().lower() in {"1", "true", "yes"}


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalized_value(value: Any) -> Any:
    value = _enum_value(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return str(value) if value is not None else None


def _display_value(value: Any) -> Any:
    return _enum_value(value)
