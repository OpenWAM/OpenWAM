from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from open_wam.configs.enums import PolicyVariantName

ALLOW_DEPRECATED_LIBERO_CONFIG_ENV = "OPEN_WAM_ALLOW_DEPRECATED_LIBERO_CONFIG"

_REMOVED_LIBERO_POLICY_CONFIG_REASONS = {
    "mot_libero_latent_local": "retired dual-expert local config; use a maintained dual_expert_libero_* program config",
    "mot_libero_latent_local_idm": "retired dual-expert IDM config; use the maintained inverse-dynamics config",
    "mot_libero_latent_local_joint": "retired dual-expert joint config; use dual_expert_libero_joint",
    "mot_libero_latent_local_joint_full_segment": "legacy dual-expert full-segment config",
    "mot_libero_latent_local_full_segment": "legacy dual-expert full-segment config",
    "mot_libero_latent_local_full_segment_non_joint_aligned": "legacy dual-expert aligned full-segment config",
    "mot_libero_latent_local_full_segment_with_latent": "legacy dual-expert full-segment latent config",
    "parallel_stream_libero_lingbot_exact_local": "legacy local parallel-stream exact config",
    "parallel_stream_libero_current_frame_action_chunk": (
        "retired current-frame action-chunk experiment"
    ),
    "parallel_stream_libero_fastwam_first_frame": (
        "retired first-frame FastWAM experiment"
    ),
    "parallel_stream_libero_joint_denoise_heng_compatible_contextual_fixed_geometry": (
        "legacy contextual-subwindow parallel-stream joint config"
    ),
    "parallel_stream_libero_joint_denoise_heng_compatible_contextual_subwindow": (
        "legacy contextual-subwindow parallel-stream joint config"
    ),
    "parallel_stream_libero_joint_denoise_heng_compatible_random_subwindow": (
        "legacy random-subwindow parallel-stream joint config"
    ),
}

_REMOVED_LIBERO_SCRIPT_REPLACEMENTS = {
    "run_libero_exact_realtime_sandbox.py": "scripts/run_libero_realtime_sandbox.py",
    "run_libero_exact_visualization.py": "scripts/run_libero_realtime_sandbox.py",
    "run_mot_non_joint_aligned_libero_A.sh": (
        "scripts/run_dual_expert_posttrain_libero.sh with a maintained CONFIG_NAME"
    ),
    "run_mot_non_joint_action_only_libero_B.sh": (
        "scripts/run_dual_expert_posttrain_libero.sh with a maintained CONFIG_NAME"
    ),
    "run_mot_full_segment_nonjoint_libero.sh": "scripts/run_dual_expert_posttrain_libero.sh",
}


def normalize_config_stem(config_path: str | Path | None) -> str:
    if config_path is None:
        return ""
    name = Path(str(config_path)).name
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def removed_libero_policy_config_reason(config_path: str | Path | None) -> str | None:
    return _REMOVED_LIBERO_POLICY_CONFIG_REASONS.get(normalize_config_stem(config_path))


def normalize_libero_script_name(script_path: str | Path | None) -> str:
    if script_path is None:
        return ""
    return Path(str(script_path)).name


def removed_libero_script_replacement(script_path: str | Path | None) -> str | None:
    return _REMOVED_LIBERO_SCRIPT_REPLACEMENTS.get(normalize_libero_script_name(script_path))


def require_current_libero_script(
    script_path: str | Path | None,
    *,
    source: str | None = None,
) -> None:
    script_label = str(source or script_path or "<unknown>")
    removed_replacement = removed_libero_script_replacement(script_path)
    if removed_replacement is not None:
        raise ValueError(
            f"{script_label} was removed from the maintained OpenWAM runtime. "
            f"Use {removed_replacement}. Git history retains the historical implementation."
        )


def collect_current_libero_policy_paradigm_issues(
    config: Any,
    *,
    config_path: str | Path | None = None,
) -> list[str]:
    """Return explicit compatibility issues for a retired LIBERO config.

    Sampling geometry, replay filtering, sequence semantics, and proprio are
    experiment choices. Their structural compatibility is validated by their
    owning typed configs and runtime components; this compatibility boundary
    must not promote one benchmark recipe into a library invariant.
    """

    policy_variant = getattr(config, "policy_variant", None)
    policy_name = _enum_value(getattr(policy_variant, "name", None))
    if policy_name not in {PolicyVariantName.PARALLEL_STREAM.value, PolicyVariantName.DUAL_EXPERT.value}:
        return []

    config_name = _enum_value(getattr(config, "name", ""))
    data = getattr(config, "data", None)
    dataset_name = _enum_value(getattr(data, "dataset_name", ""))
    if "libero" not in f"{config_name} {dataset_name} {config_path or ''}".lower():
        return []

    config_identity = config_name or config_path
    removed_reason = removed_libero_policy_config_reason(config_identity)
    return [] if removed_reason is None else [removed_reason]


def require_current_libero_policy_paradigm(
    config: Any,
    *,
    config_path: str | Path | None = None,
    source: str,
    allow_deprecated: bool = False,
) -> None:
    issues = collect_current_libero_policy_paradigm_issues(
        config,
        config_path=config_path,
    )
    if not issues or allow_deprecated or _env_allows_deprecated_libero_config():
        return

    issue_lines = "\n".join(f"  - {issue}" for issue in issues)
    config_label = str(config_path) if config_path is not None else str(getattr(config, "name", "<unknown>"))
    raise ValueError(
        f"{source} refuses retired LIBERO policy config {config_label!r}.\n"
        f"Issues:\n{issue_lines}\n"
        f"Use a maintained config, or set "
        f"{ALLOW_DEPRECATED_LIBERO_CONFIG_ENV}=1 / pass --allow-deprecated-libero-config "
        "only for historical debugging."
    )


def _env_allows_deprecated_libero_config() -> bool:
    return os.environ.get(ALLOW_DEPRECATED_LIBERO_CONFIG_ENV, "").strip().lower() in {"1", "true", "yes"}


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
