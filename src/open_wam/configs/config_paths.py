from __future__ import annotations

import warnings
from pathlib import Path
from types import MappingProxyType

from open_wam.contracts import find_repo_root

REPO_ROOT = find_repo_root(Path(__file__))
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_CONFIG_ROOT = REPO_ROOT / "configs"
_PACKAGED_CONFIG_ROOT = _PACKAGE_ROOT / "resources" / "configs"
_SOURCE_PACKAGE_ROOT = REPO_ROOT / "src" / "open_wam"
_SOURCE_CHECKOUT = (
    _SOURCE_PACKAGE_ROOT.is_dir()
    and _SOURCE_PACKAGE_ROOT.resolve() == _PACKAGE_ROOT
)
CONFIG_ROOT = (
    _SOURCE_CONFIG_ROOT
    if _SOURCE_CHECKOUT and (_SOURCE_CONFIG_ROOT / "experiments").is_dir()
    else _PACKAGED_CONFIG_ROOT
)
TEMPLATE_ROOT = _PACKAGE_ROOT / "templates"
EXPERIMENT_CONFIG_ROOT = CONFIG_ROOT / "experiments"
EVALUATION_CONFIG_ROOT = CONFIG_ROOT / "evals"
EXAMPLE_CONFIG_ROOT = CONFIG_ROOT / "examples"
SIMULATOR_CONFIG_ROOT = CONFIG_ROOT / "simulators"


class DeprecatedConfigNameWarning(FutureWarning):
    """Warning emitted when a retired public config name is resolved."""


EXPERIMENT_CONFIG_ALIASES = MappingProxyType(
    {
        "mot_libero_action_noisy_to_video": "dual_expert_libero_action_noisy_to_video",
        "mot_libero_action_then_video": "dual_expert_libero_action_then_video",
        "mot_libero_decoupled_same_step": "dual_expert_libero_decoupled_same_step",
        "mot_libero_generalist_joint_denoising": (
            "dual_expert_libero_generalist_joint_denoising"
        ),
        "mot_libero_joint": "dual_expert_libero_joint",
        "mot_libero_video_noisy_to_action": "dual_expert_libero_video_noisy_to_action",
        "mot_libero_video_then_action": "dual_expert_libero_video_then_action",
        "mot_robotwin_smoke": "dual_expert_robotwin_smoke",
        "mot_libero_latent_local_action_noisy_to_video_heng_compatible": (
            "dual_expert_libero_action_noisy_to_video"
        ),
        "mot_libero_latent_local_action_then_video_heng_compatible": (
            "dual_expert_libero_action_then_video"
        ),
        "mot_libero_latent_local_decoupled_same_step_heng_compatible": (
            "dual_expert_libero_decoupled_same_step"
        ),
        "mot_libero_latent_local_generalist_joint_denoising_heng_compatible": (
            "dual_expert_libero_generalist_joint_denoising"
        ),
        "mot_libero_latent_local_joint_heng_compatible": "dual_expert_libero_joint",
        "mot_libero_latent_local_video_noisy_to_action_heng_compatible": (
            "dual_expert_libero_video_noisy_to_action"
        ),
        "mot_libero_latent_local_video_then_action_heng_compatible": (
            "dual_expert_libero_video_then_action"
        ),
        "parallel_stream_libero_lingbot_exact_heng_compatible": (
            "parallel_stream_libero_video_then_action"
        ),
        "parallel_stream_libero_lingbot_joint_denoise": (
            "parallel_stream_libero_joint"
        ),
        "parallel_stream_libero_lingbot_joint_denoise_heng_compatible": (
            "parallel_stream_libero_joint"
        ),
        "parallel_stream_libero_lingbot_m1_action_noisy_to_video": (
            "parallel_stream_libero_action_noisy_to_video"
        ),
        "parallel_stream_libero_lingbot_m1_action_noisy_to_video_heng_compatible": (
            "parallel_stream_libero_action_noisy_to_video"
        ),
        "parallel_stream_libero_lingbot_m1_action_then_video": (
            "parallel_stream_libero_action_then_video"
        ),
        "parallel_stream_libero_lingbot_m1_action_then_video_heng_compatible": (
            "parallel_stream_libero_action_then_video"
        ),
        "parallel_stream_libero_lingbot_m1_decoupled_same_step": (
            "parallel_stream_libero_decoupled_same_step"
        ),
        "parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible": (
            "parallel_stream_libero_decoupled_same_step"
        ),
        "parallel_stream_libero_lingbot_m1_generalist_joint_denoising": (
            "parallel_stream_libero_generalist_joint_denoising"
        ),
        "parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible": (
            "parallel_stream_libero_generalist_joint_denoising"
        ),
        "parallel_stream_libero_lingbot_m1_joint": (
            "parallel_stream_libero_joint"
        ),
        "parallel_stream_libero_lingbot_m1_joint_heng_compatible": (
            "parallel_stream_libero_joint"
        ),
        "parallel_stream_libero_lingbot_m1_video_noisy_to_action": (
            "parallel_stream_libero_video_noisy_to_action"
        ),
        "parallel_stream_libero_lingbot_m1_video_noisy_to_action_heng_compatible": (
            "parallel_stream_libero_video_noisy_to_action"
        ),
        "parallel_stream_libero_lingbot_m1_video_then_action": (
            "parallel_stream_libero_video_then_action"
        ),
        "parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible": (
            "parallel_stream_libero_video_then_action"
        ),
    }
)

EVALUATION_CONFIG_ALIASES = MappingProxyType(
    {
        "mot_robotwin_smoke": "dual_expert_robotwin_smoke_eval",
        "dual_expert_robotwin_smoke": "dual_expert_robotwin_smoke_eval",
        "parallel_stream_robotwin_smoke": "parallel_stream_robotwin_smoke_eval",
    }
)

_CONFIG_ALIASES = MappingProxyType(
    {**EVALUATION_CONFIG_ALIASES, **EXPERIMENT_CONFIG_ALIASES}
)


def canonical_config_stem(value: str | Path) -> str:
    """Return the canonical stem for a public experiment or evaluation config."""

    stem = Path(value).stem
    return _CONFIG_ALIASES.get(stem, stem)


def resolve_config_reference(value: str | Path) -> Path:
    """Resolve an explicit path or a wheel-packaged ``configs/...`` reference."""

    candidate = Path(value).expanduser()
    if candidate.exists() or candidate.is_absolute():
        return resolve_config_path_alias(candidate)
    parts = candidate.parts
    if len(parts) >= 2 and parts[0] == "configs":
        packaged_candidate = CONFIG_ROOT.joinpath(*parts[1:])
        return resolve_config_path_alias(packaged_candidate)
    if len(parts) >= 2 and parts[0] == "templates":
        return TEMPLATE_ROOT.joinpath(*parts[1:])
    return resolve_config_path_alias(candidate)


def resolve_experiment_config_reference(value: str | Path) -> Path:
    """Resolve one experiment path, packaged path, or bare config name."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) == 1 and not candidate.exists():
        filename = (
            candidate.name
            if candidate.suffix in {".yaml", ".yml"}
            else f"{candidate.name}.yaml"
        )
        candidate = EXPERIMENT_CONFIG_ROOT / filename
    return resolve_config_reference(candidate)


def resolve_evaluation_config_reference(value: str | Path) -> Path:
    """Resolve one evaluation path, packaged path, or bare config name."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) == 1 and not candidate.exists():
        filename = (
            candidate.name
            if candidate.suffix in {".yaml", ".yml"}
            else f"{candidate.name}.yaml"
        )
        candidate = EVALUATION_CONFIG_ROOT / filename
    return resolve_config_reference(candidate)


def resolve_config_path_alias(path: str | Path, *, warn: bool = True) -> Path:
    """Resolve a missing retired config path to its single canonical YAML owner.

    Existing files always win. This keeps copied historical configs and
    checkpoint-local ``resolved_config.yaml`` artifacts immutable.
    """

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate

    old_stem = candidate.stem
    candidate_parent = candidate.parent.resolve()
    if candidate_parent == EVALUATION_CONFIG_ROOT.resolve():
        alias_map = EVALUATION_CONFIG_ALIASES
        canonical_root = EVALUATION_CONFIG_ROOT
    elif candidate_parent == EXPERIMENT_CONFIG_ROOT.resolve() or old_stem in EXPERIMENT_CONFIG_ALIASES:
        alias_map = EXPERIMENT_CONFIG_ALIASES
        canonical_root = EXPERIMENT_CONFIG_ROOT
    else:
        alias_map = EVALUATION_CONFIG_ALIASES
        canonical_root = EVALUATION_CONFIG_ROOT

    canonical_stem = alias_map.get(old_stem)
    if canonical_stem is None:
        return candidate

    sibling = candidate.with_name(f"{canonical_stem}.yaml")
    repository_candidate = canonical_root / f"{canonical_stem}.yaml"
    resolved = sibling if sibling.exists() else repository_candidate
    if warn:
        warnings.warn(
            f"Config name '{old_stem}' is deprecated; use '{canonical_stem}' instead. "
            f"Resolved to {resolved}.",
            DeprecatedConfigNameWarning,
            stacklevel=2,
        )
    return resolved


__all__ = [
    "CONFIG_ROOT",
    "EVALUATION_CONFIG_ALIASES",
    "EVALUATION_CONFIG_ROOT",
    "EXAMPLE_CONFIG_ROOT",
    "EXPERIMENT_CONFIG_ALIASES",
    "EXPERIMENT_CONFIG_ROOT",
    "SIMULATOR_CONFIG_ROOT",
    "TEMPLATE_ROOT",
    "DeprecatedConfigNameWarning",
    "canonical_config_stem",
    "resolve_config_reference",
    "resolve_config_path_alias",
    "resolve_evaluation_config_reference",
    "resolve_experiment_config_reference",
]
