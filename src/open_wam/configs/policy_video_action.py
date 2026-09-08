"""Architecture-independent video/action policy configuration semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .enums import (
    BackboneImplementation,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DynamicsObjective,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ProprioContextMode,
    TextConditioningMode,
    VideoActionProgram,
    VideoActionSequenceContract,
    coerce_fields,
)
from .policy_contracts import PolicyConditioningRequirements, PolicyVariantConfig
from .sequence_contract_specs import get_video_action_sequence_contract_spec


@dataclass(frozen=True, slots=True)
class _VideoActionProgramSemantics:
    """Complete shared meaning of one public video/action program."""

    current_block_coupling: CurrentBlockCoupling
    supports_dynamics_routing: bool = False
    fixed_conditioning_mode: DynamicsObjective | None = None
    has_joint_noise_clock: bool = True
    emits_video: bool = True
    emits_action: bool = True


_PROGRAM_SEMANTICS = {
    VideoActionProgram.VIDEO_THEN_ACTION: _VideoActionProgramSemantics(
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        has_joint_noise_clock=False,
    ),
    VideoActionProgram.ACTION_THEN_VIDEO: _VideoActionProgramSemantics(
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        has_joint_noise_clock=False,
    ),
    VideoActionProgram.JOINT: _VideoActionProgramSemantics(
        CurrentBlockCoupling.JOINT,
    ),
    VideoActionProgram.DECOUPLED_SAME_STEP: _VideoActionProgramSemantics(
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        has_joint_noise_clock=False,
    ),
    VideoActionProgram.VIDEO_NOISY_TO_ACTION: _VideoActionProgramSemantics(
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
    ),
    VideoActionProgram.ACTION_NOISY_TO_VIDEO: _VideoActionProgramSemantics(
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ),
    VideoActionProgram.GENERALIST_JOINT_DENOISING: _VideoActionProgramSemantics(
        CurrentBlockCoupling.JOINT,
        supports_dynamics_routing=True,
    ),
    VideoActionProgram.FORWARD_DYNAMICS: _VideoActionProgramSemantics(
        CurrentBlockCoupling.JOINT,
        supports_dynamics_routing=True,
        fixed_conditioning_mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        has_joint_noise_clock=False,
        emits_action=False,
    ),
    VideoActionProgram.INVERSE_DYNAMICS: _VideoActionProgramSemantics(
        CurrentBlockCoupling.JOINT,
        supports_dynamics_routing=True,
        fixed_conditioning_mode=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        has_joint_noise_clock=False,
        emits_video=False,
    ),
}
_DEFAULT_JOINT_TIMESTEP_COUPLING = JointTimestepCoupling.INDEPENDENT


def _program_semantics(
    program: VideoActionProgram | str,
) -> _VideoActionProgramSemantics:
    return _PROGRAM_SEMANTICS[VideoActionProgram(program)]


def fixed_conditioning_mode_for_program(
    program: VideoActionProgram | str | None,
) -> DynamicsObjective | None:
    """Return the fixed conditional mode selected by a standalone program."""

    if program is None:
        return None
    return _program_semantics(program).fixed_conditioning_mode


def supports_dynamics_routing(
    program: VideoActionProgram | str | None,
) -> bool:
    """Return whether a program supports source-and-objective routes."""

    if program is None:
        return False
    try:
        return _program_semantics(program).supports_dynamics_routing
    except ValueError:
        return False


def supports_video_conditioned_action(
    program: VideoActionProgram | str | None,
) -> bool:
    """Return whether a program can consume clean future video to emit actions."""

    if program is None:
        return False
    semantics = _program_semantics(program)
    return bool(
        VideoActionProgram(program) is VideoActionProgram.VIDEO_THEN_ACTION
        or (semantics.supports_dynamics_routing and semantics.emits_action)
    )


def resolve_fixed_conditioning_mode(
    policy_config: PolicyVariantConfig,
) -> DynamicsObjective | None:
    """Resolve the conditional mode owned by a standalone policy program."""

    return policy_config.fixed_conditioning_mode


def requires_independent_timestep_clocks(
    program: VideoActionProgram | str | None,
) -> bool:
    """Return whether a program has no meaningful joint clock to couple."""

    if program is None:
        return False
    return not _program_semantics(program).has_joint_noise_clock


def current_block_coupling_for_program(
    program: VideoActionProgram | str,
) -> CurrentBlockCoupling:
    """Return the low-level same-chunk coupling owned by one public program."""

    return _program_semantics(program).current_block_coupling


def output_flags_for_program(
    program: VideoActionProgram | str,
) -> tuple[bool, bool]:
    """Return whether normal inference emits video and action, respectively."""

    semantics = _program_semantics(program)
    return semantics.emits_video, semantics.emits_action


@dataclass(frozen=True)
class VideoActionPolicyConfig(PolicyVariantConfig):
    """Shared semantic envelope for video/action policy architectures.

    Subclasses own parameter topology and backend-specific controls. This base
    owns choices whose meaning must remain identical across architectures.
    """

    noisy_video_condition_prob: float = 0.5
    program: VideoActionProgram | None = None
    joint_timestep_coupling: JointTimestepCoupling = _DEFAULT_JOINT_TIMESTEP_COUPLING
    generalist_mode_text_token: bool = False
    text_conditioning_mode: TextConditioningMode = TextConditioningMode.TASK_PROMPT
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    # Historical action K/V is opt-in for every video/action program.
    history_stream_visibility: HistoryStreamVisibility = (
        HistoryStreamVisibility.VIDEO_ONLY
    )
    context_condition_latent_source: ContextConditionLatentSource = (
        ContextConditionLatentSource.VIDEO_LATENTS
    )
    use_condition_latents: bool = True
    # Requires an external planning prefix when the sample does not carry an
    # observed in-sequence t0, as target-only FDM/IDM samples do.
    require_condition_latents: bool = False
    sequence_contract: VideoActionSequenceContract = VideoActionSequenceContract.DEFAULT

    @property
    def supported_backbone_implementations(
        self,
    ) -> tuple[BackboneImplementation, ...]:
        return (BackboneImplementation.SHARED_TRANSFORMER,)

    @property
    def fixed_conditioning_mode(self) -> DynamicsObjective | None:
        return fixed_conditioning_mode_for_program(self.program)

    @property
    def conditioning_requirements(self) -> PolicyConditioningRequirements:
        return PolicyConditioningRequirements(
            text_conditioning_mode=self.text_conditioning_mode,
            proprio_context_mode=self.proprio_context_mode,
            dynamics_mode_context_enabled=bool(self.generalist_mode_text_token),
        )

    @property
    def current_block_coupling(self) -> CurrentBlockCoupling:
        """Low-level same-chunk coupling derived from the public program."""

        if self.program is None:  # guarded by ``__post_init__``
            raise RuntimeError("Video/action program has not been resolved.")
        return current_block_coupling_for_program(self.program)

    @property
    def requires_frame_aligned_proprio_context(self) -> bool:
        """Whether sequence assembly needs state for every model-visible frame."""

        spec = get_video_action_sequence_contract_spec(self.sequence_contract)
        return bool(spec and spec.requires_frame_aligned_proprio_context)

    def normalize_config_override_values(
        self,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize shared video/action config overrides."""

        normalized = dict(values)
        if "program" in normalized:
            resolved_program = VideoActionProgram(normalized["program"])
            if (
                resolved_program != self.program
                and "joint_timestep_coupling" not in normalized
            ):
                normalized["joint_timestep_coupling"] = _DEFAULT_JOINT_TIMESTEP_COUPLING
        return normalized

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "text_conditioning_mode": TextConditioningMode,
                "proprio_context_mode": ProprioContextMode,
                "history_stream_visibility": HistoryStreamVisibility,
                "context_condition_latent_source": ContextConditionLatentSource,
                "sequence_contract": VideoActionSequenceContract,
                "joint_timestep_coupling": JointTimestepCoupling,
            },
            optional_enum_fields={"program": VideoActionProgram},
        )
        if self.program is None:
            raise ValueError(
                "Video/action policies require an explicit "
                "`policy_variant.program`; direct runtime and coupling controls "
                "are not part of the public contract."
            )
        current_block_coupling_for_program(self.program)
        if (
            requires_independent_timestep_clocks(self.program)
            and self.joint_timestep_coupling != JointTimestepCoupling.INDEPENDENT
        ):
            raise ValueError(
                f"`program = {self.program.value}` does not define a jointly coupled "
                "video/action noise clock and requires "
                "`joint_timestep_coupling = independent`."
            )
        if not 0.0 <= float(self.noisy_video_condition_prob) <= 1.0:
            raise ValueError(
                "Video/action policies require `0 <= noisy_video_condition_prob <= 1`, "
                f"got noisy_video_condition_prob={self.noisy_video_condition_prob!r}."
            )
        if (
            self.context_condition_latent_source
            == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            and (
                not bool(self.use_condition_latents)
                or not bool(self.require_condition_latents)
            )
        ):
            raise ValueError(
                "`context_condition_latent_source = single_frame_condition_latent` "
                "requires both `use_condition_latents = true` and "
                "`require_condition_latents = true`."
            )
        if bool(self.require_condition_latents) and not bool(
            self.use_condition_latents
        ):
            raise ValueError(
                "`require_condition_latents` cannot be true when `use_condition_latents` is false."
            )
        if (
            bool(self.generalist_mode_text_token)
            and self.program != VideoActionProgram.GENERALIST_JOINT_DENOISING
        ):
            raise ValueError(
                "`generalist_mode_text_token = true` requires "
                "`program = generalist_joint_denoising`."
            )


__all__ = [
    "VideoActionPolicyConfig",
    "current_block_coupling_for_program",
]
