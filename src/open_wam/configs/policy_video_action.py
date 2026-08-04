"""Architecture-independent video/action policy configuration semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ProprioContextMode,
    VideoActionProgram,
    VideoActionSequenceContract,
    coerce_fields,
)
from .policy_compatibility import resolve_legacy_policy_field
from .policy_contracts import PolicyVariantConfig


def resolve_video_action_program_semantics(
    *,
    program: VideoActionProgram | str | None,
    current_block_coupling: CurrentBlockCoupling | str | None,
) -> tuple[VideoActionProgram | None, CurrentBlockCoupling | None]:
    """Resolve a public program into its low-level same-chunk coupling."""

    resolved_program = None if program is None else VideoActionProgram(program)
    resolved_coupling = (
        None
        if current_block_coupling is None
        else CurrentBlockCoupling(current_block_coupling)
    )
    if resolved_program is None and resolved_coupling is not None:
        resolved_program = VideoActionProgram(resolved_coupling.value)
    if resolved_program is not None:
        expected_coupling = (
            CurrentBlockCoupling.JOINT
            if resolved_program == VideoActionProgram.GENERALIST_JOINT_DENOISING
            else CurrentBlockCoupling(resolved_program.value)
        )
        if resolved_coupling is not None and resolved_coupling != expected_coupling:
            raise ValueError(
                "`policy_variant.program` conflicts with "
                "`policy_variant.current_block_coupling`: "
                f"program={resolved_program.value!r} requires "
                f"current_block_coupling={expected_coupling.value!r}, got "
                f"{resolved_coupling.value!r}."
            )
        resolved_coupling = expected_coupling
    return resolved_program, resolved_coupling


@dataclass(frozen=True)
class VideoActionPolicyConfig(PolicyVariantConfig):
    """Shared semantic envelope for video/action policy architectures.

    Subclasses own parameter topology and backend-specific controls. This base
    owns choices whose meaning must remain identical across architectures.
    """

    noisy_video_condition_prob: float = 0.5
    program: VideoActionProgram | None = None
    current_block_coupling: CurrentBlockCoupling | None = None
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA
    # Deprecated boolean alias retained for old checkpoint-era configs.
    couple_action_to_video_timesteps: bool | None = field(default=None, repr=False, compare=False)
    generalist_training_paradigm: GeneralistTrainingParadigm = GeneralistTrainingParadigm.DEMO_ONLY
    generalist_denoising_mode_probs: dict[GeneralistDenoisingMode, float] | None = None
    generalist_mode_text_token: bool = False
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    history_stream_visibility: HistoryStreamVisibility = HistoryStreamVisibility.FULL
    context_condition_latent_source: ContextConditionLatentSource = ContextConditionLatentSource.VIDEO_LATENTS
    use_condition_latents: bool = True
    require_condition_latents: bool = False
    sequence_contract: VideoActionSequenceContract = VideoActionSequenceContract.DEFAULT
    # Constructor/load alias. Maintained configs and implementation code use
    # `sequence_contract`; serialization omits this compatibility field.
    parallel_sequence_contract: VideoActionSequenceContract | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def normalize_config_override_values(
        self,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Keep the public program and its derived coupling override-safe.

        Loaded configs retain the resolved low-level coupling because runtime
        consumers need it. Replacing only ``program`` must therefore clear the
        old derived value before dataclass validation runs. A legacy
        coupling-only override similarly clears the old public program so it
        can be derived from the replacement coupling.
        """

        normalized = dict(values)
        if "program" in normalized and "current_block_coupling" not in normalized:
            normalized["current_block_coupling"] = None
        elif "current_block_coupling" in normalized and "program" not in normalized:
            normalized["program"] = None
        return normalized

    def __post_init__(self) -> None:
        super().__post_init__()
        resolved_sequence_contract = resolve_legacy_policy_field(
            canonical_value=self.sequence_contract,
            legacy_value=self.parallel_sequence_contract,
            canonical_default=VideoActionSequenceContract.DEFAULT,
            canonical_name="sequence_contract",
            legacy_name="parallel_sequence_contract",
        )
        object.__setattr__(self, "sequence_contract", resolved_sequence_contract)
        coerce_fields(
            self,
            enum_fields={
                "generalist_training_paradigm": GeneralistTrainingParadigm,
                "proprio_context_mode": ProprioContextMode,
                "history_stream_visibility": HistoryStreamVisibility,
                "context_condition_latent_source": ContextConditionLatentSource,
                "sequence_contract": VideoActionSequenceContract,
                "joint_timestep_coupling": JointTimestepCoupling,
            },
            optional_enum_fields={
                "program": VideoActionProgram,
                "current_block_coupling": CurrentBlockCoupling,
            },
        )
        resolved_program, resolved_coupling = resolve_video_action_program_semantics(
            program=self.program,
            current_block_coupling=self.current_block_coupling,
        )
        object.__setattr__(self, "program", resolved_program)
        object.__setattr__(self, "current_block_coupling", resolved_coupling)
        # Constructor aliases are input-only. Clearing them prevents
        # `dataclasses.replace()` from replaying stale aliases over a canonical
        # CLI override.
        object.__setattr__(self, "parallel_sequence_contract", None)
        if self.couple_action_to_video_timesteps is not None:
            object.__setattr__(
                self,
                "joint_timestep_coupling",
                JointTimestepCoupling.MATCH_SIGMA
                if bool(self.couple_action_to_video_timesteps)
                else JointTimestepCoupling.INDEPENDENT,
            )
        object.__setattr__(self, "couple_action_to_video_timesteps", None)
        if not 0.0 <= float(self.noisy_video_condition_prob) <= 1.0:
            raise ValueError(
                "Video/action policies require `0 <= noisy_video_condition_prob <= 1`, "
                f"got noisy_video_condition_prob={self.noisy_video_condition_prob!r}."
            )
        if bool(self.require_condition_latents) and not bool(self.use_condition_latents):
            raise ValueError(
                "`require_condition_latents` cannot be true when `use_condition_latents` is false."
            )


__all__ = [
    "VideoActionPolicyConfig",
    "resolve_video_action_program_semantics",
]
