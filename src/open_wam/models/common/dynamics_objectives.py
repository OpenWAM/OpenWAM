from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch

from open_wam.configs.enums import (
    DynamicsObjective,
    HistoryStreamVisibility,
    StrEnum,
    VideoActionProgram,
)
from open_wam.configs.policy_video_action import (
    fixed_conditioning_mode_for_program,
    supports_dynamics_routing,
)
from open_wam.contracts import (
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    ConditionalDynamicsSequenceLayout,
    SampleConstructionMetadata,
)

from .dynamics_contracts import DynamicsRolloutGeometry, DynamicsRolloutRequest
from .modality_slots import clean_noisy_slot_tensor, zero_loss_mask_like

ModeEnumT = TypeVar("ModeEnumT", bound=StrEnum)

CONDITIONAL_ROLLOUT_CHUNK_SIZE_FRAMES = 1
CONDITIONAL_SINGLE_HISTORY_VIDEO_FRAME_BLOCK_WINDOW = 3


@dataclass(frozen=True)
class DynamicsObjectiveSemantics:
    """Architecture-independent semantics for one video/action objective."""

    objective: DynamicsObjective

    @property
    def drop_text_conditioning(self) -> bool:
        return self.objective.is_conditional

    @property
    def clean_action_noisy_slot(self) -> bool:
        return self.objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO

    @property
    def clean_video_noisy_slot(self) -> bool:
        return self.objective == DynamicsObjective.VIDEO_CONDITIONED_ACTION

    @property
    def action_loss_active(self) -> bool:
        return self.objective != DynamicsObjective.ACTION_CONDITIONED_VIDEO

    @property
    def video_loss_active(self) -> bool:
        return self.objective != DynamicsObjective.VIDEO_CONDITIONED_ACTION

    @property
    def force_clean_video_condition(self) -> bool:
        return self.objective.is_conditional

    @property
    def history_frame_count(self) -> int:
        """Number of clean video frames visible to a conditional objective."""

        return 1 if self.is_conditional else 0

    @property
    def conditional_history_policy(self) -> str | None:
        if self.is_conditional:
            return DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY
        return None

    @property
    def is_joint(self) -> bool:
        return self.objective == DynamicsObjective.JOINT

    @property
    def is_conditional(self) -> bool:
        return self.objective.is_conditional

    def resolve_history_stream_visibility(
        self,
        *,
        fallback: HistoryStreamVisibility | str,
    ) -> HistoryStreamVisibility:
        """Restrict conditional dynamics to clean video history only."""

        if self.is_conditional:
            return HistoryStreamVisibility.VIDEO_ONLY
        return HistoryStreamVisibility(fallback)

    def rollout_chunk_size_frames(self, *, fallback_chunk_size: int) -> int:
        """Resolve recurrent inference geometry without changing train chunks."""

        if self.is_conditional:
            return CONDITIONAL_ROLLOUT_CHUNK_SIZE_FRAMES
        return max(1, int(fallback_chunk_size))

    def attention_window_size(self, *, fallback_window_size: int) -> int:
        if self.is_conditional:
            return CONDITIONAL_SINGLE_HISTORY_VIDEO_FRAME_BLOCK_WINDOW
        return max(1, int(fallback_window_size))


@dataclass(frozen=True, slots=True)
class DynamicsSamplePlan:
    """RNG-free program, route, and sequence decision for one sample."""

    program: VideoActionProgram
    objective: DynamicsObjective | None
    routed_objective: DynamicsObjective | None
    drop_text_conditioning: bool | None
    source: str | None
    sequence: ConditionalDynamicsSequenceLayout | None

    @property
    def uses_in_sequence_condition(self) -> bool:
        return self.sequence is not None


@dataclass(frozen=True, slots=True)
class DynamicsTrainingPlan:
    """Backend-independent objective selected for one training sample."""

    semantics: DynamicsObjectiveSemantics
    routed_objective: DynamicsObjective | None
    source: str | None
    sequence: ConditionalDynamicsSequenceLayout | None = None

    @property
    def objective(self) -> DynamicsObjective:
        return self.semantics.objective


@dataclass(frozen=True, slots=True)
class DynamicsTrainingTensors:
    """Canonical noisy slots, timesteps, and loss masks for one objective."""

    noisy_video: torch.Tensor
    video_targets: torch.Tensor
    video_timesteps: torch.Tensor
    video_loss_mask: torch.Tensor
    noisy_action: torch.Tensor
    action_targets: torch.Tensor
    action_timesteps: torch.Tensor
    action_loss_mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class DynamicsRolloutPlan:
    """Resolved rollout semantics and canonical clean-modality inputs."""

    semantics: DynamicsObjectiveSemantics
    clean_action: torch.Tensor | None = None
    clean_video: torch.Tensor | None = None
    history_action: torch.Tensor | None = None

    @property
    def objective(self) -> DynamicsObjective:
        return self.semantics.objective

    def require_generation_inputs(self) -> DynamicsRolloutPlan:
        """Validate clean inputs required to generate one conditional chunk."""

        if (
            self.objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO
            and self.clean_action is None
        ):
            raise ValueError(
                "Forward-dynamics rollout requires `clean_action` in model-space "
                "[B, T, D] layout."
            )
        if (
            self.objective == DynamicsObjective.VIDEO_CONDITIONED_ACTION
            and self.clean_video is None
        ):
            raise ValueError(
                "Inverse-dynamics rollout requires `clean_video` in latent-space "
                "[B, C, T, H, W] layout."
            )
        return self

    def resolve_geometry(
        self,
        *,
        fallback_frame_chunk_size: int,
        fallback_attention_window_size: int,
        fallback_history_stream_visibility: HistoryStreamVisibility | str,
    ) -> DynamicsRolloutGeometry:
        """Resolve the geometry every policy backend must implement identically."""

        return resolve_dynamics_rollout_geometry(
            self.semantics,
            fallback_frame_chunk_size=fallback_frame_chunk_size,
            fallback_attention_window_size=fallback_attention_window_size,
            fallback_history_stream_visibility=fallback_history_stream_visibility,
        )


def sample_conditioning_mode(
    probs: dict[ModeEnumT, float],
    *,
    enum_cls: type[ModeEnumT],
    device: torch.device,
    error_label: str,
) -> ModeEnumT:
    """Sample one enum-backed conditioning mode consistently across ranks."""

    modes = tuple(enum_cls)
    weights = torch.tensor(
        [float(probs.get(mode, 0.0)) for mode in modes],
        device=device,
        dtype=torch.float32,
    )
    if float(weights.sum().item()) <= 0.0:
        raise ValueError(
            f"{error_label} probabilities must have positive total weight."
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            index_tensor = torch.multinomial(weights, num_samples=1).to(
                device=device,
                dtype=torch.long,
            )
        else:
            index_tensor = torch.zeros(1, device=device, dtype=torch.long)
        torch.distributed.broadcast(index_tensor, src=0)
    else:
        index_tensor = torch.multinomial(weights, num_samples=1).to(
            device=device,
            dtype=torch.long,
        )
    return modes[int(index_tensor.item())]


def resolve_dynamics_training_plan(
    *,
    program: VideoActionProgram | str,
    sample_metadata: SampleConstructionMetadata | None,
    device: torch.device,
    sample_plan: DynamicsSamplePlan | None = None,
) -> DynamicsTrainingPlan | None:
    """Compile program and routed metadata into one architecture-neutral plan."""

    resolved_program = VideoActionProgram(program)
    if sample_plan is None:
        sample_plan = resolve_dynamics_sample_plan(
            program=resolved_program,
            sample_metadata=sample_metadata,
        )
    elif sample_plan.program != resolved_program:
        raise ValueError(
            "Dynamics sample plan program mismatch, "
            f"got plan={sample_plan.program.value!r} and runtime={resolved_program.value!r}."
        )
    if sample_plan is None:
        return None

    objective = sample_plan.objective
    if objective is None:
        if resolved_program != VideoActionProgram.GENERALIST_JOINT_DENOISING:
            raise AssertionError(
                f"Dynamics program {resolved_program.value!r} did not resolve an objective."
            )
        # Pure-joint GJD intentionally bypasses source routing and uses the
        # ordinary planning dataset. Preserve its established categorical draw
        # so this ablation stays checkpoint/golden compatible.
        objective = sample_conditioning_mode(
            {
                mode: float(mode == DynamicsObjective.JOINT)
                for mode in DynamicsObjective
            },
            enum_cls=DynamicsObjective,
            device=device,
            error_label="generalist dynamics training objective",
        )

    return compile_dynamics_training_plan(
        objective=objective,
        routed_objective=sample_plan.routed_objective,
        drop_text_conditioning=sample_plan.drop_text_conditioning,
        source=sample_plan.source,
        sequence=sample_plan.sequence,
    )


def compile_dynamics_training_plan(
    *,
    objective: DynamicsObjective | str,
    routed_objective: DynamicsObjective | None = None,
    drop_text_conditioning: bool | None = None,
    source: str | None = None,
    sequence: ConditionalDynamicsSequenceLayout | None = None,
) -> DynamicsTrainingPlan:
    """Compile an already-selected objective into the shared training plan."""

    semantics = resolve_dynamics_objective_semantics(
        objective,
        drop_text_conditioning=drop_text_conditioning,
    )
    if semantics.is_conditional and sequence is None:
        raise ValueError(
            "Conditional dynamics requires the canonical target-only sequence layout."
        )
    if semantics.is_joint and sequence is not None:
        raise ValueError(
            "Joint dynamics cannot use a target-only conditional sequence."
        )
    return DynamicsTrainingPlan(
        semantics=semantics,
        routed_objective=routed_objective,
        source=source,
        sequence=sequence,
    )


def resolve_dynamics_sample_plan(
    *,
    program: VideoActionProgram | str,
    sample_metadata: SampleConstructionMetadata | None,
) -> DynamicsSamplePlan | None:
    """Resolve and validate sample semantics without consuming model RNG."""

    resolved_program = VideoActionProgram(program)
    routed_objective: DynamicsObjective | None = None
    drop_text_conditioning: bool | None = None
    source: str | None = None
    if sample_metadata is not None:
        routing = sample_metadata.dynamics_routing
        routed_objective = (
            None
            if routing.mode_override is None
            else resolve_dynamics_objective(routing.mode_override)
        )
        drop_text_conditioning = routing.drop_text_conditioning
        source = routing.source

    if (
        supports_dynamics_routing(resolved_program)
        and sample_metadata is not None
        and sample_metadata.is_target_only_conditional_layout
        and (routed_objective is None or not routed_objective.is_conditional)
    ):
        routed_label = (
            "missing" if routed_objective is None else repr(routed_objective.value)
        )
        raise ValueError(
            "The target-only t0-plus-future layout requires an explicit "
            "conditional FDM/IDM route; routed objective is "
            f"{routed_label}."
        )

    fixed_objective = fixed_conditioning_mode_for_program(resolved_program)
    if fixed_objective is not None:
        if routed_objective is None:
            raise ValueError(
                f"`program={resolved_program.value}` requires dynamics-routed sample metadata."
            )
        if routed_objective != fixed_objective:
            raise ValueError(
                f"`program={resolved_program.value}` requires objective "
                f"{fixed_objective.value!r}, but the sample routes {routed_objective.value!r}."
            )
        objective: DynamicsObjective | None = fixed_objective
    elif resolved_program == VideoActionProgram.GENERALIST_JOINT_DENOISING:
        objective = routed_objective
    else:
        if routed_objective is not None:
            raise ValueError(
                "Dynamics-routed sample metadata requires a generalist, forward-dynamics, "
                f"or inverse-dynamics program; got {resolved_program.value!r}."
            )
        if supports_dynamics_routing(
            resolved_program
        ):  # pragma: no cover - enum exhaustiveness
            raise AssertionError(f"Unhandled dynamics program {resolved_program!r}.")
        return None

    sequence = None
    if objective is not None:
        semantics = resolve_dynamics_objective_semantics(
            objective,
            drop_text_conditioning=drop_text_conditioning,
        )
        if semantics.is_conditional:
            if sample_metadata is None:
                raise ValueError(
                    "Conditional dynamics requires target-only sample metadata."
                )
            sequence = sample_metadata.require_target_only_conditional_layout()
    return DynamicsSamplePlan(
        program=resolved_program,
        objective=objective,
        routed_objective=routed_objective,
        drop_text_conditioning=drop_text_conditioning,
        source=source,
        sequence=sequence,
    )


def apply_dynamics_training_plan(
    plan: DynamicsTrainingPlan,
    *,
    clean_video: torch.Tensor,
    noisy_video: torch.Tensor,
    video_targets: torch.Tensor,
    video_timesteps: torch.Tensor,
    video_loss_mask: torch.Tensor,
    clean_action: torch.Tensor,
    noisy_action: torch.Tensor,
    action_targets: torch.Tensor,
    action_timesteps: torch.Tensor,
    action_loss_mask: torch.Tensor | None,
    clean_action_mask: torch.Tensor | None = None,
) -> DynamicsTrainingTensors:
    """Apply one objective without exposing model-backend tensor conventions."""

    semantics = plan.semantics
    if semantics.is_joint:
        return DynamicsTrainingTensors(
            noisy_video=noisy_video,
            video_targets=video_targets,
            video_timesteps=video_timesteps,
            video_loss_mask=video_loss_mask,
            noisy_action=noisy_action,
            action_targets=action_targets,
            action_timesteps=action_timesteps,
            action_loss_mask=action_loss_mask,
        )
    if semantics.clean_action_noisy_slot:
        return DynamicsTrainingTensors(
            noisy_video=noisy_video,
            video_targets=video_targets,
            video_timesteps=video_timesteps,
            video_loss_mask=video_loss_mask,
            noisy_action=clean_noisy_slot_tensor(
                clean_action,
                action_mask=clean_action_mask,
            ),
            action_targets=torch.zeros_like(action_targets),
            action_timesteps=torch.zeros_like(action_timesteps),
            action_loss_mask=zero_loss_mask_like(
                action_loss_mask,
                fallback_like=noisy_action,
            ),
        )
    if semantics.clean_video_noisy_slot:
        return DynamicsTrainingTensors(
            noisy_video=clean_video,
            video_targets=torch.zeros_like(video_targets),
            video_timesteps=torch.zeros_like(video_timesteps),
            video_loss_mask=torch.zeros_like(video_loss_mask),
            noisy_action=noisy_action,
            action_targets=action_targets,
            action_timesteps=action_timesteps,
            action_loss_mask=action_loss_mask,
        )
    raise AssertionError(f"Unhandled dynamics objective {plan.objective!r}.")


def _enum_value(value: StrEnum | str) -> str:
    return value.value if isinstance(value, StrEnum) else str(value)


def resolve_dynamics_objective(
    value: DynamicsObjective | str,
) -> DynamicsObjective:
    """Resolve only the canonical public dynamics-objective vocabulary."""

    raw_value = _enum_value(value)
    try:
        return DynamicsObjective(raw_value)
    except ValueError as exc:
        supported = ", ".join(objective.value for objective in DynamicsObjective)
        raise ValueError(
            f"Unsupported dynamics objective {raw_value!r}. "
            f"Supported objectives: {supported}."
        ) from exc


def resolve_dynamics_rollout_objective(
    *,
    program: VideoActionProgram | str,
    requested_objective: DynamicsObjective | str | None = None,
) -> DynamicsObjective:
    """Resolve one rollout objective under the program-owned contract.

    Omitting the objective selects joint generation for ordinary and GJD
    programs, and selects the configured objective for fixed FDM/IDM programs.
    An explicit conditional objective is accepted only by a dynamics-capable
    program.
    """

    resolved_program = VideoActionProgram(program)
    fixed_objective = fixed_conditioning_mode_for_program(resolved_program)
    if fixed_objective is not None:
        if requested_objective is None:
            return fixed_objective
        requested = resolve_dynamics_objective(requested_objective)
        if requested != fixed_objective:
            raise ValueError(
                f"`program={resolved_program.value}` requires rollout objective "
                f"{fixed_objective.value!r}, but {requested.value!r} was requested."
            )
        return fixed_objective

    requested = (
        DynamicsObjective.JOINT
        if requested_objective is None
        else resolve_dynamics_objective(requested_objective)
    )
    if requested.is_conditional and not supports_dynamics_routing(resolved_program):
        raise ValueError(
            f"Rollout objective {requested.value!r} requires a generalist, "
            "forward-dynamics, or inverse-dynamics program; "
            f"got {resolved_program.value!r}."
        )
    return requested


def resolve_dynamics_rollout_plan(
    *,
    program: VideoActionProgram | str,
    request: DynamicsRolloutRequest | None = None,
) -> DynamicsRolloutPlan:
    """Resolve one backend-independent rollout request."""

    request = request or DynamicsRolloutRequest()
    objective = resolve_dynamics_rollout_objective(
        program=program,
        requested_objective=request.objective,
    )
    if (
        objective != DynamicsObjective.ACTION_CONDITIONED_VIDEO
        and request.clean_action is not None
    ):
        raise ValueError(
            "`clean_action` is only valid for an action-conditioned-video "
            "forward-dynamics rollout."
        )
    if (
        objective != DynamicsObjective.VIDEO_CONDITIONED_ACTION
        and request.clean_video is not None
    ):
        raise ValueError(
            "`clean_video` is only valid for a video-conditioned-action "
            "inverse-dynamics rollout."
        )
    history_action = request.history_action
    if (
        history_action is None
        and objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO
    ):
        history_action = request.clean_action
    return DynamicsRolloutPlan(
        semantics=resolve_dynamics_objective_semantics(objective),
        clean_action=request.clean_action,
        clean_video=request.clean_video,
        history_action=history_action,
    )


def resolve_dynamics_rollout_geometry(
    objective: DynamicsObjectiveSemantics | DynamicsObjective | str,
    *,
    fallback_frame_chunk_size: int,
    fallback_attention_window_size: int,
    fallback_history_stream_visibility: HistoryStreamVisibility | str,
) -> DynamicsRolloutGeometry:
    """Compile shared objective semantics into one backend-neutral geometry."""

    semantics = (
        objective
        if isinstance(objective, DynamicsObjectiveSemantics)
        else resolve_dynamics_objective_semantics(objective)
    )
    return DynamicsRolloutGeometry(
        frame_chunk_size=semantics.rollout_chunk_size_frames(
            fallback_chunk_size=fallback_frame_chunk_size,
        ),
        attention_window_size=semantics.attention_window_size(
            fallback_window_size=fallback_attention_window_size,
        ),
        history_stream_visibility=semantics.resolve_history_stream_visibility(
            fallback=fallback_history_stream_visibility,
        ),
        conditional_history_policy=semantics.conditional_history_policy,
    )


def resolve_dynamics_objective_semantics(
    objective: DynamicsObjective | str,
    *,
    drop_text_conditioning: bool | None = None,
) -> DynamicsObjectiveSemantics:
    """Resolve shared joint, FDM, or IDM video/action semantics.

    Conditional training keeps the data sampler's chunk geometry. Recurrent
    diagnostic rollouts use one generated latent frame per model call and one
    local clean video-history anchor.
    """

    resolved = resolve_dynamics_objective(objective)
    resolved_drop_text = resolved.is_conditional
    if (
        drop_text_conditioning is not None
        and bool(drop_text_conditioning) != resolved_drop_text
    ):
        if resolved_drop_text:
            raise ValueError(
                "Conditional dynamics always removes task text; routed metadata "
                "cannot disable it."
            )
        raise ValueError(
            "Joint dynamics always preserves task text; routed metadata cannot "
            "enable text removal."
        )
    return DynamicsObjectiveSemantics(objective=resolved)


def is_conditional_dynamics_objective(
    objective: DynamicsObjective | str,
) -> bool:
    return resolve_dynamics_objective_semantics(objective).is_conditional


def dynamics_objective_attention_window_size(
    objective: DynamicsObjective | str,
    *,
    fallback_window_size: int,
) -> int:
    return resolve_dynamics_objective_semantics(objective).attention_window_size(
        fallback_window_size=fallback_window_size
    )


def dynamics_objective_rollout_chunk_size(
    objective: DynamicsObjective | str,
    *,
    fallback_chunk_size: int,
) -> int:
    return resolve_dynamics_objective_semantics(objective).rollout_chunk_size_frames(
        fallback_chunk_size=fallback_chunk_size
    )
